from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from homymoly.experiments.identifiable_maps import (
    ABLATIONS,
    DegreeMaps,
    IdentifiableTypedMapDataset,
    IdentifiableTypedMapModel,
    LossWeights,
    build_annulus_map_system,
    compute_identifiable_losses,
    cone_soft_betti_loss,
    decode_ordered_markers,
    loss_weights_for_ablation,
    mapping_cone_boundaries,
)
from homymoly.topology import ChainComplex, ChainMap, cone_betti_numbers


def test_dihedral_basis_and_graph_markers_are_exact_and_identifiable() -> None:
    system = build_annulus_map_system(6)
    assert (system.num_vertices, system.num_edges, system.num_faces) == (12, 18, 6)
    assert torch.equal(
        system.boundary_1 @ system.boundary_2,
        torch.zeros((12, 6), dtype=system.boundary_1.dtype),
    )
    base_complex = ChainComplex(
        (system.num_vertices, system.num_edges, system.num_faces),
        (system.boundary_1, system.boundary_2),
    )
    assert base_complex.betti_numbers() == (1, 1, 0)
    dataset = IdentifiableTypedMapDataset(12, seed=41, sectors=6)
    seen: set[int] = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        observation = dataset.graph_observation(sample)
        assert set(observation) == {"node_features", "edge_features"}
        decoded = int(decode_ordered_markers(observation["node_features"], system))
        declared = int(torch.as_tensor(sample["transformation"]))
        assert decoded == declared
        seen.add(decoded)
        maps = DegreeMaps(*(degree[decoded] for degree in dataset.system.basis))
        torch.testing.assert_close(
            torch.as_tensor(sample["target_degree_zero"]),
            maps.degree_zero @ torch.as_tensor(sample["source_degree_zero"]),
        )
        torch.testing.assert_close(
            torch.as_tensor(sample["target_degree_two"]),
            maps.degree_two @ torch.as_tensor(sample["source_degree_two"]),
        )
        torch.testing.assert_close(
            torch.as_tensor(sample["target_sheaf_angle"]),
            maps.degree_one @ torch.as_tensor(sample["source_sheaf_angle"]),
        )
    assert seen == set(range(system.num_transformations))

    residual_one = torch.einsum(
        "ve,gej->gvj", system.boundary_1, system.basis.degree_one
    ) - torch.einsum("gvi,ie->gve", system.basis.degree_zero, system.boundary_1)
    residual_two = torch.einsum(
        "ef,gfj->gej", system.boundary_2, system.basis.degree_two
    ) - torch.einsum("gei,if->gef", system.basis.degree_one, system.boundary_2)
    assert float(residual_one.abs().max()) == 0.0
    assert float(residual_two.abs().max()) == 0.0


def test_dihedral_basis_is_signed_orthogonal_and_closed_in_all_degrees() -> None:
    system = build_annulus_map_system(6)
    identities = tuple(
        torch.eye(degree.shape[-1], dtype=degree.dtype) for degree in system.basis
    )
    for basis_index in range(system.num_transformations):
        for degree, identity in zip(system.basis, identities, strict=True):
            matrix = degree[basis_index]
            assert torch.equal(matrix.mT @ matrix, identity)

    for left in range(system.num_transformations):
        for right in range(system.num_transformations):
            common_matches = set(range(system.num_transformations))
            for degree in system.basis:
                product = degree[left] @ degree[right]
                degree_matches = {
                    index
                    for index in range(system.num_transformations)
                    if torch.equal(product, degree[index])
                }
                common_matches &= degree_matches
            assert len(common_matches) == 1


def test_cone_and_rtd_signals_are_constant_across_the_whole_hypothesis_space() -> None:
    """Lock the structural claim in docs/18-paper.md section 6.3.

    Every basis map is a signed permutation, so every candidate is an
    isomorphism of chain complexes and an isometry on signals. Cone acyclicity
    and any distance-based divergence therefore take the same value on all
    twelve hypotheses and carry zero information about which one was planted.
    This is why ``cone_only`` and ``rtd_only`` sit at chance: not an
    optimization failure, but a degenerate objective.
    """

    system = build_annulus_map_system(6)
    complex_ = ChainComplex(
        (system.num_vertices, system.num_edges, system.num_faces),
        (system.boundary_1, system.boundary_2),
    )

    # Every candidate has an acyclic cone, so acyclicity cannot discriminate.
    acyclic = tuple(
        cone_betti_numbers(
            ChainMap(
                complex_,
                complex_,
                DegreeMaps(*(degree[index] for degree in system.basis)),
            )
        )
        for index in range(system.num_transformations)
    )
    assert set(acyclic) == {(0, 0, 0, 0)}

    # Every candidate preserves pairwise distances, so the paired dissimilarity
    # matrices RTD consumes are identical and the divergence cannot discriminate.
    generator = torch.Generator().manual_seed(20260823)
    signals = torch.randn(
        (48, system.num_edges), generator=generator, dtype=torch.float64
    )
    source_distances = torch.cdist(signals, signals)
    for index in range(system.num_transformations):
        mapped = signals @ system.basis.degree_one[index].to(torch.float64).mT
        torch.testing.assert_close(
            torch.cdist(mapped, mapped), source_distances, atol=1e-6, rtol=0.0
        )


def test_uniform_mixture_has_nontrivial_cone_and_proxy_separation() -> None:
    system = build_annulus_map_system(6)
    complex_ = ChainComplex(
        (system.num_vertices, system.num_edges, system.num_faces),
        (system.boundary_1, system.boundary_2),
    )
    uniform_maps = DegreeMaps(*(degree.mean(dim=0) for degree in system.basis))
    uniform_chain_map = ChainMap(complex_, complex_, uniform_maps)
    assert cone_betti_numbers(uniform_chain_map) == (0, 1, 1, 0)

    batched_uniform = DegreeMaps(*(degree.unsqueeze(0) for degree in uniform_maps))
    pure = DegreeMaps(*(degree[:1] for degree in system.basis))
    uniform_proxy = cone_soft_betti_loss(
        system.boundary_1,
        system.boundary_2,
        batched_uniform,
        temperature=0.05,
    )
    pure_proxy = cone_soft_betti_loss(
        system.boundary_1,
        system.boundary_2,
        pure,
        temperature=0.05,
    )
    assert float(uniform_proxy) > 0.05
    assert float(uniform_proxy) > float(pure_proxy) + 0.05


def test_wrong_nonuniform_mixture_has_finite_nonzero_cone_gradient() -> None:
    system = build_annulus_map_system(6)
    logits = torch.linspace(
        -1.0,
        1.0,
        system.num_transformations,
        dtype=torch.float64,
        requires_grad=True,
    )
    weights = torch.softmax(logits, dim=0)
    maps = DegreeMaps(
        *(torch.einsum("g,gij->ij", weights, degree) for degree in system.basis)
    )
    batched = DegreeMaps(*(degree.unsqueeze(0) for degree in maps))
    loss = cone_soft_betti_loss(
        system.boundary_1,
        system.boundary_2,
        batched,
        temperature=0.05,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(torch.linalg.vector_norm(logits.grad)) > 1e-6


def test_model_uses_only_graph_inputs_and_every_relaxed_map_is_a_chain_map() -> None:
    system = build_annulus_map_system(6, dtype=torch.float32)
    dataset = IdentifiableTypedMapDataset(8, seed=12)
    node = torch.stack(
        [torch.as_tensor(dataset[index]["node_features"]) for index in range(8)]
    )
    edge = torch.stack(
        [torch.as_tensor(dataset[index]["edge_features"]) for index in range(8)]
    )
    model = IdentifiableTypedMapModel(system, hidden_dim=32)
    output = model(node, edge)
    first, second = model.residuals(output.maps)
    assert float(first.detach().abs().max()) < 1e-6
    assert float(second.detach().abs().max()) < 1e-6

    # There is no target-view argument: changing held-out targets cannot alter
    # a forward result for fixed graph observations.
    repeated = model(node, edge)
    torch.testing.assert_close(output.logits, repeated.logits)
    assert output.target_degree_two.shape == (8, system.num_faces)
    assert output.target_sheaf_angle.shape == (8, system.num_edges)


def test_three_term_cone_squares_to_the_two_declared_residuals() -> None:
    system = build_annulus_map_system(6)
    maps = DegreeMaps(*(degree[:3] for degree in system.basis))
    d1, d2, d3 = mapping_cone_boundaries(system.boundary_1, system.boundary_2, maps)
    first = system.boundary_1 @ maps.degree_one - maps.degree_zero @ system.boundary_1
    second = system.boundary_2 @ maps.degree_two - maps.degree_one @ system.boundary_2
    torch.testing.assert_close((d1 @ d2)[:, :, -system.num_edges :], first)
    torch.testing.assert_close((d2 @ d3)[:, : system.num_edges], second)

    complex_ = ChainComplex(
        (system.num_vertices, system.num_edges, system.num_faces),
        (system.boundary_1, system.boundary_2),
    )
    for index in range(3):
        chain_map = ChainMap(
            complex_, complex_, tuple(degree[index] for degree in system.basis)
        )
        assert cone_betti_numbers(chain_map) == (0, 0, 0, 0)


def test_ablation_presets_are_mutually_explicit_and_combined_backpropagates() -> None:
    expected_nonzero = {
        "task_only": {"task"},
        "reconstruction_only": {"reconstruction", "cell", "sheaf"},
        "task_reconstruction": {"task", "reconstruction", "cell", "sheaf"},
        "task_reconstruction_cone": {"task", "reconstruction", "cell", "sheaf", "cone"},
        "task_reconstruction_rtd": {"task", "reconstruction", "cell", "sheaf", "rtd"},
        "cone_only": {"cone"},
        "rtd_only": {"rtd"},
    }
    for ablation, expected in expected_nonzero.items():
        weights = loss_weights_for_ablation(ablation)  # type: ignore[arg-type]
        assert {
            name for name, value in weights.as_dict().items() if value > 0
        } == expected
    combined = LossWeights(1.0, 1.0, 0.25, 0.25, 0.1, 0.25)
    assert loss_weights_for_ablation("combined", combined=combined) == combined
    assert set(ABLATIONS) == {*expected_nonzero, "combined"}

    system = build_annulus_map_system(6, dtype=torch.float32)
    dataset = IdentifiableTypedMapDataset(12, seed=9)
    batch: dict[str, torch.Tensor | list[str]] = {}
    for key in dataset[0]:
        values = [dataset[index][key] for index in range(12)]
        batch[key] = (
            [str(value) for value in values]
            if key == "sample_id"
            else torch.stack([torch.as_tensor(value) for value in values])
        )
    model = IdentifiableTypedMapModel(system, hidden_dim=32)
    output = model(
        torch.as_tensor(batch["node_features"]),
        torch.as_tensor(batch["edge_features"]),
    )
    objective, terms = compute_identifiable_losses(
        model, output, batch, combined, cone_temperature=0.05, rtd_entities=8
    )
    assert set(terms) == {"task", "reconstruction", "cell", "sheaf", "cone", "rtd"}
    changed_tail = {
        key: value.clone() if isinstance(value, torch.Tensor) else list(value)
        for key, value in batch.items()
    }
    for key in (
        "target_degree_zero",
        "target_degree_one",
        "target_degree_two",
        "target_sheaf_angle",
    ):
        tensor = torch.as_tensor(changed_tail[key])
        tensor[8:] = tensor[8:] + 100.0
    _, changed_terms = compute_identifiable_losses(
        model,
        output,
        changed_tail,
        combined,
        cone_temperature=0.05,
        rtd_entities=8,
    )
    torch.testing.assert_close(terms["rtd"], changed_terms["rtd"])
    assert torch.isfinite(objective)
    objective.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.parametrize(
    "ablation",
    [
        "task_only",
        "task_reconstruction_cone",
        "task_reconstruction_rtd",
        "cone_only",
        "rtd_only",
    ],
)
def test_cpu_cli_writes_provenance_predictions_and_exact_cone_results(
    tmp_path: Path, ablation: str
) -> None:
    output = tmp_path / ablation
    config = {
        "experiment": {"seed": 17, "device": "cpu", "deterministic": True},
        "data": {
            "sectors": 4,
            "train_samples": 12,
            "validation_samples": 8,
            "test_samples": 8,
            "noise_std": 0.01,
        },
        "model": {"hidden_dim": 16, "dropout": 0.0, "map_temperature": 1.0},
        "training": {
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "early_stopping_patience": 1,
            "minimum_improvement": 0.0,
            "num_workers": 0,
        },
        "loss": {
            "ablation": ablation,
            "cone_temperature": 0.05,
            "rtd_training_entities": 8,
            "combined_weights": {
                "task": 1.0,
                "reconstruction": 1.0,
                "cell": 0.25,
                "sheaf": 0.25,
                "cone": 0.1,
                "rtd": 0.25,
            },
        },
        "evaluation": {
            "map_tolerance": 1e-5,
            "rank_atol": 1e-7,
            "exact_rtd_entities": 6,
            "exact_rtd_max_dim": 0,
        },
        "output": {"directory": str(output)},
    }
    config_path = tmp_path / f"{ablation}.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "src")
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/train_identifiable_maps.py"),
            "--config",
            str(config_path),
            "--seed",
            "23",
            "--ablation",
            ablation,
            "--output",
            str(output),
        ],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["ablation"] == ablation
    assert summary["seed"] == 23
    effective = yaml.safe_load((output / "effective_config.yaml").read_text())
    assert effective["experiment"]["seed"] == 23
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["seed"] == 23
    assert len(provenance["code_fingerprint"]) == 64
    for flag in ("--seed", "--ablation", "--output"):
        assert flag in provenance["command"]
    assert provenance["cublas_workspace_config"] == ":4096:8"
    assert summary["test"]["examples"] == 8
    assert summary["engineering_recovery_gate"]["applicable"] is False
    assert summary["engineering_recovery_gate"]["passed"] is None
    assert summary["test"]["soft_chain_residual_max"] <= 1e-5
    assert sum(summary["test"]["hard_cone_betti_histogram"].values()) == 8
    assert len((output / "test_predictions.jsonl").read_text().splitlines()) == 8
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert "summary.json" in manifest["artifacts"]
    assert (output / "provenance.json").is_file()
    if ablation == "task_only":
        benchmark_path = tmp_path / "benchmark.json"
        benchmark = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts/benchmark_identifiable_maps.py"),
                "--config",
                str(output / "effective_config.yaml"),
                "--checkpoint",
                str(output / "checkpoint.pt"),
                "--output",
                str(benchmark_path),
                "--warmup",
                "1",
                "--iterations",
                "2",
                "--batch-size",
                "4",
                "--device",
                "cpu",
            ],
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert benchmark.returncode == 0, benchmark.stderr
        benchmark_payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
        assert benchmark_payload["status"] == "completed"
        assert benchmark_payload["ablation"] == "task_only"
        assert benchmark_payload["seed"] == 23
        assert benchmark_payload["chain_residual_max"] <= 1e-5
        assert benchmark_payload["iterations"] == 2

        mismatched = yaml.safe_load(
            (output / "effective_config.yaml").read_text(encoding="utf-8")
        )
        mismatched["training"]["batch_size"] += 1
        mismatch_path = tmp_path / "mismatched-effective-config.yaml"
        mismatch_path.write_text(yaml.safe_dump(mismatched), encoding="utf-8")
        rejected = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts/benchmark_identifiable_maps.py"),
                "--config",
                str(mismatch_path),
                "--checkpoint",
                str(output / "checkpoint.pt"),
                "--output",
                str(tmp_path / "must-not-exist.json"),
                "--warmup",
                "0",
                "--iterations",
                "1",
                "--device",
                "cpu",
            ],
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert rejected.returncode != 0
        assert "does not exactly match" in rejected.stderr
