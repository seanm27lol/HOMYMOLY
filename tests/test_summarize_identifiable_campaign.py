from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_identifiable_campaign.py"
SPEC = importlib.util.spec_from_file_location("summarize_identifiable_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

PROJECT_ROOT = Path(__file__).parents[1]
TRAIN_RUNNER = PROJECT_ROOT / "scripts" / "train_identifiable_maps.py"
MAP_MODULE = PROJECT_ROOT / "src" / "homymoly" / "experiments" / "identifiable_maps.py"
SMALL_SPLITS = {"train": 40, "validation": 20, "test": 20}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _source_config(path: Path) -> dict[str, Any]:
    config = {
        "experiment": {"seed": 20260821, "device": "cuda", "deterministic": True},
        "data": {
            "sectors": 6,
            "train_samples": SMALL_SPLITS["train"],
            "validation_samples": SMALL_SPLITS["validation"],
            "test_samples": SMALL_SPLITS["test"],
            "noise_std": 0.05,
        },
        "model": {"hidden_dim": 32, "dropout": 0.0, "map_temperature": 0.5},
        "training": {
            "epochs": 3,
            "batch_size": 48,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "early_stopping_patience": 3,
            "minimum_improvement": 0.000001,
            "num_workers": 0,
        },
        "loss": {
            "ablation": "combined",
            "cone_temperature": 0.05,
            "rtd_training_entities": 48,
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
            "map_tolerance": 0.00001,
            "rank_atol": 0.0000001,
            "exact_rtd_entities": 4,
            "exact_rtd_max_dim": 1,
        },
        "output": {"directory": "unused"},
    }
    path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return config


def _correct_count(ablation: str, _seed_index: int) -> int:
    return {
        "task_only": 8,
        "reconstruction_only": 7,
        "task_reconstruction": 20,
        "task_reconstruction_cone": 20,
        "task_reconstruction_rtd": 20,
        "cone_only": 2,
        "rtd_only": 3,
        "combined": 20,
    }[ablation]


def _map_error(ablation: str, seed_index: int) -> float:
    return {
        "task_only": 0.3,
        "reconstruction_only": 0.18,
        "task_reconstruction": 0.0008,
        "task_reconstruction_cone": 0.0007,
        "task_reconstruction_rtd": 0.00075,
        "cone_only": 0.8,
        "rtd_only": 0.7,
        "combined": 0.0005 + 0.00005 * seed_index,
    }[ablation]


def _refresh_manifest(run_directory: Path, artifact_name: str) -> None:
    manifest_path = run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = run_directory / artifact_name
    manifest["artifacts"][artifact_name] = {
        "sha256": _sha256(artifact),
        "bytes": artifact.stat().st_size,
    }
    _json(manifest_path, manifest)


def _write_run(
    run_directory: Path,
    source_path: Path,
    source: dict[str, Any],
    *,
    seed: int,
    ablation: str,
    seed_index: int,
) -> None:
    run_directory.mkdir(parents=True)
    effective = copy.deepcopy(source)
    effective["experiment"]["seed"] = seed
    effective["loss"]["ablation"] = ablation
    effective["output"]["directory"] = str(run_directory.resolve())
    effective_path = run_directory / "effective_config.yaml"
    effective_path.write_text(
        yaml.safe_dump(effective, sort_keys=True), encoding="utf-8"
    )

    correct_count = _correct_count(ablation, seed_index)
    map_error = _map_error(ablation, seed_index)
    split_seed = seed + 3011
    records = []
    for index in range(SMALL_SPLITS["test"]):
        target = index % 12
        correct = index < correct_count
        records.append(
            {
                "sample_id": f"identifiable-{split_seed}-{index:07d}",
                "target_transformation": target,
                "predicted_transformation": target if correct else (target + 1) % 12,
                "analytic_marker_transformation": target,
                "correct": correct,
                "confidence": 0.8,
                "soft_chain_residual_max": 1e-7,
                "hard_chain_residual_max": 0.0,
                "soft_cone_betti": [0, 0, 0, 0],
                "hard_cone_betti": [0, 0, 0, 0],
                "map_mse": map_error,
                "cell_face_correct": True,
            }
        )
    predictions_path = run_directory / "test_predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    code_fingerprint = MODULE._code_fingerprint(PROJECT_ROOT, TRAIN_RUNNER)
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    provenance = {
        "schema_version": 1,
        "created_unix": 1.0,
        "command": [
            "/usr/bin/python",
            str(TRAIN_RUNNER),
            "--config",
            str(source_path.resolve()),
            "--seed",
            str(seed),
            "--ablation",
            ablation,
            "--output",
            str(run_directory.resolve()),
        ],
        "code_fingerprint": code_fingerprint,
        "working_directory": str(PROJECT_ROOT.resolve()),
        "output_directory": str(run_directory.resolve()),
        "git": {
            "revision": revision,
            "branch": "test",
            "status_porcelain": [],
        },
        "files": {
            "input_config": {
                "path": str(source_path.resolve()),
                "sha256": _sha256(source_path),
            },
            "runner": {"path": str(TRAIN_RUNNER), "sha256": _sha256(TRAIN_RUNNER)},
            "module": {"path": str(MAP_MODULE), "sha256": _sha256(MAP_MODULE)},
        },
        "python": "3.12.3",
        "platform": "Linux-aarch64",
        "torch": "2.13.0+cu130",
        "numpy": "2.5.2",
        "pyyaml": "6.0.3",
        "device": "cuda",
        "seed": seed,
        "cuda": "13.0",
        "gpu": "NVIDIA GB10",
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
    }
    _json(run_directory / "provenance.json", provenance)
    torch.save(
        {
            "schema_version": 1,
            "model_state_dict": {"fixture.weight": torch.ones(1)},
            "config": effective,
            "best_epoch": 1,
            "ablation": ablation,
        },
        run_directory / "checkpoint.pt",
    )
    _json(
        run_directory / "history.json",
        [{"epoch": 1, "train": {"objective": 1.0}, "validation": {"objective": 0.5}}],
    )

    test = {
        "examples": SMALL_SPLITS["test"],
        "transformation_accuracy": correct_count / SMALL_SPLITS["test"],
        "analytic_marker_decoder_accuracy": 1.0,
        "chance_baselines": {
            "transformation_accuracy": 1.0 / 12.0,
            "cell_face_accuracy": 1.0 / 6.0,
        },
        "cell_face_accuracy": 1.0,
        "map_mse": map_error,
        "degree_zero_mse": map_error * 2,
        "degree_one_mse": map_error * 3,
        "degree_two_mse": map_error * 4,
        "sheaf_transport_frobenius_mse": map_error * 5,
        "soft_chain_residual_max": 1e-7,
        "hard_chain_residual_max": 0.0,
        "map_tolerance": 1e-5,
        "cone_rank_oracle": {
            "method": "fixed-tolerance-float64-numerical-rank",
            "rank_atol": 1e-7,
            "map_atol": 1e-5,
        },
        "soft_cone_betti_histogram": {"[0,0,0,0]": SMALL_SPLITS["test"]},
        "hard_cone_betti_histogram": {"[0,0,0,0]": SMALL_SPLITS["test"]},
        "exact_rtd": {
            "entities": 4,
            "normalization": "full-matrix-q0.9",
            "max_dim": 1,
            "half_symmetric_rtd_by_degree": [0.0, map_error],
            "srtd_by_degree": [0.0, map_error * 2],
        },
    }
    gate_checks = {
        "transformation_accuracy": test["transformation_accuracy"] >= 0.95,
        "cell_face_accuracy": test["cell_face_accuracy"] >= 0.95,
        "map_mse": test["map_mse"] <= 0.001,
        "soft_chain_residual": test["soft_chain_residual_max"] <= 1e-5,
        "hard_chain_residual": test["hard_chain_residual_max"] <= 1e-5,
        "hard_cone_acyclic_fraction": True,
    }
    gate_applicable = ablation in {"task_reconstruction", "combined"}
    summary = {
        "schema_version": 1,
        "status": "completed",
        "experiment": "identifiable-graph-only-typed-maps",
        "scope": "finite dihedral maps on one cellular annulus; no categorical-equivalence claim",
        "ablation": ablation,
        "loss_weights": MODULE._expected_loss_weights(effective, ablation),
        "rtd_training_entities": 48,
        "seed": seed,
        "device": "cuda",
        "best_epoch": 1,
        "epochs_completed": 1,
        "best_validation_objective": 0.5,
        "engineering_recovery_gate": {
            "applicable": gate_applicable,
            "thresholds": MODULE.ENGINEERING_GATE,
            "checks": gate_checks,
            "passed": all(gate_checks.values()) if gate_applicable else None,
            "hard_cone_acyclic_fraction": 1.0,
            "status": "pre-specified development-informed engineering gate",
        },
        "elapsed_seconds": 2.0,
        "dataset": {
            "topology": "cellular_annulus",
            "sectors": 6,
            "betti_numbers": [1, 1, 0],
            "vertices": 12,
            "edges": 18,
            "faces": 6,
            "transformations": 12,
            "split_samples": SMALL_SPLITS,
            "split_seeds": {
                "train": seed + 1009,
                "validation": seed + 2017,
                "test": seed + 3011,
            },
            "graph_input_channels": {
                "node": [
                    "source_degree_zero",
                    "anchor_marker",
                    "successor_marker",
                    "noise",
                ],
                "edge": ["source_degree_one", "source_sheaf_angle", "noise"],
            },
            "held_out_targets": [
                "oriented_degree_zero",
                "oriented_degree_one",
                "oriented_cell_degree_two",
                "cell_activity",
                "rank_two_sheaf_transport",
            ],
        },
        "declared_chain_map_equations": [
            "B1 @ F1 = F0 @ B1",
            "B2 @ F2 = F1 @ B2",
        ],
        "basis_chain_residual_max": 0.0,
        "test": test,
        "environment": {"peak_cuda_memory_bytes": 4096},
    }
    _json(run_directory / "summary.json", summary)
    artifacts = {}
    for name in MODULE._REQUIRED_ARTIFACTS:
        artifact = run_directory / name
        artifacts[name] = {
            "sha256": _sha256(artifact),
            "bytes": artifact.stat().st_size,
        }
    _json(
        run_directory / "manifest.json",
        {"schema_version": 1, "source_git_revision": revision, "artifacts": artifacts},
    )


def _campaign(tmp_path: Path) -> tuple[Path, Path, str]:
    source_path = tmp_path / "gb10-full.yaml"
    source = _source_config(source_path)
    root = tmp_path / "campaign"
    for seed_index, seed in enumerate(MODULE.FROZEN_SEEDS):
        for ablation in MODULE.FROZEN_ABLATIONS:
            _write_run(
                root / f"seed-{seed}" / ablation,
                source_path,
                source,
                seed=seed,
                ablation=ablation,
                seed_index=seed_index,
            )
    return root, source_path, _sha256(source_path)


def _summarize(root: Path, source: Path, source_hash: str) -> dict[str, Any]:
    return MODULE.summarize(
        root,
        source,
        expected_source_sha256=source_hash,
        expected_split_samples=SMALL_SPLITS,
        project_root=PROJECT_ROOT,
    )


def test_strict_forty_run_summary_uses_all_paired_seeds(tmp_path: Path) -> None:
    root, source, source_hash = _campaign(tmp_path)

    report = _summarize(root, source, source_hash)

    assert report["validation"] == {
        "status": "passed",
        "expected_runs": 40,
        "included_runs": 40,
        "excluded_runs": 0,
        "missing_runs": [],
        "replaced_runs": [],
        "manifest_hashes_and_sizes_verified": True,
        "effective_configs_verified": True,
        "source_config_hash_verified": True,
        "paired_sample_ids_and_targets_verified": True,
        "per_run_sample_ids_unique_and_complete": True,
        "cuda_gb10_execution_verified": True,
        "fixed_map_tolerance_verified": True,
        "checkpoint_identity_verified": True,
        "clean_git_status_verified": True,
        "committed_revision_verified": True,
    }
    assert len(report["runs"]) == 40
    descriptive = report["paired_contrasts"]["combined_minus_task_reconstruction"]
    assert descriptive["role"] == (
        "descriptive_unadjusted_structural_plus_cone_plus_rtd"
    )
    accuracy = descriptive["endpoints"]["transformation_accuracy"]
    assert accuracy["mean_difference"] == 0.0
    assert accuracy["sensitivity_sign_test"]["ties"] == 5
    assert len(accuracy["pairs"]) == 5
    assert (
        report["by_ablation"]["combined"]["endpoints"]["map_mse"][
            "sample_standard_deviation"
        ]
        > 0
    )
    assert (
        report["qualitative_identifiability_controls"]["cone_only"][
            "test_examples_across_runs"
        ]
        == 100
    )
    assert report["engineering_recovery_gate"]["passed"] is True
    assert report["engineering_recovery_gate"]["applicable_runs"] == 10
    assert report["analysis_provenance"]["assumptions"] == {
        "paired_seed_count": 5,
        "student_t_degrees_of_freedom": 4,
        "minimum_attainable_two_sided_sign_test_pvalue_without_ties": 0.0625,
        "multiplicity_adjustment": "none",
        "inferential_structural_benefit_claim": False,
    }
    assert len(report["analysis_provenance"]["protocol"]["sha256"]) == 64
    assert len(report["analysis_provenance"]["summarizer"]["sha256"]) == 64
    assert "synthetic" in report["scope"]


def test_rejects_duplicate_or_replaced_sample_id_even_with_refreshed_manifest(
    tmp_path: Path,
) -> None:
    root, source, source_hash = _campaign(tmp_path)
    run = root / "seed-20260821" / "combined"
    predictions = run / "test_predictions.jsonl"
    rows = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
    ]
    rows[1]["sample_id"] = rows[0]["sample_id"]
    predictions.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _refresh_manifest(run, "test_predictions.jsonl")

    with pytest.raises(RuntimeError, match="duplicate sample IDs"):
        _summarize(root, source, source_hash)


def test_rejects_paired_target_replacement_even_with_internally_valid_record(
    tmp_path: Path,
) -> None:
    root, source, source_hash = _campaign(tmp_path)
    run = root / "seed-20260821" / "combined"
    predictions = run / "test_predictions.jsonl"
    rows = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["target_transformation"] = 1
    rows[0]["predicted_transformation"] = 1
    rows[0]["analytic_marker_transformation"] = 1
    predictions.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _refresh_manifest(run, "test_predictions.jsonl")

    with pytest.raises(RuntimeError, match="paired target labels differ"):
        _summarize(root, source, source_hash)


def test_rejects_non_gb10_cuda_provenance_even_with_refreshed_manifest(
    tmp_path: Path,
) -> None:
    root, source, source_hash = _campaign(tmp_path)
    run = root / "seed-20260821" / "task_only"
    provenance_path = run / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["device"] = "cpu"
    provenance["gpu"] = None
    _json(provenance_path, provenance)
    _refresh_manifest(run, "provenance.json")

    with pytest.raises(RuntimeError, match="did not execute on CUDA"):
        _summarize(root, source, source_hash)


def test_rejects_dirty_git_provenance_even_with_refreshed_manifest(
    tmp_path: Path,
) -> None:
    root, source, source_hash = _campaign(tmp_path)
    run = root / "seed-20260821" / "task_only"
    provenance_path = run / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["git"]["status_porcelain"] = [" M src/homymoly/metrics/rtd.py"]
    _json(provenance_path, provenance)
    _refresh_manifest(run, "provenance.json")

    with pytest.raises(RuntimeError, match="dirty Git worktree"):
        _summarize(root, source, source_hash)


def test_rejects_replaced_checkpoint_even_with_refreshed_manifest(
    tmp_path: Path,
) -> None:
    root, source, source_hash = _campaign(tmp_path)
    run = root / "seed-20260821" / "combined"
    checkpoint_path = run / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["ablation"] = "task_only"
    torch.save(checkpoint, checkpoint_path)
    _refresh_manifest(run, "checkpoint.pt")

    with pytest.raises(RuntimeError, match="checkpoint ablation mismatch"):
        _summarize(root, source, source_hash)


def test_rejects_extra_run_directory_and_source_hash_change(tmp_path: Path) -> None:
    root, source, source_hash = _campaign(tmp_path)
    (root / "seed-20260821" / "replacement").mkdir()
    with pytest.raises(RuntimeError, match="ablation grid mismatch"):
        _summarize(root, source, source_hash)

    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source config SHA-256 mismatch"):
        _summarize(root, source, source_hash)


def test_exact_sign_test_handles_ties_and_error_direction() -> None:
    result = MODULE._sign_test([-1.0, -2.0, 0.0, 1.0, -3.0], "lower_is_better")
    assert result["nonzero_pairs"] == 4
    assert result["ties"] == 1
    assert result["candidate_favorable"] == 3
    assert result["candidate_unfavorable"] == 1
    assert result["two_sided_pvalue"] == pytest.approx(0.625)
