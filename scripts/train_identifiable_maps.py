#!/usr/bin/env python3
"""Train and evaluate identifiable graph-only typed chain maps."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from homymoly.experiments.identifiable_maps import (
    ABLATIONS,
    Ablation,
    DegreeMaps,
    IdentifiableTypedMapDataset,
    IdentifiableTypedMapModel,
    LossWeights,
    build_annulus_map_system,
    compute_identifiable_losses,
    decode_ordered_markers,
    loss_weights_for_ablation,
    typed_representation,
)
from homymoly.metrics import (
    exact_rtd_by_degree,
    exact_srtd_by_degree,
    pairwise_euclidean_distances,
)
from homymoly.topology import ChainComplex, ChainMap, cone_betti_numbers

SCHEMA_VERSION = 1
ENGINEERING_GATE = {
    "transformation_accuracy_min": 0.95,
    "cell_face_accuracy_min": 0.95,
    "map_mse_max": 1e-3,
    "chain_residual_max": 1e-5,
    "hard_cone_acyclic_fraction_min": 1.0,
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--ablation", choices=ABLATIONS)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="bound samples, epochs, and exact evaluation for a CPU-safe run",
    )
    return parser.parse_args()


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _require_sections(config: dict[str, Any]) -> None:
    required = {
        "experiment",
        "data",
        "model",
        "training",
        "loss",
        "evaluation",
        "output",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"configuration is missing sections: {sorted(missing)}")
    unknown = config.keys() - required
    if unknown:
        raise ValueError(f"configuration has unknown sections: {sorted(unknown)}")
    for section in required:
        _mapping(config[section], section)


def load_config(path: Path) -> dict[str, Any]:
    """Load a strict top-level YAML configuration."""

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    config = _mapping(loaded, "configuration")
    _require_sections(config)
    return config


def apply_cli_overrides(
    config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    experiment = _mapping(effective["experiment"], "experiment")
    training = _mapping(effective["training"], "training")
    loss = _mapping(effective["loss"], "loss")
    model = _mapping(effective["model"], "model")
    evaluation = _mapping(effective["evaluation"], "evaluation")
    data = _mapping(effective["data"], "data")
    output = _mapping(effective["output"], "output")
    if args.device is not None:
        experiment["device"] = args.device
    if args.ablation is not None:
        previous = str(loss["ablation"])
        loss["ablation"] = args.ablation
        if args.output is None and previous != args.ablation:
            output["directory"] = f"{output['directory']}-{args.ablation}"
    if args.seed is not None:
        if args.seed < 0:
            raise ValueError("--seed must be nonnegative")
        experiment["seed"] = args.seed
        if args.output is None:
            output["directory"] = f"{output['directory']}-seed{args.seed}"
    if args.output is not None:
        output["directory"] = str(args.output)
    if args.smoke:
        data["train_samples"] = min(int(data["train_samples"]), 96)
        data["validation_samples"] = min(int(data["validation_samples"]), 48)
        data["test_samples"] = min(int(data["test_samples"]), 48)
        training["epochs"] = min(int(training["epochs"]), 6)
        training["batch_size"] = min(int(training["batch_size"]), 24)
        loss["rtd_training_entities"] = min(
            int(loss["rtd_training_entities"]), int(training["batch_size"])
        )
        training["early_stopping_patience"] = min(
            int(training["early_stopping_patience"]), 6
        )
        model["hidden_dim"] = min(int(model["hidden_dim"]), 64)
        evaluation["exact_rtd_entities"] = min(
            int(evaluation["exact_rtd_entities"]), 12
        )
        experiment["device"] = "cpu" if args.device is None else args.device
        output["directory"] = f"{output['directory']}-smoke"
    effective.update(
        {
            "experiment": experiment,
            "data": data,
            "model": model,
            "training": training,
            "loss": loss,
            "evaluation": evaluation,
            "output": output,
        }
    )
    return effective


def _validate_config(config: dict[str, Any]) -> None:
    experiment = _mapping(config["experiment"], "experiment")
    data = _mapping(config["data"], "data")
    model = _mapping(config["model"], "model")
    training = _mapping(config["training"], "training")
    loss = _mapping(config["loss"], "loss")
    evaluation = _mapping(config["evaluation"], "evaluation")
    output = _mapping(config["output"], "output")
    if int(experiment["seed"]) < 0:
        raise ValueError("experiment.seed must be nonnegative")
    if str(experiment["device"]) not in {"auto", "cpu", "cuda"}:
        raise ValueError("experiment.device must be auto, cpu, or cuda")
    if int(data["sectors"]) < 4:
        raise ValueError("data.sectors must be at least four")
    for key in ("train_samples", "validation_samples", "test_samples"):
        if int(data[key]) <= 0:
            raise ValueError(f"data.{key} must be positive")
    if float(data["noise_std"]) < 0:
        raise ValueError("data.noise_std must be nonnegative")
    if int(model["hidden_dim"]) <= 0 or float(model["map_temperature"]) <= 0:
        raise ValueError("model hidden_dim and map_temperature must be positive")
    if not 0 <= float(model["dropout"]) < 1:
        raise ValueError("model.dropout must lie in [0,1)")
    for key in ("epochs", "batch_size", "early_stopping_patience"):
        if int(training[key]) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if float(training["learning_rate"]) <= 0 or float(training["weight_decay"]) < 0:
        raise ValueError("training rate must be positive and decay nonnegative")
    if str(loss["ablation"]) not in ABLATIONS:
        raise ValueError(f"loss.ablation must be one of {ABLATIONS}")
    if float(loss["cone_temperature"]) <= 0:
        raise ValueError("loss.cone_temperature must be positive")
    rtd_training_entities = loss["rtd_training_entities"]
    if (
        isinstance(rtd_training_entities, bool)
        or not isinstance(rtd_training_entities, int)
        or not 2 <= rtd_training_entities <= int(training["batch_size"])
    ):
        raise ValueError(
            "loss.rtd_training_entities must be an integer in [2, training.batch_size]"
        )
    for key in ("task", "reconstruction", "cell", "sheaf", "cone", "rtd"):
        if float(_mapping(loss["combined_weights"], "loss.combined_weights")[key]) < 0:
            raise ValueError(f"loss.combined_weights.{key} must be nonnegative")
    if float(evaluation["map_tolerance"]) <= 0:
        raise ValueError("evaluation.map_tolerance must be positive")
    if float(evaluation["rank_atol"]) < 0:
        raise ValueError("evaluation.rank_atol must be nonnegative")
    if not 2 <= int(evaluation["exact_rtd_entities"]) <= 64:
        raise ValueError("evaluation.exact_rtd_entities must lie in [2,64]")
    if int(evaluation["exact_rtd_max_dim"]) not in (0, 1):
        raise ValueError("exact_rtd_max_dim is bounded to 0 or 1 in this experiment")
    if not str(output["directory"]):
        raise ValueError("output.directory must be non-empty")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_fingerprint(project_root: Path, runner: Path) -> str:
    """Hash the runner and every executable Python module under homymoly."""

    digest = hashlib.sha256()
    candidates = [runner.resolve()]
    candidates.extend((project_root / "src" / "homymoly").rglob("*.py"))
    for path in sorted(
        candidates, key=lambda item: item.relative_to(project_root).as_posix()
    ):
        digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def _git(command: tuple[str, ...]) -> str | None:
    result = subprocess.run(
        ("git", *command), check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic


def _to_device(
    batch: dict[str, Tensor | list[str]], device: torch.device
) -> dict[str, Tensor | list[str]]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, Tensor)
        else value
        for key, value in batch.items()
    }


def _weights(config: dict[str, Any]) -> tuple[Ablation, LossWeights]:
    loss = _mapping(config["loss"], "loss")
    combined_raw = _mapping(loss["combined_weights"], "loss.combined_weights")
    combined = LossWeights(
        **{name: float(combined_raw[name]) for name in LossWeights.__dataclass_fields__}
    )
    ablation = cast(Ablation, str(loss["ablation"]))
    return ablation, loss_weights_for_ablation(ablation, combined=combined)


def _data_loaders(
    config: dict[str, Any], device: torch.device
) -> tuple[dict[str, IdentifiableTypedMapDataset], dict[str, DataLoader]]:
    experiment = _mapping(config["experiment"], "experiment")
    data = _mapping(config["data"], "data")
    training = _mapping(config["training"], "training")
    seed = int(experiment["seed"])
    split_seeds = {"train": seed + 1009, "validation": seed + 2017, "test": seed + 3011}
    counts = {
        "train": int(data["train_samples"]),
        "validation": int(data["validation_samples"]),
        "test": int(data["test_samples"]),
    }
    datasets = {
        split: IdentifiableTypedMapDataset(
            count,
            seed=split_seeds[split],
            sectors=int(data["sectors"]),
            noise_std=float(data["noise_std"]),
        )
        for split, count in counts.items()
    }
    generator = torch.Generator().manual_seed(seed + 4049)
    workers = int(training["num_workers"])
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=int(training["batch_size"]),
            shuffle=split == "train",
            generator=generator if split == "train" else None,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
            drop_last=False,
        )
        for split, dataset in datasets.items()
    }
    return datasets, loaders


def _epoch(
    model: IdentifiableTypedMapModel,
    loader: DataLoader,
    weights: LossWeights,
    *,
    device: torch.device,
    cone_temperature: float,
    rtd_entities: int,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Counter[str] = Counter()
    examples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            node = torch.as_tensor(batch["node_features"])
            edge = torch.as_tensor(batch["edge_features"])
            output = model(node, edge)
            objective, terms = compute_identifiable_losses(
                model,
                output,
                batch,
                weights,
                cone_temperature=cone_temperature,
                rtd_entities=rtd_entities,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                optimizer.step()
            batch_size = int(node.shape[0])
            examples += batch_size
            totals["objective"] += float(objective.detach()) * batch_size
            for name, value in terms.items():
                totals[name] += float(value.detach()) * batch_size
            target = torch.as_tensor(batch["transformation"])
            totals["correct"] += int((output.logits.argmax(1) == target).sum())
    metrics = {
        name: value / examples for name, value in totals.items() if name != "correct"
    }
    metrics["accuracy"] = totals["correct"] / examples
    return metrics


def _state_on_cpu(model: torch.nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _signature_key(signature: tuple[int, ...]) -> str:
    return "[" + ",".join(str(value) for value in signature) + "]"


def _rotation_matrices(angles: Tensor) -> Tensor:
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    return torch.stack(
        (
            torch.stack((cosine, -sine), dim=-1),
            torch.stack((sine, cosine), dim=-1),
        ),
        dim=-2,
    )


def _exact_evaluation(
    model: IdentifiableTypedMapModel,
    loader: DataLoader,
    dataset: IdentifiableTypedMapDataset,
    *,
    device: torch.device,
    map_tolerance: float,
    rank_atol: float,
    exact_rtd_entities: int,
    exact_rtd_max_dim: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model.eval()
    records: list[dict[str, object]] = []
    predicted_vectors: list[Tensor] = []
    target_vectors: list[Tensor] = []
    metric_sums: Counter[str] = Counter()
    soft_signatures: Counter[str] = Counter()
    hard_signatures: Counter[str] = Counter()
    total = 0
    # Reconstruct the declared linear parameterization in float64 for the
    # discrete rank oracle. Runtime residuals below are still measured on the
    # actual float32 maps used for inference. Merely casting those rounded
    # matrices would retain avoidable ~1e-7 equation error in the cone.
    oracle_system = build_annulus_map_system(
        dataset.system.sectors, dtype=torch.float64
    )
    complex_ = ChainComplex(
        (
            oracle_system.num_vertices,
            oracle_system.num_edges,
            oracle_system.num_faces,
        ),
        (oracle_system.boundary_1, oracle_system.boundary_2),
        dtype=torch.float64,
    )
    with torch.no_grad():
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            output = model(
                torch.as_tensor(batch["node_features"]),
                torch.as_tensor(batch["edge_features"]),
            )
            transformations = torch.as_tensor(batch["transformation"]).long()
            predictions = output.logits.argmax(dim=1)
            analytic_predictions = decode_ordered_markers(
                torch.as_tensor(batch["node_features"]), dataset.system
            ).to(device=device)
            hard_maps = model.hard_maps(output.logits)
            oracle_weights = output.weights.detach().cpu().double()
            oracle_soft_maps = DegreeMaps(
                *(
                    torch.einsum("bg,gij->bij", oracle_weights, degree)
                    for degree in oracle_system.basis
                )
            )
            oracle_hard_maps = DegreeMaps(
                *(
                    degree.index_select(0, predictions.detach().cpu())
                    for degree in oracle_system.basis
                )
            )
            true_maps = DegreeMaps(
                model.basis_zero.index_select(0, transformations),
                model.basis_one.index_select(0, transformations),
                model.basis_two.index_select(0, transformations),
            )
            soft_residuals = model.residuals(output.maps)
            hard_residuals = model.residuals(hard_maps)
            batch_soft_residual = (
                torch.stack(
                    (
                        soft_residuals[0].abs().flatten(1).max(1).values,
                        soft_residuals[1].abs().flatten(1).max(1).values,
                    ),
                    dim=1,
                )
                .max(1)
                .values
            )
            batch_hard_residual = (
                torch.stack(
                    (
                        hard_residuals[0].abs().flatten(1).max(1).values,
                        hard_residuals[1].abs().flatten(1).max(1).values,
                    ),
                    dim=1,
                )
                .max(1)
                .values
            )
            if float(batch_soft_residual.max()) > map_tolerance:
                raise RuntimeError(
                    "soft learned map failed the held-out chain-map tolerance: "
                    f"{float(batch_soft_residual.max()):.3e} > {map_tolerance:.3e}"
                )
            if float(batch_hard_residual.max()) > map_tolerance:
                raise RuntimeError(
                    "decoded learned map failed the held-out chain-map tolerance: "
                    f"{float(batch_hard_residual.max()):.3e} > {map_tolerance:.3e}"
                )

            target_zero = torch.as_tensor(batch["target_degree_zero"])
            target_one = torch.as_tensor(batch["target_degree_one"])
            target_two = torch.as_tensor(batch["target_degree_two"])
            target_sheaf = torch.as_tensor(batch["target_sheaf_angle"])
            target_cell = torch.as_tensor(batch["target_cell_active"])
            predicted = typed_representation(
                output.target_degree_zero.double(),
                output.target_degree_one.double(),
                output.target_degree_two.double(),
                output.target_sheaf_angle.double(),
            ).cpu()
            target_vector = typed_representation(
                target_zero.double(),
                target_one.double(),
                target_two.double(),
                target_sheaf.double(),
            ).cpu()
            predicted_vectors.append(predicted)
            target_vectors.append(target_vector)
            map_mse = (
                (output.maps.degree_zero - true_maps.degree_zero)
                .square()
                .flatten(1)
                .mean(1)
                + (output.maps.degree_one - true_maps.degree_one)
                .square()
                .flatten(1)
                .mean(1)
                + (output.maps.degree_two - true_maps.degree_two)
                .square()
                .flatten(1)
                .mean(1)
            ) / 3.0
            zero_mse = (output.target_degree_zero - target_zero).square().mean(1)
            one_mse = (output.target_degree_one - target_one).square().mean(1)
            two_mse = (output.target_degree_two - target_two).square().mean(1)
            cell_correct = output.target_cell_active.argmax(1) == target_cell.argmax(1)
            transport_frobenius = (
                (
                    _rotation_matrices(output.target_sheaf_angle)
                    - _rotation_matrices(target_sheaf)
                )
                .square()
                .sum(dim=(-1, -2))
                .mean(dim=1)
            )
            confidence = output.weights.max(1).values

            sample_ids = cast(list[str], batch["sample_id"])
            for position, sample_id in enumerate(sample_ids):
                soft = ChainMap(
                    complex_,
                    complex_,
                    tuple(degree[position] for degree in oracle_soft_maps),
                    atol=map_tolerance,
                )
                hard = ChainMap(
                    complex_,
                    complex_,
                    tuple(degree[position] for degree in oracle_hard_maps),
                    atol=map_tolerance,
                )
                soft_signature = cone_betti_numbers(
                    soft, map_atol=map_tolerance, rank_atol=rank_atol
                )
                hard_signature = cone_betti_numbers(
                    hard, map_atol=map_tolerance, rank_atol=rank_atol
                )
                soft_signatures[_signature_key(soft_signature)] += 1
                hard_signatures[_signature_key(hard_signature)] += 1
                records.append(
                    {
                        "sample_id": sample_id,
                        "target_transformation": int(transformations[position]),
                        "predicted_transformation": int(predictions[position]),
                        "analytic_marker_transformation": int(
                            analytic_predictions[position]
                        ),
                        "correct": bool(
                            predictions[position] == transformations[position]
                        ),
                        "confidence": float(confidence[position]),
                        "soft_chain_residual_max": float(batch_soft_residual[position]),
                        "hard_chain_residual_max": float(batch_hard_residual[position]),
                        "soft_cone_betti": list(soft_signature),
                        "hard_cone_betti": list(hard_signature),
                        "map_mse": float(map_mse[position]),
                        "cell_face_correct": bool(cell_correct[position]),
                    }
                )

            batch_size = int(transformations.shape[0])
            total += batch_size
            metric_sums["correct"] += int((predictions == transformations).sum())
            metric_sums["analytic_correct"] += int(
                (analytic_predictions == transformations).sum()
            )
            metric_sums["cell_correct"] += int(cell_correct.sum())
            metric_sums["map_mse"] += float(map_mse.sum())
            metric_sums["degree_zero_mse"] += float(zero_mse.sum())
            metric_sums["degree_one_mse"] += float(one_mse.sum())
            metric_sums["degree_two_mse"] += float(two_mse.sum())
            metric_sums["sheaf_transport_frobenius_mse"] += float(
                transport_frobenius.sum()
            )
            metric_sums["soft_chain_residual_max"] = max(
                metric_sums["soft_chain_residual_max"],
                float(batch_soft_residual.max()),
            )
            metric_sums["hard_chain_residual_max"] = max(
                metric_sums["hard_chain_residual_max"],
                float(batch_hard_residual.max()),
            )

    predicted_all = torch.cat(predicted_vectors)[:exact_rtd_entities]
    target_all = torch.cat(target_vectors)[:exact_rtd_entities]
    predicted_distances = pairwise_euclidean_distances(predicted_all)
    target_distances = pairwise_euclidean_distances(target_all)
    exact_rtd = exact_rtd_by_degree(
        predicted_distances,
        target_distances,
        max_dim=exact_rtd_max_dim,
        normalization="quantile",
        normalization_quantile=0.9,
    )
    exact_srtd = exact_srtd_by_degree(
        predicted_distances,
        target_distances,
        max_dim=exact_rtd_max_dim,
        normalization="quantile",
        normalization_quantile=0.9,
    )
    summary: dict[str, object] = {
        "examples": total,
        "transformation_accuracy": metric_sums["correct"] / total,
        "analytic_marker_decoder_accuracy": metric_sums["analytic_correct"] / total,
        "chance_baselines": {
            "transformation_accuracy": 1.0 / dataset.system.num_transformations,
            "cell_face_accuracy": 1.0 / dataset.system.num_faces,
        },
        "cell_face_accuracy": metric_sums["cell_correct"] / total,
        "map_mse": metric_sums["map_mse"] / total,
        "degree_zero_mse": metric_sums["degree_zero_mse"] / total,
        "degree_one_mse": metric_sums["degree_one_mse"] / total,
        "degree_two_mse": metric_sums["degree_two_mse"] / total,
        "sheaf_transport_frobenius_mse": metric_sums["sheaf_transport_frobenius_mse"]
        / total,
        "soft_chain_residual_max": metric_sums["soft_chain_residual_max"],
        "hard_chain_residual_max": metric_sums["hard_chain_residual_max"],
        "map_tolerance": map_tolerance,
        "cone_rank_oracle": {
            "method": "fixed-tolerance-float64-numerical-rank",
            "rank_atol": rank_atol,
            "map_atol": map_tolerance,
        },
        "soft_cone_betti_histogram": dict(sorted(soft_signatures.items())),
        "hard_cone_betti_histogram": dict(sorted(hard_signatures.items())),
        "exact_rtd": {
            "entities": min(exact_rtd_entities, total),
            "normalization": "full-matrix-q0.9",
            "max_dim": exact_rtd_max_dim,
            "half_symmetric_rtd_by_degree": list(exact_rtd),
            "srtd_by_degree": list(exact_srtd),
        },
    }
    return summary, records


def _environment(
    config_path: Path, output: Path, device: torch.device, seed: int
) -> dict[str, object]:
    project_root = Path(__file__).resolve().parents[1]
    module_path = project_root / "src/homymoly/experiments/identifiable_maps.py"
    return {
        "schema_version": SCHEMA_VERSION,
        "created_unix": time.time(),
        "command": [sys.executable, *sys.argv],
        "code_fingerprint": _code_fingerprint(project_root, Path(__file__)),
        "working_directory": str(Path.cwd()),
        "output_directory": str(output.resolve()),
        "git": {
            "revision": _git(("rev-parse", "HEAD")),
            "branch": _git(("branch", "--show-current")),
            "status_porcelain": (_git(("status", "--porcelain")) or "").splitlines(),
        },
        "files": {
            "input_config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "module": {"path": str(module_path), "sha256": _sha256(module_path)},
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pyyaml": yaml.__version__,
        "device": str(device),
        "seed": seed,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def run(config_path: Path, config: dict[str, Any]) -> dict[str, object]:
    """Execute a fully configured experiment and return its summary."""

    _validate_config(config)
    experiment = _mapping(config["experiment"], "experiment")
    data = _mapping(config["data"], "data")
    model_config = _mapping(config["model"], "model")
    training = _mapping(config["training"], "training")
    loss_config = _mapping(config["loss"], "loss")
    evaluation = _mapping(config["evaluation"], "evaluation")
    output_config = _mapping(config["output"], "output")
    seed = int(experiment["seed"])
    deterministic = bool(experiment["deterministic"])
    requested_device = str(experiment["device"])
    if deterministic and requested_device in {"auto", "cuda"}:
        # Required before the first CUDA context for deterministic cuBLAS.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _seed_everything(seed, deterministic)
    device = _select_device(requested_device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output_directory = Path(str(output_config["directory"]))
    output_directory.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        output_directory / "effective_config.yaml",
        yaml.safe_dump(config, sort_keys=True),
    )
    _atomic_json(
        output_directory / "provenance.json",
        _environment(config_path, output_directory, device, seed),
    )

    system = build_annulus_map_system(int(data["sectors"]), dtype=torch.float32)
    model = IdentifiableTypedMapModel(
        system,
        hidden_dim=int(model_config["hidden_dim"]),
        dropout=float(model_config["dropout"]),
        map_temperature=float(model_config["map_temperature"]),
    ).to(device)
    datasets, loaders = _data_loaders(config, device)
    ablation, weights = _weights(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    history: list[dict[str, object]] = []
    best_objective = float("inf")
    best_epoch = 0
    best_state = _state_on_cpu(model)
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, int(training["epochs"]) + 1):
        train_metrics = _epoch(
            model,
            loaders["train"],
            weights,
            device=device,
            cone_temperature=float(loss_config["cone_temperature"]),
            rtd_entities=int(loss_config["rtd_training_entities"]),
            optimizer=optimizer,
        )
        validation_metrics = _epoch(
            model,
            loaders["validation"],
            weights,
            device=device,
            cone_temperature=float(loss_config["cone_temperature"]),
            rtd_entities=int(loss_config["rtd_training_entities"]),
            optimizer=None,
        )
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        )
        objective = validation_metrics["objective"]
        if objective < best_objective - float(training["minimum_improvement"]):
            best_objective = objective
            best_epoch = epoch
            best_state = _state_on_cpu(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(training["early_stopping_patience"]):
            break
    model.load_state_dict(best_state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    test, predictions = _exact_evaluation(
        model,
        loaders["test"],
        datasets["test"],
        device=device,
        map_tolerance=float(evaluation["map_tolerance"]),
        rank_atol=float(evaluation["rank_atol"]),
        exact_rtd_entities=int(evaluation["exact_rtd_entities"]),
        exact_rtd_max_dim=int(evaluation["exact_rtd_max_dim"]),
    )
    hard_histogram = cast(dict[str, int], test["hard_cone_betti_histogram"])
    hard_acyclic_fraction = int(hard_histogram.get("[0,0,0,0]", 0)) / int(
        test["examples"]
    )
    engineering_checks = {
        "transformation_accuracy": float(test["transformation_accuracy"])
        >= ENGINEERING_GATE["transformation_accuracy_min"],
        "cell_face_accuracy": float(test["cell_face_accuracy"])
        >= ENGINEERING_GATE["cell_face_accuracy_min"],
        "map_mse": float(test["map_mse"]) <= ENGINEERING_GATE["map_mse_max"],
        "soft_chain_residual": float(test["soft_chain_residual_max"])
        <= ENGINEERING_GATE["chain_residual_max"],
        "hard_chain_residual": float(test["hard_chain_residual_max"])
        <= ENGINEERING_GATE["chain_residual_max"],
        "hard_cone_acyclic_fraction": hard_acyclic_fraction
        >= ENGINEERING_GATE["hard_cone_acyclic_fraction_min"],
    }
    checkpoint_path = output_directory / "checkpoint.pt"
    _atomic_checkpoint(
        checkpoint_path,
        {
            "schema_version": SCHEMA_VERSION,
            "model_state_dict": best_state,
            "config": config,
            "best_epoch": best_epoch,
            "ablation": ablation,
        },
    )
    predictions_path = output_directory / "test_predictions.jsonl"
    _atomic_text(
        predictions_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
    )
    history_path = output_directory / "history.json"
    _atomic_json(history_path, history)
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "experiment": "identifiable-graph-only-typed-maps",
        "scope": (
            "finite dihedral maps on one cellular annulus; no categorical-equivalence claim"
        ),
        "ablation": ablation,
        "loss_weights": weights.as_dict(),
        "rtd_training_entities": int(loss_config["rtd_training_entities"]),
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_objective": best_objective,
        "engineering_recovery_gate": {
            "applicable": ablation in {"task_reconstruction", "combined"},
            "thresholds": ENGINEERING_GATE,
            "checks": engineering_checks,
            "passed": (
                all(engineering_checks.values())
                if ablation in {"task_reconstruction", "combined"}
                else None
            ),
            "hard_cone_acyclic_fraction": hard_acyclic_fraction,
            "status": "pre-specified development-informed engineering gate",
        },
        "elapsed_seconds": elapsed,
        "dataset": {
            "topology": "cellular_annulus",
            "sectors": system.sectors,
            "betti_numbers": [1, 1, 0],
            "vertices": system.num_vertices,
            "edges": system.num_edges,
            "faces": system.num_faces,
            "transformations": system.num_transformations,
            "split_samples": {name: len(dataset) for name, dataset in datasets.items()},
            "split_seeds": {name: dataset.seed for name, dataset in datasets.items()},
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
        "basis_chain_residual_max": model.basis_residual_max(),
        "test": test,
        "environment": {
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            )
        },
    }
    summary_path = output_directory / "summary.json"
    _atomic_json(summary_path, summary)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_git_revision": _git(("rev-parse", "HEAD")),
        "artifacts": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (
                output_directory / "effective_config.yaml",
                output_directory / "provenance.json",
                checkpoint_path,
                predictions_path,
                history_path,
                summary_path,
            )
        },
    }
    _atomic_json(output_directory / "manifest.json", manifest)
    return summary


def main() -> int:
    args = _arguments()
    config = apply_cli_overrides(load_config(args.config), args)
    summary = run(args.config, config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
