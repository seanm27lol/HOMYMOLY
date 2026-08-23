#!/usr/bin/env python3
"""Benchmark one bounded identifiable-map RTD training step."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor

from homymoly.experiments.identifiable_maps import (
    IdentifiableTypedMapDataset,
    IdentifiableTypedMapModel,
    LossWeights,
    build_annulus_map_system,
    compute_identifiable_losses,
)

SCHEMA_VERSION = 1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _stack_batch(
    dataset: IdentifiableTypedMapDataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, Tensor | list[str]]:
    batch: dict[str, Tensor | list[str]] = {}
    for key in dataset[0]:
        values = [dataset[index][key] for index in range(batch_size)]
        batch[key] = (
            [str(value) for value in values]
            if key == "sample_id"
            else torch.stack([torch.as_tensor(value) for value in values]).to(device)
        )
    return batch


def _gradient_norm(value: Tensor, parameters: tuple[Tensor, ...]) -> float:
    gradients = torch.autograd.grad(
        value, parameters, retain_graph=True, allow_unused=True
    )
    squared = value.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.square().sum()
    return float(torch.sqrt(squared).detach())


def _step(
    model: IdentifiableTypedMapModel,
    batch: dict[str, Tensor | list[str]],
    weights: LossWeights,
    *,
    temperature: float,
    rtd_entities: int,
    device: torch.device,
) -> float:
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter_ns()
    output = model(
        torch.as_tensor(batch["node_features"]),
        torch.as_tensor(batch["edge_features"]),
    )
    objective, terms = compute_identifiable_losses(
        model,
        output,
        batch,
        weights,
        cone_temperature=temperature,
        rtd_entities=rtd_entities,
    )
    objective.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if not torch.isfinite(terms["rtd"]):
        raise RuntimeError("bounded RTD term is non-finite")
    return (time.perf_counter_ns() - started) / 1e6


def main() -> int:
    args = _arguments()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations positive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("config must be a YAML mapping")
    deterministic = bool(config["experiment"]["deterministic"])
    if deterministic and args.device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed = int(config["experiment"]["seed"])
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    batch_size = int(config["training"]["batch_size"])
    rtd_entities = config["loss"]["rtd_training_entities"]
    if (
        isinstance(rtd_entities, bool)
        or not isinstance(rtd_entities, int)
        or not 2 <= rtd_entities <= batch_size
    ):
        raise ValueError("rtd_training_entities must be an integer in [2,batch_size]")
    system = build_annulus_map_system(
        int(config["data"]["sectors"]), dtype=torch.float32
    )
    model = IdentifiableTypedMapModel(
        system,
        hidden_dim=int(config["model"]["hidden_dim"]),
        dropout=float(config["model"]["dropout"]),
        map_temperature=float(config["model"]["map_temperature"]),
    ).to(device)
    dataset = IdentifiableTypedMapDataset(
        batch_size,
        seed=seed + 1009,
        sectors=int(config["data"]["sectors"]),
        noise_std=float(config["data"]["noise_std"]),
    )
    batch = _stack_batch(dataset, batch_size, device)
    raw_weights = config["loss"]["combined_weights"]
    combined = LossWeights(
        **{name: float(raw_weights[name]) for name in LossWeights.__dataclass_fields__}
    )
    rtd_only = LossWeights(0.0, 0.0, 0.0, 0.0, 0.0, combined.rtd)
    temperature = float(config["loss"]["cone_temperature"])

    output = model(
        torch.as_tensor(batch["node_features"]),
        torch.as_tensor(batch["edge_features"]),
    )
    _, terms = compute_identifiable_losses(
        model,
        output,
        batch,
        combined,
        cone_temperature=temperature,
        rtd_entities=rtd_entities,
    )
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    supervised = sum(
        combined.as_dict()[name] * terms[name]
        for name in ("task", "reconstruction", "cell", "sheaf")
    )
    gradient_norms = {
        "supervised_weighted": _gradient_norm(supervised, parameters),
        "cone_unweighted": _gradient_norm(terms["cone"], parameters),
        "cone_weighted": _gradient_norm(combined.cone * terms["cone"], parameters),
        "rtd_unweighted": _gradient_norm(terms["rtd"], parameters),
        "rtd_weighted": _gradient_norm(combined.rtd * terms["rtd"], parameters),
    }

    for _ in range(args.warmup):
        _step(
            model,
            batch,
            rtd_only,
            temperature=temperature,
            rtd_entities=rtd_entities,
            device=device,
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elapsed = [
        _step(
            model,
            batch,
            rtd_only,
            temperature=temperature,
            rtd_entities=rtd_entities,
            device=device,
        )
        for _ in range(args.iterations)
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "experiment": "identifiable-bounded-rtd-training-step-benchmark",
        "measurement_scope": (
            "full-batch model forward/backward with RTD-only objective; the cubic "
            "H0 surrogate uses the deterministic leading batch prefix"
        ),
        "seed": seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
        "batch_size": batch_size,
        "rtd_training_entities": rtd_entities,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "step_latency_ms": {
            "minimum": min(elapsed),
            "mean": float(np.mean(elapsed)),
            "median": float(np.median(elapsed)),
            "maximum": max(elapsed),
        },
        "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "loss_values_at_initialization": {
            name: float(value.detach()) for name, value in terms.items()
        },
        "gradient_l2_norms_at_initialization": gradient_norms,
        "combined_weights": combined.as_dict(),
        "provenance": {
            "created_unix": time.time(),
            "command": [sys.executable, *sys.argv],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "config": {"path": str(args.config), "sha256": _sha256(args.config)},
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
        },
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
