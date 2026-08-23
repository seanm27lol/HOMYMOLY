#!/usr/bin/env python3
"""Benchmark checkpoint-specific identifiable-map inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from homymoly.experiments.identifiable_maps import (
    IdentifiableTypedMapDataset,
    IdentifiableTypedMapModel,
    build_annulus_map_system,
)

SCHEMA_VERSION = 1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
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


def _git_revision() -> str | None:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), percentile))


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("config must be a YAML mapping")
    for section in (
        "experiment",
        "data",
        "model",
        "training",
        "loss",
        "evaluation",
        "output",
    ):
        if not isinstance(config.get(section), dict):
            raise TypeError(f"config is missing mapping section {section!r}")
    return config


def _make_batch(
    config: dict[str, Any], batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    seed = int(config["experiment"]["seed"]) + 3011
    dataset = IdentifiableTypedMapDataset(
        batch_size,
        seed=seed,
        sectors=int(config["data"]["sectors"]),
        noise_std=float(config["data"]["noise_std"]),
    )
    node = torch.stack(
        [
            torch.as_tensor(dataset[index]["node_features"])
            for index in range(batch_size)
        ]
    ).to(device)
    edge = torch.stack(
        [
            torch.as_tensor(dataset[index]["edge_features"])
            for index in range(batch_size)
        ]
    ).to(device)
    return node, edge


def _timed_forward_cuda(
    model: IdentifiableTypedMapModel,
    node: torch.Tensor,
    edge: torch.Tensor,
    iterations: int,
) -> tuple[list[float], float]:
    elapsed: list[float] = []
    checksum = 0.0
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = model(node, edge)
        end.record()
        end.synchronize()
        elapsed.append(float(start.elapsed_time(end)))
        checksum += float(output.logits[0, 0])
    return elapsed, checksum


def _timed_forward_cpu(
    model: IdentifiableTypedMapModel,
    node: torch.Tensor,
    edge: torch.Tensor,
    iterations: int,
) -> tuple[list[float], float]:
    elapsed: list[float] = []
    checksum = 0.0
    for _ in range(iterations):
        started = time.perf_counter_ns()
        output = model(node, edge)
        elapsed.append((time.perf_counter_ns() - started) / 1e6)
        checksum += float(output.logits[0, 0])
    return elapsed, checksum


def main() -> int:
    args = _arguments()
    if args.warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    config = _load_config(args.config)
    deterministic = bool(config["experiment"].get("deterministic", False))
    if deterministic and args.device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    device = torch.device(args.device)
    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(config["training"]["batch_size"])
    )
    if batch_size <= 0:
        raise ValueError("batch size must be positive")

    system = build_annulus_map_system(
        int(config["data"]["sectors"]), dtype=torch.float32
    )
    model = IdentifiableTypedMapModel(
        system,
        hidden_dim=int(config["model"]["hidden_dim"]),
        dropout=float(config["model"]["dropout"]),
        map_temperature=float(config["model"]["map_temperature"]),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint does not contain model_state_dict")
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "checkpoint schema_version does not match benchmark schema: "
            f"{checkpoint.get('schema_version')!r} != {SCHEMA_VERSION}"
        )
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise TypeError("checkpoint does not contain its effective configuration")
    if checkpoint_config != config:
        raise ValueError(
            "checkpoint effective configuration does not exactly match --config"
        )
    expected_ablation = str(config["loss"]["ablation"])
    if checkpoint.get("ablation") != expected_ablation:
        raise ValueError(
            "checkpoint ablation does not match its effective configuration: "
            f"{checkpoint.get('ablation')!r} != {expected_ablation!r}"
        )
    seed = int(config["experiment"]["seed"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    node, edge = _make_batch(config, batch_size, device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(node, edge)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            elapsed, checksum = _timed_forward_cuda(model, node, edge, args.iterations)
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
        else:
            elapsed, checksum = _timed_forward_cpu(model, node, edge, args.iterations)
            peak_allocated = 0
            peak_reserved = 0
        final_output = model(node, edge)
        residuals = model.residuals(final_output.maps)
        residual_max = max(
            float(residuals[0].abs().max()), float(residuals[1].abs().max())
        )
    tolerance = float(config["evaluation"]["map_tolerance"])
    if residual_max > tolerance:
        raise RuntimeError(
            f"benchmark map residual {residual_max:.3e} exceeds {tolerance:.3e}"
        )

    median = float(statistics.median(elapsed))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "experiment": "identifiable-map-checkpoint-inference-benchmark",
        "measurement_scope": (
            "model forward only: graph encoder, exact-map mixture, typed signal outputs; "
            "excludes data loading, training losses, exact RTD, and exact cone rank oracles"
        ),
        "ablation": expected_ablation,
        "seed": seed,
        "best_epoch": checkpoint.get("best_epoch"),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
        "batch_size": batch_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "latency_ms": {
            "median": median,
            "mean": float(statistics.fmean(elapsed)),
            "minimum": min(elapsed),
            "p10": _percentile(elapsed, 0.10),
            "p90": _percentile(elapsed, 0.90),
            "maximum": max(elapsed),
        },
        "throughput_examples_per_second_at_median": batch_size * 1000.0 / median,
        "peak_cuda_memory_allocated_bytes": peak_allocated,
        "peak_cuda_memory_reserved_bytes": peak_reserved,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "chain_residual_max": residual_max,
        "map_tolerance": tolerance,
        "materialization_checksum": checksum,
        "provenance": {
            "created_unix": time.time(),
            "command": [sys.executable, *sys.argv],
            "effective_seed": seed,
            "effective_ablation": expected_ablation,
            "code_fingerprint": _code_fingerprint(
                Path(__file__).resolve().parents[1], Path(__file__)
            ),
            "git_revision": _git_revision(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "config": {"path": str(args.config), "sha256": _sha256(args.config)},
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": _sha256(args.checkpoint),
            },
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
