#!/usr/bin/env python3
"""Benchmark routed, fixed, and dense HOMYMOLY inference on the active device."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor

from homymoly.data.collate import collate_structured
from homymoly.data.confirmatory import ConfirmatoryConfig, ConfirmatoryStructuredSignal
from homymoly.models import build_model
from homymoly.models.experts import ROUTE_ORDER
from homymoly.runtime import initialize_runtime
from homymoly.training.config import load_gate2_config
from homymoly.training.engine import _build_model_config
from homymoly.training.io import load_checkpoint


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gate2.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", default="artifacts/benchmarks/compute.json")
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _git_revision() -> str | None:
    process = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=False, capture_output=True, text=True
    )
    return process.stdout.strip() if process.returncode == 0 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark(
    name: str,
    function: Callable[[], Tensor],
    *,
    batch_size: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    for _ in range(warmup):
        function()
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    latencies: list[float] = []
    checksum = 0.0
    for _ in range(iterations):
        started = time.perf_counter()
        output = function()
        _synchronize(device)
        latencies.append((time.perf_counter() - started) * 1000.0)
        checksum += float(output.float().sum())
    total_seconds = sum(latencies) / 1000.0
    return {
        "path": name,
        "iterations": iterations,
        "batch_size": batch_size,
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_median": statistics.median(latencies),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "latency_ms_stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "examples_per_second": batch_size * iterations / total_seconds,
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "checksum": checksum,
    }


def main() -> int:
    args = _arguments()
    if args.batch_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError(
            "batch size/iterations must be positive and warmup nonnegative"
        )
    config_path = Path(args.config).resolve()
    config = load_gate2_config(config_path)
    runtime = initialize_runtime(config.runtime, seed=config.experiment.seed)
    model = build_model(_build_model_config(config)).to(runtime.device).eval()
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else None
    if checkpoint_path is not None:
        checkpoint = load_checkpoint(checkpoint_path, map_location=runtime.device)
        model.load_state_dict(checkpoint["model"])

    selected_samples = max(6, ((args.batch_size + 5) // 6) * 6)
    dataset = ConfirmatoryStructuredSignal(
        ConfirmatoryConfig(
            num_samples=selected_samples,
            seed=config.data.seed,
            min_vertices=config.data.min_vertices,
            max_vertices=config.data.max_vertices,
            node_feature_dim=config.data.node_feature_dim,
            edge_feature_dim=config.data.edge_feature_dim,
            stalk_mode=config.data.stalk_mode,
            gauge_noise_std=config.data.gauge_noise_std,
        )
    )
    batch = collate_structured([dataset[index] for index in range(args.batch_size)]).to(
        runtime.device
    )

    def autocast():  # type: ignore[no-untyped-def]
        if runtime.device.type != "cuda" or runtime.neural_dtype == torch.float32:
            return contextlib.nullcontext()
        return torch.autocast(
            device_type=runtime.device.type,
            dtype=runtime.neural_dtype,
        )

    @torch.inference_mode()
    def routed() -> Tensor:
        with autocast():
            return model(batch, hard=True).mixed_logits

    with torch.inference_mode(), autocast():
        route_profile_output = model(batch, hard=True)
    route_profile = {
        route.value: float(
            (route_profile_output.selected_routes == route_index).float().mean()
        )
        for route_index, route in enumerate(ROUTE_ORDER)
    }

    fixed_functions: dict[str, Callable[[], Tensor]] = {}
    for route in ROUTE_ORDER:
        expert = model.fixed_experts.experts[route.value]

        @torch.inference_mode()
        def fixed(expert=expert) -> Tensor:  # type: ignore[no-untyped-def]
            with autocast():
                return expert(batch).logits

        fixed_functions[f"fixed_{route.value}"] = fixed

    @torch.inference_mode()
    def dense() -> Tensor:
        with autocast():
            return model.fixed_experts(batch).logits

    paths: dict[str, Callable[[], Tensor]] = {
        "routed": routed,
        **fixed_functions,
        "dense": dense,
    }
    results = [
        _benchmark(
            name,
            function,
            batch_size=args.batch_size,
            warmup=args.warmup,
            iterations=args.iterations,
            device=runtime.device,
        )
        for name, function in paths.items()
    ]
    by_name = {str(item["path"]): item for item in results}
    routed_latency = float(by_name["routed"]["latency_ms_median"])
    dense_latency = float(by_name["dense"]["latency_ms_median"])
    payload: dict[str, object] = {
        "status": "completed",
        "device": str(runtime.device),
        "precision": str(runtime.neural_dtype),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_sha256": (
            _sha256(checkpoint_path) if checkpoint_path is not None else None
        ),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "route_profile_on_benchmark_batch": route_profile,
        "results": results,
        "routed_to_dense_latency_ratio": routed_latency / dense_latency,
        "dense_to_routed_speedup": dense_latency / routed_latency,
        "environment": {
            "git_revision": _git_revision(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(runtime.device)
                if runtime.device.type == "cuda"
                else platform.processor()
            ),
            "command": [sys.executable, *sys.argv],
        },
    }
    output = Path(args.output)
    _atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
