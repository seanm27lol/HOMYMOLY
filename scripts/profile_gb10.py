#!/usr/bin/env python3
"""Record a bounded PyTorch hardware and GEMM profile on a GB10 host."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from homymoly.config import ConfigError, load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=os.environ.get("HOMYMOLY_CONFIG", str(PROJECT_ROOT / "configs/stage1.yaml")),
    )
    parser.add_argument(
        "--matrix-size",
        type=int,
        action="append",
        dest="matrix_sizes",
        help="matrix size to profile; repeat to replace the configured sizes",
    )
    parser.add_argument("--output", help="JSON output path")
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="collect metadata without allocating benchmark matrices",
    )
    parser.add_argument(
        "--require-gb10",
        action="store_true",
        help="fail unless the CUDA device name identifies a GB10",
    )
    return parser


def _host_memory() -> dict[str, int]:
    values: dict[str, int] = {}
    path = Path("/proc/meminfo")
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, remainder = line.partition(":")
        if not separator or key not in {"MemTotal", "MemAvailable", "SwapTotal"}:
            continue
        parts = remainder.split()
        if parts and parts[0].isdigit():
            values[f"{key.lower()}_bytes"] = int(parts[0]) * 1024
    return values


def _nvidia_smi_report() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,compute_cap,memory.total,power.limit",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}
    if completed.returncode != 0:
        return {
            "available": False,
            "returncode": completed.returncode,
            "error": completed.stderr.strip(),
        }
    devices = []
    fields = ("name", "driver_version", "compute_capability", "memory_mib", "power_watts")
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        devices.append(dict(zip(fields, values, strict=False)))
    return {"available": True, "devices": devices}


def _git_report() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}
    if revision.returncode != 0 or status.returncode != 0:
        return {
            "available": False,
            "revision_error": revision.stderr.strip(),
            "status_error": status.stderr.strip(),
        }
    return {
        "available": True,
        "commit": revision.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
    }


def _dtype(torch: Any, precision: str):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[precision]


def _benchmark_size(
    torch: Any,
    *,
    size: int,
    dtype: Any,
    warmup_steps: int,
    active_steps: int,
    repetitions: int,
) -> dict[str, Any]:
    free_bytes, _ = torch.cuda.mem_get_info()
    element_size = torch.empty((), dtype=dtype).element_size()
    estimated_bytes = 3 * size * size * element_size
    if estimated_bytes > free_bytes // 2:
        return {
            "matrix_size": size,
            "status": "skipped",
            "reason": "estimated matrices exceed half of currently free CUDA memory",
            "estimated_bytes": estimated_bytes,
            "free_bytes": free_bytes,
        }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    left = torch.randn((size, size), device="cuda", dtype=dtype, generator=generator)
    right = torch.randn((size, size), device="cuda", dtype=dtype, generator=generator)
    for _ in range(warmup_steps):
        result = left @ right
    torch.cuda.synchronize()

    elapsed_seconds: list[float] = []
    for _ in range(repetitions):
        start = perf_counter()
        for _ in range(active_steps):
            result = left @ right
        torch.cuda.synchronize()
        elapsed_seconds.append((perf_counter() - start) / active_steps)

    checksum = float(result[0, 0].float().item())
    median_seconds = statistics.median(elapsed_seconds)
    return {
        "matrix_size": size,
        "status": "ok",
        "dtype": str(dtype).removeprefix("torch."),
        "warmup_steps": warmup_steps,
        "active_steps": active_steps,
        "repetitions": repetitions,
        "seconds_per_matmul": elapsed_seconds,
        "median_seconds_per_matmul": median_seconds,
        "median_tflops": (2.0 * size**3) / median_seconds / 1.0e12,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "checksum": checksum,
    }


def _torch_report(config: Any, args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        return {"available": False, "error": str(exc), "benchmarks": []}

    report: dict[str, Any] = {
        "available": True,
        "torch_version": torch.__version__,
        "cuda_build_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "benchmarks": [],
    }
    if not torch.cuda.is_available():
        return report

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    device_name = torch.cuda.get_device_name(device)
    report.update(
        {
            "device_index": device,
            "device_name": device_name,
            "is_gb10": "GB10" in device_name.upper(),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_device_memory_bytes": int(properties.total_memory),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "device_count": torch.cuda.device_count(),
        }
    )
    if args.skip_benchmark:
        return report

    torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32
    dtype = _dtype(torch, config.runtime.precision)
    sizes = tuple(args.matrix_sizes or config.profiling.matrix_sizes)
    report["benchmarks"] = [
        _benchmark_size(
            torch,
            size=size,
            dtype=dtype,
            warmup_steps=config.profiling.warmup_steps,
            active_steps=config.profiling.active_steps,
            repetitions=config.profiling.repetitions,
        )
        for size in sizes
    ]
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        artifact_paths = config.artifact_paths().create()
        timestamp = datetime.now(UTC)
        output = (
            Path(args.output).expanduser()
            if args.output
            else artifact_paths.profiles
            / f"gb10-profile-{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        if not output.is_absolute():
            output = config.project_root / output
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        pytorch_report = _torch_report(config, args)
        nvidia_smi_report = _nvidia_smi_report()
        detected_names = []
        if pytorch_report.get("device_name"):
            detected_names.append(str(pytorch_report["device_name"]))
        detected_names.extend(
            str(device.get("name", ""))
            for device in nvidia_smi_report.get("devices", [])
        )
        if args.require_gb10 and not any("GB10" in name.upper() for name in detected_names):
            raise RuntimeError(
                "--require-gb10 was set but neither PyTorch nor nvidia-smi identified a GB10"
            )
        if not args.skip_benchmark and (
            not pytorch_report.get("available")
            or not pytorch_report.get("cuda_available")
        ):
            raise RuntimeError(
                "a CUDA-enabled PyTorch runtime is required unless --skip-benchmark is set"
            )

        report = {
            "schema_version": 1,
            "created_at": timestamp.isoformat(),
            "config": str(config.source_path),
            "git": _git_report(),
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python_version": platform.python_version(),
                "processor": platform.processor(),
                **_host_memory(),
            },
            "environment": {
                key: os.environ[key]
                for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
                if key in os.environ
            },
            "nvidia_smi": nvidia_smi_report,
            "pytorch": pytorch_report,
        }
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(output)
        print(output)
        return 0
    except (ConfigError, RuntimeError, OSError, ValueError) as exc:
        print(f"GB10 profile failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
