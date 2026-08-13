#!/usr/bin/env python3
"""Train an architecturally exact bidirectional map between chain complexes."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from homymoly.models.chain_map import (
    ExactChainMapLayer,
    cone_soft_betti,
    cycle_consistency_loss,
)
from homymoly.topology import (
    ChainComplex,
    ChainMap,
    build_oriented_incidence,
    cone_betti_numbers,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--vertices", type=int, default=8)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--test-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--cycle-weight", type=float, default=0.1)
    parser.add_argument("--cone-weight", type=float, default=1e-4)
    parser.add_argument("--cone-temperature", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        default="artifacts/chain-map-exact/summary.json",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
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


def _git_revision() -> str | None:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _cycle_edges(vertices: int) -> tuple[tuple[int, int], ...]:
    if vertices < 4:
        raise ValueError("vertices must be at least four")
    return tuple((index, index + 1) for index in range(vertices - 1)) + (
        (0, vertices - 1),
    )


def _paired_signals(
    count: int,
    dimensions: tuple[int, int],
    maps: tuple[torch.Tensor, torch.Tensor],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x0 = torch.randn((count, dimensions[0]), generator=generator)
    x1 = torch.randn((count, dimensions[1]), generator=generator)
    y0 = x0 @ maps[0].cpu().mT
    y1 = x1 @ maps[1].cpu().mT
    return tuple(value.to(device) for value in (x0, x1, y0, y1))  # type: ignore[return-value]


def main() -> int:
    args = _arguments()
    if args.smoke:
        args.samples = min(args.samples, 256)
        args.test_samples = min(args.test_samples, 128)
        args.steps = min(args.steps, 20)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    selected_device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    device = torch.device(selected_device)

    if args.samples <= 0 or args.test_samples <= 0 or args.batch_size <= 0:
        raise ValueError("sample and batch counts must be positive")
    if args.steps <= 0 or args.learning_rate <= 0:
        raise ValueError("steps and learning rate must be positive")
    if args.cycle_weight < 0 or args.cone_weight < 0 or args.cone_temperature <= 0:
        raise ValueError("loss weights must be nonnegative and temperature positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    cpu_generator = torch.Generator().manual_seed(args.seed)

    incidence = build_oriented_incidence(
        args.vertices,
        _cycle_edges(args.vertices),
        dtype=torch.float64,
    )
    source_cpu = incidence.boundary_1
    vertex_order = torch.randperm(args.vertices, generator=cpu_generator)
    edge_order = torch.randperm(len(incidence.edges), generator=cpu_generator)
    true_f0_cpu = torch.eye(args.vertices, dtype=torch.float64)[vertex_order]
    true_f1_cpu = torch.eye(len(incidence.edges), dtype=torch.float64)[edge_order]
    target_cpu = true_f0_cpu @ source_cpu @ true_f1_cpu.mT

    source = source_cpu.to(device=device, dtype=torch.float32)
    target = target_cpu.to(device=device, dtype=torch.float32)
    true_f0 = true_f0_cpu.to(device=device, dtype=torch.float32)
    true_f1 = true_f1_cpu.to(device=device, dtype=torch.float32)
    forward = ExactChainMapLayer(source, target).to(device)
    reverse = ExactChainMapLayer(target, source).to(device)
    optimizer = torch.optim.AdamW(
        (*forward.parameters(), *reverse.parameters()),
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    train = _paired_signals(
        args.samples,
        forward.source_dimensions,
        (true_f0, true_f1),
        generator=cpu_generator,
        device=device,
    )
    test = _paired_signals(
        args.test_samples,
        forward.source_dimensions,
        (true_f0, true_f1),
        generator=cpu_generator,
        device=device,
    )
    started = time.perf_counter()
    history: list[dict[str, float | int]] = []
    for step in range(args.steps):
        indices = torch.randint(
            args.samples,
            (min(args.batch_size, args.samples),),
            generator=cpu_generator,
        ).to(device)
        x0, x1, y0, y1 = (value.index_select(0, indices) for value in train)
        predicted_y0, predicted_y1 = forward(x0, x1)
        predicted_x0, predicted_x1 = reverse(y0, y1)
        paired_loss = (
            F.mse_loss(predicted_y0, y0)
            + F.mse_loss(predicted_y1, y1)
            + F.mse_loss(predicted_x0, x0)
            + F.mse_loss(predicted_x1, x1)
        )
        cycle_loss = cycle_consistency_loss(forward.matrices(), reverse.matrices())
        cone_proxy = cone_soft_betti(
            source,
            target,
            forward.matrices(),
            temperature=args.cone_temperature,
        )
        loss = (
            paired_loss + args.cycle_weight * cycle_loss + args.cone_weight * cone_proxy
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "paired_loss": float(paired_loss.detach()),
                    "cycle_loss": float(cycle_loss.detach()),
                    "cone_soft_betti": float(cone_proxy.detach()),
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    with torch.no_grad():
        test_y0, test_y1 = forward(test[0], test[1])
        test_x0, test_x1 = reverse(test[2], test[3])
        test_mse = float(
            (
                F.mse_loss(test_y0, test[2])
                + F.mse_loss(test_y1, test[3])
                + F.mse_loss(test_x0, test[0])
                + F.mse_loss(test_x1, test[1])
            )
            / 4.0
        )
        learned_forward = forward.matrices()
        learned_reverse = reverse.matrices()
        forward_residual = float(forward.residual().abs().max())
        reverse_residual = float(reverse.residual().abs().max())
        cycle_error = float(cycle_consistency_loss(learned_forward, learned_reverse))
        forward_map_error = float(
            0.5
            * (
                F.mse_loss(learned_forward.degree_zero, true_f0)
                + F.mse_loss(learned_forward.degree_one, true_f1)
            )
        )

    source_complex = ChainComplex(
        (args.vertices, len(incidence.edges)),
        (source_cpu,),
        dtype=torch.float64,
    )
    target_complex = ChainComplex(
        (args.vertices, len(incidence.edges)),
        (target_cpu,),
        dtype=torch.float64,
    )
    map_tolerance = max(1e-5, 2.0 * forward_residual)
    evaluated_map = ChainMap(
        source_complex,
        target_complex,
        (
            learned_forward.degree_zero.detach().cpu().double(),
            learned_forward.degree_one.detach().cpu().double(),
        ),
        atol=map_tolerance,
    )
    cone_betti = cone_betti_numbers(
        evaluated_map,
        map_atol=map_tolerance,
        rank_atol=1e-5,
    )

    payload: dict[str, object] = {
        "status": "completed",
        "experiment": "exact-bidirectional-chain-map",
        "seed": args.seed,
        "device": str(device),
        "vertices": args.vertices,
        "edges": len(incidence.edges),
        "train_samples": args.samples,
        "test_samples": args.test_samples,
        "steps": args.steps,
        "elapsed_seconds": elapsed,
        "test_mse": test_mse,
        "forward_map_mse": forward_map_error,
        "forward_chain_residual_max": forward_residual,
        "cone_map_tolerance": map_tolerance,
        "reverse_chain_residual_max": reverse_residual,
        "cycle_consistency_mse": cycle_error,
        "forward_cone_betti": list(cone_betti),
        "history": history,
        "environment": {
            "git_revision": _git_revision(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "command": [sys.executable, *sys.argv],
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
    }
    output = Path(args.output)
    _atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
