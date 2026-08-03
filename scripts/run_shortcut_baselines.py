#!/usr/bin/env python3
"""Run the CPU confirmatory shortcut-baseline campaign and emit JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from homymoly.training.baselines import (
    ShortcutBaselineConfig,
    run_shortcut_baselines,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--data-seed", type=int, default=20260803)
    parser.add_argument("--split-seed", type=int, default=404)
    parser.add_argument("--training-seed", type=int, default=1701)
    parser.add_argument("--min-vertices", type=int, default=24)
    parser.add_argument("--max-vertices", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = run_shortcut_baselines(
        ShortcutBaselineConfig(
            num_samples=args.samples,
            data_seed=args.data_seed,
            split_seed=args.split_seed,
            training_seed=args.training_seed,
            min_vertices=args.min_vertices,
            max_vertices=args.max_vertices,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            patience=args.patience,
            num_threads=args.threads,
        )
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(output)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
