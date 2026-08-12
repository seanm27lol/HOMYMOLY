"""Command-line entry point for HOMYMOLY runtime checks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from homymoly import __version__
from homymoly.config import ConfigError, load_config


def _default_config_path() -> str:
    return os.environ.get("HOMYMOLY_CONFIG", "configs/stage1.yaml")


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=_default_config_path(),
        help="YAML configuration path (default: HOMYMOLY_CONFIG or configs/stage1.yaml)",
    )
    parser.add_argument(
        "--artifact-root",
        help="override artifacts.root without modifying the YAML file",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="homymoly",
        description="HOMYMOLY structured-representation experiments",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command")

    check = subparsers.add_parser(
        "check-config", help="validate and print the active config"
    )
    _add_config_argument(check)
    check.add_argument(
        "--create-artifacts",
        action="store_true",
        help="create the declared run directories after validation",
    )

    paths = subparsers.add_parser("paths", help="print resolved artifact paths")
    _add_config_argument(paths)
    paths.add_argument(
        "--kind",
        choices=("root", "run", "checkpoints", "tensorboard", "profiles", "metrics"),
        help="print only one path, suitable for shell scripts",
    )
    paths.add_argument(
        "--create",
        action="store_true",
        help="create all artifact directories before printing",
    )

    foundation = subparsers.add_parser(
        "validate-foundation",
        help="run the exact Stage-1 data/topology integration gate",
    )
    _add_config_argument(foundation)
    foundation.add_argument(
        "--samples",
        type=int,
        default=6,
        help="number of balanced synthetic samples to validate (default: 6)",
    )
    foundation.add_argument(
        "--vertices",
        type=int,
        help="fixed vertex count (default: data.min_vertices)",
    )
    foundation.add_argument(
        "--atol",
        type=float,
        default=1e-10,
        help="absolute tolerance for exact structural checks",
    )

    gate2 = subparsers.add_parser(
        "check-gate2-config", help="validate and print the Gate-2 experiment config"
    )
    gate2.add_argument("--config", default="configs/gate2.yaml")

    train = subparsers.add_parser(
        "train", help="train the Gate-2 experts, translators, and router"
    )
    train.add_argument("--config", default="configs/gate2.yaml")
    train.add_argument(
        "--resume", action="store_true", help="resume from the last durable checkpoint"
    )
    train.add_argument(
        "--smoke",
        action="store_true",
        help="run one bounded epoch per phase on a small deterministic subset",
    )
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="build one batch and model forward pass without optimization",
    )
    return parser


def _load(args: argparse.Namespace):
    return load_config(
        Path(args.config),
        artifact_root=args.artifact_root,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command in {"check-gate2-config", "train"}:
            from homymoly.training.config import load_gate2_config

            gate2_config = load_gate2_config(args.config)
            if args.command == "check-gate2-config":
                print(json.dumps(gate2_config.as_dict(), indent=2, sort_keys=True))
                return 0
            from homymoly.training.engine import run_training

            report = run_training(
                gate2_config,
                resume=args.resume,
                smoke=args.smoke,
                dry_run=args.dry_run,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0

        config = _load(args)
        paths = config.artifact_paths()
        if args.command == "check-config":
            if args.create_artifacts:
                paths.create()
            payload = config.as_dict()
            payload["resolved_artifacts"] = paths.as_dict()
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == "paths":
            if args.create:
                paths.create()
            values = paths.as_dict()
            if args.kind:
                print(values[args.kind])
            else:
                print(json.dumps(values, indent=2, sort_keys=True))
            return 0
        if args.command == "validate-foundation":
            from homymoly.stage1 import build_stage1_dataset, validate_foundation

            report = validate_foundation(
                num_samples=args.samples,
                seed=config.data.seed,
                num_vertices=(
                    args.vertices
                    if args.vertices is not None
                    else config.data.min_vertices
                ),
                node_feature_dim=config.data.node_feature_dim,
                edge_feature_dim=config.data.edge_feature_dim,
                atol=args.atol,
            )
            configured_dataset, splits = build_stage1_dataset(config.data)
            report["configured_dataset"] = {
                "num_samples": len(configured_dataset),
                "split_sizes": {name: len(indices) for name, indices in splits.items()},
            }
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
    except (ConfigError, RuntimeError, ValueError) as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
