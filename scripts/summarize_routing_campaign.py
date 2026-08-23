#!/usr/bin/env python3
"""Summarize a frozen five-seed routing campaign without seed replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

_T_975 = {
    1: 12.706204736,
    2: 4.30265273,
    3: 3.182446305,
    4: 2.776445105,
    5: 2.570581836,
    6: 2.446911851,
    7: 2.364624252,
    8: 2.306004135,
    9: 2.262157163,
}

FROZEN_CONFIG_SHA256 = {
    "routing-confirmatory-v2-s1.yaml": "3295c825f19f28f5edc678d51be94cf7aa1eac5bf23b37b9994b4513e69f669e",
    "routing-confirmatory-v2-s2.yaml": "f711646d1bf5c5c4ee9cb4fedf39f2b2edb5141444d3051042ecbacd4027fe13",
    "routing-confirmatory-v2-s3.yaml": "072d66d569d168a1d6150403e0254ec9320dbc2c552b6945830fd915f858e75f",
    "routing-confirmatory-v2-s4.yaml": "56b0059d4d41da4540ca186f7828cd87cfe7fc4a69f294dc21717a2a251c912b",
    "routing-confirmatory-v2-s5.yaml": "b612ed011ef8e2ec139019a3713714d6e88896b3cf2ee9eefefab15a379ccaa1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _interval(values: list[float]) -> list[float]:
    if len(values) < 2 or len(values) - 1 not in _T_975:
        raise ValueError("the t-interval supports 2--10 seeds")
    mean = statistics.mean(values)
    half_width = (
        _T_975[len(values) - 1] * statistics.stdev(values) / math.sqrt(len(values))
    )
    return [mean - half_width, mean + half_width]


def summarize(
    configs: list[Path],
    artifacts_root: Path,
    *,
    expected_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    if len(configs) != 5:
        raise ValueError("the frozen v2 protocol requires exactly five configs")
    expected_hashes = (
        FROZEN_CONFIG_SHA256 if expected_hashes is None else expected_hashes
    )
    names = {config.name for config in configs}
    if names != set(expected_hashes):
        raise ValueError(
            "config set does not match the frozen v2 protocol: "
            f"received={sorted(names)}"
        )
    rows: list[dict[str, Any]] = []
    revisions: set[str] = set()
    code_fingerprints: set[str] = set()
    environment_signatures: set[tuple[str, str, str, str]] = set()
    for config in configs:
        # The run name is intentionally extracted from the frozen YAML without
        # importing the training stack, so summarization cannot mutate runtime.
        run_name = next(
            line.split(":", 1)[1].strip()
            for line in config.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("run_name:")
        )
        run_dir = artifacts_root / run_name
        config_sha256 = _sha256(config)
        if config_sha256 != expected_hashes[config.name]:
            raise RuntimeError(
                f"frozen config hash mismatch for {config}: {config_sha256}"
            )
        summary_path = run_dir / "summary.json"
        environment_path = run_dir / "environment.json"
        if not summary_path.is_file() or not environment_path.is_file():
            raise FileNotFoundError(f"incomplete frozen run: {run_dir}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        run_status = str(summary.get("status") or "")
        if run_status not in {"completed", "gate-failed"}:
            raise RuntimeError(f"run has no analyzable final evaluation: {run_dir}")
        git_metadata = environment.get("git", {})
        revision = str(
            environment.get("git_revision")
            or (git_metadata.get("commit") if isinstance(git_metadata, dict) else None)
        )
        if revision in {"None", ""}:
            raise RuntimeError(f"run has no recorded git revision: {run_dir}")
        revisions.add(revision)
        code_fingerprint = str(environment.get("code_fingerprint") or "")
        if not code_fingerprint:
            raise RuntimeError(f"run has no recorded code fingerprint: {run_dir}")
        code_fingerprints.add(code_fingerprint)
        environment_signature = tuple(
            str(environment.get(key) or "")
            for key in ("torch_version", "cuda_version", "device", "device_name")
        )
        if not all(environment_signature):
            raise RuntimeError(f"run has incomplete environment metadata: {run_dir}")
        environment_signatures.add(environment_signature)
        git_status = (
            str(git_metadata.get("status") or "")
            if isinstance(git_metadata, dict)
            else ""
        )
        test = summary["test"]
        fixed = {
            route: float(test[f"{route}_expert_accuracy"])
            for route in ("graph", "cell", "sheaf")
        }
        best_fixed = max(fixed.values())
        hard = float(test["hard_accuracy"])
        rows.append(
            {
                "config": config.as_posix(),
                "config_sha256": config_sha256,
                "run_name": run_name,
                "git_revision": revision,
                "git_status_at_start": git_status,
                "code_fingerprint": code_fingerprint,
                "hard_accuracy": hard,
                "fixed_accuracy": fixed,
                "best_fixed_accuracy": best_fixed,
                "primary_margin": hard - best_fixed,
                "dense_accuracy": float(test["dense_accuracy"]),
                "routed_minus_dense": hard - float(test["dense_accuracy"]),
                "route_accuracy": float(test["route_accuracy"]),
                "route_mutual_information": float(
                    test["regime_route_mutual_information"]
                ),
                "route_utilization": {
                    route: float(test[f"route_utilization_{route}"])
                    for route in ("graph", "cell", "sheaf")
                },
                "failed_gate": summary.get("failed_gate"),
                "run_status": run_status,
            }
        )
    if len(revisions) != 1:
        raise RuntimeError(f"runs do not share one code revision: {sorted(revisions)}")
    if len(code_fingerprints) != 1:
        raise RuntimeError(
            "runs do not share one executable-source fingerprint: "
            f"{sorted(code_fingerprints)}"
        )
    if len(environment_signatures) != 1:
        raise RuntimeError(
            "runs do not share one runtime environment: "
            f"{sorted(environment_signatures)}"
        )
    margins = [float(row["primary_margin"]) for row in rows]
    interval = _interval(margins)
    return {
        "schema_version": 1,
        "protocol": "docs/19-routing-confirmatory-v2-protocol.md",
        "primary_endpoint": "hard_accuracy - max(fixed expert accuracies)",
        "rows": rows,
        "run_status_counts": {
            status: sum(row["run_status"] == status for row in rows)
            for status in sorted({str(row["run_status"]) for row in rows})
        },
        "primary": {
            "n": len(rows),
            "mean_margin": statistics.mean(margins),
            "sample_standard_deviation": statistics.stdev(margins),
            "student_t_95_ci": interval,
            "decision_rule": "supported iff lower confidence endpoint > 0",
            "decision": "supported" if interval[0] > 0 else "not-supported",
        },
        "shared_git_revision": next(iter(revisions)),
        "shared_code_fingerprint": next(iter(code_fingerprints)),
        "shared_environment": dict(
            zip(
                ("torch_version", "cuda_version", "device", "device_name"),
                next(iter(environment_signatures)),
                strict=True,
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.configs, args.artifacts_root)
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
