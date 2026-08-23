#!/usr/bin/env python3
"""Summarize the trained GB10 compute benchmarks as a strict compact record.

The sealed scheduler completion receipt for campaign
``identifiable-gb10-factorial-v1`` lists every file the campaign produced with
its byte count and SHA-256. This summarizer revalidates the ten identifiable-map
checkpoint benchmarks and the five routing benchmarks against that receipt and
against their own recorded configs, checkpoints, runner scripts, and hardware,
and only then aggregates them.

The two benchmark families report *different* tail statistics: the identifiable
runner records p10/p90 and the routing runner records p95. They are never pooled
and never relabelled. Both families are descriptive timing measurements from a
single runner on one machine; neither is a preregistered matched-compute Pareto
claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

CAMPAIGN_ID = "identifiable-gb10-factorial-v1"
EXPECTED_IDENTIFIABLE = 10
EXPECTED_ROUTING = 5
EXPECTED_SCHEDULER_STEPS = 56
ROUTED_PATH = "routed"
DENSE_PATH = "dense"
FIXED_PREFIX = "fixed_"

# Two-sided 97.5% Student-t quantiles indexed by degrees of freedom.
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


def _describe(values: list[float]) -> dict[str, Any]:
    """Descriptive across-seed summary with a Student-t interval when defined."""

    payload: dict[str, Any] = {
        "n": len(values),
        "values_in_seed_order": values,
        "mean": statistics.mean(values),
    }
    if len(values) >= 2:
        payload["sample_standard_deviation"] = statistics.stdev(values)
        degrees_of_freedom = len(values) - 1
        if degrees_of_freedom in _T_975:
            half_width = (
                _T_975[degrees_of_freedom]
                * statistics.stdev(values)
                / math.sqrt(len(values))
            )
            payload["student_t_95_ci"] = [
                payload["mean"] - half_width,
                payload["mean"] + half_width,
            ]
            payload["student_t_degrees_of_freedom"] = degrees_of_freedom
    return payload


def load_receipt(receipt_path: Path) -> dict[str, Any]:
    """Load the sealed completion receipt and index its outputs by path."""

    document = json.loads(receipt_path.read_text(encoding="utf-8"))
    if str(document.get("campaign_id")) != CAMPAIGN_ID:
        raise RuntimeError(
            f"receipt is for the wrong campaign: {document.get('campaign_id')}"
        )
    if str(document.get("status")) != "completed":
        raise RuntimeError(f"receipt is not sealed as completed: {receipt_path}")
    steps = list(document.get("completed_steps") or [])
    if len(steps) != EXPECTED_SCHEDULER_STEPS:
        raise RuntimeError(
            f"receipt records {len(steps)} completed steps; "
            f"expected {EXPECTED_SCHEDULER_STEPS}"
        )
    outputs = {
        str(entry["path"]): entry for entry in (document.get("outputs") or [])
    }
    if not outputs:
        raise RuntimeError(f"receipt records no outputs: {receipt_path}")
    return {
        "document": document,
        "outputs": outputs,
        "completed_steps": steps,
    }


def _verify_against_receipt(
    path: Path, project_root: Path, outputs: dict[str, dict[str, Any]]
) -> str:
    """Confirm a benchmark file is byte-identical to the sealed receipt entry."""

    relative = path.relative_to(project_root).as_posix()
    entry = outputs.get(relative)
    if entry is None:
        raise RuntimeError(f"benchmark is not listed in the sealed receipt: {relative}")
    observed = _sha256(path)
    if observed != str(entry["sha256"]):
        raise RuntimeError(
            f"benchmark hash differs from the sealed receipt for {relative}: "
            f"receipt={entry['sha256']} observed={observed}"
        )
    if path.stat().st_size != int(entry["bytes"]):
        raise RuntimeError(f"benchmark byte count differs from the receipt: {relative}")
    return observed


def _verify_referenced_file(
    reference: Path, recorded_sha256: str, label: str, project_root: Path
) -> str:
    candidate = reference if reference.is_absolute() else project_root / reference
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} is missing: {candidate}")
    observed = _sha256(candidate)
    if observed != recorded_sha256:
        raise RuntimeError(
            f"{label} hash mismatch for {candidate}: "
            f"recorded={recorded_sha256} observed={observed}"
        )
    return observed


def _identifiable_rows(
    paths: list[Path], project_root: Path, outputs: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        receipt_sha256 = _verify_against_receipt(path, project_root, outputs)
        document = json.loads(path.read_text(encoding="utf-8"))
        if str(document.get("status")) != "completed":
            raise RuntimeError(f"identifiable benchmark did not complete: {path}")
        provenance = document["provenance"]
        _verify_referenced_file(
            Path(provenance["config"]["path"]),
            str(provenance["config"]["sha256"]),
            "identifiable benchmark config",
            project_root,
        )
        _verify_referenced_file(
            Path(provenance["checkpoint"]["path"]),
            str(provenance["checkpoint"]["sha256"]),
            "identifiable benchmark checkpoint",
            project_root,
        )
        _verify_referenced_file(
            Path(provenance["runner"]["path"]),
            str(provenance["runner"]["sha256"]),
            "identifiable benchmark runner",
            project_root,
        )
        if int(provenance["effective_seed"]) != int(document["seed"]):
            raise RuntimeError(f"identifiable benchmark seed disagrees: {path}")
        if str(provenance["effective_ablation"]) != str(document["ablation"]):
            raise RuntimeError(f"identifiable benchmark ablation disagrees: {path}")
        latency = document["latency_ms"]
        if "p95" in latency:
            raise RuntimeError(
                f"identifiable benchmark unexpectedly reports p95: {path}"
            )
        if "p90" not in latency:
            raise RuntimeError(f"identifiable benchmark is missing p90: {path}")
        rows.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "sha256": receipt_sha256,
                "ablation": str(document["ablation"]),
                "seed": int(document["seed"]),
                "batch_size": int(document["batch_size"]),
                "iterations": int(document["iterations"]),
                "warmup": int(document["warmup"]),
                "best_epoch": int(document["best_epoch"]),
                "parameters": int(document["parameters"]),
                "chain_residual_max": float(document["chain_residual_max"]),
                "map_tolerance": float(document["map_tolerance"]),
                "materialization_checksum": float(document["materialization_checksum"]),
                "latency_ms": latency,
                "peak_cuda_memory_allocated_bytes": int(
                    document["peak_cuda_memory_allocated_bytes"]
                ),
                "peak_cuda_memory_reserved_bytes": int(
                    document["peak_cuda_memory_reserved_bytes"]
                ),
                "throughput_examples_per_second_at_median": float(
                    document["throughput_examples_per_second_at_median"]
                ),
                "measurement_scope": str(document["measurement_scope"]),
                "device": str(document["device"]),
                "device_name": str(document["device_name"]),
                "deterministic_algorithms": bool(
                    provenance["deterministic_algorithms"]
                ),
                "cublas_workspace_config": provenance.get("cublas_workspace_config"),
                "git_revision": str(provenance["git_revision"]),
                "code_fingerprint": str(provenance["code_fingerprint"]),
                "runner_sha256": str(provenance["runner"]["sha256"]),
                "checkpoint_sha256": str(provenance["checkpoint"]["sha256"]),
                "config_sha256": str(provenance["config"]["sha256"]),
                "torch": str(provenance["torch"]),
                "cuda": str(provenance["cuda"]),
                "python": str(provenance["python"]),
                "command": provenance["command"],
            }
        )
    return rows


def _routing_rows(
    paths: list[Path], project_root: Path, outputs: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        receipt_sha256 = _verify_against_receipt(path, project_root, outputs)
        document = json.loads(path.read_text(encoding="utf-8"))
        if str(document.get("status")) != "completed":
            raise RuntimeError(f"routing benchmark did not complete: {path}")
        checkpoint = document.get("checkpoint")
        if not checkpoint:
            raise RuntimeError(
                f"routing benchmark has no checkpoint and cannot be a trained "
                f"measurement: {path}"
            )
        _verify_referenced_file(
            Path(checkpoint),
            str(document["checkpoint_sha256"]),
            "routing benchmark checkpoint",
            project_root,
        )
        _verify_referenced_file(
            Path(document["config"]),
            str(document["config_sha256"]),
            "routing benchmark config",
            project_root,
        )
        environment = document["environment"]
        results = {str(entry["path"]): entry for entry in document["results"]}
        for required in (ROUTED_PATH, DENSE_PATH):
            if required not in results:
                raise RuntimeError(f"routing benchmark is missing {required}: {path}")
        fixed = {
            name: entry
            for name, entry in results.items()
            if name.startswith(FIXED_PREFIX)
        }
        if not fixed:
            raise RuntimeError(f"routing benchmark has no fixed routes: {path}")
        for entry in results.values():
            if "latency_ms_p95" not in entry:
                raise RuntimeError(f"routing benchmark is missing p95: {path}")
            if "latency_ms_p90" in entry:
                raise RuntimeError(
                    f"routing benchmark unexpectedly reports p90: {path}"
                )
        batch_sizes = {int(entry["batch_size"]) for entry in results.values()}
        iterations = {int(entry["iterations"]) for entry in results.values()}
        if len(batch_sizes) != 1 or len(iterations) != 1:
            raise RuntimeError(
                f"routing benchmark paths do not share batch size and iterations: {path}"
            )
        routed = results[ROUTED_PATH]
        dense = results[DENSE_PATH]
        fastest_fixed_name = min(
            fixed, key=lambda name: float(fixed[name]["latency_ms_median"])
        )
        fastest_fixed = fixed[fastest_fixed_name]
        recorded_speedup = float(document["dense_to_routed_speedup"])
        computed_speedup = float(dense["latency_ms_median"]) / float(
            routed["latency_ms_median"]
        )
        if not math.isclose(recorded_speedup, computed_speedup, rel_tol=1e-9):
            raise RuntimeError(
                f"recorded dense-to-routed speedup does not match the medians: {path}"
            )
        rows.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "sha256": receipt_sha256,
                "run_name": Path(str(document["checkpoint"])).parents[1].name,
                "config": Path(str(document["config"])).name,
                "config_sha256": str(document["config_sha256"]),
                "checkpoint_sha256": str(document["checkpoint_sha256"]),
                "batch_size": next(iter(batch_sizes)),
                "iterations": next(iter(iterations)),
                "precision": str(document["precision"]),
                "device": str(document["device"]),
                "device_name": str(environment["device_name"]),
                "git_revision": str(environment["git_revision"]),
                "torch": str(environment["torch"]),
                "cuda": str(environment["cuda"]),
                "python": str(environment["python"]),
                "platform": str(environment["platform"]),
                "command": environment["command"],
                "route_profile_on_benchmark_batch": document[
                    "route_profile_on_benchmark_batch"
                ],
                "median_latency_ms": {
                    name: float(entry["latency_ms_median"])
                    for name, entry in sorted(results.items())
                },
                "p95_latency_ms": {
                    name: float(entry["latency_ms_p95"])
                    for name, entry in sorted(results.items())
                },
                "peak_memory_bytes": {
                    name: int(entry["peak_memory_bytes"])
                    for name, entry in sorted(results.items())
                },
                "fastest_fixed_route": fastest_fixed_name,
                "dense_to_routed_median_ratio": computed_speedup,
                "routed_to_fastest_fixed_median_ratio": float(
                    routed["latency_ms_median"]
                )
                / float(fastest_fixed["latency_ms_median"]),
                "routed_peak_memory_below_dense": int(routed["peak_memory_bytes"])
                < int(dense["peak_memory_bytes"]),
            }
        )
    return rows


def _require_shared(rows: list[dict[str, Any]], key: str, label: str) -> Any:
    observed = {json.dumps(row[key], sort_keys=True) for row in rows}
    if len(observed) != 1:
        raise RuntimeError(f"{label} is not shared across runs: {sorted(observed)}")
    return json.loads(next(iter(observed)))


def summarize(
    *,
    project_root: Path,
    identifiable_dir: Path,
    routing_dir: Path,
    receipt_path: Path,
    expected_identifiable: int = EXPECTED_IDENTIFIABLE,
    expected_routing: int = EXPECTED_ROUTING,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    receipt = load_receipt(receipt_path)
    outputs = receipt["outputs"]

    identifiable_paths = sorted(identifiable_dir.glob("gb10-s*.json"))
    if len(identifiable_paths) != expected_identifiable:
        raise RuntimeError(
            f"expected {expected_identifiable} identifiable benchmarks; "
            f"found {len(identifiable_paths)}"
        )
    routing_paths = sorted(routing_dir.glob("routing-confirmatory-v2-s*-compute.json"))
    if len(routing_paths) != expected_routing:
        raise RuntimeError(
            f"expected {expected_routing} routing benchmarks; found {len(routing_paths)}"
        )

    identifiable = _identifiable_rows(identifiable_paths, project_root, outputs)
    routing = _routing_rows(routing_paths, project_root, outputs)

    identifiable_steps = [
        step for step in receipt["completed_steps"] if step.startswith("benchmark-identifiable-")
    ]
    routing_steps = [
        step for step in receipt["completed_steps"] if step.startswith("benchmark-routing-")
    ]
    if len(identifiable_steps) != expected_identifiable:
        raise RuntimeError(
            f"receipt records {len(identifiable_steps)} identifiable benchmark steps"
        )
    if len(routing_steps) != expected_routing:
        raise RuntimeError(
            f"receipt records {len(routing_steps)} routing benchmark steps"
        )

    shared_revision = _require_shared(
        identifiable + routing, "git_revision", "benchmark git revision"
    )
    shared_device = _require_shared(
        identifiable + routing, "device_name", "benchmark device"
    )
    _require_shared(identifiable + routing, "torch", "torch version")
    _require_shared(identifiable + routing, "cuda", "CUDA version")

    by_ablation: dict[str, Any] = {}
    for ablation in sorted({row["ablation"] for row in identifiable}):
        rows = sorted(
            (row for row in identifiable if row["ablation"] == ablation),
            key=lambda row: row["seed"],
        )
        peak = {row["peak_cuda_memory_allocated_bytes"] for row in rows}
        by_ablation[ablation] = {
            "n_seeds": len(rows),
            "seeds_in_order": [row["seed"] for row in rows],
            "median_latency_ms": _describe(
                [float(row["latency_ms"]["median"]) for row in rows]
            ),
            "mean_latency_ms": _describe(
                [float(row["latency_ms"]["mean"]) for row in rows]
            ),
            "p90_latency_ms": _describe(
                [float(row["latency_ms"]["p90"]) for row in rows]
            ),
            "peak_cuda_memory_allocated_bytes": sorted(peak),
            "peak_cuda_memory_allocated_bytes_identical_across_seeds": len(peak) == 1,
            "parameters": sorted({row["parameters"] for row in rows}),
            "max_chain_residual": max(float(row["chain_residual_max"]) for row in rows),
        }

    ablations = sorted(by_ablation)
    identifiable_contrast: dict[str, Any] | None = None
    if len(ablations) == 2:
        left, right = ablations
        left_rows = sorted(
            (row for row in identifiable if row["ablation"] == left),
            key=lambda row: row["seed"],
        )
        right_rows = sorted(
            (row for row in identifiable if row["ablation"] == right),
            key=lambda row: row["seed"],
        )
        if [row["seed"] for row in left_rows] != [row["seed"] for row in right_rows]:
            raise RuntimeError("identifiable benchmarks are not seed-matched")
        differences = [
            float(a["latency_ms"]["median"]) - float(b["latency_ms"]["median"])
            for a, b in zip(left_rows, right_rows, strict=True)
        ]
        identifiable_contrast = {
            "contrast": f"{left} minus {right}",
            "endpoint": "median forward latency in milliseconds",
            "paired_differences_in_seed_order": differences,
            **_describe(differences),
            "shared_inference_graph": True,
            "interpretation": (
                "Both ablations execute the same inference graph, so this contrast "
                "is a runner-noise check rather than an architectural comparison."
            ),
        }

    dense_to_routed = [row["dense_to_routed_median_ratio"] for row in routing]
    routed_to_fixed = [row["routed_to_fastest_fixed_median_ratio"] for row in routing]

    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "scope": (
            "Descriptive inference-timing measurements from one runner on one GB10 "
            "machine. Not a preregistered matched-compute Pareto claim and not an "
            "accuracy result."
        ),
        "analysis_provenance": {
            "summarizer": {
                "path": "scripts/summarize_compute_campaign.py",
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "sealed_receipt": {
                "path": receipt_path.relative_to(project_root).as_posix(),
                "sha256": _sha256(receipt_path),
                "launch_fingerprint": str(receipt["document"]["launch_fingerprint"]),
                "completed_at": str(receipt["document"]["completed_at"]),
                "completed_steps": len(receipt["completed_steps"]),
            },
            "shared_git_revision": shared_revision,
            "shared_device_name": shared_device,
        },
        "validation": {
            "identifiable_benchmarks_expected": expected_identifiable,
            "identifiable_benchmarks_validated": len(identifiable),
            "routing_benchmarks_expected": expected_routing,
            "routing_benchmarks_validated": len(routing),
            "receipt_hashes_verified": True,
            "config_hashes_verified": True,
            "checkpoint_hashes_verified": True,
            "identifiable_runner_hashes_verified": True,
            "routing_runner_hash_recorded_in_artifact": False,
            "routing_runner_hash_note": (
                "scripts/benchmark_compute.py does not embed its own SHA-256 in its "
                "output; the routing files are instead pinned by the sealed receipt "
                "hash, the recorded git revision, and the recorded command."
            ),
            "shared_hardware_verified": True,
            "tail_statistics_not_pooled": True,
            "status": "passed",
        },
        "measurement_protocol": {
            "identifiable": {
                "runner": "scripts/benchmark_identifiable_maps.py",
                "tail_statistic": "p90",
                "tail_statistic_note": "this runner records p10 and p90; it never records p95",
                "batch_size": sorted({row["batch_size"] for row in identifiable}),
                "warmup_iterations": sorted({row["warmup"] for row in identifiable}),
                "timed_iterations": sorted({row["iterations"] for row in identifiable}),
                "timing_method": (
                    "per-iteration wall-clock over a warmed, synchronized CUDA "
                    "forward pass; the reported median is the median of the timed "
                    "iterations"
                ),
                "measurement_scope": sorted(
                    {row["measurement_scope"] for row in identifiable}
                ),
                "deterministic_algorithms": sorted(
                    {row["deterministic_algorithms"] for row in identifiable}
                ),
                "exclusions": (
                    "data loading, training losses, exact RTD, and exact cone rank "
                    "oracles are outside the timed region"
                ),
                "raw_iteration_timings_retained": False,
                "raw_iteration_timings_note": (
                    "only the summary quantiles listed above were persisted; the "
                    "per-iteration series was not written to disk"
                ),
                "benchmark_order": [row["path"] for row in identifiable],
            },
            "routing": {
                "runner": "scripts/benchmark_compute.py",
                "tail_statistic": "p95",
                "tail_statistic_note": "this runner records p95; it never records p90",
                "batch_size": sorted({row["batch_size"] for row in routing}),
                "timed_iterations": sorted({row["iterations"] for row in routing}),
                "precision": sorted({row["precision"] for row in routing}),
                "timing_method": (
                    "per-iteration wall-clock over a warmed, synchronized CUDA "
                    "forward pass for each of the routed, dense, and fixed-route "
                    "paths measured in one process"
                ),
                "path_order_within_run": [
                    "routed",
                    "fixed_graph",
                    "fixed_cell",
                    "fixed_sheaf",
                    "dense",
                ],
                "order_caveat": (
                    "all paths were timed in a fixed order inside a single process, "
                    "so any residual thermal or allocator drift is confounded with "
                    "path order"
                ),
                "exclusions": "data loading and loss computation are outside the timed region",
                "raw_iteration_timings_retained": False,
                "raw_iteration_timings_note": (
                    "only mean, median, standard deviation, and p95 were persisted "
                    "per path"
                ),
                "benchmark_order": [row["path"] for row in routing],
            },
        },
        "identifiable": {
            "experiment": "identifiable-map-checkpoint-inference-benchmark",
            "by_ablation": by_ablation,
            "paired_contrast": identifiable_contrast,
            "runs": identifiable,
        },
        "routing": {
            "experiment": "routing-confirmatory-v2-compute-benchmark",
            "n_seeds": len(routing),
            "dense_to_routed_median_ratio": _describe(dense_to_routed),
            "routed_to_fastest_fixed_median_ratio": _describe(routed_to_fixed),
            "fastest_fixed_route_per_seed": {
                row["run_name"]: row["fastest_fixed_route"] for row in routing
            },
            "routed_peak_memory_below_dense_in_every_seed": all(
                row["routed_peak_memory_below_dense"] for row in routing
            ),
            "runs": routing,
        },
        "interpretation_guardrail": (
            "Routed inference is faster than dense inference and slower than the "
            "fastest single fixed route on this machine; both statements are "
            "descriptive medians from one runner. The identifiable ablations share "
            "one inference graph, so their timings are indistinguishable by "
            "construction. Identifiable tails are p90 and routing tails are p95; "
            "the two families are never pooled."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent.parent
    default_receipt = (
        project_root
        / "artifacts"
        / "scheduler"
        / CAMPAIGN_ID
        / "runs"
        / "44408d7adf8467e594879b46e25a1cb7fd89a7e7a5d5f3446548bcbf3ed1096e"
        / "campaign.complete.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--identifiable-dir",
        type=Path,
        default=project_root / "artifacts" / "identifiable-maps" / "benchmarks",
    )
    parser.add_argument(
        "--routing-dir", type=Path, default=project_root / "artifacts" / "benchmarks"
    )
    parser.add_argument("--receipt", type=Path, default=default_receipt)
    parser.add_argument("--expected-identifiable", type=int, default=EXPECTED_IDENTIFIABLE)
    parser.add_argument("--expected-routing", type=int, default=EXPECTED_ROUTING)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    report = summarize(
        project_root=project_root,
        identifiable_dir=args.identifiable_dir.expanduser().resolve(),
        routing_dir=args.routing_dir.expanduser().resolve(),
        receipt_path=args.receipt.expanduser().resolve(),
        expected_identifiable=args.expected_identifiable,
        expected_routing=args.expected_routing,
    )
    _atomic_json(args.output.expanduser(), report)
    print(
        json.dumps(
            {
                "identifiable_validated": report["validation"][
                    "identifiable_benchmarks_validated"
                ],
                "routing_validated": report["validation"][
                    "routing_benchmarks_validated"
                ],
                "dense_to_routed_median_ratio_mean": report["routing"][
                    "dense_to_routed_median_ratio"
                ]["mean"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
