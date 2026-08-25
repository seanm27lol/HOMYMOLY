#!/usr/bin/env python3
"""Export the curated, tracked publication evidence bundle under ``results/``.

This exporter is deliberately different from ``scripts/export_artifact_bundle.py``.
That script inventories an entire artifact tree and copies whatever small text
files it finds; it is a reproducibility snapshot, not journal evidence. This
script instead works from an explicit allowlist: every exported file is named by
a specification below, every specification declares what kind of evidence it is,
and anything matching the denylist is refused even if a specification asks for it.

Large raw artifacts never enter the bundle. Checkpoints, per-example prediction
dumps, training histories, scheduler logs, caches, and environments are excluded
by construction. Corruption reports are exported as compact derivatives that drop
the ``per_example`` array and keep the ``per_batch`` array, which is the unit of
analysis: every published corruption statistic is recomputable from the retained
rows, and the SHA-256 of the untruncated source report is recorded so the dropped
detail remains pinned.

The manifest carries no timestamp on purpose. Re-running this exporter over
unchanged evidence produces a byte-identical bundle, so a tracked ``results/``
directory only changes when the evidence changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, NamedTuple

SCHEMA_VERSION = 1
BUNDLE_ROOT = "results"
MANIFEST_NAME = "MANIFEST.json"
HISTORICAL_CONVERSION_RESULT = "results/campaigns/conversion-campaign-v1.json"
CORRECTED_CONVERSION_RESULT = "results/campaigns/conversion-campaign-v1-corrected.json"
CORRECTED_CONVERSION_RECORD_ID = "conversion-campaign-v1-correction-1"
CORRECTION_ID = "conversion-campaign-analysis-correction-1"
FROZEN_PROTOCOL_PATH = "docs/27-conversion-campaign-protocol.md"
FROZEN_PROTOCOL_SHA256 = (
    "503cc282f40d118ba1739c2afe1bfc77eaf2b1733baaddb91c0c3363e75ae2b8"
)
FROZEN_GENERATOR_PATH = "src/homymoly/data/conversion.py"
FROZEN_GENERATOR_SHA256 = (
    "c37ab1c725aa2101e88c1a0ad8fa3b279d72330feba35077e23fec930a4df69d"
)
FROZEN_LOCKFILE_PATH = "uv.lock"
FROZEN_LOCKFILE_SHA256 = (
    "05c6a5ad02db5b1651d426d157add170a8542634260ce8c265a3ee32693073bf"
)
EXPECTED_CAMPAIGN_ENVIRONMENT = {
    "python": "3.12.3",
    "torch_base": "2.13.0",
    "networkx": "3.6.1",
    "numpy": "2.5.2",
}
HISTORICAL_CONVERSION_SHA256 = (
    "836914d251db8d381aef9a2dcb0ac14a14562652f3e323dc840108b5f24d5ee1"
)
LIFTING_REPLICATION_V2_RESULT = "results/campaigns/lifting-replication-v2.json"
LIFTING_REPLICATION_V2_RECORD_ID = "independent-lifting-replication-v2"
LIFTING_REPLICATION_V2_PROTOCOL_PATH = (
    "docs/31-independent-lifting-replication-protocol.md"
)
LIFTING_REPLICATION_V2_RUNNER_PATH = "scripts/run_lifting_replication_v2.py"
LIFTING_REPLICATION_V2_SEAL_PATH = "docs/32-independent-lifting-replication-seal.json"
LIFTING_REPLICATION_V2_SEAL_SCHEMA = "homymoly-lifting-replication-seal/1"
LIFTING_REPLICATION_V2_DESIGN_COMMIT = "044322c7dc6a6255eec941dbcb76c45288a9666c"
# The sealed, independently validated support decisions of the frozen
# seven-claim family; H5 is the sole unsupported claim.
LIFTING_REPLICATION_V2_SUPPORTED = {
    "h1-soft-vs-ambient-adam": True,
    "h2-hard-cycle-vs-ambient-ls": True,
    "h3-hard-cycle-vs-soft-closed-form": True,
    "h4-hard-cycle-vs-hard-random": True,
    "h5-ridge-vs-ambient-ls": False,
    "h6-singular-surrogate-harm": True,
    "h7-rtd-bounded-benefit-futility": True,
}
BONFERRONI_T_DF28 = 2.546465223
T95_DF28 = 2.048407
T95_DF27 = 2.051831
T95_DF13 = 2.160368656
UNAFFECTED_NUMERIC_FIELDS = [
    "primary.*.per_topology fit endpoints",
    "primary.*.mean_log10_ratio",
    "primary.*.median_log10_ratio",
    "primary.*.sample_standard_deviation",
    "primary.*.interval_95",
    "primary.*.sensitivity_sign_test.pvalue_two_sided",
    "routing.trials raw defect and error values",
    "routing.threshold",
    "routing.mean_log10_ratio",
    "routing median error summaries",
]

# Suffixes and directory names that must never reach a tracked results bundle.
DENIED_SUFFIXES = frozenset(
    {
        ".bin",
        ".ckpt",
        ".gz",
        ".jsonl",
        ".log",
        ".npy",
        ".npz",
        ".pb",
        ".pt",
        ".pth",
        ".safetensors",
        ".tar",
        ".zip",
    }
)
DENIED_PARTS = frozenset(
    {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "checkpoints",
        "data",
        "logs",
        "metrics",
        "profiles",
        "steps",
        "tensorboard",
        "attempts",
    }
)
# A compact derivative must stay far below this; the cap is a tripwire, not a target.
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 48 * 1024 * 1024

DROPPED_CORRUPTION_KEYS = ("per_example",)


class Spec(NamedTuple):
    """One allowlisted evidence export."""

    source: str
    destination: str
    kind: str
    mode: str
    description: str


def specifications() -> list[Spec]:
    """The complete allowlist of publishable evidence."""

    specs: list[Spec] = [
        Spec(
            "results/summaries/gauge-corruption-campaign.json",
            "summaries/gauge-corruption-campaign.json",
            "compact-summary",
            "in-place",
            "Strict eight-seed gauge corruption summary (fixed-expert diagnostic).",
        ),
        Spec(
            "results/summaries/compute-campaign.json",
            "summaries/compute-campaign.json",
            "compact-summary",
            "in-place",
            "Strict trained GB10 compute-benchmark summary.",
        ),
        Spec(
            HISTORICAL_CONVERSION_RESULT,
            "campaigns/conversion-campaign-v1.json",
            "compact-summary",
            "in-place",
            (
                "Historical frozen conversion campaign v1 with the original, "
                "mis-scaled C1 correlations and incorrectly calibrated adjusted "
                "intervals retained for auditability; not the canonical source "
                "for current claims."
            ),
        ),
        Spec(
            CORRECTED_CONVERSION_RESULT,
            "campaigns/conversion-campaign-v1-corrected.json",
            "compact-summary",
            "in-place",
            (
                "Canonical corrected analysis of frozen conversion campaign v1; "
                "schema v2 records the Pearson and Bonferroni-interval corrections."
            ),
        ),
        Spec(
            LIFTING_REPLICATION_V2_RESULT,
            "campaigns/lifting-replication-v2.json",
            "compact-summary",
            "in-place",
            (
                "Completed independent edge-to-cycle lifting replication v2 on the "
                "sealed untouched seed block 20270101..20270136; an untouched-seed, "
                "outcome-informed, same-generator-family replication, not an "
                "independent-lab or independent-generator replication and not a "
                "pristine preregistration."
            ),
        ),
        Spec(
            "artifacts/identifiable-maps/campaign-summary.json",
            "summaries/identifiable-campaign-summary.json",
            "compact-summary",
            "copy",
            "Strict 40-run identifiable typed-map campaign summary.",
        ),
        Spec(
            "artifacts/routing-confirmatory-v2-summary.json",
            "summaries/routing-confirmatory-v2-summary.json",
            "endpoint-table",
            "copy",
            "Frozen five-seed routing confirmatory endpoint table.",
        ),
        Spec(
            "artifacts/gate3/paired_comparison_final.json",
            "gate3/paired_comparison_final.json",
            "gate-decision",
            "copy",
            "Corrected Gate-3 base paired comparison across three candidate kinds.",
        ),
    ]
    for run in ("full", "plus-chain", "plus-recon", "task-only"):
        specs.append(
            Spec(
                f"artifacts/gate3/{run}/corruption_report_final.json",
                f"gate3/{run}/corruption_report_final.compact.json",
                "corruption-report-derivative",
                "derive-corruption-report",
                f"Gate-3 base final corruption report for {run}, per-batch rows only.",
            )
        )
    gauge_runs = ["gauge-task-only", "gauge-plus-chain"]
    gauge_runs += [
        f"{prefix}-s{index:02d}"
        for index in range(4, 11)
        for prefix in ("gauge-task-only", "gauge-plus-chain")
    ]
    for run in gauge_runs:
        specs.append(
            Spec(
                f"artifacts/gate3g/{run}/corruption_report_final.json",
                f"gate3g/{run}/corruption_report_final.compact.json",
                "corruption-report-derivative",
                "derive-corruption-report",
                f"Gauge final corruption report for {run}, per-batch rows only.",
            )
        )
    for index in ["", *[f"-s{value:02d}" for value in range(4, 11)]]:
        run = f"gauge-task-only{index}"
        specs.append(
            Spec(
                f"artifacts/gate3g/{run}/paired_comparison_final.json",
                f"gate3g/{run}/paired_comparison_final.json",
                "gate-decision",
                "copy",
                f"Seed-matched gauge paired comparison anchored at {run}.",
            )
        )
    for seed in range(1, 6):
        for ablation in ("combined", "task_reconstruction"):
            specs.append(
                Spec(
                    f"artifacts/identifiable-maps/benchmarks/gb10-s{seed}-{ablation}.json",
                    f"benchmarks/identifiable/gb10-s{seed}-{ablation}.json",
                    "benchmark-summary",
                    "copy",
                    f"Trained identifiable-map inference benchmark, seed {seed}, {ablation}.",
                )
            )
        specs.append(
            Spec(
                f"artifacts/benchmarks/routing-confirmatory-v2-s{seed}-compute.json",
                f"benchmarks/routing/routing-confirmatory-v2-s{seed}-compute.json",
                "benchmark-summary",
                "copy",
                f"Trained routing compute benchmark, seed {seed}.",
            )
        )
    return specs


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_denied(relative: Path, label: str) -> None:
    if relative.suffix.casefold() in DENIED_SUFFIXES:
        raise ValueError(f"{label} has an excluded suffix: {relative}")
    denied = DENIED_PARTS.intersection(relative.parts)
    if denied:
        raise ValueError(
            f"{label} lives under an excluded directory {sorted(denied)}: {relative}"
        )


def _student_t_interval(
    values: list[float], critical: float, *, expected_n: int = 29
) -> list[float]:
    if len(values) != expected_n:
        raise ValueError(
            f"corrected conversion analysis expected {expected_n} observations, "
            f"found {len(values)}"
        )
    centre = statistics.mean(values)
    margin = critical * statistics.stdev(values) / math.sqrt(len(values))
    return [centre - margin, centre + margin]


def _same_interval(observed: Any, expected: list[float], *, atol: float = 1e-9) -> bool:
    return (
        isinstance(observed, list)
        and len(observed) == 2
        and all(
            math.isclose(float(actual), target, rel_tol=0.0, abs_tol=atol)
            for actual, target in zip(observed, expected, strict=True)
        )
    )


def _same_routing_trials(observed: Any, expected: Any) -> bool:
    """Compare rerun routing rows while tolerating only float-roundoff drift."""

    if not isinstance(observed, list) or not isinstance(expected, list):
        return False
    if len(observed) != len(expected):
        return False
    float_fields = {"defect", "cell_error", "graph_error"}
    exact_fields = {"seed", "split", "term_weight"}
    required = float_fields | exact_fields
    for actual_row, expected_row in zip(observed, expected, strict=True):
        if not isinstance(actual_row, dict) or not isinstance(expected_row, dict):
            return False
        if set(actual_row) != required or set(expected_row) != required:
            return False
        if any(actual_row[field] != expected_row[field] for field in exact_fields):
            return False
        if any(
            not math.isclose(
                float(actual_row[field]),
                float(expected_row[field]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for field in float_fields
        ):
            return False
    return True


def _pearson(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) != len(values_b) or not values_a:
        raise ValueError("Pearson inputs must have the same nonzero length")
    mean_a = statistics.mean(values_a)
    mean_b = statistics.mean(values_b)
    centred_a = [value - mean_a for value in values_a]
    centred_b = [value - mean_b for value in values_b]
    denominator = math.sqrt(
        sum(value * value for value in centred_a)
        * sum(value * value for value in centred_b)
    )
    if denominator <= 0.0:
        raise ValueError("Pearson inputs must both have positive variance")
    return sum(a * b for a, b in zip(centred_a, centred_b, strict=True)) / denominator


def _validate_corrected_primary_intervals(
    corrected: dict[str, Any], historical: dict[str, Any]
) -> None:
    """Recompute df=28 Bonferroni intervals from unchanged fit endpoints."""

    corrected_primary = corrected.get("primary")
    historical_primary = historical.get("primary")
    if not isinstance(corrected_primary, dict) or not isinstance(
        historical_primary, dict
    ):
        raise TypeError("conversion evidence has no primary contrast records")
    for term in ("exact", "cone", "rtd"):
        current = corrected_primary.get(term)
        previous = historical_primary.get(term)
        if not isinstance(current, dict) or not isinstance(previous, dict):
            raise TypeError(f"conversion evidence has no {term} primary contrast")
        current_rows = current.get("per_topology")
        previous_rows = previous.get("per_topology")
        if not isinstance(current_rows, list) or not isinstance(previous_rows, list):
            raise TypeError(f"{term} contrast has no per-topology endpoints")
        if not all(isinstance(row, dict) for row in [*current_rows, *previous_rows]):
            raise TypeError(f"{term} per-topology endpoints must be records")
        current_by_seed = {row.get("seed"): row for row in current_rows}
        previous_by_seed = {row.get("seed"): row for row in previous_rows}
        if (
            len(current_rows) != 29
            or len(current_by_seed) != 29
            or set(current_by_seed) != set(previous_by_seed)
        ):
            raise ValueError(f"{term} contrast does not retain the 29 historical seeds")

        ratios: list[float] = []
        for seed in sorted(current_by_seed):
            row = current_by_seed[seed]
            old_row = previous_by_seed[seed]
            held_out = float(row["held_out_mse"])
            baseline = float(row["baseline_held_out_mse"])
            if held_out != float(old_row["held_out_mse"]) or baseline != float(
                old_row["baseline_held_out_mse"]
            ):
                raise ValueError(
                    f"{term} fit endpoints changed during analysis correction"
                )
            ratios.append(math.log10(held_out / baseline))

        expected = _student_t_interval(ratios, BONFERRONI_T_DF28)
        observed = current.get("interval_bonferroni_98_33")
        if not _same_interval(observed, expected):
            raise ValueError(
                f"{term} adjusted interval was not recomputed with df=28 "
                f"t={BONFERRONI_T_DF28}"
            )
        if _same_interval(previous.get("interval_bonferroni_98_33"), expected):
            raise ValueError(
                f"historical {term} interval unexpectedly already uses the corrected quantile"
            )
        improves = expected[1] < 0.0
        harms = expected[0] > 0.0
        if current.get("improves_confirmatory") is not improves:
            raise ValueError(f"{term} improvement decision disagrees with its interval")
        if current.get("harms_confirmatory") is not harms:
            raise ValueError(f"{term} harm decision disagrees with its interval")
        if (
            previous.get("improves_confirmatory") is not improves
            or previous.get("harms_confirmatory") is not harms
        ):
            raise ValueError(f"{term} decision changed after the interval correction")


def _validate_corrected_c1(
    corrected: dict[str, Any], historical: dict[str, Any]
) -> None:
    """Recompute every Pearson coefficient from seed-keyed raw fit endpoints."""

    c1 = corrected.get("c1")
    historical_c1 = historical.get("c1")
    if not isinstance(c1, dict) or not isinstance(historical_c1, dict):
        raise TypeError("conversion evidence has no C1 record")
    weights = c1.get("weights_swept")
    rows = c1.get("per_topology")
    if not isinstance(weights, list) or not isinstance(rows, list):
        raise TypeError("corrected C1 must retain weights and per-topology raw fits")
    if c1.get("n") != 29 or len(weights) != 9 or len(rows) != 29:
        raise ValueError("corrected C1 must retain nine weights for each of 29 seeds")
    primary = corrected.get("primary")
    exact = primary.get("exact") if isinstance(primary, dict) else None
    primary_rows = exact.get("per_topology") if isinstance(exact, dict) else None
    if not isinstance(primary_rows, list):
        raise TypeError("corrected C1 cannot be matched to primary topology seeds")
    primary_seeds = {row.get("seed") for row in primary_rows}

    correlations: list[float] = []
    seen_seeds: set[Any] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("seed") in seen_seeds:
            raise ValueError("corrected C1 rows must have unique seed keys")
        seen_seeds.add(row.get("seed"))
        fits = row.get("fits")
        if not isinstance(fits, list) or len(fits) != len(weights):
            raise ValueError("each corrected C1 row must retain all nine raw fits")
        if not all(isinstance(fit, dict) for fit in fits):
            raise TypeError("corrected C1 raw fits must be records")
        if [fit.get("weight") for fit in fits] != weights:
            raise ValueError("corrected C1 fit weights do not match the frozen sweep")
        defects: list[float] = []
        held_out_errors: list[float] = []
        for fit in fits:
            defect = float(fit["boundary_compatibility_defect_frobenius"])
            held_out = float(fit["held_out_mse"])
            if not math.isfinite(defect) or not math.isfinite(held_out):
                raise ValueError("corrected C1 raw fits must be finite")
            defects.append(math.log10(max(defect, 1e-30)))
            held_out_errors.append(math.log10(max(held_out, 1e-300)))
        correlation = _pearson(defects, held_out_errors)
        if not math.isclose(
            float(row.get("correlation")), correlation, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "corrected C1 correlation is not Pearson r of its raw fits"
            )
        correlations.append(correlation)

    if seen_seeds != primary_seeds:
        raise ValueError("corrected C1 seeds do not match the primary topology seeds")
    if c1.get("per_topology_correlation") != [row["correlation"] for row in rows]:
        raise ValueError(
            "corrected C1 correlation vector does not match its seed-keyed rows"
        )
    if not math.isclose(
        float(c1.get("mean_within_topology_correlation")),
        statistics.mean(correlations),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("corrected C1 mean does not match its raw-fit correlations")
    interval = _student_t_interval(correlations, T95_DF28)
    if not _same_interval(c1.get("interval_95"), interval):
        raise ValueError(
            "corrected C1 interval does not match its raw-fit correlations"
        )
    if c1.get("positive_topologies") != sum(value > 0.0 for value in correlations):
        raise ValueError("corrected C1 positive-topology count is inconsistent")
    supported = interval[0] > 0.0
    if c1.get("supported") is not supported:
        raise ValueError(
            "corrected C1 path-association flag disagrees with its interval"
        )
    if c1.get("supported_claim") != (
        "positive within-seed regularization-path association only"
    ):
        raise ValueError("corrected C1 supported claim is not narrowly identified")
    if c1.get("does_not_establish") != [
        "independent predictive information",
        "off-path calibration",
        "causal effect",
    ]:
        raise ValueError("corrected C1 claim limits are missing")
    if historical_c1.get("supported") is not supported:
        raise ValueError("C1 path-association flag changed after correcting Pearson r")


def _validate_locked_environment(project_root: Path, document: dict[str, Any]) -> None:
    provenance = document.get("provenance")
    environment = (
        provenance.get("environment") if isinstance(provenance, dict) else None
    )
    if not isinstance(environment, dict):
        raise TypeError("corrected campaign has no locked environment provenance")
    if environment.get("expected") != EXPECTED_CAMPAIGN_ENVIRONMENT:
        raise ValueError("corrected campaign records an unexpected locked environment")
    if environment.get("matches_expected") is not True:
        raise ValueError("corrected campaign did not run in its locked environment")
    actual = environment.get("actual")
    if not isinstance(actual, dict) or any(
        actual.get(name) != expected
        for name, expected in EXPECTED_CAMPAIGN_ENVIRONMENT.items()
    ):
        raise ValueError("recorded runtime does not match the lock-derived versions")
    lockfile = environment.get("lockfile")
    if lockfile != {
        "path": FROZEN_LOCKFILE_PATH,
        "sha256": FROZEN_LOCKFILE_SHA256,
        "frozen_campaign_sha256": FROZEN_LOCKFILE_SHA256,
    }:
        raise ValueError("corrected campaign does not pin the frozen lockfile")
    local_lock = project_root / FROZEN_LOCKFILE_PATH
    if not local_lock.is_file() or _sha256(local_lock) != FROZEN_LOCKFILE_SHA256:
        raise ValueError("frozen campaign lockfile is missing or has changed")
    dependencies = provenance.get("dependencies")
    if not isinstance(dependencies, dict) or any(
        dependencies.get(name) != actual.get(name)
        for name in ("torch", "networkx", "numpy")
    ):
        raise ValueError(
            "top-level dependency provenance disagrees with the locked runtime"
        )
    execution = provenance.get("execution")
    if not isinstance(execution, dict) or execution.get("tensor_device") != "cpu":
        raise ValueError("corrected conversion campaign was not recorded as a CPU run")
    runner = project_root / "scripts" / "run_conversion_campaign.py"
    if not runner.is_file() or provenance.get("runner_sha256") != _sha256(runner):
        raise ValueError("corrected campaign runner is missing or has changed")
    if provenance.get("git_status") != "":
        raise ValueError("corrected campaign provenance records a dirty worktree")


def _validate_design_audit(document: dict[str, Any]) -> None:
    design = document.get("design")
    if not isinstance(design, dict):
        raise TypeError("corrected campaign has no design record")
    if design.get("training_pairs") != 16 or design.get("held_out_pairs") != 3072:
        raise ValueError(
            "corrected campaign changed the frozen train/test sample counts"
        )
    if design.get("training_label_noise") != {
        "distribution": "independent zero-mean Gaussian",
        "standard_deviation": 0.02,
    }:
        raise ValueError("corrected campaign does not disclose training-label noise")
    if design.get("held_out_targets") != "noiseless ground-truth linear responses":
        raise ValueError(
            "corrected campaign does not identify noiseless held-out targets"
        )
    if design.get("primary_inference_unit") != (
        "one eligible generator seed, jointly determining topology, training "
        "predictors and noise, and noiseless held-out predictors"
    ):
        raise ValueError(
            "corrected campaign does not identify the seed-level inference unit"
        )
    if design.get("exchangeability_assumption") != (
        "eligible seed-level joint replicates are exchangeable for the Student-t "
        "interval; the design does not separate topology heterogeneity from "
        "data/noise-realisation heterogeneity"
    ):
        raise ValueError("corrected campaign omits the exchangeability limitation")

    dimensions = design.get("eligible_topology_dimensions")
    if not isinstance(dimensions, list) or not all(
        isinstance(row, dict) for row in dimensions
    ):
        raise TypeError("corrected campaign has no topology-dimension audit")
    if len(dimensions) != 29 or len({row.get("seed") for row in dimensions}) != 29:
        raise ValueError("topology-dimension audit must retain 29 unique seeds")
    edges = [int(row["edges"]) for row in dimensions]
    faces = [int(row["faces"]) for row in dimensions]
    computed = {
        "training_pairs": 16,
        "median_edges_per_output_row": statistics.median(edges),
        "median_cycle_subspace_dimension": statistics.median(faces),
        "full_row_regressions_with_edges_gt_training_pairs": sum(
            value > 16 for value in edges
        ),
        "cycle_subspaces_with_faces_le_training_pairs": sum(
            value <= 16 for value in faces
        ),
        "seeds_moving_from_edges_gt_n_to_faces_le_n": sum(
            edge > 16 and face <= 16 for edge, face in zip(edges, faces, strict=True)
        ),
    }
    canonical = {
        "training_pairs": 16,
        "median_edges_per_output_row": 23,
        "median_cycle_subspace_dimension": 11,
        "full_row_regressions_with_edges_gt_training_pairs": 21,
        "cycle_subspaces_with_faces_le_training_pairs": 24,
        "seeds_moving_from_edges_gt_n_to_faces_le_n": 16,
    }
    geometry = design.get("scarce_probe_geometry")
    if not isinstance(geometry, dict):
        raise TypeError("corrected campaign has no scarce-probe geometry audit")
    if computed != canonical or any(
        geometry.get(key) != value for key, value in canonical.items()
    ):
        raise ValueError(
            "scarce-probe geometry counts do not match the canonical 29 seeds"
        )

    definitions = design.get("objective_definitions")
    compatibility = definitions.get("exact") if isinstance(definitions, dict) else None
    rtd = definitions.get("rtd") if isinstance(definitions, dict) else None
    if not isinstance(compatibility, dict) or any(
        (
            compatibility.get("display_name") != "boundary-compatibility penalty",
            compatibility.get("formula") != "mean((B1 @ W.T)^2)",
            compatibility.get("frozen_key_is_historical_shorthand") is not True,
            compatibility.get("is_exactness_of_a_sequence") is not False,
        )
    ):
        raise ValueError(
            "frozen key 'exact' is not explicitly identified as boundary compatibility"
        )
    if compatibility.get("structural_side_information") != (
        "B1 determines the target cycle subspace ker(B1); the penalty does not "
        "directly use B2 or response labels, but the committed deterministic "
        "generator algorithm can recover its noncanonical B2 basis from the graph"
    ):
        raise ValueError("boundary-compatibility side information is not disclosed")
    if (
        not isinstance(rtd, dict)
        or rtd.get("is_representation_topology_divergence") is not False
        or rtd.get("target_alignment")
        != (
            "the generated truth discards cut-space directions while this surrogate "
            "asks the lower-dimensional output to preserve the full source distance geometry"
        )
    ):
        raise ValueError("RTD-inspired surrogate target misalignment is not disclosed")


def _validate_withdrawn_routing(
    corrected: dict[str, Any], historical: dict[str, Any]
) -> None:
    routing = corrected.get("routing")
    previous = historical.get("routing")
    if not isinstance(routing, dict) or not isinstance(previous, dict):
        raise TypeError("conversion evidence has no routing audit")
    if "interval_95" in routing:
        raise ValueError("withdrawn H5 must not expose an inferential interval_95")
    if (
        routing.get("supported") is not None
        or routing.get("decision") != "withdrawn-non-informative"
        or routing.get("decision_informative") is not False
    ):
        raise ValueError("H5 inferential support was not withdrawn")
    if not _same_routing_trials(routing.get("trials"), previous.get("trials")):
        raise ValueError("routing raw trials changed during the H5 audit correction")
    for key in (
        "threshold",
        "mean_log10_ratio",
        "median_routed",
        "median_always_cell",
        "median_always_graph",
    ):
        if not math.isclose(
            float(routing.get(key)),
            float(previous.get(key)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"routing {key} changed during the H5 audit correction")

    trials = routing.get("trials")
    if not isinstance(trials, list) or not all(isinstance(row, dict) for row in trials):
        raise TypeError("routing audit has no retained trial records")
    evaluation = [row for row in trials if row.get("split") == "evaluation"]
    threshold = float(routing["threshold"])
    values: list[float] = []
    clustered: dict[int, list[float]] = {}
    selected_lower = 0
    for row in evaluation:
        cell = float(row["cell_error"])
        graph = float(row["graph_error"])
        routed = cell if float(row["defect"]) <= threshold else graph
        best = min(cell, graph)
        if routed == best:
            selected_lower += 1
        value = math.log10(routed / best)
        if value < -1e-12:
            raise ValueError("routing endpoint violates its algebraic nonnegativity")
        values.append(value)
        clustered.setdefault(int(row["seed"]), []).append(value)
    if (
        len(values) != 28
        or len(clustered) != 14
        or any(len(seed_values) != 2 for seed_values in clustered.values())
    ):
        raise ValueError("H5 audit must retain 14 topology clusters with two fits each")
    naive = _student_t_interval(values, T95_DF27, expected_n=28)
    if not _same_interval(
        routing.get("historical_pseudoreplicated_interval_95"), naive
    ):
        raise ValueError(
            "historical row-naive H5 interval was not independently reproduced"
        )
    if not _same_interval(previous.get("interval_95"), naive):
        raise ValueError("schema-v1 H5 interval does not match its retained raw trials")

    cluster_means = [statistics.mean(clustered[seed]) for seed in sorted(clustered)]
    descriptive = _student_t_interval(cluster_means, T95_DF13, expected_n=14)
    if not _same_interval(
        routing.get("topology_clustered_descriptive_interval_95"), descriptive
    ):
        raise ValueError("topology-clustered descriptive H5 interval is inconsistent")
    summaries = routing.get("topology_cluster_summaries")
    if not isinstance(summaries, list) or len(summaries) != 14:
        raise ValueError("H5 audit must publish one summary per topology cluster")
    by_seed = {row.get("seed"): row for row in summaries if isinstance(row, dict)}
    if set(by_seed) != set(clustered):
        raise ValueError("H5 cluster summaries do not match evaluation topology seeds")
    for seed, seed_values in clustered.items():
        summary = by_seed[seed]
        if summary.get("n_correlated_fits") != 2 or not _same_interval(
            [summary.get("mean_log10_ratio"), summary.get("mean_log10_ratio")],
            [statistics.mean(seed_values), statistics.mean(seed_values)],
            atol=1e-12,
        ):
            raise ValueError("H5 cluster summary is inconsistent with retained trials")
        observed_values = summary.get("endpoint_values")
        if (
            not isinstance(observed_values, list)
            or len(observed_values) != 2
            or any(
                not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
                for actual, expected in zip(observed_values, seed_values, strict=True)
            )
        ):
            raise ValueError("H5 cluster endpoint values are inconsistent")
    if (
        routing.get("evaluation_trials") != 28
        or routing.get("evaluation_topology_clusters") != 14
    ):
        raise ValueError("H5 routing counts do not declare 28 rows in 14 clusters")
    if routing.get("selected_lower_error_rows") != selected_lower:
        raise ValueError("H5 selected-lower-error count is inconsistent")


def validate_corrected_conversion_record(
    project_root: Path, document: dict[str, Any]
) -> None:
    """Reject a canonical correction that cannot prove its frozen lineage."""

    schema = document.get("schema")
    if document.get("schema_version") != 2 or schema != {
        "name": "homymoly.conversion-campaign-result",
        "version": 2,
        "record_id": CORRECTED_CONVERSION_RECORD_ID,
    }:
        raise ValueError(
            "corrected conversion campaign has an unexpected schema-v2 identity"
        )
    if document.get("campaign") != "conversion-campaign-v1":
        raise ValueError(
            "corrected conversion campaign has an unexpected campaign identity"
        )

    correction = document.get("correction")
    if not isinstance(correction, dict) or correction.get("id") != CORRECTION_ID:
        raise ValueError(
            "corrected conversion campaign has no recognized correction record"
        )
    if correction.get("protocol_modified") is not False:
        raise ValueError(
            "analysis correction must not claim that the frozen protocol changed"
        )
    if correction.get("data_or_fit_settings_modified") is not False:
        raise ValueError(
            "analysis correction must not claim that data or fit settings changed"
        )
    reasons = correction.get("reasons")
    if not isinstance(reasons, list) or not all(
        isinstance(reason, dict) for reason in reasons
    ):
        raise TypeError("analysis correction reasons must be a list of records")
    by_id = {reason.get("id"): reason for reason in reasons}
    if len(reasons) != 3 or set(by_id) != {
        "c1-pearson-normalisation",
        "bonferroni-critical-value",
        "h5-pseudoreplication-and-impossible-decision",
    }:
        raise ValueError("analysis correction does not identify all three audit issues")
    if by_id["c1-pearson-normalisation"].get("corrected_estimator") != (
        "sum((x-x_bar)*(y-y_bar)) / sqrt(sum((x-x_bar)^2)*sum((y-y_bar)^2))"
    ):
        raise ValueError(
            "C1 correction does not name the conventional Pearson estimator"
        )
    if by_id["bonferroni-critical-value"].get("corrected_quantile") != (
        "t.ppf(1 - (0.05 / 3) / 2, df) = t.ppf(0.991666..., df)"
    ):
        raise ValueError(
            "interval correction does not name the frozen Bonferroni quantile"
        )
    if by_id["h5-pseudoreplication-and-impossible-decision"].get(
        "corrected_handling"
    ) != (
        "withdraw inferential support; retain the historical naive interval by "
        "name and add a topology-clustered descriptive interval over 14 "
        "within-topology means"
    ):
        raise ValueError(
            "H5 correction does not name the withdrawal and clustered audit"
        )
    if correction.get("decision_changes") != {
        "exact": False,
        "cone": False,
        "rtd": False,
        "c1": False,
    }:
        raise ValueError("analysis correction must record that no decisions changed")
    if correction.get("unaffected_numeric_fields") != UNAFFECTED_NUMERIC_FIELDS:
        raise ValueError(
            "analysis correction does not enumerate the granular unchanged fields"
        )
    withdrawal = correction.get("decision_withdrawals")
    if not isinstance(withdrawal, dict) or withdrawal.get("routing") != (
        "the endpoint's support rule was impossible and its schema-v1 interval "
        "treated correlated rows as independent"
    ):
        raise ValueError("analysis correction does not withdraw the routing decision")

    supersedes = correction.get("supersedes")
    if not isinstance(supersedes, dict):
        raise TypeError(
            "analysis correction does not identify the superseded schema-v1 record"
        )
    if supersedes.get("path") != HISTORICAL_CONVERSION_RESULT:
        raise ValueError(
            "analysis correction points to an unexpected historical result"
        )
    if supersedes.get("schema_version") != 1:
        raise ValueError("analysis correction does not supersede schema version 1")
    if supersedes.get("sha256") != HISTORICAL_CONVERSION_SHA256:
        raise ValueError(
            "analysis correction records the wrong historical result SHA-256"
        )
    historical = project_root / HISTORICAL_CONVERSION_RESULT
    if not historical.is_file() or _sha256(historical) != HISTORICAL_CONVERSION_SHA256:
        raise ValueError("historical conversion result is missing or has changed")
    historical_document = json.loads(historical.read_text(encoding="utf-8"))

    protocol = document.get("protocol")
    if not isinstance(protocol, dict):
        raise TypeError("corrected conversion campaign has no protocol provenance")
    if (
        protocol.get("path") != FROZEN_PROTOCOL_PATH
        or protocol.get("sha256") != FROZEN_PROTOCOL_SHA256
        or protocol.get("frozen_sha256") != FROZEN_PROTOCOL_SHA256
        or protocol.get("document_hash_matches_frozen") is not True
        or protocol.get("execution_matches_frozen_text") is not False
    ):
        raise ValueError(
            "corrected conversion campaign does not distinguish frozen text from execution"
        )
    deviations = protocol.get("implementation_deviations")
    if not isinstance(deviations, list) or len(deviations) != 1:
        raise ValueError(
            "corrected conversion campaign must disclose one protocol deviation"
        )
    deviation = deviations[0]
    if not isinstance(deviation, dict) or deviation.get("id") != (
        "compatibility-mean-normalisation"
    ):
        raise ValueError(
            "corrected conversion campaign omits the known execution deviation"
        )
    if (
        deviation.get("frozen_text") != "squared Frobenius norm ||B1 @ W.T||_F^2"
        or deviation.get("executed_objective") != "elementwise mean((B1 @ W.T)^2)"
        or deviation.get("fit_implementation_changed_in_correction") is not False
    ):
        raise ValueError(
            "protocol deviation does not faithfully describe the executed fit"
        )
    frozen_protocol = project_root / FROZEN_PROTOCOL_PATH
    if (
        not frozen_protocol.is_file()
        or _sha256(frozen_protocol) != FROZEN_PROTOCOL_SHA256
    ):
        raise ValueError("frozen conversion protocol is missing or has changed")

    provenance = document.get("provenance")
    generator = provenance.get("generator") if isinstance(provenance, dict) else None
    if not isinstance(generator, dict):
        raise TypeError("corrected conversion campaign has no generator provenance")
    if (
        generator.get("class") != "homymoly.data.conversion.ConversionDataset"
        or generator.get("path") != FROZEN_GENERATOR_PATH
        or generator.get("sha256") != FROZEN_GENERATOR_SHA256
        or generator.get("frozen_campaign_sha256") != FROZEN_GENERATOR_SHA256
        or generator.get("matches_frozen_campaign") is not True
    ):
        raise ValueError(
            "corrected conversion campaign does not pin the frozen generator"
        )
    frozen_generator = project_root / FROZEN_GENERATOR_PATH
    if (
        not frozen_generator.is_file()
        or _sha256(frozen_generator) != FROZEN_GENERATOR_SHA256
    ):
        raise ValueError("frozen conversion generator is missing or has changed")

    _validate_locked_environment(project_root, document)
    _validate_design_audit(document)
    _validate_corrected_primary_intervals(document, historical_document)
    _validate_corrected_c1(document, historical_document)
    _validate_withdrawn_routing(document, historical_document)


def _is_full_git_revision(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _support_decision_keys(node: Any, path: str = "c1") -> list[str]:
    """Locate inferential-decision keys that must never appear under C1."""

    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            location = f"{path}.{key}"
            lowered = str(key).lower()
            if "support" in lowered or "multiplicity" in lowered:
                found.append(location)
            found.extend(_support_decision_keys(value, location))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_support_decision_keys(value, f"{path}[{index}]"))
    return found


def validate_lifting_replication_v2_record(
    project_root: Path, document: dict[str, Any]
) -> None:
    """Reject a v2 replication result that cannot prove its sealed lineage."""

    if document.get("schema") != {
        "name": "homymoly.independent-lifting-replication-result",
        "version": 1,
        "record_id": LIFTING_REPLICATION_V2_RECORD_ID,
    }:
        raise ValueError("v2 lifting replication has an unexpected schema identity")
    if document.get("campaign") != LIFTING_REPLICATION_V2_RECORD_ID:
        raise ValueError("v2 lifting replication has an unexpected campaign identity")
    if document.get("status") != "complete":
        raise ValueError("only a complete v2 lifting replication permits inference")

    eligibility = document.get("eligibility")
    if not isinstance(eligibility, dict):
        raise TypeError("v2 lifting replication has no eligibility accounting")
    eligible_seeds = eligibility.get("eligible_seeds")
    ineligible = eligibility.get("ineligible")
    generation_failures = eligibility.get("generation_failures")
    if (
        eligibility.get("declared") != 36
        or eligibility.get("eligible") != 33
        or not isinstance(eligible_seeds, list)
        or len(eligible_seeds) != 33
        or len(set(eligible_seeds)) != 33
    ):
        raise ValueError(
            "v2 lifting replication must retain 33 unique eligible seeds of 36 declared"
        )
    if not isinstance(ineligible, list) or not isinstance(generation_failures, list):
        raise TypeError("v2 ineligible and generation-failure rows must be lists")
    if generation_failures:
        raise ValueError("v2 lifting replication must record zero generation failures")
    if len(ineligible) != 3 or any(
        not isinstance(row, dict)
        or not isinstance(row.get("seed"), int)
        or not row.get("reason")
        for row in ineligible
    ):
        raise ValueError("v2 ineligible rows must carry a seed and an explicit reason")
    if set(eligible_seeds) | {row["seed"] for row in ineligible} != set(
        range(20270101, 20270137)
    ):
        raise ValueError("v2 seed accounting does not reconstruct the sealed block")
    audit = document.get("audit")
    if not isinstance(audit, dict):
        raise TypeError("v2 lifting replication has no audit block")
    if (
        audit.get("declared_seeds") != 36
        or audit.get("eligible_seeds") != 33
        or audit.get("eligible_seed_ids") != eligible_seeds
        or audit.get("ineligible_seed_rows") != 3
        or audit.get("generation_failure_rows") != 0
    ):
        raise ValueError("v2 audit block disagrees with the eligibility accounting")

    provenance = document.get("provenance")
    if not isinstance(provenance, dict):
        raise TypeError("v2 lifting replication has no provenance record")
    seal_path = project_root / LIFTING_REPLICATION_V2_SEAL_PATH
    if not seal_path.is_file():
        raise ValueError("v2 design-seal record is missing")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("schema") != LIFTING_REPLICATION_V2_SEAL_SCHEMA:
        raise ValueError("v2 design-seal record has an unexpected schema")
    recorded_seal = provenance.get("seal")
    if not isinstance(recorded_seal, dict):
        raise TypeError("v2 lifting replication does not record the design seal")
    if (
        recorded_seal.get("path") != LIFTING_REPLICATION_V2_SEAL_PATH
        or recorded_seal.get("schema") != LIFTING_REPLICATION_V2_SEAL_SCHEMA
        or recorded_seal.get("committed_at_head") is not True
        or recorded_seal.get("sha256") != _sha256(seal_path)
    ):
        raise ValueError("v2 provenance does not pin the committed design-seal record")

    pinned = {
        "protocol": LIFTING_REPLICATION_V2_PROTOCOL_PATH,
        "runner": LIFTING_REPLICATION_V2_RUNNER_PATH,
        "generator": FROZEN_GENERATOR_PATH,
    }
    for name, relative in pinned.items():
        record = provenance.get(name)
        if not isinstance(record, dict):
            raise TypeError(f"v2 lifting replication has no {name} provenance")
        local = project_root / relative
        if not local.is_file():
            raise ValueError(f"v2 {name} file is missing: {relative}")
        digest = _sha256(local)
        if record.get("path") != relative or record.get("sha256") != digest:
            raise ValueError(f"v2 {name} provenance does not match the actual file")
        if seal.get(f"{name}_sha256") != digest:
            raise ValueError(f"v2 {name} hash disagrees with the design-seal record")
    environment = provenance.get("environment")
    lockfile = environment.get("lockfile") if isinstance(environment, dict) else None
    local_lock = project_root / FROZEN_LOCKFILE_PATH
    if not local_lock.is_file() or _sha256(local_lock) != FROZEN_LOCKFILE_SHA256:
        raise ValueError("frozen campaign lockfile is missing or has changed")
    if (
        not isinstance(lockfile, dict)
        or lockfile.get("path") != FROZEN_LOCKFILE_PATH
        or lockfile.get("sha256") != FROZEN_LOCKFILE_SHA256
    ):
        raise ValueError("v2 lifting replication does not pin the frozen lockfile")
    if seal.get("lock_sha256") != FROZEN_LOCKFILE_SHA256:
        raise ValueError("v2 lock hash disagrees with the design-seal record")

    design_commit = provenance.get("design_commit")
    if design_commit != LIFTING_REPLICATION_V2_DESIGN_COMMIT:
        raise ValueError("v2 design commit is not the sealed design commit")
    if (
        seal.get("design_commit") != design_commit
        or recorded_seal.get("design_commit") != design_commit
    ):
        raise ValueError("v2 design commit disagrees with the design-seal record")
    execution_revision = provenance.get("execution_revision")
    if not _is_full_git_revision(design_commit) or not _is_full_git_revision(
        execution_revision
    ):
        raise ValueError("v2 commits must be recorded as full 40-hex revisions")
    # HEAD has legitimately advanced since the sealed execution, so the revision
    # is checked only against the record itself, never against live git state.
    if execution_revision != provenance.get("git_revision"):
        raise ValueError(
            "v2 execution revision disagrees with the recorded git revision"
        )
    if execution_revision == design_commit:
        raise ValueError("v2 execution revision must postdate the design commit")

    primary = document.get("primary")
    if not isinstance(primary, dict):
        raise TypeError("v2 lifting replication has no primary claim family")
    if primary.get("family_size") != 7:
        raise ValueError("v2 primary family must contain exactly seven claims")
    claims = primary.get("claims")
    seal_family = seal.get("primary_family")
    if (
        not isinstance(claims, list)
        or not all(isinstance(claim, dict) for claim in claims)
        or not isinstance(seal_family, list)
    ):
        raise TypeError("v2 primary family and seal family must be claim records")
    if [claim.get("id") for claim in claims] != [
        claim.get("id") for claim in seal_family
    ]:
        raise ValueError("v2 claim ids do not match the sealed primary family")
    observed = {claim.get("id"): claim.get("supported") for claim in claims}
    if observed != LIFTING_REPLICATION_V2_SUPPORTED:
        raise ValueError("v2 support decisions disagree with the validated outcome")

    c1 = document.get("c1")
    if not isinstance(c1, dict):
        raise TypeError("v2 lifting replication has no C1 record")
    forbidden = sorted(_support_decision_keys(c1))
    if forbidden:
        raise ValueError(
            f"v2 C1 must not carry support or multiplicity decisions: {forbidden}"
        )


# Where each evidence shape records the revision that generated it. The first
# path that resolves wins; every exported file must agree on the result.
REVISION_PATHS = (
    ("analysis_provenance", "git_commit"),
    ("analysis_provenance", "shared_git_revision"),
    ("shared_provenance", "git_revision"),
    ("shared_git_revision",),
    ("provenance", "git_revision"),
    ("environment", "git_revision"),
    ("git", "commit"),
    ("inputs", "baseline", "git", "commit"),
)


def evidence_revision(document: Any) -> str | None:
    """Read the generating git revision out of one piece of evidence."""

    for path in REVISION_PATHS:
        cursor: Any = document
        for key in path:
            if not isinstance(cursor, dict) or key not in cursor:
                cursor = None
                break
            cursor = cursor[key]
        if isinstance(cursor, str) and cursor:
            return cursor
    return None


def compact_corruption_report(document: dict[str, Any]) -> dict[str, Any]:
    """Drop the per-example array and keep everything the analysis consumes.

    The published corruption statistics are computed over ``per_batch`` rows, so
    removing ``per_example`` changes no reported number. The derivative records
    which keys were dropped and how many rows each held.
    """

    if "per_batch" not in document:
        raise ValueError("corruption report has no per_batch rows to retain")
    compact = {
        key: value
        for key, value in document.items()
        if key not in DROPPED_CORRUPTION_KEYS
    }
    compact["_derivative"] = {
        "derivation": "per-batch-lossless-v1",
        "dropped_keys": {
            key: {"rows": len(document[key])}
            for key in DROPPED_CORRUPTION_KEYS
            if key in document
        },
        "retained_per_batch_rows": len(document["per_batch"]),
        "guarantee": (
            "Every published corruption statistic in this report is computed from "
            "the retained per_batch rows; the dropped arrays are per-example detail "
            "that no reported number depends on."
        ),
    }
    return compact


def export(
    *,
    project_root: Path,
    output_root: Path,
    specs: list[Spec] | None = None,
    max_file_bytes: int = MAX_FILE_BYTES,
    max_total_bytes: int = MAX_TOTAL_BYTES,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()
    specs = specifications() if specs is None else specs

    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for spec in specs:
        source = project_root / spec.source
        destination_relative = Path(spec.destination)
        _reject_denied(Path(spec.source), "source")
        _reject_denied(destination_relative, "destination")
        if not source.is_file():
            raise FileNotFoundError(f"required evidence is missing: {spec.source}")

        destination = output_root / destination_relative
        source_sha256 = _sha256(source)
        entry: dict[str, Any] = {
            "path": destination_relative.as_posix(),
            "kind": spec.kind,
            "description": spec.description,
            "source": spec.source,
            "source_sha256": source_sha256,
            "derivation": None,
        }

        if spec.mode == "in-place":
            if destination.resolve() != source.resolve():
                raise ValueError(
                    f"in-place evidence must already sit at its destination: {spec.source}"
                )
            payload = source.read_bytes()
        elif spec.mode == "copy":
            payload = source.read_bytes()
        elif spec.mode == "derive-corruption-report":
            document = json.loads(source.read_text(encoding="utf-8"))
            compact = compact_corruption_report(document)
            payload = (json.dumps(compact, indent=2, sort_keys=True) + "\n").encode()
            entry["derivation"] = compact["_derivative"]["derivation"]
            entry["source_bytes"] = source.stat().st_size
        else:
            raise ValueError(f"unknown export mode: {spec.mode}")

        document = json.loads(payload)
        if spec.destination == "campaigns/conversion-campaign-v1-corrected.json":
            validate_corrected_conversion_record(project_root, document)
        if spec.destination == "campaigns/lifting-replication-v2.json":
            validate_lifting_replication_v2_record(project_root, document)
        entry["evidence_revision"] = evidence_revision(document)

        if len(payload) > max_file_bytes:
            raise ValueError(
                f"exported evidence exceeds the per-file cap ({len(payload)} bytes): "
                f"{spec.destination}"
            )
        total_bytes += len(payload)
        if total_bytes > max_total_bytes:
            raise ValueError("exported bundle exceeds the total byte cap")

        if spec.mode != "in-place":
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(
                destination.suffix + f".{os.getpid()}.tmp"
            )
            try:
                temporary.write_bytes(payload)
                temporary.replace(destination)
            finally:
                if temporary.exists():
                    temporary.unlink()

        entry["bytes"] = len(payload)
        entry["sha256"] = _sha256_bytes(payload)
        entries.append(entry)

    entries.sort(key=lambda item: item["path"])
    kinds: dict[str, int] = {}
    for entry in entries:
        kinds[entry["kind"]] = kinds.get(entry["kind"], 0) + 1

    unattributed = [
        entry["path"] for entry in entries if entry["evidence_revision"] is None
    ]
    if unattributed:
        raise ValueError(
            f"exported evidence has no recorded generating commit: {unattributed}"
        )
    revision_counts: dict[str, int] = {}
    for entry in entries:
        revision = str(entry["evidence_revision"])
        revision_counts[revision] = revision_counts.get(revision, 0) + 1

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "bundle": BUNDLE_ROOT,
        # Commits are read from the evidence itself, never from the exporter's
        # own HEAD: a manifest that named its own commit could not be committed
        # without dangling. Every exported file must name one, and the campaigns
        # were frozen at different times, so more than one commit is expected.
        "source_commits": dict(sorted(revision_counts.items())),
        "source_commit_note": (
            "Generating revision recorded inside each exported artifact, with the "
            "number of exported files attributed to it. The routing confirmatory "
            "campaign was frozen before the identifiable-map campaign, so the two "
            "families legitimately carry different commits."
        ),
        "generating_command": [
            "python",
            "scripts/export_publication_evidence.py",
            "--output-root",
            BUNDLE_ROOT,
        ],
        "generating_pipeline": [
            [
                "python",
                "scripts/summarize_gauge_corruption_campaign.py",
                "--output",
                "results/summaries/gauge-corruption-campaign.json",
            ],
            [
                "python",
                "scripts/summarize_compute_campaign.py",
                "--output",
                "results/summaries/compute-campaign.json",
            ],
            [
                "python",
                "scripts/export_publication_evidence.py",
                "--output-root",
                BUNDLE_ROOT,
            ],
        ],
        "exclusions": {
            "denied_suffixes": sorted(DENIED_SUFFIXES),
            "denied_directories": sorted(DENIED_PARTS),
            "policy": (
                "Checkpoints, per-example prediction dumps, training histories, "
                "scheduler logs, caches, and environments are excluded by "
                "construction and are not journal evidence. The untracked "
                "/artifacts/ tree remains the only home for those files."
            ),
        },
        "determinism": (
            "This manifest carries no timestamp; re-exporting unchanged evidence "
            "reproduces it byte for byte."
        ),
        "summary": {
            "files": len(entries),
            "bytes": total_bytes,
            "by_kind": dict(sorted(kinds.items())),
        },
        "files": entries,
    }

    manifest_path = output_root / MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    temporary = manifest_path.with_suffix(manifest_path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(manifest_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest


def verify(project_root: Path, output_root: Path) -> list[str]:
    """Recheck an exported bundle against its own manifest."""

    manifest = json.loads((output_root / MANIFEST_NAME).read_text(encoding="utf-8"))
    problems: list[str] = []
    for entry in manifest["files"]:
        path = output_root / entry["path"]
        if not path.is_file():
            problems.append(f"missing: {entry['path']}")
            continue
        if _sha256(path) != entry["sha256"]:
            problems.append(f"hash mismatch: {entry['path']}")
        if path.stat().st_size != entry["bytes"]:
            problems.append(f"byte count mismatch: {entry['path']}")
        source = project_root / entry["source"]
        if source.is_file() and _sha256(source) != entry["source_sha256"]:
            problems.append(f"source changed since export: {entry['source']}")
    tracked = {entry["path"] for entry in manifest["files"]} | {MANIFEST_NAME}
    for path in sorted(output_root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(output_root).as_posix()
            if relative not in tracked:
                problems.append(f"unlisted file in bundle: {relative}")
    return problems


def _parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--output-root", type=Path, default=project_root / BUNDLE_ROOT)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="recheck an existing bundle against its manifest without rewriting it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    try:
        if args.verify_only:
            problems = verify(project_root, output_root)
            for problem in problems:
                print(f"publication evidence problem: {problem}", file=sys.stderr)
            if problems:
                return 2
            print(json.dumps({"verified": True, "bundle": str(output_root)}))
            return 0
        manifest = export(project_root=project_root, output_root=output_root)
    except (OSError, TypeError, ValueError) as exc:
        print(f"publication evidence export failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
