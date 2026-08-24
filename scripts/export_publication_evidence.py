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
HISTORICAL_CONVERSION_SHA256 = (
    "836914d251db8d381aef9a2dcb0ac14a14562652f3e323dc840108b5f24d5ee1"
)
BONFERRONI_T_DF28 = 2.546465223
T95_DF28 = 2.048407
UNAFFECTED_NUMERIC_FIELDS = [
    "primary.*.per_topology fit endpoints",
    "primary.*.mean_log10_ratio",
    "primary.*.median_log10_ratio",
    "primary.*.sample_standard_deviation",
    "primary.*.interval_95",
    "primary.*.sensitivity_sign_test.pvalue_two_sided",
    "routing numeric fields",
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


def _student_t_interval(values: list[float], critical: float) -> list[float]:
    if len(values) != 29:
        raise ValueError(
            "corrected conversion analysis must retain 29 topologies (df=28)"
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
        raise ValueError("corrected C1 decision disagrees with its interval")
    if historical_c1.get("supported") is not supported:
        raise ValueError("C1 support decision changed after correcting Pearson r")


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
    if len(reasons) != 2 or set(by_id) != {
        "c1-pearson-normalisation",
        "bonferroni-critical-value",
    }:
        raise ValueError("analysis correction does not identify both corrected issues")
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

    design = document.get("design")
    definitions = (
        design.get("objective_definitions") if isinstance(design, dict) else None
    )
    compatibility = definitions.get("exact") if isinstance(definitions, dict) else None
    if compatibility != {
        "display_name": "boundary-compatibility penalty",
        "formula": "mean((B1 @ W.T)^2)",
        "frozen_key_is_historical_shorthand": True,
        "is_exactness_of_a_sequence": False,
    }:
        raise ValueError(
            "frozen key 'exact' is not explicitly identified as boundary compatibility"
        )

    _validate_corrected_primary_intervals(document, historical_document)
    _validate_corrected_c1(document, historical_document)


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
