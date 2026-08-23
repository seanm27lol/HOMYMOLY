"""Paired cross-model inference for schema-v3 corruption reports.

The comparison operates on the fixed-expert batch diagnostics emitted by
``scripts/eval_corruption.py``.  It does not evaluate a translator, a learned
chain map, or conversion quality.

Example::

    python scripts/compare_corruption_reports.py \
        --baseline artifacts/gate3/task-only/corruption_report.json \
        --candidate artifacts/gate3/full/corruption_report.json \
        --output artifacts/gate3/paired_comparison.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from homymoly.training.io import atomic_json

try:  # Supports both ``python -m scripts...`` and direct script execution.
    from scripts.eval_corruption import _partial_spearman, _stable_hash_seed
except ModuleNotFoundError:  # pragma: no cover - exercised by direct CLI use
    from eval_corruption import _partial_spearman, _stable_hash_seed


_METHOD = "paired-cross-model-corruption-v1"
_PAIRING_FIELDS = (
    "block_seed",
    "num_examples",
    "sigma_min",
    "sigma_mean",
    "sigma_max",
)
_MEASURE_FIELDS = (
    "topological_defect",
    "damage_rate",
    "mean_embedding_displacement",
)
_CLAIM_BOUNDARY = (
    "Fixed-expert embedding diagnostic only; this paired analysis does not "
    "evaluate representation conversion, a translator, or a learned chain map."
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON report {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"report {path} must contain a JSON object")
    return value


def _finite_number(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be a finite number")
    return result


def _positive_integer(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _validate_report(
    report: Mapping[str, Any], *, path: Path
) -> tuple[list[float], dict[tuple[str, float, str], dict[str, Any]]]:
    schema_version = report.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version < 3
    ):
        raise ValueError(f"report {path} requires schema_version >= 3")

    sampling = report.get("sampling")
    if not isinstance(sampling, dict):
        raise TypeError(f"report {path} is missing sampling metadata")
    for field in ("protocol", "data_seed", "experiment_seed"):
        if field not in sampling:
            raise ValueError(f"report {path} sampling is missing {field!r}")
    if not isinstance(sampling["protocol"], str) or not sampling["protocol"]:
        raise ValueError(f"report {path} sampling protocol must be nonempty")
    for field in ("data_seed", "experiment_seed"):
        value = sampling[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"report {path} sampling {field} must be an integer")

    raw_severities = report.get("severities")
    if not isinstance(raw_severities, list) or not raw_severities:
        raise ValueError(f"report {path} severities must be a nonempty list")
    severities = [
        _finite_number(value, context=f"report {path} severity")
        for value in raw_severities
    ]
    if len(set(severities)) != len(severities):
        raise ValueError(f"report {path} severities must be unique")

    rows = report.get("per_batch")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"report {path} per_batch must be a nonempty list")
    indexed: dict[tuple[str, float, str], dict[str, Any]] = {}
    severity_set = set(severities)
    for position, raw_row in enumerate(rows):
        context = f"report {path} per_batch[{position}]"
        if not isinstance(raw_row, dict):
            raise TypeError(f"{context} must be an object")
        kind = raw_row.get("kind")
        block_id = raw_row.get("block_id")
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"{context}.kind must be a nonempty string")
        if not isinstance(block_id, str) or not block_id:
            raise ValueError(f"{context}.block_id must be a nonempty string")
        severity = _finite_number(
            raw_row.get("severity"), context=f"{context}.severity"
        )
        if severity not in severity_set:
            raise ValueError(f"{context}.severity is absent from report severities")
        key = (kind, severity, block_id)
        if key in indexed:
            raise ValueError(f"report {path} has duplicate per_batch key {key!r}")

        _positive_integer(
            raw_row.get("num_examples"), context=f"{context}.num_examples"
        )
        block_seed = raw_row.get("block_seed")
        if isinstance(block_seed, bool) or not isinstance(block_seed, int):
            raise TypeError(f"{context}.block_seed must be an integer")
        for field in (*_PAIRING_FIELDS[2:], *_MEASURE_FIELDS):
            _finite_number(raw_row.get(field), context=f"{context}.{field}")
        if not (
            float(raw_row["sigma_min"])
            <= float(raw_row["sigma_mean"])
            <= float(raw_row["sigma_max"])
        ):
            raise ValueError(f"{context} sigma summaries are not ordered")
        indexed[key] = dict(raw_row)

    by_kind_block: dict[tuple[str, str], set[float]] = {}
    for kind, severity, block_id in indexed:
        by_kind_block.setdefault((kind, block_id), set()).add(severity)
    for (kind, block_id), observed in by_kind_block.items():
        if observed != severity_set:
            raise ValueError(
                f"report {path} block {(kind, block_id)!r} is incomplete: "
                f"expected severities {sorted(severity_set)!r}, "
                f"observed {sorted(observed)!r}"
            )
    for kind in {key[0] for key in indexed}:
        blocks = {key[2] for key in indexed if key[0] == kind}
        if len(blocks) < 2:
            raise ValueError(
                f"report {path} kind {kind!r} needs at least two complete blocks"
            )
    return severities, indexed


def _sampling_signature(report: Mapping[str, Any]) -> dict[str, Any]:
    sampling = report["sampling"]
    execution = report.get("execution")
    if not isinstance(execution, dict):
        raise TypeError("report is missing execution metadata")
    for field in ("batch_size", "max_batches"):
        _positive_integer(execution.get(field), context=f"execution.{field}")
    metric = report.get("topological_metric")
    if not isinstance(metric, dict):
        raise TypeError("report is missing topological_metric metadata")
    return {
        "sampling_metadata": sampling,
        "sampling_protocol": sampling["protocol"],
        "data_seed": sampling["data_seed"],
        "experiment_seed": sampling["experiment_seed"],
        "batch_size": execution["batch_size"],
        "max_batches": execution["max_batches"],
        "topological_metric": metric,
    }


def _validate_pairing(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    baseline_rows: Mapping[tuple[str, float, str], Mapping[str, Any]],
    candidate_rows: Mapping[tuple[str, float, str], Mapping[str, Any]],
    *,
    baseline_severities: Sequence[float],
    candidate_severities: Sequence[float],
    candidate_path: Path,
) -> None:
    if list(candidate_severities) != list(baseline_severities):
        raise ValueError(f"candidate {candidate_path} has different severities")
    if _sampling_signature(candidate) != _sampling_signature(baseline):
        raise ValueError(
            f"candidate {candidate_path} has a different sampling protocol, "
            "seed, execution sampling configuration, or topological metric"
        )
    baseline_keys = set(baseline_rows)
    candidate_keys = set(candidate_rows)
    if candidate_keys != baseline_keys:
        missing = sorted(baseline_keys - candidate_keys)
        extra = sorted(candidate_keys - baseline_keys)
        raise ValueError(
            f"candidate {candidate_path} per_batch join is not exact: "
            f"missing={missing!r}, extra={extra!r}"
        )
    for key in sorted(baseline_keys):
        baseline_row = baseline_rows[key]
        candidate_row = candidate_rows[key]
        for field in _PAIRING_FIELDS:
            if candidate_row[field] != baseline_row[field]:
                raise ValueError(
                    f"candidate {candidate_path} pairing mismatch at {key!r}: "
                    f"{field} differs ({baseline_row[field]!r} != "
                    f"{candidate_row[field]!r})"
                )


def _ordered_kind_rows(
    indexed: Mapping[tuple[str, float, str], Mapping[str, Any]], kind: str
) -> list[dict[str, Any]]:
    keys = sorted(
        (key for key in indexed if key[0] == kind),
        key=lambda key: (key[2], key[1]),
    )
    return [dict(indexed[key]) for key in keys]


def _adjusted_estimate(
    rows: Sequence[Mapping[str, Any]], *, block_field: str = "block_id"
) -> float:
    return float(
        _partial_spearman(
            [float(row["topological_defect"]) for row in rows],
            [float(row["damage_rate"]) for row in rows],
            [float(row["mean_embedding_displacement"]) for row in rows],
            [float(row["severity"]) for row in rows],
            blocks=[str(row[block_field]) for row in rows],
        )
    )


def _rows_by_block(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["block_id"]), []).append(dict(row))
    for values in grouped.values():
        values.sort(key=lambda row: float(row["severity"]))
    return grouped


def _paired_bootstrap_interval(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> list[float]:
    baseline_by_block = _rows_by_block(baseline_rows)
    candidate_by_block = _rows_by_block(candidate_rows)
    blocks = sorted(baseline_by_block)
    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(replicates):
        selected = [blocks[rng.randrange(len(blocks))] for _ in blocks]
        sampled_baseline: list[dict[str, Any]] = []
        sampled_candidate: list[dict[str, Any]] = []
        for draw, block in enumerate(selected):
            bootstrap_block = f"{draw}:{block}"
            for source, target in (
                (baseline_by_block[block], sampled_baseline),
                (candidate_by_block[block], sampled_candidate),
            ):
                for row in source:
                    copied = dict(row)
                    copied["_bootstrap_block"] = bootstrap_block
                    target.append(copied)
        differences.append(
            _adjusted_estimate(sampled_candidate, block_field="_bootstrap_block")
            - _adjusted_estimate(sampled_baseline, block_field="_bootstrap_block")
        )
    lower, upper = np.quantile(np.asarray(differences), (0.025, 0.975))
    return [float(lower), float(upper)]


def _swapped_difference(
    baseline_by_block: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_by_block: Mapping[str, Sequence[Mapping[str, Any]]],
    blocks: Sequence[str],
    swaps: Sequence[bool],
) -> float:
    permuted_baseline: list[Mapping[str, Any]] = []
    permuted_candidate: list[Mapping[str, Any]] = []
    for block, swap in zip(blocks, swaps, strict=True):
        if swap:
            permuted_baseline.extend(candidate_by_block[block])
            permuted_candidate.extend(baseline_by_block[block])
        else:
            permuted_baseline.extend(baseline_by_block[block])
            permuted_candidate.extend(candidate_by_block[block])
    return _adjusted_estimate(permuted_candidate) - _adjusted_estimate(
        permuted_baseline
    )


def _paired_randomization(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    observed_difference: float,
    seed: int,
    monte_carlo_replicates: int,
) -> dict[str, Any]:
    baseline_by_block = _rows_by_block(baseline_rows)
    candidate_by_block = _rows_by_block(candidate_rows)
    blocks = sorted(baseline_by_block)
    threshold = abs(observed_difference) - 1e-15
    exceedances = 0
    if len(blocks) <= 16:
        assignments = 1 << len(blocks)
        for mask in range(assignments):
            swaps = [bool(mask & (1 << index)) for index in range(len(blocks))]
            if (
                abs(
                    _swapped_difference(
                        baseline_by_block, candidate_by_block, blocks, swaps
                    )
                )
                >= threshold
            ):
                exceedances += 1
        pvalue = exceedances / assignments
        return {
            "mode": "exact",
            "pvalue_two_sided": pvalue,
            "assignments_evaluated": assignments,
            "exceedances": exceedances,
            "seed": None,
        }

    rng = random.Random(seed)
    for _ in range(monte_carlo_replicates):
        swaps = [bool(rng.getrandbits(1)) for _ in blocks]
        if (
            abs(
                _swapped_difference(
                    baseline_by_block, candidate_by_block, blocks, swaps
                )
            )
            >= threshold
        ):
            exceedances += 1
    return {
        "mode": "deterministic-monte-carlo",
        "pvalue_two_sided": (exceedances + 1) / (monte_carlo_replicates + 1),
        "assignments_evaluated": monte_carlo_replicates,
        "exceedances": exceedances,
        "seed": seed,
    }


def _report_provenance(
    path: Path, report: Mapping[str, Any], *, digest: str
) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": digest,
        "schema_version": report["schema_version"],
        "checkpoint": report.get("checkpoint"),
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "checkpoint_load": report.get("checkpoint_load"),
        "config": report.get("config"),
        "config_sha256": report.get("config_sha256"),
        "script_sha256": report.get("script_sha256"),
        "source_command": report.get("command"),
        "git": report.get("git"),
    }


def compare_reports(
    baseline_path: Path,
    candidate_paths: Sequence[Path],
    *,
    bootstrap_replicates: int = 2000,
    randomization_replicates: int = 10000,
    analysis_seed: int = 20260813,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate and compare one baseline with one or more candidate reports."""
    if not candidate_paths:
        raise ValueError("at least one candidate report is required")
    if bootstrap_replicates <= 0 or randomization_replicates <= 0:
        raise ValueError("inference replicate counts must be positive")

    baseline_path = baseline_path.resolve()
    resolved_candidates = [path.resolve() for path in candidate_paths]
    baseline = _load_json(baseline_path)
    baseline_severities, baseline_index = _validate_report(baseline, path=baseline_path)
    baseline_digest = _file_sha256(baseline_path)
    comparisons: list[dict[str, Any]] = []
    candidate_provenance: list[dict[str, Any]] = []

    for candidate_number, candidate_path in enumerate(resolved_candidates, start=1):
        candidate = _load_json(candidate_path)
        candidate_severities, candidate_index = _validate_report(
            candidate, path=candidate_path
        )
        _validate_pairing(
            baseline,
            candidate,
            baseline_index,
            candidate_index,
            baseline_severities=baseline_severities,
            candidate_severities=candidate_severities,
            candidate_path=candidate_path,
        )
        candidate_digest = _file_sha256(candidate_path)
        candidate_provenance.append(
            _report_provenance(candidate_path, candidate, digest=candidate_digest)
        )
        kinds = sorted({key[0] for key in baseline_index})
        by_kind: dict[str, dict[str, Any]] = {}
        for kind in kinds:
            baseline_rows = _ordered_kind_rows(baseline_index, kind)
            candidate_rows = _ordered_kind_rows(candidate_index, kind)
            baseline_estimate = _adjusted_estimate(baseline_rows)
            candidate_estimate = _adjusted_estimate(candidate_rows)
            difference = candidate_estimate - baseline_estimate
            bootstrap_seed = _stable_hash_seed(
                _METHOD,
                analysis_seed,
                baseline_digest,
                candidate_digest,
                kind,
                "paired-complete-block-bootstrap",
            )
            randomization_seed = _stable_hash_seed(
                _METHOD,
                analysis_seed,
                baseline_digest,
                candidate_digest,
                kind,
                "whole-block-model-label-randomization",
            )
            blocks = sorted({str(row["block_id"]) for row in baseline_rows})
            by_kind[kind] = {
                "counts": {
                    "paired_batch_observations": len(baseline_rows),
                    "complete_blocks": len(blocks),
                    "severity_levels": len(baseline_severities),
                },
                "baseline_adjusted_partial_spearman": baseline_estimate,
                "candidate_adjusted_partial_spearman": candidate_estimate,
                "candidate_minus_baseline": difference,
                "paired_complete_block_bootstrap_95_ci": _paired_bootstrap_interval(
                    baseline_rows,
                    candidate_rows,
                    seed=bootstrap_seed,
                    replicates=bootstrap_replicates,
                ),
                "bootstrap_replicates": bootstrap_replicates,
                "bootstrap_seed": bootstrap_seed,
                "paired_whole_block_model_label_randomization": _paired_randomization(
                    baseline_rows,
                    candidate_rows,
                    observed_difference=difference,
                    seed=randomization_seed,
                    monte_carlo_replicates=randomization_replicates,
                ),
            }
        comparisons.append(
            {
                "candidate_number": candidate_number,
                "candidate_path": str(candidate_path),
                "candidate_sha256": candidate_digest,
                "counts": {
                    "paired_batch_observations": len(baseline_index),
                    "kinds": len(kinds),
                },
                "by_kind": by_kind,
            }
        )

    return {
        "schema_version": 1,
        "analysis_method": _METHOD,
        "command": list(command) if command is not None else None,
        "script_sha256": _file_sha256(Path(__file__).resolve()),
        "inputs": {
            "baseline": _report_provenance(
                baseline_path, baseline, digest=baseline_digest
            ),
            "candidates": candidate_provenance,
        },
        "validated_pairing_contract": {
            "join_key": ["kind", "severity", "block_id"],
            "equal_fields": list(_PAIRING_FIELDS),
            "sampling_signature": _sampling_signature(baseline),
            "severities": baseline_severities,
            "complete_blocks_required": True,
        },
        "method": {
            "report_statistic": (
                "Pearson correlation of rank residuals for topological_defect "
                "and damage_rate, adjusted for ranked severity, ranked "
                "mean_embedding_displacement, and block fixed effects"
            ),
            "contrast": "candidate adjusted statistic minus baseline adjusted statistic",
            "interval": (
                "paired complete-block bootstrap; the same whole blocks are "
                "resampled for both reports"
            ),
            "test": (
                "two-sided paired model-label randomization at the whole-block "
                "level; exact for at most 16 blocks, otherwise deterministic "
                "Monte Carlo with add-one correction"
            ),
            "analysis_seed": analysis_seed,
            "randomization_replicates_if_monte_carlo": randomization_replicates,
            "inferential_scope": (
                "Conditional on the two fixed trained checkpoints and sampled "
                "held-out blocks; this does not estimate variation across "
                "training seeds."
            ),
            "multiplicity": (
                "No adjustment across candidates or corruption kinds; p-values "
                "are per candidate-kind contrast."
            ),
        },
        "claim_boundary": _CLAIM_BOUNDARY,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument(
        "--candidate",
        required=True,
        type=Path,
        nargs="+",
        action="append",
        help="one or more candidate reports; the option may be repeated",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--randomization-replicates", type=int, default=10000)
    parser.add_argument("--analysis-seed", type=int, default=20260813)
    args = parser.parse_args()
    candidates = [path for group in args.candidate for path in group]
    try:
        result = compare_reports(
            args.baseline,
            candidates,
            bootstrap_replicates=args.bootstrap_replicates,
            randomization_replicates=args.randomization_replicates,
            analysis_seed=args.analysis_seed,
            command=[sys.executable, *sys.argv],
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    atomic_json(args.output, result)
    print(f"paired comparison written to {args.output}")


if __name__ == "__main__":
    main()
