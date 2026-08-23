#!/usr/bin/env python3
"""Summarize the seed-matched gauge corruption campaign as a strict compact record.

The campaign trains one ``gauge-task-only`` baseline and one ``gauge-plus-chain``
candidate per training seed, evaluates each trained checkpoint with the
fixed-expert corruption evaluator, and pairs the two reports of a seed with
``scripts/compare_corruption_reports.py``. This summarizer never recomputes a
statistic from raw embeddings: it revalidates the recorded provenance of every
paired comparison and then aggregates the already-published
candidate-minus-baseline adjusted statistics across training seeds.

Scope is deliberately narrow. Every number produced here is a fixed-expert
embedding diagnostic. Nothing in this file evaluates representation conversion,
a translator, or a learned chain map.
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

SCOPE = (
    "Fixed-expert embedding diagnostic only; this campaign does not evaluate "
    "representation conversion, a translator, or a learned chain map."
)
BASELINE_PREFIX = "gauge-task-only"
CANDIDATE_PREFIX = "gauge-plus-chain"
EXPECTED_PAIRS = 8
PAIRED_FILENAME = "paired_comparison_final.json"
REPORT_FILENAME = "corruption_report_final.json"

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


def _student_t_interval(values: list[float]) -> list[float]:
    degrees_of_freedom = len(values) - 1
    if degrees_of_freedom not in _T_975:
        raise ValueError("the t-interval supports 2--10 paired seeds")
    half_width = (
        _T_975[degrees_of_freedom]
        * statistics.stdev(values)
        / math.sqrt(len(values))
    )
    mean = statistics.mean(values)
    return [mean - half_width, mean + half_width]


def exact_two_sided_sign_test(values: list[float]) -> dict[str, Any]:
    """Exact two-sided sign test on nonzero differences.

    Zero differences are discarded before the test, which is the conventional
    treatment and is reported explicitly so the effective denominator is never
    silently inflated.
    """

    favorable = sum(value > 0 for value in values)
    unfavorable = sum(value < 0 for value in values)
    ties = sum(value == 0 for value in values)
    trials = favorable + unfavorable
    if trials == 0:
        pvalue: float | None = None
        minimum_attainable: float | None = None
    else:
        smaller = min(favorable, unfavorable)
        tail = sum(math.comb(trials, index) for index in range(smaller + 1))
        pvalue = min(1.0, 2.0 * tail / (2.0**trials))
        minimum_attainable = min(1.0, 2.0 / (2.0**trials))
    return {
        "definition": (
            "exact two-sided binomial sign test on nonzero "
            "candidate-minus-baseline differences"
        ),
        "candidate_favorable": favorable,
        "candidate_unfavorable": unfavorable,
        "ties_discarded": ties,
        "nonzero_trials": trials,
        "pvalue_two_sided": pvalue,
        "minimum_attainable_pvalue_without_ties": minimum_attainable,
    }


def _config_seed(config: Path) -> int:
    """Read the frozen seed from a gate3g YAML without importing the training stack."""

    seeds: set[int] = set()
    for line in config.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("seed:"):
            seeds.add(int(stripped.split(":", 1)[1].strip()))
    if len(seeds) != 1:
        raise RuntimeError(
            f"config does not declare exactly one seed value: {config} -> {sorted(seeds)}"
        )
    return next(iter(seeds))


def _verify_side(side: dict[str, Any], role: str, project_root: Path) -> dict[str, Any]:
    """Revalidate one side of a pair and return its compact provenance row."""

    report_path = Path(side["path"])
    config_path = Path(side["config"])
    checkpoint_path = Path(side["checkpoint"])
    for path in (report_path, config_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"{role} input is missing: {path}")

    for path, recorded, label in (
        (report_path, side["sha256"], "corruption report"),
        (config_path, side["config_sha256"], "config"),
        (checkpoint_path, side["checkpoint_sha256"], "checkpoint"),
    ):
        observed = _sha256(path)
        if observed != recorded:
            raise RuntimeError(
                f"{role} {label} hash mismatch for {path}: "
                f"recorded={recorded} observed={observed}"
            )

    git = side.get("git") or {}
    commit = str(git.get("commit") or "")
    if not commit:
        raise RuntimeError(f"{role} report has no recorded git commit: {report_path}")
    if str(git.get("status") or "") != "":
        raise RuntimeError(
            f"{role} report was generated from a dirty worktree: {report_path}"
        )

    # Run directories are named ...-sNN while seeds are dates, so the config
    # seed is authoritative and is never inferred from the directory name.
    run_name = report_path.parent.name
    seed = _config_seed(config_path)
    return {
        "role": role,
        "run_name": run_name,
        "seed": seed,
        "report": report_path.relative_to(project_root).as_posix(),
        "report_sha256": side["sha256"],
        "report_schema_version": side["schema_version"],
        "config": config_path.relative_to(project_root).as_posix(),
        "config_sha256": side["config_sha256"],
        "checkpoint": checkpoint_path.relative_to(project_root).as_posix(),
        "checkpoint_sha256": side["checkpoint_sha256"],
        "evaluator_sha256": side["script_sha256"],
        "git_commit": commit,
        "git_status_at_generation": "",
        "checkpoint_load": side.get("checkpoint_load"),
        "source_command": side.get("source_command"),
    }


def summarize(
    gate3g_root: Path,
    *,
    project_root: Path,
    expected_pairs: int = EXPECTED_PAIRS,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    paired_paths = sorted(gate3g_root.glob(f"*/{PAIRED_FILENAME}"))
    if len(paired_paths) != expected_pairs:
        raise RuntimeError(
            f"expected {expected_pairs} paired comparisons under {gate3g_root}; "
            f"found {len(paired_paths)}"
        )

    pairs: list[dict[str, Any]] = []
    comparator_hashes: set[str] = set()
    evaluator_hashes: set[str] = set()
    commits: set[str] = set()
    analysis_methods: set[str] = set()
    claim_boundaries: set[str] = set()
    join_keys: set[tuple[str, ...]] = set()
    equal_fields: set[tuple[str, ...]] = set()
    severity_sets: set[tuple[float, ...]] = set()
    block_counts: set[int] = set()
    observation_counts: set[int] = set()
    analysis_seeds: set[int] = set()
    per_kind: dict[str, list[dict[str, Any]]] = {}

    for paired_path in paired_paths:
        document = json.loads(paired_path.read_text(encoding="utf-8"))
        comparator_hashes.add(str(document["script_sha256"]))
        analysis_methods.add(str(document["analysis_method"]))
        claim_boundaries.add(str(document["claim_boundary"]))
        analysis_seeds.add(int(document["method"]["analysis_seed"]))

        candidates = document["inputs"]["candidates"]
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected exactly one candidate per seed-matched pair: {paired_path}"
            )
        baseline_row = _verify_side(
            document["inputs"]["baseline"], "baseline", project_root
        )
        candidate_row = _verify_side(candidates[0], "candidate", project_root)

        if not baseline_row["run_name"].startswith(BASELINE_PREFIX):
            raise RuntimeError(
                f"baseline is not a {BASELINE_PREFIX} run: {baseline_row['run_name']}"
            )
        if not candidate_row["run_name"].startswith(CANDIDATE_PREFIX):
            raise RuntimeError(
                f"candidate is not a {CANDIDATE_PREFIX} run: {candidate_row['run_name']}"
            )
        if baseline_row["seed"] != candidate_row["seed"]:
            raise RuntimeError(
                "pair is not seed-matched: "
                f"baseline={baseline_row['seed']} candidate={candidate_row['seed']}"
            )
        evaluator_hashes.update(
            {baseline_row["evaluator_sha256"], candidate_row["evaluator_sha256"]}
        )
        commits.update({baseline_row["git_commit"], candidate_row["git_commit"]})

        contract = document["validated_pairing_contract"]
        if not contract.get("complete_blocks_required"):
            raise RuntimeError(f"pair does not require complete blocks: {paired_path}")
        join_keys.add(tuple(contract["join_key"]))
        equal_fields.add(tuple(contract["equal_fields"]))
        severity_sets.add(tuple(float(value) for value in contract["severities"]))
        signature = contract["sampling_signature"]
        seed = baseline_row["seed"]
        if int(signature["data_seed"]) != seed or int(signature["experiment_seed"]) != seed:
            raise RuntimeError(
                f"sampling signature does not match the training seed for {paired_path}"
            )
        metadata = signature["sampling_metadata"]
        if int(metadata["data_seed"]) != seed or int(metadata["experiment_seed"]) != seed:
            raise RuntimeError(
                f"sampling metadata does not match the training seed for {paired_path}"
            )

        comparison = document["comparisons"][0]
        kinds = comparison["by_kind"]
        for kind, payload in sorted(kinds.items()):
            counts = payload["counts"]
            block_counts.add(int(counts["complete_blocks"]))
            observation_counts.add(int(counts["paired_batch_observations"]))
            if int(counts["severity_levels"]) != len(contract["severities"]):
                raise RuntimeError(
                    f"severity count disagrees with the pairing contract: {paired_path}"
                )
            per_kind.setdefault(kind, []).append(
                {
                    "seed": seed,
                    "baseline_adjusted_partial_spearman": float(
                        payload["baseline_adjusted_partial_spearman"]
                    ),
                    "candidate_adjusted_partial_spearman": float(
                        payload["candidate_adjusted_partial_spearman"]
                    ),
                    "candidate_minus_baseline": float(payload["candidate_minus_baseline"]),
                    "within_pair_randomization_pvalue_two_sided": float(
                        payload["paired_whole_block_model_label_randomization"][
                            "pvalue_two_sided"
                        ]
                    ),
                    "within_pair_randomization_mode": payload[
                        "paired_whole_block_model_label_randomization"
                    ]["mode"],
                }
            )

        pairs.append(
            {
                "seed": seed,
                "paired_comparison": paired_path.relative_to(project_root).as_posix(),
                "paired_comparison_sha256": _sha256(paired_path),
                "comparator_sha256": str(document["script_sha256"]),
                "baseline": baseline_row,
                "candidate": candidate_row,
                "sampling_signature": signature,
            }
        )

    seeds = [pair["seed"] for pair in pairs]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError(f"training seeds are not distinct: {sorted(seeds)}")
    for name, observed in (
        ("comparator script", comparator_hashes),
        ("evaluator script", evaluator_hashes),
        ("git commit", commits),
        ("analysis method", analysis_methods),
        ("claim boundary", claim_boundaries),
        ("pairing join key", join_keys),
        ("pairing equal-fields contract", equal_fields),
        ("severity grid", severity_sets),
        ("complete-block count", block_counts),
        ("paired batch observation count", observation_counts),
        ("analysis seed", analysis_seeds),
    ):
        if len(observed) != 1:
            raise RuntimeError(
                f"{name} is not shared across the campaign: {sorted(observed)}"
            )

    kind_names = sorted(per_kind)
    for kind in kind_names:
        rows = per_kind[kind]
        if len(rows) != expected_pairs:
            raise RuntimeError(
                f"corruption kind {kind} is missing seeds: "
                f"{len(rows)} of {expected_pairs}"
            )

    seed_order = sorted(seeds)
    aggregates: dict[str, Any] = {}
    for kind in kind_names:
        rows = sorted(per_kind[kind], key=lambda row: row["seed"])
        if [row["seed"] for row in rows] != seed_order:
            raise RuntimeError(f"corruption kind {kind} has inconsistent seeds")
        differences = [row["candidate_minus_baseline"] for row in rows]
        aggregates[kind] = {
            "endpoint": (
                "adjusted partial Spearman correlation between the topological "
                "defect diagnostic and the damage rate, severity- and "
                "displacement-adjusted with block fixed effects"
            ),
            "contrast": "candidate (gauge-plus-chain) minus baseline (gauge-task-only)",
            "n_paired_seeds": len(rows),
            "seeds_in_order": seed_order,
            "differences_in_seed_order": differences,
            "mean_difference": statistics.mean(differences),
            "sample_standard_deviation_of_differences": statistics.stdev(differences),
            "student_t_95_ci": _student_t_interval(differences),
            "student_t_degrees_of_freedom": len(differences) - 1,
            "interval_includes_zero": (
                _student_t_interval(differences)[0]
                <= 0.0
                <= _student_t_interval(differences)[1]
            ),
            "sensitivity_sign_test": exact_two_sided_sign_test(differences),
            "per_seed": rows,
        }

    supported = [
        kind
        for kind in kind_names
        if not aggregates[kind]["interval_includes_zero"]
    ]
    contract = json.loads(paired_paths[0].read_text(encoding="utf-8"))[
        "validated_pairing_contract"
    ]

    return {
        "schema_version": 1,
        "scope": SCOPE,
        "campaign": "gate3g gauge corruption, seed-matched chain/translator contrast",
        "contrast_definition": {
            "baseline": f"{BASELINE_PREFIX}: translator_weight=0.0, chain_weight=0.0",
            "candidate": f"{CANDIDATE_PREFIX}: translator_weight=0.1, chain_weight=0.05",
            "pairing": "one baseline and one candidate per training seed",
        },
        "analysis_provenance": {
            "summarizer": {
                "path": "scripts/summarize_gauge_corruption_campaign.py",
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "comparator_sha256": next(iter(comparator_hashes)),
            "evaluator_sha256": next(iter(evaluator_hashes)),
            "git_commit": next(iter(commits)),
            "analysis_seed": next(iter(analysis_seeds)),
            "analysis_method": next(iter(analysis_methods)),
            "claim_boundary": next(iter(claim_boundaries)),
        },
        "validation": {
            "expected_pairs": expected_pairs,
            "validated_pairs": len(pairs),
            "distinct_training_seeds": len(set(seeds)),
            "report_hashes_verified": True,
            "checkpoint_hashes_verified": True,
            "config_hashes_verified": True,
            "evaluator_hash_shared": True,
            "comparator_hash_shared": True,
            "clean_git_status_verified": True,
            "complete_blocks_required": True,
            "complete_blocks_per_kind": next(iter(block_counts)),
            "paired_batch_observations_per_kind": next(iter(observation_counts)),
            "pairing_join_key": list(next(iter(join_keys))),
            "pairing_equal_fields": list(next(iter(equal_fields))),
            "severity_grid": list(next(iter(severity_sets))),
            "sampling_protocol": contract["sampling_signature"]["sampling_protocol"],
            "topological_metric": contract["sampling_signature"]["topological_metric"],
            "status": "passed",
        },
        "aggregation": {
            "unit_of_analysis": "training seed",
            "n_paired_seeds": expected_pairs,
            "student_t_degrees_of_freedom": expected_pairs - 1,
            "multiplicity_adjustment": "none",
            "minimum_attainable_two_sided_sign_test_pvalue_without_ties": min(
                1.0, 2.0 / (2.0**expected_pairs)
            ),
            "corruption_kinds": kind_names,
        },
        "by_kind": aggregates,
        "decision": {
            "rule": (
                "a kind is called a difference only if its across-seed t interval "
                "excludes zero"
            ),
            "kinds_with_interval_excluding_zero": supported,
            "structural_benefit_established": bool(supported),
        },
        "interpretation_guardrail": (
            "These are descriptive, unadjusted, seed-level summaries of a "
            "fixed-expert embedding diagnostic on synthetic gauge data. Intervals "
            "are Student-t summaries over eight training seeds and carry no "
            "multiplicity correction; the exact sign test cannot fall below "
            "p=0.0078125 at eight untied seeds. No representation-conversion, "
            "translator, or chain-map capability is evaluated here."
        ),
        "pairs": sorted(pairs, key=lambda pair: pair["seed"]),
    }


def _parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--gate3g-root", type=Path, default=project_root / "artifacts" / "gate3g"
    )
    parser.add_argument("--expected-pairs", type=int, default=EXPECTED_PAIRS)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = summarize(
        args.gate3g_root.expanduser().resolve(),
        project_root=args.project_root.expanduser().resolve(),
        expected_pairs=args.expected_pairs,
    )
    _atomic_json(args.output.expanduser(), report)
    print(
        json.dumps(
            {
                "validated_pairs": report["validation"]["validated_pairs"],
                "corruption_kinds": report["aggregation"]["corruption_kinds"],
                "structural_benefit_established": report["decision"][
                    "structural_benefit_established"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
