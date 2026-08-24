#!/usr/bin/env python3
"""Re-run and correct the prospectively frozen campaign declared in docs/27.

The frozen document fixed the topology seeds, training size, weights, endpoints,
decision rules, multiplicity adjustment, and routing split. The executed runner
used an elementwise mean-square boundary-compatibility penalty where the document
wrote an unnormalised squared Frobenius norm. That implementation deviation is
preserved here so the corrected analysis re-runs the same fits, and it is
recorded explicitly in schema v2.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
from pathlib import Path
from typing import Any

import torch

from homymoly.data.conversion import ConversionDataset

PROTOCOL = "docs/27-conversion-campaign-protocol.md"
GENERATOR_SOURCE = "src/homymoly/data/conversion.py"
FROZEN_PROTOCOL_SHA256 = (
    "503cc282f40d118ba1739c2afe1bfc77eaf2b1733baaddb91c0c3363e75ae2b8"
)
FROZEN_GENERATOR_SHA256 = (
    "c37ab1c725aa2101e88c1a0ad8fa3b279d72330feba35077e23fec930a4df69d"
)
RESULT_SCHEMA_VERSION = 2
RESULT_RECORD_ID = "conversion-campaign-v1-correction-1"
CORRECTED_RESULT_PATH = "results/campaigns/conversion-campaign-v1-corrected.json"
ORIGINAL_CAMPAIGN_GIT_REVISION = "11644c68ec0b8c28416a14ce4d8799e4c9ca0860"
SUPERSEDED_RESULT_PATH = "results/campaigns/conversion-campaign-v1.json"
SUPERSEDED_RESULT_SHA256 = (
    "836914d251db8d381aef9a2dcb0ac14a14562652f3e323dc840108b5f24d5ee1"
)
SUPERSEDED_RUNNER_SHA256 = (
    "8a478e5a3906d5bb5cfc3645159f8739cc3e840a50bbce851564533b2ce89fb6"
)
SEEDS = tuple(range(20261001, 20261031))
MIN_FACES = 3
MIN_ELIGIBLE = 24
N_TRAIN = 16
N_HELD_OUT = 3072
NOISE = 0.02
STEPS = 2500
LEARNING_RATE = 0.05
WEIGHTS = {"exact": 3.0, "cone": 0.01, "rtd": 0.1}
C1_WEIGHTS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
FAMILY_SIZE = 3

# Student-t quantiles for a two-sided 95% interval and a two-sided
# 98.333...% Bonferroni interval. For the latter, three primary contrasts give
# alpha_per_contrast = 0.05 / 3 and the upper quantile is
# 1 - alpha_per_contrast / 2 = 0.991666....
_T975 = {
    9: 2.262157,
    14: 2.144787,
    19: 2.093024,
    23: 2.068658,
    24: 2.063899,
    25: 2.059539,
    26: 2.055529,
    27: 2.051831,
    28: 2.048407,
    29: 2.045230,
}
_T_ADJ = {
    9: 2.933324088,
    14: 2.717755159,
    19: 2.625105913,
    23: 2.582017198,
    24: 2.573641017,
    25: 2.565978552,
    26: 2.558942386,
    27: 2.552458806,
    28: 2.546465223,
    29: 2.540908149,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_sha256(
    project_root: Path, relative_path: str, expected: str, *, label: str
) -> str:
    """Hash a frozen input and stop before fitting if it has changed."""

    path = project_root / relative_path
    try:
        actual = _sha256(path)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"stop condition: {label} is missing at {relative_path}"
        ) from error
    if actual != expected:
        raise RuntimeError(
            f"stop condition: {label} SHA-256 is {actual}, expected {expected} "
            f"for {relative_path}"
        )
    return actual


def _generator_provenance(project_root: Path) -> dict[str, Any]:
    actual = _verified_sha256(
        project_root,
        GENERATOR_SOURCE,
        FROZEN_GENERATOR_SHA256,
        label="conversion generator",
    )
    return {
        "class": "homymoly.data.conversion.ConversionDataset",
        "path": GENERATOR_SOURCE,
        "sha256": actual,
        "frozen_campaign_sha256": FROZEN_GENERATOR_SHA256,
        "frozen_campaign_git_revision": ORIGINAL_CAMPAIGN_GIT_REVISION,
        "matches_frozen_campaign": True,
    }


def _correction_record() -> dict[str, Any]:
    """Describe the analysis correction without renaming the frozen campaign."""

    return {
        "id": "conversion-campaign-analysis-correction-1",
        "date": "2026-08-24",
        "scope": (
            "C1 Pearson correlations and primary Bonferroni-adjusted intervals; "
            "fits, raw endpoints, unadjusted intervals, and routing values are unchanged"
        ),
        "reasons": [
            {
                "id": "c1-pearson-normalisation",
                "issue": (
                    "schema v1 averaged products after standardising each nine-point "
                    "vector with Bessel-corrected sample standard deviations; apart "
                    "from its epsilon, that estimator equals (n-1)/n times "
                    "conventional Pearson r"
                ),
                "corrected_estimator": (
                    "sum((x-x_bar)*(y-y_bar)) / sqrt(sum((x-x_bar)^2)*sum((y-y_bar)^2))"
                ),
            },
            {
                "id": "bonferroni-critical-value",
                "issue": (
                    "schema v1 used two-sided 99% Student-t critical values while "
                    "labelling the intervals as the protocol-required Bonferroni "
                    "98.333...% intervals"
                ),
                "corrected_quantile": (
                    "t.ppf(1 - (0.05 / 3) / 2, df) = t.ppf(0.991666..., df)"
                ),
            },
        ],
        "affected_fields": [
            "c1.per_topology_correlation",
            "c1.mean_within_topology_correlation",
            "c1.interval_95",
            "c1.supported",
            "primary.*.interval_bonferroni_98_33",
            "primary.*.improves_confirmatory",
            "primary.*.harms_confirmatory",
        ],
        "unaffected_numeric_fields": [
            "primary.*.per_topology fit endpoints",
            "primary.*.mean_log10_ratio",
            "primary.*.median_log10_ratio",
            "primary.*.sample_standard_deviation",
            "primary.*.interval_95",
            "primary.*.sensitivity_sign_test.pvalue_two_sided",
            "routing numeric fields",
        ],
        "decision_changes": {
            "exact": False,
            "cone": False,
            "rtd": False,
            "c1": False,
        },
        "protocol_modified": False,
        "data_or_fit_settings_modified": False,
        "supersedes": {
            "path": SUPERSEDED_RESULT_PATH,
            "schema_version": 1,
            "sha256": SUPERSEDED_RESULT_SHA256,
            "runner_sha256": SUPERSEDED_RUNNER_SHA256,
            "git_revision": ORIGINAL_CAMPAIGN_GIT_REVISION,
        },
    }


def _protocol_implementation_deviations() -> list[dict[str, Any]]:
    return [
        {
            "id": "compatibility-mean-normalisation",
            "frozen_text": "squared Frobenius norm ||B1 @ W.T||_F^2",
            "executed_objective": "elementwise mean((B1 @ W.T)^2)",
            "consequence": (
                "the executed penalty is divided by num_vertices * num_faces; "
                "because topology dimensions vary, weight 3.0 has a "
                "topology-dependent scale relative to the frozen notation"
            ),
            "fit_implementation_changed_in_correction": False,
        }
    ]


def _git(project_root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


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


def _interval(values: list[float], table: dict[int, float]) -> list[float]:
    degrees = len(values) - 1
    if degrees not in table:
        raise RuntimeError(f"no t quantile for df={degrees}")
    half = table[degrees] * statistics.stdev(values) / math.sqrt(len(values))
    mean = statistics.mean(values)
    return [mean - half, mean + half]


def _sign_test(values: list[float]) -> dict[str, Any]:
    positive = sum(1 for value in values if value > 0)
    negative = sum(1 for value in values if value < 0)
    trials = positive + negative
    ties = len(values) - trials
    if trials == 0:
        return {
            "pvalue_two_sided": None,
            "negative": 0,
            "positive": 0,
            "ties_discarded": ties,
        }
    smaller = min(positive, negative)
    tail = sum(math.comb(trials, index) for index in range(smaller + 1))
    return {
        "pvalue_two_sided": min(1.0, 2.0 * tail / (2.0**trials)),
        "negative": negative,
        "positive": positive,
        "ties_discarded": ties,
    }


def _fit(sample: Any, term: str | None, weight: float) -> tuple[float, float]:
    """Fit W and return (held-out MSE, Frobenius compatibility defect)."""

    edges, faces = sample.num_edges, sample.num_faces
    truth = sample.boundary_2.mT
    boundary_1 = sample.boundary_1
    generator = torch.Generator().manual_seed(
        int(hashlib.sha256(sample.sample_id.encode()).hexdigest()[:12], 16)
    )
    train_x = torch.randn((N_TRAIN, edges), generator=generator, dtype=torch.float64)
    test_x = torch.randn((N_HELD_OUT, edges), generator=generator, dtype=torch.float64)
    train_y = train_x @ truth.mT + NOISE * torch.randn(
        (N_TRAIN, faces), generator=generator, dtype=torch.float64
    )
    test_y = test_x @ truth.mT

    learned = torch.zeros((faces, edges), dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.Adam([learned], lr=LEARNING_RATE)
    for _ in range(STEPS):
        predicted = train_x @ learned.mT
        loss = (predicted - train_y).pow(2).mean()
        if term == "exact":
            loss = loss + weight * (boundary_1 @ learned.mT).pow(2).mean()
        elif term == "cone":
            loss = loss + weight * torch.exp(-torch.linalg.svdvals(learned).min() * 2.0)
        elif term == "rtd":
            source = torch.cdist(train_x, train_x)
            mapped = torch.cdist(predicted, predicted)
            loss = (
                loss
                + weight
                * (mapped / (mapped.mean() + 1e-12) - source / (source.mean() + 1e-12))
                .pow(2)
                .mean()
            )
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        held_out = float(((test_x @ learned.mT) - test_y).pow(2).mean())
        violation = float(torch.linalg.matrix_norm(boundary_1 @ learned.mT))
    if not math.isfinite(held_out) or not math.isfinite(violation):
        raise RuntimeError(f"non-finite result for {sample.sample_id} term={term}")
    return held_out, violation


def _routing_trial(sample: Any, weight: float) -> tuple[float, float, float]:
    """Return (measured defect, cell-route error, graph-route error)."""

    edges, faces = sample.num_edges, sample.num_faces
    truth = sample.boundary_2.mT
    boundary_1 = sample.boundary_1
    generator = torch.Generator().manual_seed(
        int(hashlib.sha256((sample.sample_id + "route").encode()).hexdigest()[:12], 16)
    )
    readout = torch.randn((faces,), generator=generator, dtype=torch.float64)
    train_x = torch.randn((N_TRAIN, edges), generator=generator, dtype=torch.float64)
    test_x = torch.randn((N_HELD_OUT, edges), generator=generator, dtype=torch.float64)
    train_y = train_x @ truth.mT + NOISE * torch.randn(
        (N_TRAIN, faces), generator=generator, dtype=torch.float64
    )
    train_t = (train_x @ truth.mT) @ readout + NOISE * torch.randn(
        (N_TRAIN,), generator=generator, dtype=torch.float64
    )
    test_t = (test_x @ truth.mT) @ readout

    learned = torch.zeros((faces, edges), dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.Adam([learned], lr=LEARNING_RATE)
    for _ in range(STEPS):
        loss = ((train_x @ learned.mT) - train_y).pow(2).mean()
        if weight:
            loss = loss + weight * (boundary_1 @ learned.mT).pow(2).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        defect = float(torch.linalg.matrix_norm(boundary_1 @ learned.mT))
        cell = float((((test_x @ learned.mT) @ readout) - test_t).pow(2).mean())
    direct = torch.linalg.lstsq(train_x, train_t.unsqueeze(-1)).solution.squeeze(-1)
    graph = float(((test_x @ direct) - test_t).pow(2).mean())
    return defect, cell, graph


def _correlation(left: list[float], right: list[float]) -> float:
    """Return the conventional Pearson product-moment correlation.

    Pearson's ratio uses the same unnormalised centred sums in its numerator and
    denominator.  The schema-v1 implementation instead averaged products after
    dividing by sample standard deviations, which scales the answer by
    ``(n - 1) / n`` (apart from its epsilon).  Computing the ratio directly
    avoids any dependence on a population-versus-sample standard-deviation
    convention.
    """

    if len(left) != len(right):
        raise ValueError("correlation inputs must have equal length")
    if len(left) < 2:
        raise ValueError("correlation requires at least two paired observations")
    if not all(math.isfinite(value) for value in (*left, *right)):
        raise ValueError("correlation inputs must be finite")

    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_centred = [value - left_mean for value in left]
    right_centred = [value - right_mean for value in right]
    cross_product = math.fsum(
        a * b for a, b in zip(left_centred, right_centred, strict=True)
    )
    left_sum_squares = math.fsum(value * value for value in left_centred)
    right_sum_squares = math.fsum(value * value for value in right_centred)
    denominator = math.sqrt(left_sum_squares * right_sum_squares)
    if denominator == 0.0:
        raise ValueError("correlation is undefined for a constant input")

    # Round-off can produce a value a few ulps outside the mathematical range.
    return min(1.0, max(-1.0, cross_product / denominator))


def run(project_root: Path) -> dict[str, Any]:
    protocol_sha256 = _verified_sha256(
        project_root,
        PROTOCOL,
        FROZEN_PROTOCOL_SHA256,
        label="frozen conversion protocol",
    )
    generator_provenance = _generator_provenance(project_root)

    eligible, skipped = [], []
    for seed in SEEDS:
        sample = ConversionDataset(1, seed=seed, dtype=torch.float64)[0]
        (eligible if sample.num_faces >= MIN_FACES else skipped).append((seed, sample))
    if len(eligible) < MIN_ELIGIBLE:
        raise RuntimeError(
            f"stop condition: only {len(eligible)} eligible topologies, "
            f"minimum is {MIN_ELIGIBLE}"
        )

    baseline = {seed: _fit(sample, None, 0.0) for seed, sample in eligible}
    primary: dict[str, Any] = {}
    for term, weight in WEIGHTS.items():
        differences, rows = [], []
        for seed, sample in eligible:
            held_out, violation = _fit(sample, term, weight)
            reference = baseline[seed][0]
            differences.append(math.log10(held_out / reference))
            rows.append(
                {
                    "seed": seed,
                    "held_out_mse": held_out,
                    "baseline_held_out_mse": reference,
                    "boundary_compatibility_defect_frobenius": violation,
                }
            )
        adjusted = _interval(differences, _T_ADJ)
        primary[term] = {
            "weight": weight,
            "n": len(differences),
            "mean_log10_ratio": statistics.mean(differences),
            "median_log10_ratio": statistics.median(differences),
            "sample_standard_deviation": statistics.stdev(differences),
            "interval_95": _interval(differences, _T975),
            "interval_bonferroni_98_33": adjusted,
            "improves_confirmatory": adjusted[1] < 0.0,
            "harms_confirmatory": adjusted[0] > 0.0,
            "sensitivity_sign_test": _sign_test(differences),
            "per_topology": rows,
        }

    c1_correlations = []
    c1_rows = []
    for seed, sample in eligible:
        fits = [_fit(sample, "exact", weight) for weight in C1_WEIGHTS]
        correlation = _correlation(
            [math.log10(max(v, 1e-30)) for _, v in fits],
            [math.log10(max(h, 1e-300)) for h, _ in fits],
        )
        c1_correlations.append(correlation)
        c1_rows.append(
            {
                "seed": seed,
                "correlation": correlation,
                "fits": [
                    {
                        "weight": weight,
                        "held_out_mse": held_out,
                        "boundary_compatibility_defect_frobenius": defect,
                    }
                    for weight, (held_out, defect) in zip(C1_WEIGHTS, fits, strict=True)
                ],
            }
        )
    c1_interval = _interval(c1_correlations, _T975)

    routing_rows = []
    for index, (seed, sample) in enumerate(eligible):
        for weight in (0.0, WEIGHTS["exact"]):
            defect, cell, graph = _routing_trial(sample, weight)
            routing_rows.append(
                {
                    "seed": seed,
                    "split": "threshold" if index % 2 == 0 else "evaluation",
                    "term_weight": weight,
                    "defect": defect,
                    "cell_error": cell,
                    "graph_error": graph,
                }
            )
    threshold_pool = [r["defect"] for r in routing_rows if r["split"] == "threshold"]
    threshold = statistics.median(threshold_pool)
    evaluation = [r for r in routing_rows if r["split"] == "evaluation"]
    routed_differences = [
        math.log10(
            (row["cell_error"] if row["defect"] <= threshold else row["graph_error"])
            / min(row["cell_error"], row["graph_error"])
        )
        for row in evaluation
    ]

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "schema": {
            "name": "homymoly.conversion-campaign-result",
            "version": RESULT_SCHEMA_VERSION,
            "record_id": RESULT_RECORD_ID,
        },
        "campaign": "conversion-campaign-v1",
        "correction": _correction_record(),
        "protocol": {
            "path": PROTOCOL,
            "sha256": protocol_sha256,
            "frozen_sha256": FROZEN_PROTOCOL_SHA256,
            "document_hash_matches_frozen": True,
            "execution_matches_frozen_text": False,
            "implementation_deviations": _protocol_implementation_deviations(),
        },
        "provenance": {
            "git_revision": _git(project_root, "rev-parse", "HEAD"),
            "git_status": _git(project_root, "status", "--short"),
            "python": platform.python_version(),
            "dependencies": {
                "torch": torch.__version__,
                "networkx": importlib.metadata.version("networkx"),
                "numpy": importlib.metadata.version("numpy"),
            },
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "generator": generator_provenance,
        },
        "design": {
            "declared_seeds": list(SEEDS),
            "eligible_topologies": len(eligible),
            "skipped_topologies": [seed for seed, _ in skipped],
            "training_pairs": N_TRAIN,
            "held_out_pairs": N_HELD_OUT,
            "observation_noise_standard_deviation": NOISE,
            "steps": STEPS,
            "optimiser": {
                "name": "Adam",
                "learning_rate": LEARNING_RATE,
                "initialisation": "W = 0",
                "dtype": "float64",
            },
            "fit_scope": "one independently trained free matrix W per topology",
            "weights": WEIGHTS,
            "weight_provenance": (
                "selected and frozen after exploratory work; no machine-readable "
                "selection criterion was retained"
            ),
            "eligible_topology_dimensions": [
                {
                    "seed": seed,
                    "vertices": sample.num_vertices,
                    "edges": sample.num_edges,
                    "faces": sample.num_faces,
                    "free_parameters": sample.num_edges * sample.num_faces,
                }
                for seed, sample in eligible
            ],
            "free_parameters": {
                "definition": "num_edges * num_faces for each independently fitted W",
                "minimum": min(
                    sample.num_edges * sample.num_faces for _, sample in eligible
                ),
                "median": statistics.median(
                    sample.num_edges * sample.num_faces for _, sample in eligible
                ),
                "maximum": max(
                    sample.num_edges * sample.num_faces for _, sample in eligible
                ),
            },
            "objective_definitions": {
                "exact": {
                    "display_name": "boundary-compatibility penalty",
                    "formula": "mean((B1 @ W.T)^2)",
                    "frozen_key_is_historical_shorthand": True,
                    "is_exactness_of_a_sequence": False,
                },
                "cone": {
                    "display_name": "singular-value cone surrogate",
                    "formula": "exp(-2 * sigma_min(W))",
                    "is_mapping_cone_homology": False,
                },
                "rtd": {
                    "display_name": "RTD-inspired normalized-distance surrogate",
                    "formula": (
                        "MSE(normalized pairwise distances of X @ W.T, "
                        "normalized pairwise distances of X)"
                    ),
                    "is_representation_topology_divergence": False,
                },
            },
            "multiplicity": "Bonferroni across the three primary contrasts",
            "family_size": FAMILY_SIZE,
        },
        "primary": primary,
        "c1": {
            "weights_swept": list(C1_WEIGHTS),
            "n": len(c1_correlations),
            "mean_within_topology_correlation": statistics.mean(c1_correlations),
            "interval_95": c1_interval,
            "positive_topologies": sum(1 for value in c1_correlations if value > 0),
            "supported": c1_interval[0] > 0.0,
            "inference_role": (
                "prespecified secondary analysis; unadjusted two-sided 95% "
                "Student-t interval"
            ),
            "per_topology_correlation": c1_correlations,
            "per_topology": c1_rows,
        },
        "routing": {
            "threshold_split_size": len(threshold_pool),
            "evaluation_trials": len(evaluation),
            "threshold": threshold,
            "endpoint": (
                "log10(routed / per-trial minimum of cell and graph errors) "
                "on the evaluation split"
            ),
            "decision_informative": False,
            "protocol_design_note": (
                "The routed numerator is one of the two errors in the per-trial "
                "denominator, so every endpoint value is nonnegative and the "
                "preregistered upper-bound-below-zero support rule is impossible."
            ),
            "mean_log10_ratio": statistics.mean(routed_differences),
            "interval_95": _interval(routed_differences, _T975),
            "supported": _interval(routed_differences, _T975)[1] < 0.0,
            "median_routed": statistics.median(
                [
                    row["cell_error"]
                    if row["defect"] <= threshold
                    else row["graph_error"]
                    for row in evaluation
                ]
            ),
            "median_always_cell": statistics.median(
                [row["cell_error"] for row in evaluation]
            ),
            "median_always_graph": statistics.median(
                [row["graph_error"] for row in evaluation]
            ),
            "trials": routing_rows,
        },
    }


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=f"write the corrected record here (publication path: {CORRECTED_RESULT_PATH})",
    )
    args = parser.parse_args(argv)
    report = run(args.project_root.expanduser().resolve())
    _atomic_json(args.output.expanduser(), report)
    summary = {
        term: {
            "improves": payload["improves_confirmatory"],
            "harms": payload["harms_confirmatory"],
            "interval_bonferroni": payload["interval_bonferroni_98_33"],
        }
        for term, payload in report["primary"].items()
    }
    summary["c1_supported"] = report["c1"]["supported"]
    summary["routing_supported"] = report["routing"]["supported"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
