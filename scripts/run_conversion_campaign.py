#!/usr/bin/env python3
"""Run the frozen conversion campaign declared in docs/27.

The protocol is preregistered. This script implements it and nothing else: the
topologies, training size, weights, decision rules, multiplicity adjustment, and
the out-of-sample routing threshold are all fixed by that document. It records
the protocol's SHA-256 so a reader can confirm the design was not revised after
the fact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any

import torch

from homymoly.data.conversion import ConversionDataset

PROTOCOL = "docs/27-conversion-campaign-protocol.md"
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

# Two-sided Student-t quantiles: 97.5% for the reported interval, and
# 99.1667% for the Bonferroni-adjusted confirmatory interval (0.05 / 3).
_T975 = {
    9: 2.262157, 14: 2.144787, 19: 2.093024, 23: 2.068658, 24: 2.063899,
    25: 2.059539, 26: 2.055529, 27: 2.051831, 28: 2.048407, 29: 2.045230,
}
_T_ADJ = {
    9: 3.249836, 14: 2.976843, 19: 2.860935, 23: 2.807336, 24: 2.797170,
    25: 2.787436, 26: 2.778715, 27: 2.770683, 28: 2.763262, 29: 2.756386,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), check=False, capture_output=True, text=True, timeout=15
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
    if trials == 0:
        return {"pvalue_two_sided": None, "favourable": 0, "unfavourable": 0}
    smaller = min(positive, negative)
    tail = sum(math.comb(trials, index) for index in range(smaller + 1))
    return {
        "pvalue_two_sided": min(1.0, 2.0 * tail / (2.0**trials)),
        "favourable": negative,
        "unfavourable": positive,
    }


def _fit(sample: Any, term: str | None, weight: float) -> tuple[float, float]:
    """Fit W under one term and return (held-out MSE, exactness violation)."""

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
            loss = loss + weight * torch.exp(
                -torch.linalg.svdvals(learned).min() * 2.0
            )
        elif term == "rtd":
            source = torch.cdist(train_x, train_x)
            mapped = torch.cdist(predicted, predicted)
            loss = loss + weight * (
                mapped / (mapped.mean() + 1e-12) - source / (source.mean() + 1e-12)
            ).pow(2).mean()
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
    a = torch.tensor(left, dtype=torch.float64)
    b = torch.tensor(right, dtype=torch.float64)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    return float((a * b).mean())


def run(project_root: Path) -> dict[str, Any]:
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
                    "exactness_violation": violation,
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

    c1 = []
    for _seed, sample in eligible:
        fits = [_fit(sample, "exact", weight) for weight in C1_WEIGHTS]
        c1.append(
            _correlation(
                [math.log10(max(v, 1e-30)) for _, v in fits],
                [math.log10(max(h, 1e-300)) for h, _ in fits],
            )
        )
    c1_interval = _interval(c1, _T975)

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
        "schema_version": 1,
        "campaign": "conversion-campaign-v1",
        "protocol": {
            "path": PROTOCOL,
            "sha256": _sha256(project_root / PROTOCOL),
        },
        "provenance": {
            "git_revision": _git("rev-parse", "HEAD"),
            "git_status": _git("status", "--short"),
            "torch": torch.__version__,
            "runner_sha256": _sha256(Path(__file__).resolve()),
        },
        "design": {
            "declared_seeds": list(SEEDS),
            "eligible_topologies": len(eligible),
            "skipped_topologies": [seed for seed, _ in skipped],
            "training_pairs": N_TRAIN,
            "steps": STEPS,
            "weights": WEIGHTS,
            "multiplicity": "Bonferroni across the three primary contrasts",
            "family_size": FAMILY_SIZE,
        },
        "primary": primary,
        "c1": {
            "weights_swept": list(C1_WEIGHTS),
            "n": len(c1),
            "mean_within_topology_correlation": statistics.mean(c1),
            "interval_95": c1_interval,
            "positive_topologies": sum(1 for value in c1 if value > 0),
            "supported": c1_interval[0] > 0.0,
            "per_topology_correlation": c1,
        },
        "routing": {
            "threshold_split_size": len(threshold_pool),
            "evaluation_trials": len(evaluation),
            "threshold": threshold,
            "endpoint": "log10(routed / best fixed strategy) on the evaluation split",
            "mean_log10_ratio": statistics.mean(routed_differences),
            "interval_95": _interval(routed_differences, _T975),
            "supported": _interval(routed_differences, _T975)[1] < 0.0,
            "median_routed": statistics.median(
                [
                    row["cell_error"] if row["defect"] <= threshold else row["graph_error"]
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
    parser.add_argument("--output", type=Path, required=True)
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
