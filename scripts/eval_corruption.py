"""Gate-3 corruption-suite evaluation.

Loads a trained Gate-2 checkpoint, applies graded structural corruptions to
held-out samples, and measures for each corruption kind and severity:

* conversion damage — the task-accuracy drop of the affected route's expert
  and translator on that route's regime;
* the translator's structural diagnostics (per-sample reconstruction and
  consistency);
* the exact SRTD between clean and corrupted expert embeddings per batch —
  the topological conversion defect.

The report answers the plan's Gate-3 question: does a structural diagnostic
add predictive value for conversion damage after controlling for
reconstruction error?  Usage:

    python scripts/eval_corruption.py \
        --checkpoint artifacts/gate2-run10-translators-competent/checkpoints/last.pt \
        --config configs/gate2.yaml \
        --output artifacts/gate3/corruption_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
from torch.nn import functional as F

from homymoly.data.collate import collate_structured
from homymoly.data.corruptions import apply_corruption, corruption_kinds
from homymoly.metrics import exact_srtd, pairwise_euclidean_distances
from homymoly.models import build_model
from homymoly.runtime import initialize_runtime
from homymoly.training import engine
from homymoly.training.config import load_gate2_config

ROUTE_FOR_KIND = {
    "transport_rotation": "sheaf",
    "edge_cochain_noise": "cell",
    "node_anchor_noise": "graph",
}
TRANSLATOR_FOR_KIND = {
    "transport_rotation": "graph_to_sheaf",
    "edge_cochain_noise": "graph_to_cell",
    "node_anchor_noise": None,
}
SEVERITIES = (0.05, 0.1, 0.2, 0.4, 0.8)


def _rankdata(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _spearman(x, y):
    rx, ry = _rankdata(x), _rankdata(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def _partial_spearman(x, y, control):
    rx, ry, rc = _rankdata(x), _rankdata(y), _rankdata(control)
    mc = statistics.mean(rc)

    def residualize(r):
        mr = statistics.mean(r)
        denom = sum((c - mc) ** 2 for c in rc) or 1.0
        slope = sum((c - mc) * (v - mr) for c, v in zip(rc, r)) / denom
        return [v - slope * (c - mc) for v, c in zip(r, rc)]

    return _spearman(residualize(rx), residualize(ry))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/gate2.yaml")
    parser.add_argument("--output", default="artifacts/gate3/corruption_report.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=12)
    args = parser.parse_args()

    config = load_gate2_config(args.config)
    runtime = initialize_runtime(config.runtime, seed=config.experiment.seed)
    dataset = engine._build_dataset(config, smoke=False)
    splits = dataset.split_indices(
        train_fraction=config.data.train_fraction,
        validation_fraction=config.data.validation_fraction,
    )
    test_indices = splits["test"]
    model = build_model(engine._build_model_config(config)).to(runtime.device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=runtime.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    per_example_rows: list[dict] = []
    per_batch_rows: list[dict] = []

    for kind in corruption_kinds():
        route = ROUTE_FOR_KIND[kind]
        regime_examples = [
            index
            for index in test_indices
            if dataset.regimes[index].value == route
        ]
        for severity in SEVERITIES:
            for batch_start in range(0, len(regime_examples), args.batch_size):
                batch_indices = regime_examples[
                    batch_start : batch_start + args.batch_size
                ][: args.batch_size]
                if len(batch_indices) < 8:
                    continue
                batch_number = batch_start // args.batch_size
                if batch_number >= args.max_batches:
                    break
                clean = collate_structured([dataset[i] for i in batch_indices])
                rng = torch.Generator().manual_seed(
                    hash((kind, severity, batch_start)) & 0x7FFFFFFF
                )
                corrupted = collate_structured(
                    [
                        apply_corruption(
                            dataset[i],
                            kind=kind,
                            sigma=float(
                                torch.rand((), generator=rng).item() * severity
                            ),
                            seed=config.data.seed,
                        )
                        for i in batch_indices
                    ]
                )
                clean = clean.to(runtime.device)
                corrupted = corrupted.to(runtime.device)
                with torch.no_grad(), torch.autocast(
                    device_type=runtime.device.type,
                    dtype=runtime.neural_dtype,
                    enabled=runtime.device.type == "cuda",
                ):
                    clean_out = model.fixed_experts.experts[route](clean)
                    corrupted_out = model.fixed_experts.experts[route](corrupted)
                clean_pred = clean_out.logits.float().argmax(dim=-1)
                corrupted_pred = corrupted_out.logits.float().argmax(dim=-1)
                labels = clean.labels
                clean_correct = clean_pred == labels
                damage = (clean_correct & (corrupted_pred != labels)).float()
                corrupted_ce = F.cross_entropy(
                    corrupted_out.logits.float(), labels, reduction="none"
                )
                clean_ce = F.cross_entropy(
                    clean_out.logits.float(), labels, reduction="none"
                )
                ce_increase = (corrupted_ce - clean_ce).float()

                def _paired_distances(embedding: torch.Tensor) -> torch.Tensor:
                    # Subsample to the engine's rtd_max_points convention:
                    # the exact cone is exponential in entity count.
                    points = embedding.float()
                    if points.shape[0] > config.loss.rtd_max_points:
                        selector = torch.linspace(
                            0,
                            points.shape[0] - 1,
                            config.loss.rtd_max_points,
                        ).long()
                        points = points[selector]
                    distances = pairwise_euclidean_distances(points)
                    # The diagonal is zero by construction; the expansion in
                    # the distance helper can leave cancellation residue.
                    distances.fill_diagonal_(0.0)
                    return distances

                embedding_clean = _paired_distances(clean_out.embedding)
                embedding_corrupted = _paired_distances(corrupted_out.embedding)
                topological_defect = exact_srtd(embedding_clean, embedding_corrupted)

                reconstruction = (
                    (corrupted_out.embedding.float() - clean_out.embedding.float())
                    .square()
                    .mean(dim=-1)
                )
                diagnostic = corrupted_out.diagnostics.float().mean(dim=-1)
                per_batch_rows.append(
                    {
                        "kind": kind,
                        "severity": severity,
                        "damage_rate": float(damage.mean()),
                        "mean_ce_increase": float(ce_increase.mean()),
                        "topological_defect": float(topological_defect),
                        "mean_reconstruction": float(reconstruction.mean()),
                        "mean_diagnostic": float(diagnostic.mean()),
                    }
                )
                for k in range(len(labels)):
                    per_example_rows.append(
                        {
                            "kind": kind,
                            "severity": severity,
                            "damage": float(damage[k]),
                            "ce_increase": float(ce_increase[k]),
                            "reconstruction": float(reconstruction[k]),
                            "diagnostic": float(diagnostic[k]),
                        }
                    )

    analysis: dict[str, dict] = {}
    for kind in corruption_kinds():
        rows = [row for row in per_example_rows if row["kind"] == kind]
        batch_rows = [row for row in per_batch_rows if row["kind"] == kind]
        analysis[kind] = {
            "examples": len(rows),
            "damage_rate": statistics.mean(row["damage"] for row in rows),
            "spearman(diagnostic, damage)": _spearman(
                [row["diagnostic"] for row in rows],
                [row["damage"] for row in rows],
            ),
            "partial_spearman(diagnostic, damage | reconstruction)": _partial_spearman(
                [row["diagnostic"] for row in rows],
                [row["damage"] for row in rows],
                [row["reconstruction"] for row in rows],
            ),
            "batch_spearman(topological_defect, damage_rate)": _spearman(
                [row["topological_defect"] for row in batch_rows],
                [row["damage_rate"] for row in batch_rows],
            ),
            "batch_partial(topological_defect, damage_rate | reconstruction)": (
                _partial_spearman(
                    [row["topological_defect"] for row in batch_rows],
                    [row["damage_rate"] for row in batch_rows],
                    [row["mean_reconstruction"] for row in batch_rows],
                )
            ),
            "batch_spearman(reconstruction, damage_rate)": _spearman(
                [row["mean_reconstruction"] for row in batch_rows],
                [row["damage_rate"] for row in batch_rows],
            ),
        }

    report = {
        "checkpoint": args.checkpoint,
        "severities": list(SEVERITIES),
        "analysis": analysis,
        "per_batch": per_batch_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    for kind, result in analysis.items():
        print(kind, json.dumps(result, indent=2))
    print(f"report written to {output}")


if __name__ == "__main__":
    main()
