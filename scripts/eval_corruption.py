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
import hashlib
import json
import math
import random
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from homymoly.data.collate import collate_structured
from homymoly.data.corruptions import apply_corruption, corruption_kinds
from homymoly.metrics import exact_srtd, pairwise_euclidean_distances
from homymoly.models import build_model
from homymoly.runtime import initialize_runtime
from homymoly.training import engine
from homymoly.training.config import load_gate2_config
from homymoly.training.io import atomic_json

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
_DRAW_PROTOCOL = "sha256-block-and-sample-v1"
_ANALYSIS_METHOD = "rank-residual-partial-spearman-v1"


def _rankdata(values: Sequence[float]) -> list[float]:
    if len(values) == 0:
        raise ValueError("rank data must be nonempty")
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("rank data must be finite")
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


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("correlation inputs must have equal length of at least two")
    mx, my = statistics.mean(x), statistics.mean(y)
    centered_x = [float(value) - mx for value in x]
    centered_y = [float(value) - my for value in y]
    numerator = sum(
        left * right for left, right in zip(centered_x, centered_y, strict=True)
    )
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    return numerator / denominator if denominator else 0.0


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y):
        raise ValueError("Spearman inputs must have equal length")
    return _pearson(_rankdata(x), _rankdata(y))


def _rank_residuals(
    values: Sequence[float],
    controls: Sequence[Sequence[float]],
    *,
    blocks: Sequence[str | int] | None = None,
) -> list[float]:
    count = len(values)
    if count < 2:
        raise ValueError("partial correlation requires at least two observations")
    if any(len(control) != count for control in controls):
        raise ValueError("controls must match the data length")
    if blocks is not None and len(blocks) != count:
        raise ValueError("block labels must match the data length")
    columns: list[np.ndarray] = [np.ones(count, dtype=np.float64)]
    columns.extend(
        np.asarray(_rankdata(control), dtype=np.float64) for control in controls
    )
    if blocks is not None:
        levels = sorted(set(blocks), key=str)
        for level in levels[1:]:
            columns.append(
                np.asarray([item == level for item in blocks], dtype=np.float64)
            )
    design = np.column_stack(columns)
    ranked = np.asarray(_rankdata(values), dtype=np.float64)
    fitted = design @ np.linalg.lstsq(design, ranked, rcond=None)[0]
    return (ranked - fitted).tolist()


def _partial_spearman(
    x: Sequence[float],
    y: Sequence[float],
    *controls: Sequence[float],
    blocks: Sequence[str | int] | None = None,
) -> float:
    """Pearson correlation of rank residuals (standard partial Spearman)."""
    if len(x) != len(y):
        raise ValueError("partial-Spearman inputs must have equal length")
    residual_x = _rank_residuals(x, controls, blocks=blocks)
    residual_y = _rank_residuals(y, controls, blocks=blocks)
    return _pearson(residual_x, residual_y)


def _stable_hash_seed(*parts: object) -> int:
    encoded = json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode()
    return (
        int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
        & 0x7FFF_FFFF_FFFF_FFFF
    )


def _corruption_sigmas(
    sample_ids: Sequence[str],
    *,
    data_seed: int,
    experiment_seed: int,
    kind: str,
    severity: float,
    block_id: int,
    batch_start: int,
) -> tuple[int, list[float]]:
    """Stable paired draws independent of Python's process-salted hash."""
    if severity < 0 or not math.isfinite(severity):
        raise ValueError("severity must be finite and nonnegative")
    block_seed = _stable_hash_seed(
        _DRAW_PROTOCOL,
        data_seed,
        experiment_seed,
        kind,
        float(severity).hex(),
        block_id,
        batch_start,
    )
    sigmas = []
    for position, sample_id in enumerate(sample_ids):
        value = _stable_hash_seed(block_seed, position, sample_id)
        sigmas.append(((value + 0.5) / float(1 << 63)) * severity)
    return block_seed, sigmas


def _residual_permutation_pvalue(
    x: Sequence[float],
    y: Sequence[float],
    controls: Sequence[Sequence[float]],
    blocks: Sequence[str | int],
    *,
    seed: int,
    replicates: int,
) -> float:
    residual_x = _rank_residuals(x, controls, blocks=blocks)
    residual_y = _rank_residuals(y, controls, blocks=blocks)
    observed = abs(_pearson(residual_x, residual_y))
    grouped = {
        block: [index for index, value in enumerate(blocks) if value == block]
        for block in sorted(set(blocks), key=str)
    }
    rng = random.Random(seed)
    exceedances = 0
    for _ in range(replicates):
        permuted = list(residual_y)
        for indices in grouped.values():
            values = [permuted[index] for index in indices]
            rng.shuffle(values)
            for index, value in zip(indices, values, strict=True):
                permuted[index] = value
        if abs(_pearson(residual_x, permuted)) >= observed - 1e-15:
            exceedances += 1
    return (exceedances + 1) / (replicates + 1)


def _block_bootstrap_interval(
    x: Sequence[float],
    y: Sequence[float],
    controls: Sequence[Sequence[float]],
    blocks: Sequence[str | int],
    *,
    seed: int,
    replicates: int,
) -> list[float]:
    levels = sorted(set(blocks), key=str)
    grouped = {
        block: [index for index, value in enumerate(blocks) if value == block]
        for block in levels
    }
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        selected = [levels[rng.randrange(len(levels))] for _ in levels]
        indices: list[int] = []
        sampled_blocks: list[str] = []
        for draw, block in enumerate(selected):
            block_indices = grouped[block]
            indices.extend(block_indices)
            sampled_blocks.extend([f"{draw}:{block}"] * len(block_indices))
        sampled_controls = [
            [control[index] for index in indices] for control in controls
        ]
        estimates.append(
            _partial_spearman(
                [x[index] for index in indices],
                [y[index] for index in indices],
                *sampled_controls,
                blocks=sampled_blocks,
            )
        )
    lower, upper = np.quantile(np.asarray(estimates), (0.025, 0.975))
    return [float(lower), float(upper)]


def _repeated_measure_inference(
    x: Sequence[float],
    y: Sequence[float],
    controls: Sequence[Sequence[float]],
    blocks: Sequence[str | int],
    *,
    seed: int,
    bootstrap_replicates: int,
    permutation_replicates: int,
) -> dict[str, Any]:
    return {
        "estimate": _partial_spearman(x, y, *controls, blocks=blocks),
        "block_bootstrap_95_ci": _block_bootstrap_interval(
            x,
            y,
            controls,
            blocks,
            seed=_stable_hash_seed(seed, "bootstrap"),
            replicates=bootstrap_replicates,
        ),
        "within_block_permutation_pvalue_two_sided": _residual_permutation_pvalue(
            x,
            y,
            controls,
            blocks,
            seed=_stable_hash_seed(seed, "permutation"),
            replicates=permutation_replicates,
        ),
        "bootstrap_replicates": bootstrap_replicates,
        "permutation_replicates": permutation_replicates,
        "seed": seed,
        "method": (
            "Pearson correlation of rank residuals after ranked numeric controls "
            "and block fixed effects; complete-block bootstrap; within-block "
            "residual permutation"
        ),
    }


def _load_model_state_compatibly(model: torch.nn.Module, state: dict[str, Any]) -> None:
    incompatible = model.load_state_dict(state, strict=False)
    allowed_prefix = "graph_to_sheaf.transport_angle."
    missing = [
        key for key in incompatible.missing_keys if not key.startswith(allowed_prefix)
    ]
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"incompatible checkpoint: missing={missing}, unexpected={unexpected}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/gate2.yaml")
    parser.add_argument("--output", default="artifacts/gate3/corruption_report.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=12)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--permutation-replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap_replicates <= 0 or args.permutation_replicates <= 0:
        parser.error("inference replicate counts must be positive")

    config = load_gate2_config(args.config)
    runtime = initialize_runtime(config.runtime, seed=config.experiment.seed)
    dataset = engine._build_dataset(config, smoke=False)
    splits = dataset.split_indices(
        train_fraction=config.data.train_fraction,
        validation_fraction=config.data.validation_fraction,
    )
    test_indices = splits["test"]
    model = build_model(engine._build_model_config(config)).to(runtime.device).eval()
    checkpoint = torch.load(
        args.checkpoint, map_location=runtime.device, weights_only=False
    )
    _load_model_state_compatibly(model, checkpoint["model"])

    per_example_rows: list[dict] = []
    per_batch_rows: list[dict] = []

    for kind in corruption_kinds():
        route = ROUTE_FOR_KIND[kind]
        regime_examples = [
            index for index in test_indices if dataset.regimes[index].value == route
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
                block_id = f"{kind}:{batch_number:04d}"
                block_seed, sigmas = _corruption_sigmas(
                    clean.sample_ids,
                    data_seed=config.data.seed,
                    experiment_seed=config.experiment.seed,
                    kind=kind,
                    severity=severity,
                    block_id=batch_number,
                    batch_start=batch_start,
                )
                corrupted = collate_structured(
                    [
                        apply_corruption(
                            dataset[i],
                            kind=kind,
                            sigma=sigma,
                            seed=config.data.seed,
                        )
                        for i, sigma in zip(batch_indices, sigmas, strict=True)
                    ]
                )
                clean = clean.to(runtime.device)
                corrupted = corrupted.to(runtime.device)
                with (
                    torch.no_grad(),
                    torch.autocast(
                        device_type=runtime.device.type,
                        dtype=runtime.neural_dtype,
                        enabled=runtime.device.type == "cuda",
                    ),
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
                        "block_id": block_id,
                        "block_number": batch_number,
                        "batch_start": batch_start,
                        "block_seed": block_seed,
                        "num_examples": len(batch_indices),
                        "sigma_min": min(sigmas),
                        "sigma_mean": statistics.mean(sigmas),
                        "sigma_max": max(sigmas),
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
                            "sample_id": clean.sample_ids[k],
                            "block_id": block_id,
                            "block_seed": block_seed,
                            "sigma": sigmas[k],
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
        topology = [row["topological_defect"] for row in batch_rows]
        damage_rates = [row["damage_rate"] for row in batch_rows]
        reconstruction = [row["mean_reconstruction"] for row in batch_rows]
        severities = [row["severity"] for row in batch_rows]
        blocks = [row["block_id"] for row in batch_rows]
        inference_seed = _stable_hash_seed(
            _ANALYSIS_METHOD,
            config.data.seed,
            config.experiment.seed,
            kind,
        )
        repeated = _repeated_measure_inference(
            topology,
            damage_rates,
            (reconstruction, severities),
            blocks,
            seed=inference_seed,
            bootstrap_replicates=args.bootstrap_replicates,
            permutation_replicates=args.permutation_replicates,
        )
        analysis[kind] = {
            # Backwards-readable: this remains the number of repeated
            # example-severity observations, not the number of unique samples.
            "examples": len(rows),
            "example_observations": len(rows),
            "unique_examples": len({row["sample_id"] for row in rows}),
            "batch_observations": len(batch_rows),
            "unique_blocks": len(set(blocks)),
            "severity_levels": len(set(severities)),
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
                topology,
                damage_rates,
            ),
            "batch_partial(topological_defect, damage_rate | reconstruction)": (
                _partial_spearman(topology, damage_rates, reconstruction)
            ),
            "batch_partial(topological_defect, damage_rate | reconstruction, severity, block)": repeated[
                "estimate"
            ],
            "repeated_measures_inference": repeated,
            "batch_spearman(reconstruction, damage_rate)": _spearman(
                [row["mean_reconstruction"] for row in batch_rows],
                [row["damage_rate"] for row in batch_rows],
            ),
        }

    report = {
        "schema_version": 2,
        "checkpoint": args.checkpoint,
        "severities": list(SEVERITIES),
        "sampling": {
            "protocol": _DRAW_PROTOCOL,
            "data_seed": config.data.seed,
            "experiment_seed": config.experiment.seed,
            "pairing_contract": (
                "Runs with equal data_seed, experiment_seed, split, batch size, "
                "kind, severity, block, and sample IDs receive identical draws."
            ),
        },
        "analysis_protocol": {
            "method": _ANALYSIS_METHOD,
            "numeric_controls": ["mean_reconstruction", "severity"],
            "repeated_measure_block": "batch block across severity levels",
        },
        "analysis": analysis,
        "per_batch": per_batch_rows,
        "per_example": per_example_rows,
    }
    output = Path(args.output)
    atomic_json(output, report)
    for kind, result in analysis.items():
        print(kind, json.dumps(result, indent=2))
    print(f"report written to {output}")


if __name__ == "__main__":
    main()
