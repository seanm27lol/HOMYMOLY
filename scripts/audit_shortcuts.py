#!/usr/bin/env python3
"""Measure cheap scalar shortcuts in the Stage-1 bring-up benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from homymoly.config import load_config
from homymoly.data import MixedStructuredSignal, SignalRegime

REGIMES = tuple(SignalRegime)
REGIME_TO_INDEX = {regime: index for index, regime in enumerate(REGIMES)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/stage1.yaml"))
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--vertices", type=int, default=24)
    parser.add_argument("--output", type=Path)
    return parser


def _features(sample: Any) -> tuple[list[float], list[float]]:
    node = sample.node_features
    edge = sample.edge_features
    tails, heads = sample.edge_index

    route_features = [
        float(node[:, 0].abs().max()),
        float(torch.topk(edge[:, 1].abs(), k=min(3, sample.num_edges)).values.mean()),
        float(torch.linalg.vector_norm(node[:, -2:], dim=1).mean()),
    ]

    edge_products = node[tails, 0] * node[heads, 0]
    strongest = int(edge_products.abs().argmax())
    graph_relation = float(edge_products[strongest])
    cell_edge_sum = float(edge[:, 1].sum())
    transported = torch.einsum("eij,ej->ei", sample.transport, node[tails, -2:])
    sheaf_residual = float(
        torch.linalg.vector_norm(node[heads, -2:] - transported, dim=1).max()
    )
    return route_features, [graph_relation, cell_edge_sum, sheaf_residual]


def _fit_stump(values: np.ndarray, labels: np.ndarray) -> tuple[float, int, float]:
    unique = np.unique(values)
    if unique.size == 1:
        candidates = unique
    else:
        candidates = np.concatenate(
            (
                [unique[0] - 1.0],
                (unique[:-1] + unique[1:]) / 2.0,
                [unique[-1] + 1.0],
            )
        )
    best = (float(candidates[0]), 1, -1.0)
    for threshold in candidates:
        for polarity in (-1, 1):
            predictions = (polarity * (values - threshold) >= 0).astype(np.int64)
            accuracy = float(np.mean(predictions == labels))
            if accuracy > best[2]:
                best = (float(threshold), polarity, accuracy)
    return best


def audit_shortcuts(
    *,
    num_samples: int,
    seed: int,
    num_vertices: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, Any]:
    dataset = MixedStructuredSignal(
        num_samples=num_samples,
        seed=seed,
        num_vertices=num_vertices,
    )
    splits = dataset.split_indices(
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    route_features: list[list[float]] = []
    label_features: list[list[float]] = []
    regimes: list[int] = []
    labels: list[int] = []
    for sample in dataset:
        route, label = _features(sample)
        route_features.append(route)
        label_features.append(label)
        regimes.append(REGIME_TO_INDEX[sample.regime])
        labels.append(int(sample.label))

    route_array = np.asarray(route_features, dtype=np.float64)
    label_array = np.asarray(label_features, dtype=np.float64)
    regime_array = np.asarray(regimes, dtype=np.int64)
    target_array = np.asarray(labels, dtype=np.int64)
    train = np.asarray(splits["train"], dtype=np.int64)
    test = np.asarray(splits["test"], dtype=np.int64)

    mean = route_array[train].mean(axis=0)
    scale = route_array[train].std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (route_array - mean) / scale
    centroids = np.stack(
        [
            standardized[train][regime_array[train] == index].mean(axis=0)
            for index in range(3)
        ]
    )
    distances = np.linalg.norm(
        standardized[test, None, :] - centroids[None, :, :],
        axis=2,
    )
    route_predictions = distances.argmin(axis=1)
    route_accuracy = float(np.mean(route_predictions == regime_array[test]))

    stump_reports: dict[str, dict[str, float | int]] = {}
    thresholds: list[float] = []
    polarities: list[int] = []
    for index, regime in enumerate(REGIMES):
        regime_train = train[regime_array[train] == index]
        threshold, polarity, training_accuracy = _fit_stump(
            label_array[regime_train, index],
            target_array[regime_train],
        )
        regime_test = test[regime_array[test] == index]
        predictions = (
            polarity * (label_array[regime_test, index] - threshold) >= 0
        ).astype(np.int64)
        stump_reports[regime.value] = {
            "threshold": threshold,
            "polarity": polarity,
            "train_accuracy": training_accuracy,
            "test_accuracy": float(np.mean(predictions == target_array[regime_test])),
        }
        thresholds.append(threshold)
        polarities.append(polarity)

    scalar_predictions = np.empty(len(test), dtype=np.int64)
    for position, predicted_regime in enumerate(route_predictions):
        value = label_array[test[position], predicted_regime]
        scalar_predictions[position] = int(
            polarities[predicted_regime] * (value - thresholds[predicted_regime]) >= 0
        )

    return {
        "schema_version": 1,
        "benchmark_tier": "bringup",
        "warning": "High scalar accuracy disqualifies this tier from routing-novelty claims.",
        "num_samples": len(dataset),
        "split_sizes": {name: len(indices) for name, indices in splits.items()},
        "amplitude_route_accuracy": route_accuracy,
        "conditional_scalar_label_baselines": stump_reports,
        "routed_scalar_system_accuracy": float(
            np.mean(scalar_predictions == target_array[test])
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    config = load_config(args.config)
    report = audit_shortcuts(
        num_samples=args.samples,
        seed=config.data.seed,
        num_vertices=args.vertices,
        train_fraction=config.data.train_fraction,
        validation_fraction=config.data.validation_fraction,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(output)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
