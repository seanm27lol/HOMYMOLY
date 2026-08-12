from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from homymoly.data.confirmatory import (
    ConfirmatoryConfig,
    ConfirmatoryStructuredSignal,
)
from homymoly.data.types import SignalRegime
from homymoly.topology.incidence import build_oriented_incidence


def _assert_same_sample(left: object, right: object) -> None:
    assert type(left) is type(right)
    assert left.sample_id == right.sample_id  # type: ignore[attr-defined]
    assert left.regime is right.regime  # type: ignore[attr-defined]
    assert int(left.label) == int(right.label)  # type: ignore[attr-defined]
    for name in (
        "node_features",
        "edge_features",
        "edge_index",
        "face_index",
        "face_active",
        "transport",
    ):
        assert torch.equal(getattr(left, name), getattr(right, name))


def _pair_by_regime(
    samples: list[object], regime: SignalRegime
) -> tuple[object, object]:
    selected = sorted(
        (sample for sample in samples if sample.regime is regime),  # type: ignore[attr-defined]
        key=lambda sample: int(sample.label),  # type: ignore[attr-defined]
    )
    assert len(selected) == 2
    return selected[0], selected[1]


def _route_scalars(sample: object) -> np.ndarray:
    return np.asarray(
        [
            float(sample.node_features[:, 0].abs().max()),  # type: ignore[attr-defined]
            float(
                sample.edge_features[:, 1]  # type: ignore[attr-defined]
                .abs()
                .topk(3)
                .values.mean()
            ),
            float(
                torch.linalg.vector_norm(
                    sample.node_features[:, -2:],
                    dim=1,  # type: ignore[attr-defined]
                ).mean()
            ),
        ],
        dtype=np.float64,
    )


def _local_residual_max(sample: object) -> float:
    tail, head = sample.edge_index  # type: ignore[attr-defined]
    transported = torch.einsum(
        "eij,ej->ei",
        sample.transport,  # type: ignore[attr-defined]
        sample.node_features[tail, -2:],  # type: ignore[attr-defined]
    )
    residual = transported - sample.node_features[head, -2:]  # type: ignore[attr-defined]
    return float(torch.linalg.vector_norm(residual, dim=1).max())


def _graph_relation_prediction(sample: object) -> int:
    tail, head = sample.edge_index  # type: ignore[attr-defined]
    anchor_mask = (
        (sample.node_features[tail, 0].abs() > 0.5)  # type: ignore[attr-defined]
        & (sample.node_features[head, 0].abs() > 0.5)  # type: ignore[attr-defined]
    )
    products = (
        sample.node_features[tail[anchor_mask], 0]  # type: ignore[attr-defined]
        * sample.node_features[head[anchor_mask], 0]  # type: ignore[attr-defined]
    )
    return int(float(products.sum()) > 0)


def _cell_probe_prediction(sample: object) -> int:
    edge_positions = {
        tuple(edge): position
        for position, edge in enumerate(
            sample.edge_index.t().tolist()  # type: ignore[attr-defined]
        )
    }
    circulations = []
    for a, b, c in sample.face_index.t().tolist():  # type: ignore[attr-defined]
        circulations.append(
            sample.edge_features[edge_positions[(b, c)], 1]  # type: ignore[attr-defined]
            - sample.edge_features[edge_positions[(a, c)], 1]  # type: ignore[attr-defined]
            + sample.edge_features[edge_positions[(a, b)], 1]  # type: ignore[attr-defined]
        )
    probe_position = int(torch.stack(circulations).abs().argmax())
    return int(sample.face_active[probe_position])  # type: ignore[attr-defined]


def _maximum_holonomy_defect(sample: object) -> float:
    edge_positions = {
        tuple(edge): position
        for position, edge in enumerate(
            sample.edge_index.t().tolist()  # type: ignore[attr-defined]
        )
    }
    identity = torch.eye(2, dtype=sample.transport.dtype)  # type: ignore[attr-defined]
    defects = []
    for a, b, c in sample.face_index.t().tolist():  # type: ignore[attr-defined]
        transport_ab = sample.transport[edge_positions[(a, b)]]  # type: ignore[attr-defined]
        transport_ac = sample.transport[edge_positions[(a, c)]]  # type: ignore[attr-defined]
        transport_bc = sample.transport[edge_positions[(b, c)]]  # type: ignore[attr-defined]
        holonomy = transport_ac.transpose(-1, -2) @ transport_bc @ transport_ab
        defects.append(float(torch.linalg.matrix_norm(holonomy - identity)))
    return max(defects)


def _nearest_centroid_accuracy(
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> float:
    mean = features[train_indices].mean(axis=0)
    scale = features[train_indices].std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (features - mean) / scale
    classes = np.unique(targets)
    centroids = np.stack(
        [
            standardized[train_indices][targets[train_indices] == item].mean(axis=0)
            for item in classes
        ]
    )
    distances = (
        (standardized[test_indices, None, :] - centroids[None, :, :]) ** 2
    ).sum(axis=2)
    predictions = classes[distances.argmin(axis=1)]
    return float(np.mean(predictions == targets[test_indices]))


def _best_threshold_accuracy(
    values: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> float:
    train_values = values[train_indices]
    ordered = np.sort(np.unique(train_values))
    thresholds = np.concatenate(
        (
            [ordered[0] - 1e-9],
            (ordered[:-1] + ordered[1:]) / 2,
            [ordered[-1] + 1e-9],
        )
    )
    best_accuracy = -1.0
    best_threshold = 0.0
    best_direction = 1
    for threshold in thresholds:
        for direction in (1, -1):
            predictions = (
                train_values > threshold if direction == 1 else train_values < threshold
            ).astype(np.int64)
            accuracy = float(np.mean(predictions == targets[train_indices]))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = float(threshold)
                best_direction = direction
    predictions = (
        values[test_indices] > best_threshold
        if best_direction == 1
        else values[test_indices] < best_threshold
    ).astype(np.int64)
    return float(np.mean(predictions == targets[test_indices]))


def test_confirmatory_config_requires_complete_overlapping_groups() -> None:
    with pytest.raises(ValueError, match="divisible by six"):
        ConfirmatoryConfig(num_samples=20)
    with pytest.raises(ValueError, match="overlap"):
        ConfirmatoryConfig(
            reliable_strength_min=1.3,
            reliable_strength_max=1.5,
            nuisance_strength_min=0.8,
            nuisance_strength_max=1.2,
        )
    dataset = ConfirmatoryStructuredSignal(ConfirmatoryConfig(num_samples=18))
    assert dataset.config.num_samples == 18


def test_confirmatory_generation_is_deterministic_balanced_and_group_disjoint() -> None:
    first = ConfirmatoryStructuredSignal(60, seed=27, min_vertices=24, max_vertices=36)
    second = ConfirmatoryStructuredSignal(60, seed=27, min_vertices=24, max_vertices=36)
    for index in (0, 5, 6, 31, 59):
        _assert_same_sample(first[index], first[index])
        _assert_same_sample(first[index], second[index])

    assert first.label_counts == {0: 30, 1: 30}
    assert first.regime_counts == {regime: 20 for regime in SignalRegime}
    assert set(first.joint_counts.values()) == {10}
    for group_id in range(10):
        indices = range(group_id * 6, (group_id + 1) * 6)
        assert Counter((first.regimes[i], first.labels[i]) for i in indices) == Counter(
            (regime, label) for regime in SignalRegime for label in (0, 1)
        )

    split = first.split_indices(seed=991)
    repeated = first.split_indices(seed=991)
    assert split == repeated
    split_sets = {name: set(indices) for name, indices in split.items()}
    assert set.union(*split_sets.values()) == set(range(len(first)))
    assert split_sets["train"].isdisjoint(split_sets["validation"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["validation"].isdisjoint(split_sets["test"])
    group_sets = {
        name: {first.group_ids[index] for index in indices}
        for name, indices in split.items()
    }
    assert group_sets["train"].isdisjoint(group_sets["validation"])
    assert group_sets["train"].isdisjoint(group_sets["test"])
    assert group_sets["validation"].isdisjoint(group_sets["test"])
    for indices in split.values():
        assert set(first.distribution(indices)["joint_counts"].values()) == {
            len(indices) // 6
        }


def test_counterfactual_pairs_match_structure_and_unary_marginals() -> None:
    dataset = ConfirmatoryStructuredSignal(18, seed=42, num_vertices=30)
    group = [dataset[index] for index in range(6)]
    reference = group[0]
    for sample in group[1:]:
        assert torch.equal(sample.edge_index, reference.edge_index)
        assert torch.equal(sample.face_index, reference.face_index)
        assert sample.num_vertices == reference.num_vertices
        assert int(sample.face_active.sum()) == int(reference.face_active.sum())

    graph_zero, graph_one = _pair_by_regime(group, SignalRegime.GRAPH)
    torch.testing.assert_close(
        torch.sort(graph_zero.node_features[:, 0]).values,
        torch.sort(graph_one.node_features[:, 0]).values,
        rtol=0,
        atol=0,
    )
    assert torch.equal(graph_zero.node_features[:, 1:], graph_one.node_features[:, 1:])
    assert torch.equal(graph_zero.edge_features, graph_one.edge_features)
    assert torch.equal(graph_zero.face_active, graph_one.face_active)
    assert torch.equal(graph_zero.transport, graph_one.transport)
    torch.testing.assert_close(_route_scalars(graph_zero), _route_scalars(graph_one))

    cell_zero, cell_one = _pair_by_regime(group, SignalRegime.CELL)
    assert torch.equal(cell_zero.node_features, cell_one.node_features)
    assert torch.equal(cell_zero.edge_features, cell_one.edge_features)
    assert torch.equal(cell_zero.transport, cell_one.transport)
    assert not torch.equal(cell_zero.face_active, cell_one.face_active)
    assert float(cell_zero.edge_features[:, 1].sum()) == float(
        cell_one.edge_features[:, 1].sum()
    )

    sheaf_zero, sheaf_one = _pair_by_regime(group, SignalRegime.SHEAF)
    assert torch.equal(sheaf_zero.node_features, sheaf_one.node_features)
    assert torch.equal(sheaf_zero.edge_features, sheaf_one.edge_features)
    assert torch.equal(sheaf_zero.face_active, sheaf_one.face_active)
    assert not torch.equal(sheaf_zero.transport, sheaf_one.transport)
    torch.testing.assert_close(_route_scalars(sheaf_zero), _route_scalars(sheaf_one))


def test_boundaries_and_route_scoped_views_are_valid() -> None:
    dataset = ConfirmatoryStructuredSignal(
        30, seed=71, min_vertices=24, max_vertices=40
    )
    for index in range(len(dataset)):
        sample = dataset[index]
        incidence = build_oriented_incidence(
            sample.num_vertices,
            sample.edge_index.t().tolist(),
            sample.face_index.t().tolist(),
        )
        torch.testing.assert_close(
            incidence.boundary_1 @ incidence.boundary_2,
            torch.zeros((sample.num_vertices, sample.num_faces), dtype=torch.float64),
            rtol=0,
            atol=0,
        )
        assert torch.isfinite(sample.node_features).all()
        assert torch.isfinite(sample.edge_features).all()
        orthogonality_error = sample.transport.transpose(
            -1, -2
        ) @ sample.transport - torch.eye(2)
        assert float(orthogonality_error.abs().max()) < 1e-5

    sample = dataset[0]
    assert set(sample.model_inputs("graph")) == {
        "node_features",
        "edge_features",
        "edge_index",
    }
    assert {"face_index", "face_active"}.issubset(sample.model_inputs("cell"))
    assert "transport" not in sample.model_inputs("cell")
    assert "transport" in sample.model_inputs("sheaf")
    assert "face_active" not in sample.model_inputs("sheaf")
    for route in SignalRegime:
        inputs = sample.model_inputs(route)
        assert "label" not in inputs
        assert "regime" not in inputs
        assert sample.observations_for(route) is sample.observations


def test_relational_mechanisms_recover_their_conditional_targets() -> None:
    dataset = ConfirmatoryStructuredSignal(
        180, seed=903, min_vertices=24, max_vertices=64
    )
    graph_samples = [dataset[index] for index in dataset.indices_for(regime="graph")]
    cell_samples = [dataset[index] for index in dataset.indices_for(regime="cell")]
    sheaf_samples = [dataset[index] for index in dataset.indices_for(regime="sheaf")]

    graph_accuracy = np.mean(
        [
            _graph_relation_prediction(sample) == int(sample.label)
            for sample in graph_samples
        ]
    )
    cell_accuracy = np.mean(
        [_cell_probe_prediction(sample) == int(sample.label) for sample in cell_samples]
    )
    sheaf_accuracy = np.mean(
        [
            (_maximum_holonomy_defect(sample) > 1.0) == int(sample.label)
            for sample in sheaf_samples
        ]
    )
    assert graph_accuracy > 0.90
    assert cell_accuracy > 0.95
    assert sheaf_accuracy > 0.95


def test_basic_scalar_shortcuts_remain_below_confirmatory_ceiling() -> None:
    dataset = ConfirmatoryStructuredSignal(
        600, seed=20260803, min_vertices=24, max_vertices=64
    )
    samples = [dataset[index] for index in range(len(dataset))]
    split = dataset.split_indices(seed=404)
    train_indices = np.asarray(split["train"], dtype=np.int64)
    test_indices = np.asarray(split["test"], dtype=np.int64)

    route_features = np.stack([_route_scalars(sample) for sample in samples])
    regime_targets = np.asarray(
        [tuple(SignalRegime).index(sample.regime) for sample in samples], dtype=np.int64
    )
    route_accuracy = _nearest_centroid_accuracy(
        route_features, regime_targets, train_indices, test_indices
    )
    assert 1 / 3 < route_accuracy < 0.80

    scalar_features = np.stack(
        [
            np.concatenate(
                (
                    np.asarray(
                        [
                            sample.num_vertices,
                            sample.num_edges,
                            sample.num_faces,
                            int(sample.face_active.sum()),
                            float(sample.edge_features[:, 1].sum()),
                            float(sample.node_features[:, 0].sum()),
                            float(
                                sample.transport.diagonal(dim1=-2, dim2=-1)
                                .sum(-1)
                                .mean()
                            ),
                            _local_residual_max(sample),
                        ]
                    ),
                    _route_scalars(sample),
                )
            )
            for sample in samples
        ]
    )
    label_targets = np.asarray(
        [int(sample.label) for sample in samples], dtype=np.int64
    )
    label_accuracy = _nearest_centroid_accuracy(
        scalar_features, label_targets, train_indices, test_indices
    )
    assert label_accuracy < 0.65

    cell_indices = np.asarray(
        [
            index
            for index, sample in enumerate(samples)
            if sample.regime is SignalRegime.CELL
        ],
        dtype=np.int64,
    )
    train_cell = np.intersect1d(train_indices, cell_indices)
    test_cell = np.intersect1d(test_indices, cell_indices)
    edge_sums = np.asarray(
        [float(sample.edge_features[:, 1].sum()) for sample in samples]
    )
    assert _best_threshold_accuracy(
        edge_sums, label_targets, train_cell, test_cell
    ) == pytest.approx(0.5)

    sheaf_indices = np.asarray(
        [
            index
            for index, sample in enumerate(samples)
            if sample.regime is SignalRegime.SHEAF
        ],
        dtype=np.int64,
    )
    train_sheaf = np.intersect1d(train_indices, sheaf_indices)
    test_sheaf = np.intersect1d(test_indices, sheaf_indices)
    local_residuals = np.asarray([_local_residual_max(sample) for sample in samples])
    assert (
        _best_threshold_accuracy(
            local_residuals, label_targets, train_sheaf, test_sheaf
        )
        < 0.70
    )
