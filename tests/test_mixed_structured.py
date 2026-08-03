from __future__ import annotations

from collections import Counter

import torch

from homymoly.data import MixedStructuredSignal, SignalRegime


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


def test_dataset_is_deterministic_by_seed_and_index() -> None:
    first = MixedStructuredSignal(18, seed=123, min_vertices=24, max_vertices=30)
    second = MixedStructuredSignal(18, seed=123, min_vertices=24, max_vertices=30)

    for index in (0, 5, 6, 17):
        _assert_same_sample(first[index], first[index])
        _assert_same_sample(first[index], second[index])

    changed = MixedStructuredSignal(18, seed=124, min_vertices=24, max_vertices=30)
    assert first[0].sample_id != changed[0].sample_id
    assert not torch.equal(first[0].node_features, changed[0].node_features) or (
        first[0].num_vertices != changed[0].num_vertices
    )


def test_joint_schedule_is_balanced_and_inspectable() -> None:
    dataset = MixedStructuredSignal(60, seed=5, num_vertices=24)

    assert dataset.label_counts == {0: 30, 1: 30}
    assert dataset.regime_counts == {
        SignalRegime.GRAPH: 20,
        SignalRegime.CELL: 20,
        SignalRegime.SHEAF: 20,
    }
    assert set(dataset.joint_counts.values()) == {10}
    assert len(dataset.indices_for(regime="graph")) == 20
    assert len(dataset.indices_for(regime="cell", label=1)) == 10

    summary = dataset.distribution()
    assert summary["num_samples"] == 60
    assert summary["joint_counts"] == dataset.joint_counts


def test_nonmultiple_schedule_balances_both_marginals() -> None:
    for size in range(6, 18):
        dataset = MixedStructuredSignal(size, seed=10, num_vertices=24)
        assert max(dataset.label_counts.values()) - min(dataset.label_counts.values()) <= 1
        assert max(dataset.regime_counts.values()) - min(dataset.regime_counts.values()) <= 1
        assert max(dataset.joint_counts.values()) - min(dataset.joint_counts.values()) <= 1


def test_each_counterfactual_group_shares_canonical_structure() -> None:
    dataset = MixedStructuredSignal(12, seed=72, min_vertices=24, max_vertices=40)

    for start in (0, 6):
        group = [dataset[index] for index in range(start, start + 6)]
        assert Counter((sample.regime, int(sample.label)) for sample in group) == Counter(
            (regime, label) for regime in SignalRegime for label in (0, 1)
        )
        reference = group[0]
        for sample in group[1:]:
            assert sample.num_vertices == reference.num_vertices
            assert torch.equal(sample.edge_index, reference.edge_index)
            assert torch.equal(sample.face_index, reference.face_index)


def test_generated_structures_are_canonical_and_within_size_contract() -> None:
    dataset = MixedStructuredSignal(18, seed=44, min_vertices=24, max_vertices=36)

    for sample in (dataset[index] for index in range(len(dataset))):
        assert 24 <= sample.num_vertices <= 36
        edges = [tuple(edge) for edge in sample.edge_index.t().tolist()]
        faces = [tuple(face) for face in sample.face_index.t().tolist()]
        assert edges == sorted(set(edges))
        assert all(u < v for u, v in edges)
        assert faces == sorted(set(faces))
        assert all(a < b < c for a, b, c in faces)
        edge_set = set(edges)
        for a, b, c in faces:
            assert {(a, b), (a, c), (b, c)}.issubset(edge_set)

        assert sample.face_active.dtype == torch.bool
        assert 0 < int(sample.face_active.sum()) <= sample.num_faces
        identity = torch.eye(2).expand(sample.num_edges, 2, 2)
        torch.testing.assert_close(
            sample.transport.transpose(-1, -2) @ sample.transport,
            identity,
            rtol=1e-5,
            atol=1e-5,
        )
        assert "label" not in sample.metadata
        assert "regime" not in sample.metadata
        for regime in SignalRegime:
            assert sample.observations_for(regime) is sample.observations


def test_group_disjoint_splits_cover_dataset_and_expose_counts() -> None:
    dataset = MixedStructuredSignal(60, seed=13, num_vertices=(24, 30))
    split = dataset.split_indices(seed=99)
    repeated = dataset.split_indices(seed=99)

    assert split == repeated
    assert set(split) == {"train", "validation", "test"}
    split_sets = {name: set(indices) for name, indices in split.items()}
    assert split_sets["train"].isdisjoint(split_sets["validation"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["validation"].isdisjoint(split_sets["test"])
    assert set.union(*split_sets.values()) == set(range(len(dataset)))

    group_sets = {
        name: {dataset.group_ids[index] for index in indices}
        for name, indices in split.items()
    }
    assert group_sets["train"].isdisjoint(group_sets["validation"])
    assert group_sets["train"].isdisjoint(group_sets["test"])
    assert group_sets["validation"].isdisjoint(group_sets["test"])
    assert sum(dataset.distribution(indices)["num_samples"] for indices in split.values()) == 60


def test_cell_label_has_no_global_edge_sum_shortcut() -> None:
    dataset = MixedStructuredSignal(300, seed=2026, num_vertices=24)
    cell_samples = [
        dataset[index]
        for index in dataset.indices_for(regime=SignalRegime.CELL)
    ]
    predictions = [
        int(float(sample.edge_features[:, 1].sum()) >= 0.0)
        for sample in cell_samples
    ]
    labels = [int(sample.label) for sample in cell_samples]
    accuracy = sum(left == right for left, right in zip(predictions, labels, strict=True)) / len(labels)

    # Check both possible sign conventions. This caught a previous >90%
    # first-order leak caused by tying circulation orientation to the label.
    best_polarity_accuracy = max(accuracy, 1.0 - accuracy)
    assert best_polarity_accuracy < 0.65
    assert {sample.metadata["benchmark_tier"] for sample in cell_samples} == {"bringup"}
