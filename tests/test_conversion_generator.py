from __future__ import annotations

import numpy as np
import pytest
import torch

from homymoly.data.conversion import (
    ConversionConfig,
    ConversionDataset,
    ConversionSample,
)
from homymoly.topology import (
    ChainComplex,
    exactness_defects,
    graph_to_cell_inclusion,
)

DTYPE = torch.float64


def _dataset(count: int = 64, **kwargs: object) -> ConversionDataset:
    return ConversionDataset(count, seed=20260823, dtype=DTYPE, **kwargs)  # type: ignore[arg-type]


def _complexes(sample: ConversionSample) -> tuple[ChainComplex, ChainComplex]:
    graph = ChainComplex((sample.num_vertices, sample.num_edges), (sample.boundary_1,))
    cell = ChainComplex(
        (sample.num_vertices, sample.num_edges, sample.num_faces),
        (sample.boundary_1, sample.boundary_2),
    )
    return graph, cell


def test_samples_are_deterministic_for_a_seed() -> None:
    first, second = _dataset(8)[3], _dataset(8)[3]

    assert first.sample_id == second.sample_id
    assert torch.equal(first.boundary_2, second.boundary_2)
    assert torch.equal(
        first.observations.edge_features, second.observations.edge_features
    )
    assert torch.equal(first.face_active, second.face_active)


def test_the_cycle_basis_is_a_valid_boundary_on_every_sample() -> None:
    """B1 @ B2 == 0 is what makes the cell complex a complex at all."""

    for sample in _dataset(64):
        if not sample.num_faces:
            continue
        product = sample.boundary_1 @ sample.boundary_2
        assert float(product.abs().max()) == 0.0


def test_cell_structure_is_a_function_of_the_graph_observation() -> None:
    """Face activity thresholds B2^T x1, so a perfect converter exists."""

    for sample in _dataset(32):
        if not sample.num_faces:
            continue
        cochain = sample.observations.edge_features[:, 0]
        torch.testing.assert_close(
            sample.boundary_2.mT @ cochain, sample.face_circulation, atol=1e-12, rtol=0
        )
        threshold = sample.metadata["activity_threshold"]
        expected = sample.face_circulation.abs() > threshold
        assert torch.equal(sample.face_active, expected)


def test_sheaf_holonomy_is_carried_by_the_twist_channel() -> None:
    """A frame difference telescopes around a cycle, so the twist supplies holonomy."""

    for sample in _dataset(32):
        if not sample.num_faces:
            continue
        twist = sample.observations.edge_features[:, 1]
        torch.testing.assert_close(
            sample.boundary_2.mT @ twist,
            sample.cycle_holonomy_angle,
            atol=1e-12,
            rtol=0,
        )
        # Transport is the endpoint frame difference plus that twist.
        tails, heads = sample.edge_index[0], sample.edge_index[1]
        angle = torch.atan2(
            sample.observations.node_features[:, 1],
            sample.observations.node_features[:, 0],
        )
        torch.testing.assert_close(
            sample.edge_transport_angle,
            angle[heads] - angle[tails] + twist,
            atol=1e-6,
            rtol=0,
        )


def test_frame_differences_alone_would_make_every_holonomy_trivial() -> None:
    """The reason the twist channel exists, stated as a check."""

    for sample in _dataset(16):
        if not sample.num_faces:
            continue
        angle = torch.atan2(
            sample.observations.node_features[:, 1],
            sample.observations.node_features[:, 0],
        )
        tails, heads = sample.edge_index[0], sample.edge_index[1]
        frame_only = angle[heads] - angle[tails]
        assert float((sample.boundary_2.mT @ frame_only).abs().max()) < 1e-8


def test_homological_defects_vary_across_examples() -> None:
    """The failure mode of every earlier design: a constant defect informs nothing."""

    profiles, ranks = set(), set()
    for sample in _dataset(96):
        if not sample.num_faces:
            continue
        graph, cell = _complexes(sample)
        defects = exactness_defects(graph_to_cell_inclusion(graph, cell))
        profiles.add(tuple((d.homology_kernel, d.homology_cokernel) for d in defects))
        ranks.add(sample.cycle_rank)

    assert len(ranks) > 5, "graph topology must vary"
    assert len(profiles) > 5, "defects must vary with it"


def test_the_destroyed_class_count_equals_the_cycle_rank() -> None:
    """The inclusion sends every independent cycle to a face boundary."""

    for sample in _dataset(32):
        if not sample.num_faces:
            continue
        graph, cell = _complexes(sample)
        defects = exactness_defects(graph_to_cell_inclusion(graph, cell))
        assert defects[1].homology_kernel == sample.cycle_rank
        assert defects[1].homology_cokernel == 0


def test_face_activity_is_close_to_balanced() -> None:
    rates = [
        float(sample.face_active.float().mean())
        for sample in _dataset(96)
        if sample.num_faces
    ]
    assert 0.35 < float(np.mean(rates)) < 0.65


def test_observation_shapes_and_finiteness() -> None:
    for sample in _dataset(16):
        observations = sample.observations
        assert observations.node_features.shape == (sample.num_vertices, 4)
        assert observations.edge_features.shape == (sample.num_edges, 3)
        assert sample.edge_index.shape == (2, sample.num_edges)
        assert torch.isfinite(observations.node_features).all()
        assert torch.isfinite(observations.edge_features).all()


def test_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="min_vertices"):
        ConversionConfig(min_vertices=3)
    with pytest.raises(ValueError, match="max_vertices"):
        ConversionConfig(min_vertices=10, max_vertices=8)
    with pytest.raises(ValueError, match="densities"):
        ConversionConfig(min_density=0.5, max_density=0.2)
    with pytest.raises(ValueError, match="activity_quantile"):
        ConversionConfig(activity_quantile=1.0)
    with pytest.raises(ValueError, match="twist_scale"):
        ConversionConfig(twist_scale=0.0)


def test_dataset_rejects_bad_indices_and_sizes() -> None:
    with pytest.raises(ValueError, match="num_samples must be positive"):
        ConversionDataset(0, seed=1)
    with pytest.raises(TypeError, match="num_samples must be an integer"):
        ConversionDataset(True, seed=1)  # type: ignore[arg-type]
    dataset = _dataset(4)
    with pytest.raises(IndexError):
        dataset[9]
    assert dataset[-1].sample_id == dataset[3].sample_id
