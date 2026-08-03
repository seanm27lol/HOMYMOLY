from __future__ import annotations

import pickle
from dataclasses import replace

import pytest
import torch

from homymoly.data import (
    MixedStructuredSignal,
    SignalRegime,
    collate_structured,
)
from homymoly.topology.incidence import build_oriented_incidence


def test_sample_shares_one_observation_bundle_and_isolates_metadata() -> None:
    sample = MixedStructuredSignal(6, seed=9, num_vertices=24)[0]

    assert sample.observations_for(SignalRegime.GRAPH) is sample.observations
    assert sample.observations_for(SignalRegime.CELL) is sample.observations
    assert sample.observations_for(SignalRegime.SHEAF) is sample.observations
    assert sample.node_features.data_ptr() == sample.observations.node_features.data_ptr()
    assert sample.edge_features.data_ptr() == sample.observations.edge_features.data_ptr()

    graph_inputs = sample.model_inputs("graph")
    cell_inputs = sample.model_inputs("cell")
    sheaf_inputs = sample.model_inputs("sheaf")
    assert set(graph_inputs) == {"node_features", "edge_features", "edge_index"}
    assert set(cell_inputs) == {
        "node_features",
        "edge_features",
        "edge_index",
        "face_index",
        "face_active",
    }
    assert set(sheaf_inputs) == {
        "node_features",
        "edge_features",
        "edge_index",
        "transport",
        "face_index",
    }
    assert "face_active" not in graph_inputs
    assert "transport" not in graph_inputs
    for model_inputs in (graph_inputs, cell_inputs, sheaf_inputs):
        assert "label" not in model_inputs
        assert "regime" not in model_inputs
        assert "metadata" not in model_inputs
    assert "label" not in sample.metadata
    assert "regime" not in sample.metadata
    with pytest.raises(TypeError):
        sample.metadata["new_key"] = "not mutable"  # type: ignore[index]


def test_sample_rejects_noncanonical_edges_and_reserved_metadata() -> None:
    sample = MixedStructuredSignal(6, seed=4, num_vertices=24)[0]
    reversed_edge = sample.edge_index.clone()
    reversed_edge[:, 0] = reversed_edge[:, 0].flip(0)

    with pytest.raises(ValueError, match="u < v"):
        replace(sample, edge_index=reversed_edge)
    with pytest.raises(ValueError, match="reserved"):
        replace(sample, metadata={"label": 0})


def test_generated_faces_obey_the_boundary_law() -> None:
    sample = MixedStructuredSignal(6, seed=18, num_vertices=24)[2]
    incidence = build_oriented_incidence(
        sample.num_vertices,
        sample.edge_index.t().tolist(),
        sample.face_index.t().tolist(),
    )

    assert incidence.boundary_1.shape[1] == sample.num_edges
    assert incidence.boundary_2.shape[1] == sample.num_faces
    torch.testing.assert_close(
        incidence.boundary_1 @ incidence.boundary_2,
        torch.zeros(
            (sample.num_vertices, sample.num_faces), dtype=torch.float64
        ),
        rtol=0,
        atol=0,
    )


def test_collate_pads_structure_and_keeps_activity_distinct_from_validity() -> None:
    small = MixedStructuredSignal(6, seed=1, num_vertices=24)[0]
    large = MixedStructuredSignal(6, seed=2, num_vertices=31)[0]
    batch = collate_structured([small, large])

    assert len(batch) == 2
    assert batch.node_features.shape == (2, 31, small.node_features.shape[1])
    assert batch.edge_features.shape[0] == 2
    assert batch.edge_index.shape[:2] == (2, 2)
    assert batch.face_index.shape[:2] == (2, 3)
    assert batch.node_mask.sum(dim=1).tolist() == [small.num_vertices, large.num_vertices]
    assert batch.edge_mask.sum(dim=1).tolist() == [small.num_edges, large.num_edges]
    assert batch.face_mask.sum(dim=1).tolist() == [small.num_faces, large.num_faces]
    assert not torch.any(batch.face_active & ~batch.face_mask)

    assert torch.all(batch.edge_index[0, :, small.num_edges :] == -1)
    assert torch.all(batch.face_index[0, :, small.num_faces :] == -1)
    assert torch.count_nonzero(batch.node_features[0, small.num_vertices :]) == 0
    assert torch.count_nonzero(batch.edge_features[0, small.num_edges :]) == 0
    assert batch.labels.tolist() == [int(small.label), int(large.label)]
    assert batch.metadata[0] is small.metadata

    graph_inputs = batch.model_inputs("graph")
    cell_inputs = batch.model_inputs("cell")
    sheaf_inputs = batch.model_inputs("sheaf")
    assert "face_active" not in graph_inputs
    assert "transport" not in graph_inputs
    assert {"face_index", "face_mask", "face_active"}.issubset(cell_inputs)
    assert "transport" not in cell_inputs
    assert "transport" in sheaf_inputs
    assert "face_active" not in sheaf_inputs
    for inputs in (graph_inputs, cell_inputs, sheaf_inputs):
        assert "labels" not in inputs
        assert "regimes" not in inputs
        assert "metadata" not in inputs
    assert batch.observations_for("graph") is batch.observations
    assert batch.observations_for("cell") is batch.observations
    assert batch.observations_for("sheaf") is batch.observations


def test_device_transfer_preserves_discrete_dtypes() -> None:
    sample = MixedStructuredSignal(6, seed=3, num_vertices=24)[0]
    moved_sample = sample.to("cpu")
    batch = collate_structured([sample]).to("cpu")

    assert moved_sample.edge_index.dtype == torch.long
    assert moved_sample.face_index.dtype == torch.long
    assert moved_sample.face_active.dtype == torch.bool
    assert moved_sample.label.dtype == torch.long
    assert batch.edge_index.dtype == torch.long
    assert batch.face_index.dtype == torch.long
    assert batch.node_mask.dtype == torch.bool
    assert batch.labels.dtype == torch.long


def test_batch_rejects_inconsistent_counts_and_padding() -> None:
    small = MixedStructuredSignal(6, seed=31, num_vertices=24)[0]
    large = MixedStructuredSignal(6, seed=32, num_vertices=30)[0]
    batch = collate_structured([small, large])

    with pytest.raises(ValueError, match="num_vertices"):
        replace(batch, num_vertices=batch.num_vertices + 1)

    corrupted_edges = batch.edge_index.clone()
    corrupted_edges[0, 0, small.num_edges] = 0
    with pytest.raises(ValueError, match="-1 sentinel"):
        replace(batch, edge_index=corrupted_edges)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_sample_rejects_mixed_devices() -> None:
    sample = MixedStructuredSignal(6, seed=33, num_vertices=24)[0]
    with pytest.raises(ValueError, match="share the observation device"):
        replace(sample, label=sample.label.to("cuda"))


def test_sample_and_metadata_are_picklable_for_worker_processes() -> None:
    sample = MixedStructuredSignal(6, seed=21, num_vertices=24)[0]
    restored = pickle.loads(pickle.dumps(sample))

    assert restored.sample_id == sample.sample_id
    assert dict(restored.metadata) == dict(sample.metadata)
    assert torch.equal(restored.node_features, sample.node_features)
    with pytest.raises(TypeError):
        restored.metadata["new_key"] = "not mutable"  # type: ignore[index]


def test_empty_collation_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        collate_structured([])
