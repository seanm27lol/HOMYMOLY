"""Equivalence tests for the padded boundary-edge face representation.

Triangles converted to boundary lists must produce identical structural
quantities as the ``face_index`` path, and longer cycles (the molecular
case) must satisfy the exact ``B1 @ B2 == 0`` cycle contract.
"""

from __future__ import annotations

import torch

from homymoly.data import MixedStructuredSignal, triangles_to_boundary_lists
from homymoly.data.boundary import cycles_to_boundary_lists
from homymoly.data.collate import collate_structured
from homymoly.models.ops import (
    face_boundary_coefficients,
    face_holonomy,
    face_vertex_mean,
    scatter_faces_to_nodes,
)


def _batch_with_lists():
    samples = []
    for index in range(4):
        sample = MixedStructuredSignal(24, seed=40 + index, num_vertices=24)[index]
        boundary, vertices = triangles_to_boundary_lists(sample.edge_index, sample.face_index)
        samples.append(
            type(sample)(
                observations=sample.observations,
                edge_index=sample.edge_index,
                face_index=sample.face_index,
                face_active=sample.face_active,
                transport=sample.transport,
                label=sample.label,
                regime=sample.regime,
                sample_id=sample.sample_id,
                metadata=sample.metadata,
                face_boundary=boundary,
                face_vertices=vertices,
            )
        )
    return collate_structured(samples)


def test_coefficients_match_endpoint_matching() -> None:
    batch = _batch_with_lists()
    face_valid = batch.face_mask
    from_lists = face_boundary_coefficients(
        batch.edge_index,
        batch.edge_mask,
        batch.face_index,
        face_valid,
        dtype=torch.float32,
        face_boundary=batch.face_boundary,
    )
    from_matching = face_boundary_coefficients(
        batch.edge_index,
        batch.edge_mask,
        batch.face_index,
        face_valid,
        dtype=torch.float32,
    )
    torch.testing.assert_close(from_lists, from_matching, rtol=0, atol=0)


def test_holonomy_and_vertex_paths_match() -> None:
    batch = _batch_with_lists()
    face_valid = batch.face_mask
    holonomy_lists = face_holonomy(
        batch.transport,
        batch.edge_index,
        batch.edge_mask,
        batch.face_index,
        face_valid,
        face_boundary=batch.face_boundary,
    )
    holonomy_matching = face_holonomy(
        batch.transport,
        batch.edge_index,
        batch.edge_mask,
        batch.face_index,
        face_valid,
    )
    torch.testing.assert_close(holonomy_lists, holonomy_matching, rtol=0, atol=0)

    node_hidden = torch.randn(len(batch), batch.node_mask.shape[1], 8)
    mean_lists = face_vertex_mean(
        node_hidden, batch.face_index, face_valid, face_vertices=batch.face_vertices
    )
    mean_matching = face_vertex_mean(node_hidden, batch.face_index, face_valid)
    torch.testing.assert_close(mean_lists, mean_matching, rtol=0, atol=0)

    face_hidden = torch.randn(len(batch), batch.face_mask.shape[1], 8)
    scatter_lists = scatter_faces_to_nodes(
        face_hidden,
        batch.face_index,
        face_valid,
        num_nodes=node_hidden.shape[1],
        face_vertices=batch.face_vertices,
    )
    scatter_matching = scatter_faces_to_nodes(
        face_hidden, batch.face_index, face_valid, num_nodes=node_hidden.shape[1]
    )
    # Identical semantics; only floating scatter ordering differs.
    torch.testing.assert_close(scatter_lists, scatter_matching, rtol=1e-5, atol=1e-6)


def test_longer_cycles_satisfy_the_exact_cycle_contract() -> None:
    # A square (4-cycle) plus a triangle: the boundary coefficient matrix
    # must satisfy B1 @ B2 == 0 exactly, and no 4-ring may appear as a face.
    # Edges: square 0-1-2-3-0 plus diagonal (0, 2).
    edge_index = torch.tensor([[0, 1, 2, 0, 0], [1, 2, 3, 3, 2]])
    cycles = [(0, 1, 2), (0, 1, 2, 3)]
    boundary, vertices = cycles_to_boundary_lists(edge_index, cycles)
    assert boundary.shape == (2, 4, 2)  # K = longest cycle length
    num_edges = edge_index.shape[1]
    b1 = torch.zeros((4, num_edges))
    for position, (u, v) in enumerate(edge_index.t().tolist()):
        b1[u, position] = -1.0
        b1[v, position] = 1.0
    b2 = torch.zeros((num_edges, 2))
    for face in range(2):
        for slot in range(4):
            coeff = int(boundary[face, slot, 1])
            if coeff:
                b2[int(boundary[face, slot, 0]), face] = coeff
    assert torch.all(b1 @ b2 == 0)
    assert int(vertices[1, 3]) == 3 and int(vertices[0, 3]) == -1


def test_batch_roundtrip_and_slicing_with_lists() -> None:
    from homymoly.models.system import _slice_batch

    batch = _batch_with_lists()
    assert batch.face_boundary.shape[2] == 3
    moved = batch.to("cpu")
    assert moved.face_boundary is not None and moved.face_vertices is not None
    indices = torch.tensor([1, 3])
    sliced = _slice_batch(batch, indices)
    assert sliced.face_boundary.shape[0] == 2
    assert torch.equal(sliced.face_boundary, batch.face_boundary[indices])
    inputs = batch.model_inputs("cell")
    assert "face_boundary" in inputs and "face_vertices" in inputs
