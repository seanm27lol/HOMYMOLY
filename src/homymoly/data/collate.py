"""Padding-aware collation for structured samples."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from .types import StructuredBatch, StructuredObservations, StructuredSample


def collate_structured(samples: Sequence[StructuredSample]) -> StructuredBatch:
    """Pad structured samples without mixing masks, metadata, or supervision.

    ``face_mask`` marks real candidate-face slots, whereas ``face_active`` marks
    which of those candidates are actual 2-cells.  The distinction is retained
    in the batch so a padded face can never be interpreted as an inactive cell.
    """

    samples = tuple(samples)
    if not samples:
        raise ValueError("collate_structured requires at least one sample")
    if any(not isinstance(sample, StructuredSample) for sample in samples):
        raise TypeError("collate_structured accepts only StructuredSample values")

    first = samples[0]
    node_dim = int(first.node_features.shape[1])
    edge_dim = int(first.edge_features.shape[1])
    dtype = first.node_features.dtype
    edge_dtype = first.edge_features.dtype
    transport_dtype = first.transport.dtype
    device = first.node_features.device

    for sample in samples[1:]:
        if sample.node_features.shape[1] != node_dim:
            raise ValueError("all samples must use the same node feature dimension")
        if sample.edge_features.shape[1] != edge_dim:
            raise ValueError("all samples must use the same edge feature dimension")
        if sample.node_features.dtype != dtype or sample.edge_features.dtype != edge_dtype:
            raise TypeError("all samples must use matching observation dtypes")
        if sample.transport.dtype != transport_dtype:
            raise TypeError("all samples must use the same transport dtype")
        tensors = (
            sample.node_features,
            sample.edge_features,
            sample.edge_index,
            sample.face_index,
            sample.face_active,
            sample.transport,
            sample.label,
        )
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("all sample tensors must be on the same device")

    boundary_present = [sample.face_boundary is not None for sample in samples]
    if any(boundary_present) and not all(boundary_present):
        raise ValueError("boundary lists must be present on every sample or none")

    batch_size = len(samples)
    max_vertices = max(sample.num_vertices for sample in samples)
    max_edges = max(sample.num_edges for sample in samples)
    max_faces = max(sample.num_faces for sample in samples)
    max_length = (
        max(int(sample.face_boundary.shape[1]) for sample in samples)
        if all(boundary_present)
        else 0
    )

    node_features = torch.zeros(
        (batch_size, max_vertices, node_dim), dtype=dtype, device=device
    )
    edge_features = torch.zeros(
        (batch_size, max_edges, edge_dim), dtype=edge_dtype, device=device
    )
    node_mask = torch.zeros((batch_size, max_vertices), dtype=torch.bool, device=device)
    edge_mask = torch.zeros((batch_size, max_edges), dtype=torch.bool, device=device)
    face_mask = torch.zeros((batch_size, max_faces), dtype=torch.bool, device=device)
    face_active = torch.zeros((batch_size, max_faces), dtype=torch.bool, device=device)
    edge_index = torch.full(
        (batch_size, 2, max_edges), -1, dtype=torch.long, device=device
    )
    face_index = torch.full(
        (batch_size, 3, max_faces), -1, dtype=torch.long, device=device
    )
    transport = torch.zeros(
        (batch_size, max_edges, 2, 2), dtype=transport_dtype, device=device
    )

    for batch_index, sample in enumerate(samples):
        num_vertices = sample.num_vertices
        num_edges = sample.num_edges
        num_faces = sample.num_faces
        node_features[batch_index, :num_vertices] = sample.node_features
        edge_features[batch_index, :num_edges] = sample.edge_features
        edge_index[batch_index, :, :num_edges] = sample.edge_index
        face_index[batch_index, :, : sample.face_index.shape[1]] = sample.face_index
        transport[batch_index, :num_edges] = sample.transport
        node_mask[batch_index, :num_vertices] = True
        edge_mask[batch_index, :num_edges] = True
        face_mask[batch_index, :num_faces] = True
        face_active[batch_index, :num_faces] = sample.face_active

    face_boundary = None
    face_vertices = None
    if all(boundary_present):
        face_boundary = torch.zeros(
            (batch_size, max_faces, max_length, 2), dtype=torch.long, device=device
        )
        face_vertices = torch.full(
            (batch_size, max_faces, max_length), -1, dtype=torch.long, device=device
        )
        for batch_index, sample in enumerate(samples):
            length = int(sample.face_boundary.shape[1])
            face_boundary[batch_index, : sample.num_faces, :length] = sample.face_boundary
            face_vertices[batch_index, : sample.num_faces, :length] = sample.face_vertices

    return StructuredBatch(
        observations=StructuredObservations(
            node_features=node_features,
            edge_features=edge_features,
        ),
        node_mask=node_mask,
        edge_index=edge_index,
        edge_mask=edge_mask,
        face_index=face_index,
        face_mask=face_mask,
        face_active=face_active,
        transport=transport,
        labels=torch.stack([sample.label for sample in samples]),
        regimes=tuple(sample.regime for sample in samples),
        sample_ids=tuple(sample.sample_id for sample in samples),
        metadata=tuple(sample.metadata for sample in samples),
        num_vertices=torch.tensor(
            [sample.num_vertices for sample in samples], dtype=torch.long, device=device
        ),
        num_edges=torch.tensor(
            [sample.num_edges for sample in samples], dtype=torch.long, device=device
        ),
        num_faces=torch.tensor(
            [sample.num_faces for sample in samples], dtype=torch.long, device=device
        ),
        face_boundary=face_boundary,
        face_vertices=face_vertices,
    )


def make_structured_collate() -> Callable[[Sequence[StructuredSample]], StructuredBatch]:
    """Return a picklable DataLoader collate function."""

    return collate_structured


# A domain-specific spelling useful at call sites.
collate_mixed_structured = collate_structured


__all__ = [
    "collate_mixed_structured",
    "collate_structured",
    "make_structured_collate",
]
