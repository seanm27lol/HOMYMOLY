"""Executable validation gate for the Stage-1 mathematical foundation."""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch

from .config import DataConfig
from .data import MixedStructuredSignal, SignalRegime
from .topology import (
    ChainComplex,
    ChainMap,
    build_oriented_incidence,
    cone_betti_numbers,
    connection_coboundary,
    connection_residual,
    graph_to_cell_inclusion,
    mapping_cone,
)


def build_stage1_dataset(
    config: DataConfig,
) -> tuple[MixedStructuredSignal, dict[str, tuple[int, ...]]]:
    """Build one dataset and group-disjoint splits from the runtime contract."""

    dataset = MixedStructuredSignal(
        num_samples=config.num_samples,
        seed=config.seed,
        min_vertices=config.min_vertices,
        max_vertices=config.max_vertices,
        node_feature_dim=config.node_feature_dim,
        edge_feature_dim=config.edge_feature_dim,
    )
    splits = dataset.split_indices(
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    return dataset, splits


def _triangle_oracles() -> dict[str, list[int]]:
    """Return hand-checkable homology sentinels for the declared convention."""

    incidence = build_oriented_incidence(
        3,
        ((0, 1), (0, 2), (1, 2)),
        ((0, 1, 2),),
        dtype=torch.float64,
    )
    graph = ChainComplex((3, 3), (incidence.boundary_1,))
    filled = ChainComplex(
        (3, 3, 1),
        (incidence.boundary_1, incidence.boundary_2),
    )
    inclusion = graph_to_cell_inclusion(graph, filled)
    return {
        "triangle_graph_betti": list(graph.betti_numbers()),
        "filled_triangle_betti": list(filled.betti_numbers()),
        "identity_cone_betti": list(cone_betti_numbers(ChainMap.identity(graph))),
        "inclusion_cone_betti": list(cone_betti_numbers(inclusion)),
    }


def validate_foundation(
    *,
    num_samples: int = 6,
    seed: int = 20260802,
    num_vertices: int = 24,
    node_feature_dim: int = 4,
    edge_feature_dim: int = 2,
    atol: float = 1e-10,
) -> dict[str, Any]:
    """Validate generated structures against the Stage-1 chain-map contract.

    This is an exact-oracle smoke gate, not a training metric. Incidences and
    homological checks use float64 even when the generated observations use a
    lower-precision neural dtype.
    """

    if atol < 0:
        raise ValueError("atol must be nonnegative")
    dataset = MixedStructuredSignal(
        num_samples=num_samples,
        seed=seed,
        num_vertices=num_vertices,
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
    )

    max_boundary_residual = 0.0
    max_chain_map_residual = 0.0
    max_cone_chain_residual = 0.0
    max_sheaf_operator_residual = 0.0
    max_transport_orthogonality_residual = 0.0
    total_edges = 0
    total_candidate_faces = 0
    total_active_faces = 0

    for sample in dataset:
        edges = tuple(tuple(edge) for edge in sample.edge_index.t().tolist())
        candidate_faces = tuple(
            tuple(face) for face in sample.face_index.t().tolist()
        )
        active_faces = tuple(
            face
            for face, active in zip(
                candidate_faces,
                sample.face_active.tolist(),
                strict=True,
            )
            if active
        )

        candidates = build_oriented_incidence(
            sample.num_vertices,
            edges,
            candidate_faces,
            dtype=torch.float64,
        )
        active = build_oriented_incidence(
            sample.num_vertices,
            edges,
            active_faces,
            dtype=torch.float64,
        )
        if candidates.edges != active.edges:
            raise RuntimeError("candidate and active complexes disagree on edge ordering")

        graph = ChainComplex(
            (sample.num_vertices, sample.num_edges),
            (active.boundary_1,),
        )
        cell = ChainComplex(
            (sample.num_vertices, sample.num_edges, len(active_faces)),
            (active.boundary_1, active.boundary_2),
        )
        inclusion = graph_to_cell_inclusion(graph, cell, atol=atol)
        cone = mapping_cone(inclusion, atol=atol)

        transport = sample.transport.to(torch.float64)
        node_values = sample.node_features[:, -2:].to(torch.float64)
        coboundary = connection_coboundary(
            sample.edge_index,
            transport,
            num_vertices=sample.num_vertices,
        )
        residual = connection_residual(
            node_values,
            sample.edge_index,
            transport,
        )
        operator_residual = coboundary @ node_values.flatten() - residual.flatten()
        if operator_residual.numel():
            max_sheaf_operator_residual = max(
                max_sheaf_operator_residual,
                float(operator_residual.abs().max().item()),
            )
        identity = torch.eye(2, dtype=torch.float64).expand(sample.num_edges, 2, 2)
        orthogonality_residual = transport.mT @ transport - identity
        if orthogonality_residual.numel():
            max_transport_orthogonality_residual = max(
                max_transport_orthogonality_residual,
                float(orthogonality_residual.abs().max().item()),
            )

        candidate_residual = (
            candidates.boundary_1 @ candidates.boundary_2
        )
        active_residual = active.boundary_1 @ active.boundary_2
        for residual in (candidate_residual, active_residual):
            if residual.numel():
                max_boundary_residual = max(
                    max_boundary_residual,
                    float(residual.abs().max().item()),
                )
        max_chain_map_residual = max(
            max_chain_map_residual,
            inclusion.max_residual(),
        )
        max_cone_chain_residual = max(
            max_cone_chain_residual,
            cone.max_chain_residual(),
        )
        total_edges += sample.num_edges
        total_candidate_faces += sample.num_faces
        total_active_faces += len(active_faces)

    if max_boundary_residual > atol:
        raise RuntimeError(
            f"boundary-law residual {max_boundary_residual:.3e} exceeds {atol:.3e}"
        )
    if max_chain_map_residual > atol:
        raise RuntimeError(
            f"chain-map residual {max_chain_map_residual:.3e} exceeds {atol:.3e}"
        )
    if max_cone_chain_residual > atol:
        raise RuntimeError(
            f"cone chain residual {max_cone_chain_residual:.3e} exceeds {atol:.3e}"
        )
    if max_sheaf_operator_residual > atol:
        raise RuntimeError(
            "connection-sheaf operator residual "
            f"{max_sheaf_operator_residual:.3e} exceeds {atol:.3e}"
        )
    if max_transport_orthogonality_residual > 1e-5:
        raise RuntimeError(
            "transport orthogonality residual "
            f"{max_transport_orthogonality_residual:.3e} exceeds 1.000e-05"
        )

    regime_counts = Counter(regime.value for regime in dataset.regimes)
    label_counts = Counter(dataset.labels)
    return {
        "schema_version": 1,
        "status": "passed",
        "num_samples": len(dataset),
        "seed": seed,
        "num_vertices": num_vertices,
        "regime_counts": dict(sorted(regime_counts.items())),
        "label_counts": {str(key): value for key, value in sorted(label_counts.items())},
        "total_edges": total_edges,
        "total_candidate_faces": total_candidate_faces,
        "total_active_faces": total_active_faces,
        "max_boundary_residual": max_boundary_residual,
        "max_chain_map_residual": max_chain_map_residual,
        "max_cone_chain_residual": max_cone_chain_residual,
        "max_sheaf_operator_residual": max_sheaf_operator_residual,
        "max_transport_orthogonality_residual": (
            max_transport_orthogonality_residual
        ),
        "observation_routes": [regime.value for regime in SignalRegime],
        "oracles": _triangle_oracles(),
    }


__all__ = ["build_stage1_dataset", "validate_foundation"]
