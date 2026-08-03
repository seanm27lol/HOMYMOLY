"""Deterministic oriented incidence matrices for finite 2-complexes.

The convention throughout this module is homological: ``boundary_1`` maps
oriented edges to vertices and ``boundary_2`` maps oriented faces to edges.
Edges are always oriented from the smaller vertex id to the larger one.  A
face is rotated to start at its smallest vertex and then assigned the
lexicographically smaller of its two possible orientations.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

Edge = tuple[int, int]
Face = tuple[int, ...]


def canonical_edge(u: int, v: int) -> Edge:
    """Return the deterministic orientation of a non-loop edge."""

    u, v = int(u), int(v)
    if u == v:
        raise ValueError(f"self-loops do not define 1-cells: ({u}, {v})")
    return (u, v) if u < v else (v, u)


def _rotate_to_minimum(vertices: tuple[int, ...]) -> tuple[int, ...]:
    minimum_index = min(range(len(vertices)), key=vertices.__getitem__)
    return vertices[minimum_index:] + vertices[:minimum_index]


def canonical_cycle(vertices: Sequence[int]) -> Face:
    """Canonicalize a simple cycle independently of start and direction.

    A repeated closing vertex is accepted, so ``(0, 1, 2, 0)`` and
    ``(0, 1, 2)`` describe the same face.
    """

    cycle = tuple(int(vertex) for vertex in vertices)
    if len(cycle) >= 2 and cycle[0] == cycle[-1]:
        cycle = cycle[:-1]
    if len(cycle) < 3:
        raise ValueError("a face boundary must contain at least three vertices")
    if len(set(cycle)) != len(cycle):
        raise ValueError(f"face boundaries must be simple cycles: {cycle}")

    forward = _rotate_to_minimum(cycle)
    reverse = _rotate_to_minimum(tuple(reversed(cycle)))
    return min(forward, reverse)


def _validate_vertices(num_vertices: int, cells: Iterable[Sequence[int]]) -> None:
    for cell in cells:
        for vertex in cell:
            if vertex < 0 or vertex >= num_vertices:
                raise ValueError(
                    f"vertex {vertex} is outside the declared range "
                    f"[0, {num_vertices})"
                )


def build_boundary_1(
    num_vertices: int,
    edges: Iterable[Sequence[int]],
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, tuple[Edge, ...]]:
    """Build ``B1: C_1 -> C_0`` and return its canonical edge ordering."""

    num_vertices = int(num_vertices)
    if num_vertices < 0:
        raise ValueError("num_vertices must be nonnegative")

    canonical_edges = tuple(sorted(canonical_edge(*edge) for edge in edges))
    if len(set(canonical_edges)) != len(canonical_edges):
        raise ValueError("parallel or duplicate edges are not supported in v0.1")
    _validate_vertices(num_vertices, canonical_edges)

    boundary = torch.zeros(
        (num_vertices, len(canonical_edges)), dtype=dtype, device=device
    )
    for edge_index, (tail, head) in enumerate(canonical_edges):
        boundary[tail, edge_index] = -1
        boundary[head, edge_index] = 1
    return boundary, canonical_edges


def build_boundary_2(
    edges: Iterable[Sequence[int]],
    faces: Iterable[Sequence[int]],
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, tuple[Face, ...]]:
    """Build ``B2: C_2 -> C_1`` for a supplied edge-row ordering.

    Each edge is canonically oriented, but the order of ``edges`` is
    preserved so the returned rows align with an existing ``B1``.
    Face columns are sorted after canonicalization for determinism.
    """

    canonical_edges = tuple(canonical_edge(*edge) for edge in edges)
    if len(set(canonical_edges)) != len(canonical_edges):
        raise ValueError("parallel or duplicate edges are not supported in v0.1")
    edge_to_index = {edge: index for index, edge in enumerate(canonical_edges)}

    canonical_faces = tuple(sorted(canonical_cycle(face) for face in faces))
    if len(set(canonical_faces)) != len(canonical_faces):
        raise ValueError("duplicate faces are not supported")

    boundary = torch.zeros(
        (len(canonical_edges), len(canonical_faces)), dtype=dtype, device=device
    )
    for face_index, face in enumerate(canonical_faces):
        for start, end in zip(face, face[1:] + face[:1]):
            edge = canonical_edge(start, end)
            try:
                edge_index = edge_to_index[edge]
            except KeyError as error:
                raise ValueError(
                    f"face {face} uses boundary edge {edge}, which is not present"
                ) from error
            boundary[edge_index, face_index] += 1 if (start, end) == edge else -1
    return boundary, canonical_faces


@dataclass(frozen=True)
class OrientedIncidence:
    """Canonical cells and their degree-one and degree-two boundaries."""

    num_vertices: int
    edges: tuple[Edge, ...]
    faces: tuple[Face, ...]
    boundary_1: torch.Tensor
    boundary_2: torch.Tensor

    @property
    def b1(self) -> torch.Tensor:
        return self.boundary_1

    @property
    def b2(self) -> torch.Tensor:
        return self.boundary_2


def build_oriented_incidence(
    num_vertices: int,
    edges: Iterable[Sequence[int]],
    faces: Iterable[Sequence[int]] = (),
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> OrientedIncidence:
    """Construct a deterministic oriented graph or finite 2-complex."""

    boundary_1, canonical_edges = build_boundary_1(
        num_vertices, edges, dtype=dtype, device=device
    )
    boundary_2, canonical_faces = build_boundary_2(
        canonical_edges, faces, dtype=dtype, device=device
    )
    _validate_vertices(int(num_vertices), canonical_faces)
    validate_boundary_squared_zero(boundary_1, boundary_2)
    return OrientedIncidence(
        num_vertices=int(num_vertices),
        edges=canonical_edges,
        faces=canonical_faces,
        boundary_1=boundary_1,
        boundary_2=boundary_2,
    )


def validate_boundary_squared_zero(
    boundary_1: torch.Tensor,
    boundary_2: torch.Tensor,
    *,
    atol: float = 1e-10,
    raise_on_error: bool = True,
) -> float:
    """Validate ``B1 @ B2 = 0`` and return the maximum absolute residual."""

    boundary_1 = torch.as_tensor(boundary_1)
    boundary_2 = torch.as_tensor(boundary_2)
    if boundary_1.ndim != 2 or boundary_2.ndim != 2:
        raise ValueError("boundary matrices must be two-dimensional")
    if boundary_1.shape[1] != boundary_2.shape[0]:
        raise ValueError(
            "incompatible boundary shapes: "
            f"B1={tuple(boundary_1.shape)}, B2={tuple(boundary_2.shape)}"
        )
    if not torch.isfinite(boundary_1).all() or not torch.isfinite(boundary_2).all():
        raise ValueError("boundary matrices must contain only finite values")

    residual_matrix = boundary_1 @ boundary_2
    residual = (
        float(residual_matrix.abs().max().item()) if residual_matrix.numel() else 0.0
    )
    if raise_on_error and residual > atol:
        raise ValueError(
            f"boundary law failed: max|B1 @ B2|={residual:.3e} > {atol:.3e}"
        )
    return residual
