"""Converters for the padded oriented boundary-edge face representation.

``face_boundary[f, k] = (edge_position, coefficient)`` and
``face_vertices[f, k]`` are the sparse 2-cell contract required for
non-triangular faces (molecular rings), per the plan's migration note:
triangles are the length-three special case and convert exactly; nothing
longer may be encoded as a nonexistent triangle.
"""

from __future__ import annotations

import torch
from torch import Tensor


def triangles_to_boundary_lists(
    edge_index: Tensor,
    face_index: Tensor,
) -> tuple[Tensor, Tensor]:
    """Convert ascending triangles ``[3, F]`` to boundary lists.

    The coefficient convention matches ``face_boundary_coefficients`` and
    ``_edge_boundary``: for a face ``(a, b, c)`` with ``a < b < c``, the
    oriented boundary is ``+[ab] +[bc] -[ac]``.
    """

    if face_index.ndim != 2 or face_index.shape[0] != 3:
        raise ValueError("face_index must have shape [3, F]")
    num_faces = int(face_index.shape[1])
    edge_positions = {
        (int(u), int(v)): position
        for position, (u, v) in enumerate(edge_index.t().tolist())
    }
    face_boundary = torch.zeros((num_faces, 3, 2), dtype=torch.long)
    face_vertices = face_index.t().contiguous().clone()
    for face, (a, b, c) in enumerate(face_index.t().tolist()):
        entries = (((a, b), 1), ((b, c), 1), ((a, c), -1))
        for slot, (edge, coefficient) in enumerate(entries):
            face_boundary[face, slot, 0] = edge_positions[(int(edge[0]), int(edge[1]))]
            face_boundary[face, slot, 1] = coefficient
    return face_boundary, face_vertices


def cycles_to_boundary_lists(
    edge_index: Tensor,
    cycles: list[tuple[int, ...]],
    max_length: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Convert oriented simple cycles (molecular rings) to boundary lists.

    Each cycle is a vertex tuple in boundary order; consecutive pairs form
    the oriented boundary edges with coefficient +1 when the stored
    canonical edge ``(u, v)`` satisfies ``u < v`` matches the walk
    direction, -1 otherwise (the walk closes from last back to first).
    Coefficient-zero padding follows the real entries.
    """

    if max_length is None:
        max_length = max((len(cycle) for cycle in cycles), default=0)
    edge_positions = {
        (min(int(u), int(v)), max(int(u), int(v))): position
        for position, (u, v) in enumerate(edge_index.t().tolist())
    }
    face_boundary = torch.zeros((len(cycles), max_length, 2), dtype=torch.long)
    face_vertices = torch.full((len(cycles), max_length), -1, dtype=torch.long)
    for face, cycle in enumerate(cycles):
        if len(cycle) < 3:
            raise ValueError("a 2-cell cycle must contain at least three vertices")
        for slot in range(len(cycle)):
            u = int(cycle[slot])
            v = int(cycle[(slot + 1) % len(cycle)])
            canonical = (min(u, v), max(u, v))
            if canonical not in edge_positions:
                raise ValueError(f"cycle edge {canonical} is absent from edge_index")
            face_boundary[face, slot, 0] = edge_positions[canonical]
            face_boundary[face, slot, 1] = 1 if u < v else -1
            face_vertices[face, slot] = u
    return face_boundary, face_vertices


__all__ = ["cycles_to_boundary_lists", "triangles_to_boundary_lists"]
