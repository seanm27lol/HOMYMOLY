"""Rank-r connection-sheaf operators for canonically oriented graphs."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _validate_connection(
    edge_index: torch.Tensor,
    transport: torch.Tensor,
    *,
    num_vertices: int | None,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    edge_index = torch.as_tensor(edge_index)
    transport = torch.as_tensor(transport)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must have dtype torch.long")
    if transport.ndim != 3 or transport.shape[1] != transport.shape[2]:
        raise ValueError("transport must have shape [E, rank, rank]")
    if transport.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_index and transport disagree on the number of edges")
    if not transport.is_floating_point() or not torch.isfinite(transport).all():
        raise ValueError("transport must be finite and floating point")
    if edge_index.device != transport.device:
        raise ValueError("edge_index and transport must share a device")
    if edge_index.numel() and torch.any(edge_index[0] >= edge_index[1]):
        raise ValueError("connection edges must use the canonical tail < head orientation")

    inferred_vertices = int(edge_index.max()) + 1 if edge_index.numel() else 0
    if num_vertices is None:
        num_vertices = inferred_vertices
    if not isinstance(num_vertices, int) or num_vertices < inferred_vertices:
        raise ValueError("num_vertices must contain every endpoint")
    if edge_index.numel() and int(edge_index.min()) < 0:
        raise ValueError("edge indices must be nonnegative")
    return edge_index, transport, num_vertices, int(transport.shape[1])


def connection_coboundary(
    edge_index: torch.Tensor,
    transport: torch.Tensor,
    *,
    num_vertices: int | None = None,
) -> torch.Tensor:
    """Build the degree-zero connection-sheaf coboundary.

    For each canonical edge ``tail < head``, ``transport[e]`` maps a vector in
    the tail stalk into the head frame. The oriented edge residual is

    ``x_head - transport[e] @ x_tail``.

    Equivalently, the restrictions into the edge stalk are
    ``rho_tail = transport[e]`` and ``rho_head = I``.
    """

    edge_index, transport, num_vertices, rank = _validate_connection(
        edge_index,
        transport,
        num_vertices=num_vertices,
    )
    num_edges = int(edge_index.shape[1])
    coboundary = torch.zeros(
        (num_edges * rank, num_vertices * rank),
        dtype=transport.dtype,
        device=transport.device,
    )
    identity = torch.eye(rank, dtype=transport.dtype, device=transport.device)
    for edge in range(num_edges):
        tail = int(edge_index[0, edge])
        head = int(edge_index[1, edge])
        rows = slice(edge * rank, (edge + 1) * rank)
        coboundary[rows, tail * rank : (tail + 1) * rank] = -transport[edge]
        coboundary[rows, head * rank : (head + 1) * rank] = identity
    return coboundary


def connection_residual(
    node_values: torch.Tensor,
    edge_index: torch.Tensor,
    transport: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ``x_head - T_(tail,head) x_tail`` on every edge."""

    edge_index, transport, num_vertices, rank = _validate_connection(
        edge_index,
        transport,
        num_vertices=int(node_values.shape[0]) if node_values.ndim == 2 else None,
    )
    if node_values.ndim != 2 or tuple(node_values.shape) != (num_vertices, rank):
        raise ValueError("node_values must have shape [V, rank]")
    if node_values.device != transport.device or node_values.dtype != transport.dtype:
        raise ValueError("node values and transport must share dtype and device")
    tails, heads = edge_index
    transported = torch.einsum("eij,ej->ei", transport, node_values[tails])
    return node_values[heads] - transported


def cycle_holonomy(
    cycle: Sequence[int],
    edge_index: torch.Tensor,
    transport: torch.Tensor,
) -> torch.Tensor:
    """Compose connection transports once around an oriented vertex cycle."""

    edge_index, transport, _, rank = _validate_connection(
        edge_index,
        transport,
        num_vertices=None,
    )
    vertices = tuple(int(vertex) for vertex in cycle)
    if len(vertices) >= 2 and vertices[0] == vertices[-1]:
        vertices = vertices[:-1]
    if len(vertices) < 3 or len(set(vertices)) != len(vertices):
        raise ValueError("cycle must contain at least three distinct vertices")

    edge_lookup = {
        (int(edge_index[0, index]), int(edge_index[1, index])): index
        for index in range(edge_index.shape[1])
    }
    holonomy = torch.eye(rank, dtype=transport.dtype, device=transport.device)
    for start, end in zip(vertices, vertices[1:] + vertices[:1], strict=True):
        edge = (min(start, end), max(start, end))
        if edge not in edge_lookup:
            raise ValueError(f"cycle uses missing edge {edge}")
        step = transport[edge_lookup[edge]]
        if start > end:
            step = torch.linalg.inv(step)
        holonomy = step @ holonomy
    return holonomy


__all__ = ["connection_coboundary", "connection_residual", "cycle_holonomy"]
