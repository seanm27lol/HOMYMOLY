"""Pure-PyTorch masked operations used by all Gate-2 model routes."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _work_dtype(tensor: Tensor) -> torch.dtype:
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return tensor.dtype


def apply_mask(values: Tensor, mask: Tensor) -> Tensor:
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    zero = torch.zeros((), dtype=values.dtype, device=values.device)
    return torch.where(expanded, values, zero)


def masked_mean(values: Tensor, mask: Tensor, *, dim: int = 1) -> Tensor:
    """Masked mean with FP32 reduction for low-precision neural tensors."""

    if dim != 1:
        raise ValueError("Gate-2 masked_mean currently supports the padded axis dim=1")
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    dtype = _work_dtype(values)
    masked_values = torch.where(
        expanded,
        values.to(dtype),
        torch.zeros((), dtype=dtype, device=values.device),
    )
    numerator = masked_values.sum(dim=dim)
    denominator = expanded.to(dtype).sum(dim=dim).clamp_min(1.0)
    return (numerator / denominator).to(values.dtype)


def masked_max(values: Tensor, mask: Tensor, *, dim: int = 1) -> Tensor:
    """Masked max with FP32 computation for low-precision neural tensors."""

    if dim != 1:
        raise ValueError("Gate-2 masked_max currently supports the padded axis dim=1")
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    dtype = _work_dtype(values)
    filled = torch.where(
        expanded,
        values.to(dtype),
        torch.full((), -1.0e30, dtype=dtype, device=values.device),
    )
    result = filled.amax(dim=dim)
    # Fully masked samples are meaningless; report zero instead of the fill.
    any_valid = expanded.any(dim=dim)
    return torch.where(any_valid, result, torch.zeros_like(result)).to(values.dtype)


def masked_feature_energy(values: Tensor, mask: Tensor) -> Tensor:
    """Per-sample mean squared feature magnitude in FP32/FP64."""

    batch, items = values.shape[:2]
    flattened = values.reshape(batch, items, -1).to(_work_dtype(values))
    per_item = flattened.square().mean(dim=-1)
    per_item = torch.where(mask, per_item, torch.zeros_like(per_item))
    numerator = per_item.sum(dim=1)
    denominator = mask.sum(dim=1).to(per_item.dtype).clamp_min(1.0)
    return numerator / denominator


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    batch, items = prediction.shape[:2]
    difference = (prediction - target).reshape(batch, items, -1)
    per_item = difference.to(_work_dtype(difference)).square().mean(dim=-1)
    per_item = torch.where(mask, per_item, torch.zeros_like(per_item))
    numerator = per_item.sum()
    denominator = mask.sum().to(per_item.dtype).clamp_min(1.0)
    return numerator / denominator


def safe_gather_nodes(node_values: Tensor, indices: Tensor) -> Tensor:
    """Gather padded batched node values, safely clamping masked sentinels."""

    if indices.ndim != 2 or indices.shape[0] != node_values.shape[0]:
        raise ValueError("indices must have shape [B, K]")
    maximum = max(int(node_values.shape[1]) - 1, 0)
    safe = indices.clamp(min=0, max=maximum)
    batches = torch.arange(node_values.shape[0], device=node_values.device).unsqueeze(1)
    return node_values[batches, safe]


def scatter_mean_to_nodes(
    messages: Tensor,
    indices: Tensor,
    mask: Tensor,
    *,
    num_nodes: int,
) -> Tensor:
    """Masked scatter-mean from padded items to padded vertices."""

    safe = indices.clamp(min=0, max=max(num_nodes - 1, 0))
    source = apply_mask(messages, mask)
    result = messages.new_zeros((messages.shape[0], num_nodes, messages.shape[-1]))
    result.scatter_add_(1, safe.unsqueeze(-1).expand_as(source), source)
    counts = messages.new_zeros((messages.shape[0], num_nodes, 1))
    counts.scatter_add_(
        1,
        safe.unsqueeze(-1),
        mask.unsqueeze(-1).to(dtype=messages.dtype),
    )
    return result / counts.clamp_min(1.0)


def face_holonomy(
    transport: Tensor,
    edge_index: Tensor,
    edge_mask: Tensor,
    face_index: Tensor,
    face_valid: Tensor,
    *,
    face_boundary: Tensor | None = None,
) -> Tensor:
    """Per-face transport holonomy ``[B, F, 2, 2]`` for rank-2 connections.

    The connection matrices produced by the generators are planar rotations,
    so they commute and the oriented cycle product can be evaluated exactly as
    a product of unit complex numbers (``cos + i sin`` per transport, with
    inverse-orientation edges contributing the conjugate).  Padded faces and
    edges contribute identity.  The computation runs in FP32; callers should
    cast the result to the surrounding working dtype.
    """

    coefficients = face_boundary_coefficients(
        edge_index,
        edge_mask,
        face_index,
        face_valid,
        dtype=torch.float32,
        face_boundary=face_boundary,
    )
    planar = transport.to(torch.float32)
    unit = torch.complex(planar[..., 0, 0], planar[..., 1, 0])
    expanded = unit.unsqueeze(1).expand(coefficients.shape)
    identity = torch.ones_like(expanded)
    factor = torch.where(
        coefficients > 0,
        expanded,
        torch.where(coefficients < 0, expanded.conj(), identity),
    )
    holonomy = factor.prod(dim=2)
    return torch.stack(
        (
            torch.stack((holonomy.real, -holonomy.imag), dim=-1),
            torch.stack((holonomy.imag, holonomy.real), dim=-1),
        ),
        dim=-2,
    )


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.layers(values)


class GraphMessageLayer(nn.Module):
    """Undirected edge-conditioned message layer with explicit masks."""

    def __init__(self, hidden_dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.forward_message = MLP(
            3 * hidden_dim, hidden_dim, hidden_dim, dropout=dropout
        )
        self.reverse_message = MLP(
            3 * hidden_dim, hidden_dim, hidden_dim, dropout=dropout
        )
        self.node_update = MLP(2 * hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.edge_update = MLP(3 * hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_hidden: Tensor,
        edge_hidden: Tensor,
        edge_index: Tensor,
        node_mask: Tensor,
        edge_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        tails = edge_index[:, 0]
        heads = edge_index[:, 1]
        tail_hidden = safe_gather_nodes(node_hidden, tails)
        head_hidden = safe_gather_nodes(node_hidden, heads)

        to_head = self.forward_message(
            torch.cat((tail_hidden, head_hidden, edge_hidden), dim=-1)
        )
        to_tail = self.reverse_message(
            torch.cat((head_hidden, tail_hidden, edge_hidden), dim=-1)
        )
        aggregated = scatter_mean_to_nodes(
            to_head, heads, edge_mask, num_nodes=node_hidden.shape[1]
        ) + scatter_mean_to_nodes(
            to_tail, tails, edge_mask, num_nodes=node_hidden.shape[1]
        )
        node_delta = self.node_update(torch.cat((node_hidden, aggregated), dim=-1))
        next_nodes = apply_mask(self.node_norm(node_hidden + node_delta), node_mask)

        edge_delta = self.edge_update(
            torch.cat((tail_hidden, head_hidden, edge_hidden), dim=-1)
        )
        next_edges = apply_mask(self.edge_norm(edge_hidden + edge_delta), edge_mask)
        return next_nodes, next_edges


def face_boundary_coefficients(
    edge_index: Tensor,
    edge_mask: Tensor,
    face_index: Tensor,
    face_valid: Tensor,
    *,
    dtype: torch.dtype,
    face_boundary: Tensor | None = None,
) -> Tensor:
    """Return triangle-boundary coefficients ``[bc] - [ac] + [ab]``.

    When ``face_boundary`` is provided (the padded oriented boundary-edge
    representation ``[B, F, K, 2]`` of ``(edge_position, coefficient)``), the
    dense ``[B, F, E]`` matrix is assembled from the stored lists instead of
    endpoint matching — this is the path for non-triangular cells.
    """

    if face_boundary is not None:
        coefficients = torch.zeros(
            (*face_valid.shape, edge_mask.shape[1]),
            dtype=dtype,
            device=edge_mask.device,
        )
        valid_entries = face_boundary[..., 1] != 0
        positions = face_boundary[..., 0].clamp(min=0, max=max(edge_mask.shape[1] - 1, 0))
        contributions = torch.where(
            valid_entries,
            face_boundary[..., 1].to(dtype),
            torch.zeros((), dtype=dtype, device=edge_mask.device),
        )
        coefficients.scatter_add_(2, positions, contributions)
        valid = face_valid.unsqueeze(2) & edge_mask.unsqueeze(1)
        return coefficients * valid.to(dtype)

    edge_u = edge_index[:, 0].unsqueeze(1)
    edge_v = edge_index[:, 1].unsqueeze(1)
    a = face_index[:, 0].unsqueeze(2)
    b = face_index[:, 1].unsqueeze(2)
    c = face_index[:, 2].unsqueeze(2)

    ab = (edge_u == a) & (edge_v == b)
    ac = (edge_u == a) & (edge_v == c)
    bc = (edge_u == b) & (edge_v == c)
    coefficients = ab.to(dtype) - ac.to(dtype) + bc.to(dtype)
    valid = face_valid.unsqueeze(2) & edge_mask.unsqueeze(1)
    return coefficients * valid.to(dtype)


def face_boundary_aggregate(
    edge_hidden: Tensor,
    edge_index: Tensor,
    edge_mask: Tensor,
    face_index: Tensor,
    face_valid: Tensor,
    *,
    face_boundary: Tensor | None = None,
) -> Tensor:
    coefficients = face_boundary_coefficients(
        edge_index,
        edge_mask,
        face_index,
        face_valid,
        dtype=edge_hidden.dtype,
        face_boundary=face_boundary,
    )
    return torch.einsum("bfe,beh->bfh", coefficients, edge_hidden)


def face_vertex_mean(
    node_hidden: Tensor,
    face_index: Tensor,
    face_valid: Tensor,
    *,
    face_vertices: Tensor | None = None,
) -> Tensor:
    if face_vertices is not None:
        vertex_valid = face_vertices != -1
        batch_size = face_vertices.shape[0]
        gathered = safe_gather_nodes(
            node_hidden, face_vertices.clamp(min=0).reshape(batch_size, -1)
        ).reshape(*face_vertices.shape, node_hidden.shape[-1])
        batch, faces = face_vertices.shape[:2]
        pooled = masked_mean(
            gathered.reshape(batch * faces, face_vertices.shape[2], node_hidden.shape[-1]),
            vertex_valid.reshape(batch * faces, face_vertices.shape[2]),
            dim=1,
        ).reshape(batch, faces, node_hidden.shape[-1])
        return apply_mask(pooled, face_valid)
    gathered = [
        safe_gather_nodes(node_hidden, face_index[:, index]) for index in range(3)
    ]
    return apply_mask(torch.stack(gathered, dim=0).mean(dim=0), face_valid)


def scatter_faces_to_nodes(
    face_hidden: Tensor,
    face_index: Tensor,
    face_valid: Tensor,
    *,
    num_nodes: int,
    face_vertices: Tensor | None = None,
) -> Tensor:
    if face_vertices is not None:
        source = apply_mask(face_hidden, face_valid)
        per_vertex_source = source.unsqueeze(2).expand(
            *face_vertices.shape, face_hidden.shape[-1]
        )
        vertex_valid = face_vertices != -1
        result = face_hidden.new_zeros(
            (face_hidden.shape[0], num_nodes, face_hidden.shape[-1])
        )
        counts = face_hidden.new_zeros((face_hidden.shape[0], num_nodes, 1))
        indices = face_vertices.clamp(min=0, max=max(num_nodes - 1, 0))
        contributions = apply_mask(per_vertex_source, vertex_valid)
        flat_index = indices.reshape(indices.shape[0], -1, 1).expand(
            -1, -1, face_hidden.shape[-1]
        )
        result.scatter_add_(1, flat_index, contributions.reshape(contributions.shape[0], -1, contributions.shape[-1]))
        counts.scatter_add_(
            1,
            indices.reshape(indices.shape[0], -1, 1),
            vertex_valid.reshape(vertex_valid.shape[0], -1, 1).to(face_hidden.dtype),
        )
        return result / counts.clamp_min(1.0)
    result = face_hidden.new_zeros(
        (face_hidden.shape[0], num_nodes, face_hidden.shape[-1])
    )
    counts = face_hidden.new_zeros((face_hidden.shape[0], num_nodes, 1))
    source = apply_mask(face_hidden, face_valid)
    for vertex_position in range(3):
        indices = face_index[:, vertex_position].clamp(min=0, max=max(num_nodes - 1, 0))
        result.scatter_add_(1, indices.unsqueeze(-1).expand_as(source), source)
        counts.scatter_add_(
            1,
            indices.unsqueeze(-1),
            face_valid.unsqueeze(-1).to(face_hidden.dtype),
        )
    return result / counts.clamp_min(1.0)


__all__ = [
    "MLP",
    "GraphMessageLayer",
    "apply_mask",
    "face_boundary_aggregate",
    "face_holonomy",
    "face_vertex_mean",
    "masked_feature_energy",
    "masked_max",
    "masked_mean",
    "masked_mse",
    "safe_gather_nodes",
    "scatter_faces_to_nodes",
    "scatter_mean_to_nodes",
]
