"""Differentiable chain-map residual metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from torch import Tensor

MatrixLike = Tensor | np.ndarray
ResidualReduction = Literal["mean", "sum", "frobenius"]


def _matrix(name: str, value: MatrixLike) -> Tensor:
    tensor = torch.as_tensor(value)
    if tensor.is_complex():
        raise TypeError(f"{name} must be real")
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float64)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a matrix, got shape {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _common_dtype(matrices: Sequence[Tensor]) -> torch.dtype:
    dtype = matrices[0].dtype
    for matrix in matrices[1:]:
        dtype = torch.promote_types(dtype, matrix.dtype)
    return dtype


def chain_map_residual_matrix(
    source_boundary: MatrixLike,
    target_boundary: MatrixLike,
    map_degree: MatrixLike,
    map_previous_degree: MatrixLike,
) -> Tensor:
    """Return ``d_target F_n - F_(n-1) d_source`` at one degree."""

    source = _matrix("source_boundary", source_boundary)
    target = _matrix("target_boundary", target_boundary)
    current = _matrix("map_degree", map_degree)
    previous = _matrix("map_previous_degree", map_previous_degree)
    matrices = (source, target, current, previous)
    if len({matrix.device for matrix in matrices}) != 1:
        raise ValueError("chain-map matrices must share a device")
    dtype = _common_dtype(matrices)
    source, target, current, previous = (matrix.to(dtype=dtype) for matrix in matrices)

    source_previous, source_current = source.shape
    target_previous, target_current = target.shape
    if tuple(current.shape) != (target_current, source_current):
        raise ValueError(
            "map_degree has incompatible shape; expected "
            f"{(target_current, source_current)}, got {tuple(current.shape)}"
        )
    if tuple(previous.shape) != (target_previous, source_previous):
        raise ValueError(
            "map_previous_degree has incompatible shape; expected "
            f"{(target_previous, source_previous)}, got {tuple(previous.shape)}"
        )
    return target @ current - previous @ source


def _reduce_residual(residual: Tensor, reduction: ResidualReduction) -> Tensor:
    if reduction not in {"mean", "sum", "frobenius"}:
        raise ValueError("reduction must be one of: mean, sum, frobenius")
    squared = residual.square()
    if reduction == "sum":
        return squared.sum()
    if reduction == "frobenius":
        return torch.linalg.vector_norm(residual)
    if squared.numel() == 0:
        return residual.sum() * 0.0
    return squared.mean()


def chain_map_residual(
    source_boundary: MatrixLike,
    target_boundary: MatrixLike,
    map_degree: MatrixLike,
    map_previous_degree: MatrixLike,
    *,
    reduction: ResidualReduction = "mean",
) -> Tensor:
    """Return a scalar residual for one degree of a proposed chain map.

    ``mean`` and ``sum`` aggregate squared residual entries. ``frobenius``
    returns the unsquared Frobenius norm. This metric alone does not prevent a
    trivial zero map and must be paired with task or non-collapse objectives.
    """

    residual = chain_map_residual_matrix(
        source_boundary,
        target_boundary,
        map_degree,
        map_previous_degree,
    )
    return _reduce_residual(residual, reduction)


def chain_complex_residual(
    source_boundaries: Sequence[MatrixLike],
    target_boundaries: Sequence[MatrixLike],
    degree_maps: Sequence[MatrixLike],
    *,
    reduction: ResidualReduction = "mean",
) -> Tensor:
    """Aggregate chain-map residuals over all positive degrees.

    Boundary sequences use ``(d_1, ..., d_N)`` and degree maps use
    ``(F_0, ..., F_N)``. Source and target may have different dimensions at a
    degree, but both must be represented through the same maximum degree, using
    correctly shaped zero maps where needed.
    """

    if len(source_boundaries) != len(target_boundaries):
        raise ValueError("source and target must declare the same maximum degree")
    if len(degree_maps) != len(source_boundaries) + 1:
        raise ValueError("expected exactly one degree map more than boundaries")
    if not degree_maps:
        raise ValueError("at least the degree-zero map is required")

    maps = tuple(
        _matrix(f"degree_maps[{index}]", value)
        for index, value in enumerate(degree_maps)
    )
    if not source_boundaries:
        return maps[0].sum() * 0.0
    residuals = tuple(
        chain_map_residual_matrix(
            source_boundaries[index],
            target_boundaries[index],
            maps[index + 1],
            maps[index],
        )
        for index in range(len(source_boundaries))
    )
    if len({residual.device for residual in residuals}) != 1:
        raise ValueError("all chain degrees must share a device")
    dtype = _common_dtype(residuals)
    flattened = torch.cat(
        tuple(residual.to(dtype=dtype).reshape(-1) for residual in residuals)
    )
    return _reduce_residual(flattened, reduction)


__all__ = [
    "chain_complex_residual",
    "chain_map_residual",
    "chain_map_residual_matrix",
]
