"""Paired-representation dissimilarity adapters and validation."""

from __future__ import annotations

from math import isfinite
from typing import Literal

import numpy as np
import torch
from torch import Tensor

Normalization = Literal["none", "max", "mean", "quantile"]
MatrixLike = Tensor | np.ndarray


def _validate_atol(atol: float) -> float:
    atol = float(atol)
    if not isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    return atol


def validate_dissimilarity_matrix(
    matrix: MatrixLike,
    *,
    atol: float = 1e-7,
) -> Tensor:
    """Validate and canonicalize a finite symmetric dissimilarity matrix.

    Integer inputs are converted to float64. Floating tensors retain their
    dtype, device, and autograd connection. Values within ``atol`` of the
    declared invariants are symmetrized, clamped nonnegative, and assigned an
    exact zero diagonal.
    """

    atol = _validate_atol(atol)
    value = torch.as_tensor(matrix)
    if value.is_complex():
        raise TypeError("dissimilarity matrices must be real")
    if not value.is_floating_point():
        value = value.to(torch.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(
            f"a dissimilarity matrix must be square; got shape {tuple(value.shape)}"
        )
    if not torch.isfinite(value).all():
        raise ValueError("dissimilarity matrices must contain only finite values")
    if not torch.allclose(value, value.mT, atol=atol, rtol=0.0):
        raise ValueError("dissimilarity matrices must be symmetric")
    if value.numel() and float(value.min().item()) < -atol:
        raise ValueError("dissimilarity matrices must be nonnegative")
    diagonal = torch.diagonal(value)
    if diagonal.numel() and float(diagonal.abs().max().item()) > atol:
        raise ValueError("dissimilarity matrices must have a zero diagonal")

    canonical = (value + value.mT) * 0.5
    canonical = canonical.clamp_min(0.0)
    if canonical.shape[0]:
        diagonal_mask = torch.eye(
            canonical.shape[0],
            dtype=torch.bool,
            device=canonical.device,
        )
        canonical = torch.where(
            diagonal_mask,
            torch.zeros((), dtype=canonical.dtype, device=canonical.device),
            canonical,
        )
    return canonical


def pairwise_euclidean_distances(points: MatrixLike) -> Tensor:
    """Return the Euclidean within-view distances for ``[entities, features]``.

    Different views may use different feature dimensions. Pairing is by row
    index; no cross-view distance is constructed.
    """

    value = torch.as_tensor(points)
    if value.is_complex():
        raise TypeError("point features must be real")
    if not value.is_floating_point():
        value = value.to(torch.float64)
    if value.ndim != 2:
        raise ValueError(f"points must have shape [N, D], got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError("point features must contain only finite values")
    if value.shape[0] == 0:
        return value.new_zeros((0, 0))
    distances = torch.cdist(value, value, p=2)
    # The arithmetic average removes insignificant backend asymmetry while
    # preserving gradients. The diagonal of cdist is already exactly zero.
    return (distances + distances.mT) * 0.5


def normalize_dissimilarity(
    matrix: MatrixLike,
    *,
    mode: Normalization = "max",
    quantile: float = 0.9,
    eps: float | None = None,
    atol: float = 1e-7,
) -> Tensor:
    """Normalize off-diagonal distances independently within one view.

    ``quantile`` divides by the requested all-matrix-entry quantile (0.9 by
    default), matching the distance rescaling used by the published RTD and
    SRTD algorithms. ``max`` and ``mean`` remain explicit alternatives. A
    collapsed view is left at zero rather than divided by zero. Normalization
    is differentiable almost everywhere, including with respect to a tensor
    scale statistic.
    """

    value = validate_dissimilarity_matrix(matrix, atol=atol)
    if mode not in {"none", "max", "mean", "quantile"}:
        raise ValueError("mode must be one of: none, max, mean, quantile")
    quantile = float(quantile)
    if not isfinite(quantile) or not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be finite and in (0, 1]")
    if mode == "none" or value.shape[0] < 2:
        return value

    if eps is None:
        eps = 16.0 * torch.finfo(value.dtype).eps
    eps = float(eps)
    if not isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    indices = torch.triu_indices(
        value.shape[0],
        value.shape[1],
        offset=1,
        device=value.device,
    )
    off_diagonal = value[indices[0], indices[1]]
    if mode == "max":
        scale = off_diagonal.max()
    elif mode == "mean":
        scale = off_diagonal.mean()
    else:
        # The primary RTD implementation applies np.quantile to the complete
        # N x N distance matrix, including the diagonal and both triangles.
        scale = torch.quantile(value.flatten(), quantile)
    return value / scale.clamp_min(eps)


__all__ = [
    "Normalization",
    "normalize_dissimilarity",
    "pairwise_euclidean_distances",
    "validate_dissimilarity_matrix",
]
