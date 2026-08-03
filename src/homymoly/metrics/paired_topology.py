"""Differentiable paired-distance topology surrogates in homological degree 0.

These functions compare the exact single-linkage connectivity thresholds of
two paired finite dissimilarity spaces. They preserve entity localization and
give useful directional and symmetric training signals. They do not construct
the cross-barcodes from the published RTD or SRTD algorithms, do not inspect
higher homological degrees, and must be reported as ``H0 RTD-style surrogate``
or equivalent qualified language.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .distances import MatrixLike, Normalization, normalize_dissimilarity

Reduction = Literal["mean", "sum"]


@dataclass(frozen=True, slots=True)
class PairedH0TopologySurrogates:
    """Directional scores and their half-sum for one paired comparison."""

    forward: Tensor
    reverse: Tensor
    symmetric: Tensor


def single_linkage_connectivity(
    distance_matrix: MatrixLike,
    *,
    atol: float = 1e-7,
) -> Tensor:
    """Return all-pairs H0 connectivity thresholds by min-max closure.

    Entry ``(i, j)`` is the smallest filtration value at which entities ``i``
    and ``j`` lie in the same connected component of the threshold graph. This
    is the minimax path distance and is computed by Floyd-Warshall over the
    ``(min, max)`` semiring. The result is exact for the supplied floating edge
    weights and differentiable almost everywhere with respect to them.
    """

    closure = normalize_dissimilarity(
        distance_matrix,
        mode="none",
        atol=atol,
    )
    for pivot in range(closure.shape[0]):
        through_pivot = torch.maximum(
            closure[:, pivot].unsqueeze(1),
            closure[pivot, :].unsqueeze(0),
        )
        closure = torch.minimum(closure, through_pivot)
    return (closure + closure.mT) * 0.5


def _paired_connectivity_difference(
    source_distances: MatrixLike,
    target_distances: MatrixLike,
    *,
    normalization: Normalization,
    atol: float,
) -> tuple[Tensor, Tensor, Tensor]:
    source = normalize_dissimilarity(
        source_distances,
        mode=normalization,
        atol=atol,
    )
    target = normalize_dissimilarity(
        target_distances,
        mode=normalization,
        atol=atol,
    )
    if source.shape != target.shape:
        raise ValueError(
            "paired representations must contain the same number of entities; "
            f"got {tuple(source.shape)} and {tuple(target.shape)}"
        )
    if source.device != target.device:
        raise ValueError("paired distance matrices must be on the same device")
    dtype = torch.promote_types(source.dtype, target.dtype)
    source = source.to(dtype=dtype)
    target = target.to(dtype=dtype)
    source_connectivity = single_linkage_connectivity(source, atol=atol)
    target_connectivity = single_linkage_connectivity(target, atol=atol)
    difference = target_connectivity - source_connectivity
    indices = torch.triu_indices(
        source.shape[0],
        source.shape[1],
        offset=1,
        device=source.device,
    )
    return source, target, difference[indices[0], indices[1]]


def _reduce(
    values: Tensor, source: Tensor, target: Tensor, reduction: Reduction
) -> Tensor:
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be one of: mean, sum")
    if values.numel() == 0:
        # Retain a zero autograd connection to both potentially learned views.
        return (source.sum() + target.sum()) * 0.0
    return values.mean() if reduction == "mean" else values.sum()


def paired_h0_rtd_surrogates(
    source_distances: MatrixLike,
    target_distances: MatrixLike,
    *,
    normalization: Normalization = "max",
    reduction: Reduction = "mean",
    atol: float = 1e-7,
) -> PairedH0TopologySurrogates:
    """Compute forward, reverse, and symmetric H0 RTD-style surrogates.

    Forward measures connectivity that occurs later in the target than in the
    source. Reverse exchanges the roles. Symmetric is their arithmetic
    half-sum. These are paired-localization diagnostics, not full RTD/SRTD.
    """

    source, target, difference = _paired_connectivity_difference(
        source_distances,
        target_distances,
        normalization=normalization,
        atol=atol,
    )
    forward = _reduce(torch.relu(difference), source, target, reduction)
    reverse = _reduce(torch.relu(-difference), source, target, reduction)
    return PairedH0TopologySurrogates(
        forward=forward,
        reverse=reverse,
        symmetric=0.5 * (forward + reverse),
    )


def directional_h0_rtd_surrogate(
    source_distances: MatrixLike,
    target_distances: MatrixLike,
    *,
    normalization: Normalization = "max",
    reduction: Reduction = "mean",
    atol: float = 1e-7,
) -> Tensor:
    """Return the qualified source-to-target H0 RTD-style surrogate."""

    return paired_h0_rtd_surrogates(
        source_distances,
        target_distances,
        normalization=normalization,
        reduction=reduction,
        atol=atol,
    ).forward


def symmetric_h0_srtd_surrogate(
    first_distances: MatrixLike,
    second_distances: MatrixLike,
    *,
    normalization: Normalization = "max",
    reduction: Reduction = "mean",
    atol: float = 1e-7,
) -> Tensor:
    """Return the half-sum H0 SRTD-style surrogate.

    This function is symmetric by construction, but it is not the published
    SRTD union/intersection mapping-cone construction.
    """

    return paired_h0_rtd_surrogates(
        first_distances,
        second_distances,
        normalization=normalization,
        reduction=reduction,
        atol=atol,
    ).symmetric


__all__ = [
    "PairedH0TopologySurrogates",
    "directional_h0_rtd_surrogate",
    "paired_h0_rtd_surrogates",
    "single_linkage_connectivity",
    "symmetric_h0_srtd_surrogate",
]
