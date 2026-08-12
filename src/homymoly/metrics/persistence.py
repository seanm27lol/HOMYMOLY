"""Small-sample exact H0 persistence references.

The routines here are non-differentiable evaluation diagnostics. They compute
ordinary zero-dimensional Vietoris-Rips merge times by Kruskal/union-find.
They are not RTD cross-barcodes and, unlike the paired surrogate, discard the
entity localization of persistence events.
"""

from __future__ import annotations

from math import inf, isfinite
from typing import Literal

import numpy as np

from .distances import MatrixLike, Normalization, normalize_dissimilarity

ReferenceReduction = Literal["mean", "sum"]


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def zero_dimensional_persistence_deaths(
    distance_matrix: MatrixLike,
    *,
    normalization: Normalization = "max",
    atol: float = 1e-7,
) -> np.ndarray:
    """Return the ``N-1`` finite H0 death times in nondecreasing order.

    The essential connected-component bar is omitted. Values are an exact MST
    reference for the supplied finite floating dissimilarities. Tensor inputs
    are detached and copied to CPU intentionally; this function is not a loss.
    """

    normalized = normalize_dissimilarity(
        distance_matrix,
        mode=normalization,
        atol=atol,
    )
    # NumPy has no native bfloat16 representation; float64 also makes this
    # evaluation oracle independent of the neural tensor precision.
    matrix = normalized.detach().to(device="cpu").double().numpy()
    size = int(matrix.shape[0])
    if size < 2:
        return np.empty((0,), dtype=np.float64)

    edges = sorted(
        (float(matrix[left, right]), left, right)
        for left in range(size)
        for right in range(left + 1, size)
    )
    components = _UnionFind(size)
    deaths: list[float] = []
    for weight, left, right in edges:
        if components.union(left, right):
            deaths.append(weight)
            if len(deaths) == size - 1:
                break
    return np.asarray(deaths, dtype=np.float64)


def h0_persistence_death_discrepancy(
    first_distances: MatrixLike,
    second_distances: MatrixLike,
    *,
    normalization: Normalization = "max",
    p: float = 1.0,
    reduction: ReferenceReduction = "mean",
    atol: float = 1e-7,
) -> float:
    """Compare sorted ordinary H0 death times for equal-size spaces.

    This reference can be zero when paired event localization differs, so it is
    deliberately reported separately from paired topology surrogates. It is not
    advertised as persistence-diagram Wasserstein distance or as RTD.
    """

    first = zero_dimensional_persistence_deaths(
        first_distances,
        normalization=normalization,
        atol=atol,
    )
    second = zero_dimensional_persistence_deaths(
        second_distances,
        normalization=normalization,
        atol=atol,
    )
    if first.shape != second.shape:
        raise ValueError("H0 death discrepancy requires equal entity counts")
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be one of: mean, sum")
    p = float(p)
    if p != inf and (not isfinite(p) or p < 1.0):
        raise ValueError("p must be at least 1 or positive infinity")
    if first.size == 0:
        return 0.0

    difference = np.abs(first - second)
    if p == inf:
        return float(difference.max())
    powered = difference**p
    aggregate = powered.mean() if reduction == "mean" else powered.sum()
    return float(aggregate ** (1.0 / p))


__all__ = [
    "h0_persistence_death_discrepancy",
    "zero_dimensional_persistence_deaths",
]
