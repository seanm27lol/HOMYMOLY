"""Degree-specific RTD/SRTD cross-barcodes via a filtered mapping cone.

This module is the repository's exact, non-differentiable reference for
paired-representation topology comparison.  It complements the differentiable
H0 surrogates in :mod:`homymoly.metrics.paired_topology`, which remain
training-time signals with qualified names; measured on identical inputs the
two families can disagree even in directional ordering, so they must not be
reported interchangeably.

Construction.  On the shared vertex set of two paired dissimilarity matrices,
every simplex ``sigma`` carries the Vietoris-Rips filtration values
``f_R(sigma)`` and ``f_S(sigma)`` (maximum edge weight).  The union complex
``U`` filters by ``min(f_R, f_S)``.  Three subcomplex choices yield the
reported variants:

* ``"forward"``/``"reverse"``: subcomplex filtered by ``f_R`` (resp. ``f_S``)
  — the directional R-Cross-Barcode semantics: a discrepancy is born when the
  union joins components the source has not joined, and dies when the source
  joins them;
* ``"srtd"``: subcomplex filtered by ``max(f_R, f_S)`` — the symmetric
  union/intersection construction whose chain complex is homotopy equivalent
  to the mapping cone of the intersection inclusion.

The cross-barcode is the ordinary persistent homology of the filtered mapping
cone ``Cone(sub -> U)`` computed over GF(2). Published RTD is normally the
degree-one score, while SRTD is defined per homological degree; scalar helpers
therefore select an explicit degree rather than adding unrelated degrees.
Routines run on CPU in float64 and are evaluation references, not losses.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import inf
from typing import Literal

import numpy as np
import torch

from .distances import MatrixLike, Normalization, normalize_dissimilarity

ConeMode = Literal["forward", "reverse", "srtd"]

_MAX_ENTITIES = 64


@dataclass(frozen=True, slots=True)
class CrossBarcodeInterval:
    """One cone cross-barcode interval; ``death`` is ``inf`` for essential classes."""

    degree: int
    birth: float
    death: float

    @property
    def length(self) -> float:
        return self.death - self.birth


def _to_float64(
    matrix: MatrixLike,
    normalization: Normalization,
    normalization_quantile: float,
    atol: float,
) -> np.ndarray:
    normalized = normalize_dissimilarity(
        matrix,
        mode=normalization,
        quantile=normalization_quantile,
        atol=atol,
    )
    return normalized.detach().to(device="cpu", dtype=torch.float64).numpy()


def _validate_degree_range(max_dim: int, degree: int | None = None) -> None:
    if isinstance(max_dim, bool) or not isinstance(max_dim, int) or max_dim < 0:
        raise ValueError("max_dim must be a nonnegative integer")
    if degree is None:
        return
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    if degree > max_dim:
        raise ValueError("degree must not exceed max_dim")


def _simplex_filtration_values(
    distances: np.ndarray, max_simplex_dim: int
) -> dict[tuple[int, ...], float]:
    """Vietoris-Rips filtration value of every simplex: maximum edge weight."""

    n = distances.shape[0]
    values: dict[tuple[int, ...], float] = {}
    for size in range(1, max_simplex_dim + 2):
        for sigma in itertools.combinations(range(n), size):
            if size == 1:
                values[sigma] = 0.0
            else:
                values[sigma] = max(
                    distances[i, j] for i, j in itertools.combinations(sigma, 2)
                )
    return values


def cone_cross_barcode(
    source_distances: MatrixLike,
    target_distances: MatrixLike,
    *,
    mode: ConeMode = "srtd",
    max_dim: int = 2,
    normalization: Normalization = "quantile",
    normalization_quantile: float = 0.9,
    atol: float = 1e-7,
) -> tuple[CrossBarcodeInterval, ...]:
    """Return the exact cone cross-barcode for one paired comparison.

    ``mode`` selects the subcomplex: ``"forward"`` uses the source filtration
    (directional, source-to-target), ``"reverse"`` the target filtration, and
    ``"srtd"`` the intersection (``max``) filtration for the symmetric
    construction. ``max_dim`` is the largest returned homological degree.
    One additional chain degree is constructed internally to determine deaths;
    those truncation-frontier generators are never returned. ``quantile``
    normalization with quantile 0.9 matches the published RTD/SRTD convention;
    ``none``, ``max``, and ``mean`` remain explicit alternatives.
    """

    if mode not in ("forward", "reverse", "srtd"):
        raise ValueError(f"unknown cone mode: {mode}")
    _validate_degree_range(max_dim)
    source = _to_float64(source_distances, normalization, normalization_quantile, atol)
    target = _to_float64(target_distances, normalization, normalization_quantile, atol)
    if source.shape != target.shape:
        raise ValueError(
            "paired representations must contain the same number of entities; "
            f"got {tuple(source.shape)} and {tuple(target.shape)}"
        )
    n = int(source.shape[0])
    if n > _MAX_ENTITIES:
        raise ValueError(
            f"exact cone cross-barcodes are bounded to {_MAX_ENTITIES} entities "
            f"(the simplex enumeration is exponential); got {n}"
        )
    if n < 2:
        return ()

    # Deaths of degree-max_dim classes need cofaces one dimension higher.
    f_source = _simplex_filtration_values(source, max_dim + 1)
    f_target = _simplex_filtration_values(target, max_dim + 1)

    generators: list[tuple[float, int, tuple[str, tuple[int, ...]]]] = []
    for sigma in f_source:
        union = min(f_source[sigma], f_target[sigma])
        if mode == "srtd":
            sub = max(f_source[sigma], f_target[sigma])
        elif mode == "forward":
            sub = f_source[sigma]
        else:
            sub = f_target[sigma]
        generators.append((union, len(sigma) - 1, ("U", sigma)))
        # The cone shifts the subcomplex part up one degree.
        generators.append((sub, len(sigma), ("I", sigma)))
    order = sorted(
        range(len(generators)),
        key=lambda i: (generators[i][0], generators[i][1]),
    )
    gens = [generators[i] for i in order]
    index_of = {g[2]: i for i, g in enumerate(gens)}

    columns: list[set[int]] = []
    for _, _, (part, sigma) in gens:
        column: set[int] = set()
        faces = [sigma[:i] + sigma[i + 1 :] for i in range(len(sigma))]
        if part == "U":
            for face in faces:
                face_index = index_of.get(("U", face))
                if face_index is not None:
                    column.add(face_index)
        else:
            # Cone differential on the subcomplex part: inclusion plus
            # boundary, d(I sigma) = U sigma + I d(sigma).  The inclusion
            # term applies to every simplex, vertices included; dropping it
            # for vertices silently leaves the union part unpaired.
            self_index = index_of.get(("U", sigma))
            if self_index is not None:
                column.add(self_index)
            for face in faces:
                face_index = index_of.get(("I", face))
                if face_index is not None:
                    column.add(face_index)
        columns.append(column)

    reduced = [set(column) for column in columns]
    low_of: dict[int, int] = {}
    paired_births: set[int] = set()
    intervals: list[CrossBarcodeInterval] = []
    for column_index in range(len(gens)):
        column = reduced[column_index]
        while column:
            low = max(column)
            if low not in low_of:
                low_of[low] = column_index
                break
            column ^= reduced[low_of[low]]
            reduced[column_index] = column
        if column:
            low = max(column)
            if gens[column_index][0] > gens[low][0]:
                intervals.append(
                    CrossBarcodeInterval(
                        degree=gens[low][1],
                        birth=gens[low][0],
                        death=gens[column_index][0],
                    )
                )
            paired_births.add(low)
    for index, generator in enumerate(gens):
        if not reduced[index] and index not in paired_births:
            intervals.append(
                CrossBarcodeInterval(degree=generator[1], birth=generator[0], death=inf)
            )
    return tuple(interval for interval in intervals if interval.degree <= max_dim)


def total_persistence(
    barcode: tuple[CrossBarcodeInterval, ...], *, degree: int | None = None
) -> float:
    """Sum finite interval lengths, optionally restricted to one degree."""

    return float(
        sum(
            interval.length
            for interval in barcode
            if interval.death != inf and (degree is None or interval.degree == degree)
        )
    )


def persistence_by_degree(
    barcode: tuple[CrossBarcodeInterval, ...], *, max_dim: int
) -> tuple[float, ...]:
    """Return finite total persistence indexed by homological degree."""

    _validate_degree_range(max_dim)
    return tuple(
        total_persistence(barcode, degree=degree) for degree in range(max_dim + 1)
    )


def exact_rtd_directional_by_degree(
    source_distances: MatrixLike,
    target_distances: MatrixLike,
    *,
    max_dim: int = 2,
    normalization: Normalization = "quantile",
    normalization_quantile: float = 0.9,
    atol: float = 1e-7,
) -> tuple[float, ...]:
    """Return directional RTD scores indexed by homological degree."""

    barcode = cone_cross_barcode(
        source_distances,
        target_distances,
        mode="forward",
        max_dim=max_dim,
        normalization=normalization,
        normalization_quantile=normalization_quantile,
        atol=atol,
    )
    return persistence_by_degree(barcode, max_dim=max_dim)


def exact_rtd_directional(
    source_distances: MatrixLike,
    target_distances: MatrixLike,
    *,
    degree: int = 1,
    max_dim: int = 2,
    normalization: Normalization = "quantile",
    normalization_quantile: float = 0.9,
    atol: float = 1e-7,
) -> float:
    """Return the directional RTD score in one homological degree.

    Degree one is the scalar used by the published RTD algorithm. ``max_dim``
    controls the highest degree constructed and must be at least ``degree``.
    """

    _validate_degree_range(max_dim, degree)
    return exact_rtd_directional_by_degree(
        source_distances,
        target_distances,
        max_dim=max_dim,
        normalization=normalization,
        normalization_quantile=normalization_quantile,
        atol=atol,
    )[degree]


def exact_rtd_by_degree(
    source_distances: MatrixLike,
    target_distances: MatrixLike,
    *,
    max_dim: int = 2,
    normalization: Normalization = "quantile",
    normalization_quantile: float = 0.9,
    atol: float = 1e-7,
) -> tuple[float, ...]:
    """Return half-symmetrized RTD scores indexed by homological degree."""

    forward = exact_rtd_directional_by_degree(
        source_distances,
        target_distances,
        max_dim=max_dim,
        normalization=normalization,
        normalization_quantile=normalization_quantile,
        atol=atol,
    )
    reverse = exact_rtd_directional_by_degree(
        target_distances,
        source_distances,
        max_dim=max_dim,
        normalization=normalization,
        normalization_quantile=normalization_quantile,
        atol=atol,
    )
    return tuple(
        0.5 * (left + right) for left, right in zip(forward, reverse, strict=True)
    )


def exact_rtd(
    source_distances: MatrixLike,
    target_distances: MatrixLike,
    *,
    degree: int = 1,
    max_dim: int = 2,
    normalization: Normalization = "quantile",
    normalization_quantile: float = 0.9,
    atol: float = 1e-7,
) -> float:
    """Return the half-symmetrized RTD score in one homological degree."""

    _validate_degree_range(max_dim, degree)
    return exact_rtd_by_degree(
        source_distances,
        target_distances,
        max_dim=max_dim,
        normalization=normalization,
        normalization_quantile=normalization_quantile,
        atol=atol,
    )[degree]


def exact_srtd_by_degree(
    source_distances: MatrixLike,
    target_distances: MatrixLike,
    *,
    max_dim: int = 2,
    normalization: Normalization = "quantile",
    normalization_quantile: float = 0.9,
    atol: float = 1e-7,
) -> tuple[float, ...]:
    """Return symmetric union/intersection cone scores by homological degree."""

    barcode = cone_cross_barcode(
        source_distances,
        target_distances,
        mode="srtd",
        max_dim=max_dim,
        normalization=normalization,
        normalization_quantile=normalization_quantile,
        atol=atol,
    )
    return persistence_by_degree(barcode, max_dim=max_dim)


def exact_srtd(
    source_distances: MatrixLike,
    target_distances: MatrixLike,
    *,
    degree: int = 1,
    max_dim: int = 2,
    normalization: Normalization = "quantile",
    normalization_quantile: float = 0.9,
    atol: float = 1e-7,
) -> float:
    """Return the SRTD score in one homological degree.

    The SRTD algorithm defines one score per degree. This scalar convenience
    function defaults to degree one; use :func:`exact_srtd_by_degree` when
    several degrees are part of the declared analysis.
    """

    _validate_degree_range(max_dim, degree)
    return exact_srtd_by_degree(
        source_distances,
        target_distances,
        max_dim=max_dim,
        normalization=normalization,
        normalization_quantile=normalization_quantile,
        atol=atol,
    )[degree]


__all__ = [
    "ConeMode",
    "CrossBarcodeInterval",
    "cone_cross_barcode",
    "exact_rtd",
    "exact_rtd_by_degree",
    "exact_rtd_directional",
    "exact_rtd_directional_by_degree",
    "exact_srtd",
    "exact_srtd_by_degree",
    "persistence_by_degree",
    "total_persistence",
]
