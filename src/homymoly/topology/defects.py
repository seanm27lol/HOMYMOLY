"""Directional exactness defects of a chain map.

A mapping cone answers one question with one number: is this map a
quasi-isomorphism? That bundles two different facts together. This module keeps
them apart:

- the **kernel** of the induced map on homology is what the conversion
  *destroys* — structure present in the source that does not survive;
- the **cokernel** is what the conversion cannot *reach* — structure present in
  the target that no source class maps onto.

The mapping cone's homology is exactly the sum of the two, degree by degree,

    dim H_n(cone F) = dim coker H_n(F) + dim ker H_(n-1)(F),

so the cone is strictly less informative than the pair. ``cone_betti_from_defects``
recomputes the cone from a defect profile, which makes that relationship a
runnable check rather than an assertion.

Homology classes are represented by harmonic cochains: under the Hodge
decomposition ``C_n = im d_(n+1) + Harm_n + im d_n^T``, every cycle is a boundary
plus a harmonic part, so orthogonal projection onto ``Harm_n`` reads off the
homology class.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from .chain import ChainComplex, ChainMap, numerical_rank


class DegreeDefect(NamedTuple):
    """Directional defect measurements at one degree."""

    degree: int
    chain_kernel: int
    chain_cokernel: int
    homology_kernel: int
    homology_cokernel: int
    source_betti: int
    target_betti: int

    @property
    def is_iso_on_homology(self) -> bool:
        return self.homology_kernel == 0 and self.homology_cokernel == 0


def harmonic_basis(
    complex_: ChainComplex,
    degree: int,
    *,
    rtol: float | None = None,
    atol: float = 0.0,
) -> torch.Tensor:
    """Return an orthonormal basis of the harmonic subspace at ``degree``.

    The columns span a canonical copy of ``H_degree`` inside the chain space.
    """

    dimension = complex_.space_dim(degree)
    betti = complex_.betti(degree, rtol=rtol, atol=atol)
    if dimension == 0 or betti == 0:
        return torch.zeros((dimension, 0), dtype=torch.float64)
    laplacian = complex_.hodge_laplacian(degree).to(torch.float64)
    # eigh returns ascending eigenvalues, so the harmonic space is the leading
    # block; its size is the Betti number under the repository rank convention.
    _, vectors = torch.linalg.eigh(laplacian)
    return vectors[:, :betti].contiguous()


def induced_homology_map(
    chain_map: ChainMap,
    degree: int,
    *,
    rtol: float | None = None,
    atol: float = 0.0,
) -> torch.Tensor:
    """Return the matrix of ``H_degree(F)`` in harmonic bases."""

    source = harmonic_basis(chain_map.source, degree, rtol=rtol, atol=atol)
    target = harmonic_basis(chain_map.target, degree, rtol=rtol, atol=atol)
    if source.shape[1] == 0 or target.shape[1] == 0:
        return torch.zeros((target.shape[1], source.shape[1]), dtype=torch.float64)
    return target.mT @ chain_map.map(degree).to(torch.float64) @ source


def degree_defect(
    chain_map: ChainMap,
    degree: int,
    *,
    rtol: float | None = None,
    atol: float = 0.0,
) -> DegreeDefect:
    """Measure what one degree of a chain map destroys and cannot reach."""

    matrix = chain_map.map(degree).to(torch.float64)
    chain_rank = numerical_rank(matrix, rtol=rtol, atol=atol)
    source_dim = chain_map.source.space_dim(degree)
    target_dim = chain_map.target.space_dim(degree)

    induced = induced_homology_map(chain_map, degree, rtol=rtol, atol=atol)
    source_betti = chain_map.source.betti(degree, rtol=rtol, atol=atol)
    target_betti = chain_map.target.betti(degree, rtol=rtol, atol=atol)
    # The induced map must be ranked against the scale of the chain map, not its
    # own largest singular value. A map that kills homology produces an induced
    # matrix that is uniformly at noise level, and a purely relative threshold
    # would rescale to that noise and always report full rank.
    induced_rank = 0
    if induced.numel():
        scale = float(torch.linalg.matrix_norm(matrix, 2)) if matrix.numel() else 0.0
        floor = max(induced.shape) * torch.finfo(torch.float64).eps * max(scale, 1.0)
        induced_rank = numerical_rank(induced, rtol=0.0, atol=max(atol, floor))
    return DegreeDefect(
        degree=degree,
        chain_kernel=source_dim - chain_rank,
        chain_cokernel=target_dim - chain_rank,
        homology_kernel=source_betti - induced_rank,
        homology_cokernel=target_betti - induced_rank,
        source_betti=source_betti,
        target_betti=target_betti,
    )


def exactness_defects(
    chain_map: ChainMap,
    *,
    rtol: float | None = None,
    atol: float = 0.0,
) -> tuple[DegreeDefect, ...]:
    """Measure destroyed and unreachable structure at every degree."""

    return tuple(
        degree_defect(chain_map, degree, rtol=rtol, atol=atol)
        for degree in range(chain_map.max_degree + 1)
    )


def cone_betti_from_defects(defects: tuple[DegreeDefect, ...]) -> tuple[int, ...]:
    """Predict the mapping cone's Betti numbers from a defect profile.

    Implements ``dim H_n(cone) = dim coker H_n(F) + dim ker H_(n-1)(F)``. Agreement
    with :func:`homymoly.topology.cone_betti_numbers` is the runnable statement
    that a cone bundles the two directional defects into a single number.
    """

    by_degree = {defect.degree: defect for defect in defects}
    highest = max(by_degree) if by_degree else -1
    predicted: list[int] = []
    for degree in range(highest + 2):
        cokernel = by_degree[degree].homology_cokernel if degree in by_degree else 0
        kernel = (
            by_degree[degree - 1].homology_kernel if degree - 1 in by_degree else 0
        )
        predicted.append(cokernel + kernel)
    return tuple(predicted)


def is_quasi_isomorphism(
    chain_map: ChainMap,
    *,
    rtol: float | None = None,
    atol: float = 0.0,
) -> bool:
    """Return whether the induced map is an isomorphism at every degree."""

    return all(
        defect.is_iso_on_homology
        for defect in exactness_defects(chain_map, rtol=rtol, atol=atol)
    )
