"""Mapping cones and Hodge projections for finite real chain complexes."""

from __future__ import annotations

import torch

from .chain import ChainComplex, ChainMap


def mapping_cone(chain_map: ChainMap, *, atol: float = 1e-10) -> ChainComplex:
    """Construct the homological mapping cone of an exact chain map.

    The grading and sign convention are

    ``Cone(F)_n = D_n + C_(n-1)`` and

    ``d_cone = [[d_D, F], [0, -d_C]]``.

    The chain-map law is checked before construction; an approximate map does
    not define cone homology and is rejected.
    """

    chain_map.assert_valid(atol=atol)
    source, target = chain_map.source, chain_map.target
    top_degree = max(target.max_degree, source.max_degree + 1)
    dimensions = tuple(
        target.space_dim(degree) + source.space_dim(degree - 1)
        for degree in range(top_degree + 1)
    )

    boundaries: list[torch.Tensor] = []
    for degree in range(1, top_degree + 1):
        top_left = target.boundary(degree)
        top_right = chain_map.map(degree - 1)
        bottom_left = torch.zeros(
            (source.space_dim(degree - 2), target.space_dim(degree)),
            dtype=source.dtype,
            device=source.device,
        )
        bottom_right = -source.boundary(degree - 1)
        top = torch.cat((top_left, top_right), dim=1)
        bottom = torch.cat((bottom_left, bottom_right), dim=1)
        boundaries.append(torch.cat((top, bottom), dim=0))

    return ChainComplex(
        dimensions,
        boundaries,
        dtype=source.dtype,
        device=source.device,
        validate=True,
        atol=atol,
    )


def cone_betti_numbers(
    chain_map: ChainMap,
    *,
    map_atol: float = 1e-10,
    rank_rtol: float | None = None,
    rank_atol: float = 0.0,
) -> tuple[int, ...]:
    """Construct a mapping cone and compute its float64 Betti oracle."""

    return mapping_cone(chain_map, atol=map_atol).betti_numbers(
        rtol=rank_rtol, atol=rank_atol
    )


def hodge_projector(
    boundary_2: torch.Tensor,
    *,
    rtol: float | None = None,
) -> torch.Tensor:
    """Project edge chains orthogonally away from ``im(B2)``.

    Returns ``P1 = I - B2 B2^+``.  This is the Euclidean Hodge projector used
    by the canonical cell-to-graph chain map.
    """

    boundary_2 = torch.as_tensor(boundary_2)
    if boundary_2.ndim != 2:
        raise ValueError("B2 must be a two-dimensional matrix")
    if not boundary_2.is_floating_point():
        boundary_2 = boundary_2.to(torch.float64)
    if not torch.isfinite(boundary_2).all():
        raise ValueError("B2 must contain only finite entries")

    num_edges = boundary_2.shape[0]
    identity = torch.eye(
        num_edges, dtype=boundary_2.dtype, device=boundary_2.device
    )
    if boundary_2.shape[1] == 0:
        return identity
    pseudoinverse = (
        torch.linalg.pinv(boundary_2)
        if rtol is None
        else torch.linalg.pinv(boundary_2, rtol=rtol)
    )
    return identity - boundary_2 @ pseudoinverse
