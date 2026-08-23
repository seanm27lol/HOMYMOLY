"""Learnable maps constrained to be exact finite chain maps.

The Gate-2 translators operate on padded neural features.  This module is the
smaller algebraic core promised by the mathematical contract: degree maps are
parameterized in the nullspace of the chain-map equations, so every forward
pass satisfies ``d_D F = F d_C`` up to floating-point roundoff.  Task or paired
signal losses are still required because the zero chain map is always legal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn


def _validated_boundary(name: str, value: Tensor) -> Tensor:
    boundary = torch.as_tensor(value)
    if boundary.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not boundary.is_floating_point():
        boundary = boundary.to(torch.float64)
    if not torch.isfinite(boundary).all():
        raise ValueError(f"{name} must contain only finite values")
    return boundary


def _chain_constraint_matrix(
    source_boundary: Tensor, target_boundary: Tensor
) -> Tensor:
    """Return ``A`` such that ``A @ [vec(F0), vec(F1)] == 0``."""

    source = _validated_boundary("source_boundary", source_boundary).to(torch.float64)
    target = _validated_boundary("target_boundary", target_boundary).to(torch.float64)
    source_vertices, source_edges = source.shape
    target_vertices, target_edges = target.shape
    parameter_count = target_vertices * source_vertices + target_edges * source_edges
    columns: list[Tensor] = []
    for parameter in range(parameter_count):
        f0 = torch.zeros(
            (target_vertices, source_vertices),
            dtype=torch.float64,
            device=source.device,
        )
        f1 = torch.zeros(
            (target_edges, source_edges), dtype=torch.float64, device=source.device
        )
        if parameter < f0.numel():
            f0.reshape(-1)[parameter] = 1.0
        else:
            f1.reshape(-1)[parameter - f0.numel()] = 1.0
        columns.append((target @ f1 - f0 @ source).reshape(-1))
    return torch.stack(columns, dim=1)


def _nullspace(matrix: Tensor, *, rtol: float | None = None) -> Tensor:
    matrix = torch.as_tensor(matrix, dtype=torch.float64)
    if matrix.ndim != 2:
        raise ValueError("nullspace input must be a matrix")
    if matrix.shape[1] == 0:
        return matrix.new_zeros((0, 0))
    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=True)
    if singular_values.numel() == 0:
        rank = 0
    else:
        tolerance = (
            max(matrix.shape) * torch.finfo(matrix.dtype).eps
            if rtol is None
            else float(rtol)
        ) * float(singular_values[0])
        rank = int((singular_values > tolerance).sum())
    return vh[rank:].mT.contiguous()


@dataclass(frozen=True, slots=True)
class ChainMapMatrices:
    degree_zero: Tensor
    degree_one: Tensor


class ExactChainMapLayer(nn.Module):
    """A learnable map between two-term chain complexes.

    The source boundary has shape ``[C0, C1]`` and the target boundary has
    shape ``[D0, D1]``. Parameters are coordinates in the nullspace of the
    linear chain-map constraint. This makes exactness architectural rather
    than a soft penalty.
    """

    def __init__(
        self,
        source_boundary: Tensor,
        target_boundary: Tensor,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        source = _validated_boundary("source_boundary", source_boundary)
        target = _validated_boundary("target_boundary", target_boundary)
        if source.device != target.device:
            raise ValueError("source and target boundaries must share a device")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("chain-map parameters require a floating dtype")
        constraint = _chain_constraint_matrix(source, target)
        basis = _nullspace(constraint)
        if basis.shape[1] == 0:
            raise ValueError(
                "the chain-map constraint has only the zero-dimensional solution"
            )
        self.register_buffer("source_boundary", source.to(dtype=dtype))
        self.register_buffer("target_boundary", target.to(dtype=dtype))
        self.register_buffer("nullspace_basis", basis.to(dtype=dtype))
        self.coefficients = nn.Parameter(torch.zeros(basis.shape[1], dtype=dtype))

    @property
    def source_dimensions(self) -> tuple[int, int]:
        return (int(self.source_boundary.shape[0]), int(self.source_boundary.shape[1]))

    @property
    def target_dimensions(self) -> tuple[int, int]:
        return (int(self.target_boundary.shape[0]), int(self.target_boundary.shape[1]))

    def matrices(self) -> ChainMapMatrices:
        flat = self.nullspace_basis @ self.coefficients
        source_vertices, source_edges = self.source_dimensions
        target_vertices, target_edges = self.target_dimensions
        f0_count = target_vertices * source_vertices
        return ChainMapMatrices(
            degree_zero=flat[:f0_count].reshape(target_vertices, source_vertices),
            degree_one=flat[f0_count:].reshape(target_edges, source_edges),
        )

    def residual(self) -> Tensor:
        maps = self.matrices()
        return (
            self.target_boundary @ maps.degree_one
            - maps.degree_zero @ self.source_boundary
        )

    def forward(self, degree_zero: Tensor, degree_one: Tensor) -> tuple[Tensor, Tensor]:
        maps = self.matrices()
        if degree_zero.shape[-1] != self.source_dimensions[0]:
            raise ValueError("degree-zero signals have the wrong source dimension")
        if degree_one.shape[-1] != self.source_dimensions[1]:
            raise ValueError("degree-one signals have the wrong source dimension")
        degree_zero = degree_zero.to(
            device=maps.degree_zero.device, dtype=maps.degree_zero.dtype
        )
        degree_one = degree_one.to(
            device=maps.degree_one.device, dtype=maps.degree_one.dtype
        )
        return degree_zero @ maps.degree_zero.mT, degree_one @ maps.degree_one.mT


def mapping_cone_boundaries(
    source_boundary: Tensor,
    target_boundary: Tensor,
    maps: ChainMapMatrices,
) -> tuple[Tensor, Tensor]:
    """Return the two differentials of ``Cone(F)`` for a two-term complex."""

    source = torch.as_tensor(
        source_boundary,
        dtype=maps.degree_zero.dtype,
        device=maps.degree_zero.device,
    )
    target = torch.as_tensor(
        target_boundary,
        dtype=maps.degree_zero.dtype,
        device=maps.degree_zero.device,
    )
    d1 = torch.cat((target, maps.degree_zero), dim=1)
    d2 = torch.cat((maps.degree_one, -source), dim=0)
    return d1, d2


def cone_soft_betti(
    source_boundary: Tensor,
    target_boundary: Tensor,
    maps: ChainMapMatrices,
    *,
    temperature: float = 0.05,
) -> Tensor:
    """Differentiable soft nullity of the mapping-cone Hodge Laplacians.

    This is a spectral proxy, not an exact Betti number. It is appropriate as
    an acyclicity objective only when the declared conversion should be a
    quasi-isomorphism. Exact cone homology remains an evaluation oracle.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    d1, d2 = mapping_cone_boundaries(source_boundary, target_boundary, maps)
    laplacians = (
        d1 @ d1.mT,
        d1.mT @ d1 + d2 @ d2.mT,
        d2.mT @ d2,
    )
    soft_counts = [
        torch.exp(
            -torch.linalg.eigvalsh(laplacian.float()).clamp_min(0.0) / temperature
        ).sum()
        for laplacian in laplacians
    ]
    return torch.stack(soft_counts).sum()


def cycle_consistency_loss(
    forward: ChainMapMatrices,
    reverse: ChainMapMatrices,
) -> Tensor:
    """Squared unit/counit error for a pair of degree-wise linear maps."""

    if forward.degree_zero.shape[::-1] != reverse.degree_zero.shape:
        raise ValueError("degree-zero forward/reverse map shapes are incompatible")
    if forward.degree_one.shape[::-1] != reverse.degree_one.shape:
        raise ValueError("degree-one forward/reverse map shapes are incompatible")
    source_zero = torch.eye(
        forward.degree_zero.shape[1],
        dtype=forward.degree_zero.dtype,
        device=forward.degree_zero.device,
    )
    target_zero = torch.eye(
        forward.degree_zero.shape[0],
        dtype=forward.degree_zero.dtype,
        device=forward.degree_zero.device,
    )
    source_one = torch.eye(
        forward.degree_one.shape[1],
        dtype=forward.degree_one.dtype,
        device=forward.degree_one.device,
    )
    target_one = torch.eye(
        forward.degree_one.shape[0],
        dtype=forward.degree_one.dtype,
        device=forward.degree_one.device,
    )
    terms: Sequence[Tensor] = (
        reverse.degree_zero @ forward.degree_zero - source_zero,
        forward.degree_zero @ reverse.degree_zero - target_zero,
        reverse.degree_one @ forward.degree_one - source_one,
        forward.degree_one @ reverse.degree_one - target_one,
    )
    return torch.stack([term.square().mean() for term in terms]).mean()


__all__ = [
    "ChainMapMatrices",
    "ExactChainMapLayer",
    "cone_soft_betti",
    "cycle_consistency_loss",
    "mapping_cone_boundaries",
]
