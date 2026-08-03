"""Finite real chain complexes, chain maps, and float64 rank oracles."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

ArrayLike = np.ndarray | torch.Tensor


def _as_numpy_float64(matrix: ArrayLike) -> np.ndarray:
    if isinstance(matrix, torch.Tensor):
        array = matrix.detach().to(device="cpu", dtype=torch.float64).numpy()
    else:
        array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("rank oracles require finite entries")
    return array


def numerical_rank(
    matrix: ArrayLike,
    *,
    rtol: float | None = None,
    atol: float = 0.0,
) -> int:
    """Compute a reproducible matrix rank from a CPU float64 SVD.

    The default relative tolerance is ``max(m, n) * eps(float64)`` times the
    largest singular value.  Callers should report any non-default tolerance
    used in experiments.
    """

    array = _as_numpy_float64(matrix)
    if array.size == 0:
        return 0
    singular_values = np.linalg.svd(array, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        return 0
    if rtol is None:
        rtol = max(array.shape) * np.finfo(np.float64).eps
    if rtol < 0 or atol < 0:
        raise ValueError("rank tolerances must be nonnegative")
    threshold = max(float(atol), float(rtol) * float(singular_values[0]))
    return int(np.count_nonzero(singular_values > threshold))


def nullity(
    matrix: ArrayLike,
    *,
    rtol: float | None = None,
    atol: float = 0.0,
) -> int:
    """Return the dimension of the matrix kernel."""

    array = _as_numpy_float64(matrix)
    return int(array.shape[1] - numerical_rank(array, rtol=rtol, atol=atol))


class ChainComplex:
    """A finite homological chain complex in degrees ``0..N``.

    ``boundaries[n - 1]`` is ``d_n: C_n -> C_(n-1)`` and therefore has
    shape ``(dimensions[n - 1], dimensions[n])``.
    """

    def __init__(
        self,
        dimensions: Sequence[int],
        boundaries: Sequence[ArrayLike],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        validate: bool = True,
        atol: float = 1e-10,
    ) -> None:
        self.dimensions = tuple(int(dimension) for dimension in dimensions)
        if not self.dimensions:
            raise ValueError("a chain complex must declare at least degree zero")
        if any(dimension < 0 for dimension in self.dimensions):
            raise ValueError("chain-space dimensions must be nonnegative")
        if len(boundaries) != len(self.dimensions) - 1:
            raise ValueError(
                "expected one boundary per positive degree: "
                f"got {len(boundaries)} for dimensions {self.dimensions}"
            )

        if dtype is None:
            first_tensor = next(
                (value for value in boundaries if isinstance(value, torch.Tensor)), None
            )
            dtype = (
                first_tensor.dtype
                if first_tensor is not None and first_tensor.is_floating_point()
                else torch.float64
            )
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("chain complexes require a floating-point dtype")

        tensors: list[torch.Tensor] = []
        for degree, value in enumerate(boundaries, start=1):
            tensor = torch.as_tensor(value, dtype=dtype, device=device)
            expected = (self.dimensions[degree - 1], self.dimensions[degree])
            if tensor.ndim != 2 or tuple(tensor.shape) != expected:
                raise ValueError(
                    f"d_{degree} has shape {tuple(tensor.shape)}; expected {expected}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"d_{degree} contains non-finite entries")
            tensors.append(tensor)

        self.boundaries = tuple(tensors)
        self.dtype = dtype
        self.device = (
            self.boundaries[0].device
            if self.boundaries
            else torch.device(device or "cpu")
        )
        if validate:
            self.assert_valid(atol=atol)

    @property
    def max_degree(self) -> int:
        return len(self.dimensions) - 1

    def space_dim(self, degree: int) -> int:
        return self.dimensions[degree] if 0 <= degree <= self.max_degree else 0

    def boundary(self, degree: int) -> torch.Tensor:
        """Return ``d_degree``, including correctly shaped zero maps outside range."""

        if 1 <= degree <= self.max_degree:
            return self.boundaries[degree - 1]
        return torch.zeros(
            (self.space_dim(degree - 1), self.space_dim(degree)),
            dtype=self.dtype,
            device=self.device,
        )

    def chain_residuals(self) -> dict[int, torch.Tensor]:
        """Return ``d_(n-1) @ d_n`` for every degree where it is defined."""

        return {
            degree: self.boundary(degree - 1) @ self.boundary(degree)
            for degree in range(2, self.max_degree + 1)
        }

    def max_chain_residual(self) -> float:
        residuals = self.chain_residuals().values()
        return max(
            (
                float(residual.abs().max().item()) if residual.numel() else 0.0
                for residual in residuals
            ),
            default=0.0,
        )

    def assert_valid(self, *, atol: float = 1e-10, rtol: float = 0.0) -> None:
        for degree, residual in self.chain_residuals().items():
            scale = float(
                torch.linalg.matrix_norm(self.boundary(degree - 1)).item()
                * torch.linalg.matrix_norm(self.boundary(degree)).item()
            )
            tolerance = float(atol) + float(rtol) * scale
            norm = float(torch.linalg.matrix_norm(residual).item())
            if norm > tolerance:
                raise ValueError(
                    f"chain law failed at degree {degree}: "
                    f"||d_{degree - 1} d_{degree}||={norm:.3e} > {tolerance:.3e}"
                )

    def betti(
        self,
        degree: int,
        *,
        rtol: float | None = None,
        atol: float = 0.0,
    ) -> int:
        """Compute ``dim ker(d_n) - rank(d_(n+1))`` over the reals."""

        if degree < 0 or degree > self.max_degree:
            return 0
        self.assert_valid(atol=max(1e-10, atol))
        rank_out = numerical_rank(self.boundary(degree), rtol=rtol, atol=atol)
        rank_in = numerical_rank(self.boundary(degree + 1), rtol=rtol, atol=atol)
        value = self.space_dim(degree) - rank_out - rank_in
        if value < 0:
            raise RuntimeError(
                f"negative Betti estimate at degree {degree}; check rank tolerance"
            )
        return int(value)

    def betti_numbers(
        self,
        *,
        rtol: float | None = None,
        atol: float = 0.0,
    ) -> tuple[int, ...]:
        return tuple(
            self.betti(degree, rtol=rtol, atol=atol)
            for degree in range(self.max_degree + 1)
        )

    def hodge_laplacian(self, degree: int) -> torch.Tensor:
        """Return ``d_n^T d_n + d_(n+1) d_(n+1)^T``."""

        if degree < 0 or degree > self.max_degree:
            raise ValueError(f"degree {degree} is outside this complex")
        outgoing = self.boundary(degree)
        incoming = self.boundary(degree + 1)
        return outgoing.mT @ outgoing + incoming @ incoming.mT


class ChainMap:
    """A degree-preserving linear map between finite chain complexes."""

    def __init__(
        self,
        source: ChainComplex,
        target: ChainComplex,
        maps: Sequence[ArrayLike],
        *,
        validate: bool = True,
        atol: float = 1e-10,
    ) -> None:
        if source.dtype != target.dtype or source.device != target.device:
            raise ValueError("source and target must share dtype and device")
        self.source = source
        self.target = target
        self.max_degree = max(source.max_degree, target.max_degree)
        if len(maps) != self.max_degree + 1:
            raise ValueError(
                f"expected {self.max_degree + 1} degree maps, got {len(maps)}"
            )

        tensors: list[torch.Tensor] = []
        for degree, value in enumerate(maps):
            tensor = torch.as_tensor(
                value, dtype=source.dtype, device=source.device
            )
            expected = (target.space_dim(degree), source.space_dim(degree))
            if tensor.ndim != 2 or tuple(tensor.shape) != expected:
                raise ValueError(
                    f"F_{degree} has shape {tuple(tensor.shape)}; expected {expected}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"F_{degree} contains non-finite entries")
            tensors.append(tensor)
        self.maps = tuple(tensors)
        if validate:
            self.assert_valid(atol=atol)

    def map(self, degree: int) -> torch.Tensor:
        if 0 <= degree <= self.max_degree:
            return self.maps[degree]
        return torch.zeros(
            (self.target.space_dim(degree), self.source.space_dim(degree)),
            dtype=self.source.dtype,
            device=self.source.device,
        )

    def residuals(self) -> dict[int, torch.Tensor]:
        """Return ``d_D F_n - F_(n-1) d_C`` in every positive degree."""

        return {
            degree: self.target.boundary(degree) @ self.map(degree)
            - self.map(degree - 1) @ self.source.boundary(degree)
            for degree in range(1, self.max_degree + 1)
        }

    def max_residual(self) -> float:
        return max(
            (
                float(residual.abs().max().item()) if residual.numel() else 0.0
                for residual in self.residuals().values()
            ),
            default=0.0,
        )

    def assert_valid(self, *, atol: float = 1e-10, rtol: float = 0.0) -> None:
        self.source.assert_valid(atol=atol, rtol=rtol)
        self.target.assert_valid(atol=atol, rtol=rtol)
        for degree, residual in self.residuals().items():
            left = self.target.boundary(degree) @ self.map(degree)
            right = self.map(degree - 1) @ self.source.boundary(degree)
            scale = float(
                torch.linalg.matrix_norm(left).item()
                + torch.linalg.matrix_norm(right).item()
            )
            tolerance = float(atol) + float(rtol) * scale
            norm = float(torch.linalg.matrix_norm(residual).item())
            if norm > tolerance:
                raise ValueError(
                    f"chain-map law failed at degree {degree}: "
                    f"residual={norm:.3e} > {tolerance:.3e}"
                )

    @classmethod
    def identity(cls, complex_: ChainComplex) -> ChainMap:
        maps = [
            torch.eye(
                complex_.space_dim(degree),
                dtype=complex_.dtype,
                device=complex_.device,
            )
            for degree in range(complex_.max_degree + 1)
        ]
        return cls(complex_, complex_, maps)

    @classmethod
    def zero(cls, source: ChainComplex, target: ChainComplex) -> ChainMap:
        top_degree = max(source.max_degree, target.max_degree)
        maps = [
            torch.zeros(
                (target.space_dim(degree), source.space_dim(degree)),
                dtype=source.dtype,
                device=source.device,
            )
            for degree in range(top_degree + 1)
        ]
        return cls(source, target, maps)


def graph_to_cell_inclusion(
    graph: ChainComplex,
    cell: ChainComplex,
    *,
    atol: float = 1e-10,
) -> ChainMap:
    """Return the canonical inclusion of a graph as a cell 1-skeleton."""

    if graph.max_degree != 1 or cell.max_degree < 1:
        raise ValueError("expected a degree-1 graph and a degree-1-or-2 cell complex")
    if graph.dimensions[:2] != cell.dimensions[:2]:
        raise ValueError("graph and cell complex must share vertex and edge spaces")
    if not torch.allclose(
        graph.boundary(1), cell.boundary(1), atol=atol, rtol=0.0
    ):
        raise ValueError("graph and cell complex must have the same B1")

    top_degree = max(graph.max_degree, cell.max_degree)
    maps: list[torch.Tensor] = []
    for degree in range(top_degree + 1):
        if degree <= 1:
            maps.append(
                torch.eye(
                    graph.space_dim(degree),
                    dtype=graph.dtype,
                    device=graph.device,
                )
            )
        else:
            maps.append(
                torch.zeros(
                    (cell.space_dim(degree), graph.space_dim(degree)),
                    dtype=graph.dtype,
                    device=graph.device,
                )
            )
    return ChainMap(graph, cell, maps, atol=atol)
