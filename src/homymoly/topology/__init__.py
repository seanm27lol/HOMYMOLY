"""Topological primitives used by HOMYMOLY's typed representations."""

from .chain import (
    ChainComplex,
    ChainMap,
    graph_to_cell_inclusion,
    nullity,
    numerical_rank,
)
from .cone import cone_betti_numbers, hodge_projector, mapping_cone
from .incidence import (
    Edge,
    Face,
    OrientedIncidence,
    build_boundary_1,
    build_boundary_2,
    build_oriented_incidence,
    canonical_cycle,
    canonical_edge,
    validate_boundary_squared_zero,
)
from .sheaf import connection_coboundary, connection_residual, cycle_holonomy

__all__ = [
    "ChainComplex",
    "ChainMap",
    "Edge",
    "Face",
    "OrientedIncidence",
    "build_boundary_1",
    "build_boundary_2",
    "build_oriented_incidence",
    "canonical_cycle",
    "canonical_edge",
    "cone_betti_numbers",
    "connection_coboundary",
    "connection_residual",
    "cycle_holonomy",
    "graph_to_cell_inclusion",
    "hodge_projector",
    "mapping_cone",
    "nullity",
    "numerical_rank",
    "validate_boundary_squared_zero",
]
