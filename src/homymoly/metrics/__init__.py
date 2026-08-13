"""Gate-2 structural metrics with explicit claim boundaries.

Two families live here and must not be reported interchangeably.  The
differentiable paired-distance functions in ``paired_topology`` are exact for
the 0-dimensional connectivity filtration they compute and serve as practical
RTD/SRTD-style *training* surrogates; they are not the published cross-barcode
constructions (measured: directional ordering can disagree with the exact
reference).  The non-differentiable routines in ``exact_rtd`` are the exact
evaluation reference: directional R-Cross-Barcode semantics and the symmetric
union/intersection mapping-cone construction, computed over GF(2) in float64.
Small-sample MST persistence routines remain reference diagnostics.
"""

from .chain import (
    chain_complex_residual,
    chain_map_residual,
    chain_map_residual_matrix,
)
from .distances import (
    normalize_dissimilarity,
    pairwise_euclidean_distances,
    validate_dissimilarity_matrix,
)
from .exact_rtd import (
    ConeMode,
    CrossBarcodeInterval,
    cone_cross_barcode,
    exact_rtd,
    exact_rtd_by_degree,
    exact_rtd_directional,
    exact_rtd_directional_by_degree,
    exact_srtd,
    exact_srtd_by_degree,
    persistence_by_degree,
    total_persistence,
)
from .paired_topology import (
    PairedH0TopologySurrogates,
    directional_h0_rtd_surrogate,
    paired_h0_rtd_surrogates,
    single_linkage_connectivity,
    symmetric_h0_srtd_surrogate,
)
from .persistence import (
    h0_persistence_death_discrepancy,
    zero_dimensional_persistence_deaths,
)
from .routing import aggregate_route_diagnostics

__all__ = [
    "ConeMode",
    "CrossBarcodeInterval",
    "PairedH0TopologySurrogates",
    "aggregate_route_diagnostics",
    "chain_complex_residual",
    "chain_map_residual",
    "chain_map_residual_matrix",
    "cone_cross_barcode",
    "directional_h0_rtd_surrogate",
    "exact_rtd",
    "exact_rtd_by_degree",
    "exact_rtd_directional",
    "exact_rtd_directional_by_degree",
    "exact_srtd",
    "exact_srtd_by_degree",
    "h0_persistence_death_discrepancy",
    "normalize_dissimilarity",
    "paired_h0_rtd_surrogates",
    "pairwise_euclidean_distances",
    "persistence_by_degree",
    "single_linkage_connectivity",
    "symmetric_h0_srtd_surrogate",
    "total_persistence",
    "validate_dissimilarity_matrix",
    "zero_dimensional_persistence_deaths",
]
