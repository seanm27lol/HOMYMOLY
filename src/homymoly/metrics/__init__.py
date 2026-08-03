"""Gate-2 structural metrics with explicit claim boundaries.

The paired-distance functions in this package are differentiable, exact for the
0-dimensional connectivity filtration they compute, and useful as practical
RTD/SRTD-style surrogates.  They are *not* implementations of the complete
published RTD or SRTD cross-barcode constructions and must not be reported as
such.  Non-differentiable MST persistence routines are included as small-sample
reference diagnostics.
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
    "PairedH0TopologySurrogates",
    "aggregate_route_diagnostics",
    "chain_complex_residual",
    "chain_map_residual",
    "chain_map_residual_matrix",
    "directional_h0_rtd_surrogate",
    "h0_persistence_death_discrepancy",
    "normalize_dissimilarity",
    "paired_h0_rtd_surrogates",
    "pairwise_euclidean_distances",
    "single_linkage_connectivity",
    "symmetric_h0_srtd_surrogate",
    "validate_dissimilarity_matrix",
    "zero_dimensional_persistence_deaths",
]
