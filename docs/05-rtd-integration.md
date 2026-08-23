# RTD integration

## Decision

HOMYMOLY will use Representation Topology Divergence. RTD is a validated comparison tool, an auxiliary loss for paired metric views, and a baseline for the proposed map-aware cone loss. It is not renamed or presented as a HOMYMOLY contribution.

## Why RTD belongs in the project

RTD already captures three properties central to the original idea:

- it compares representations in different ambient spaces;
- it respects the one-to-one identity of underlying samples;
- its cross-barcode has an exact-sequence/mapping-cone interpretation.

Discarding RTD would remove the closest established operationalization of structural representation mismatch. The correct move is to reproduce it faithfully and identify exactly where HOMYMOLY goes beyond it.

## Input contract

For a paired minibatch

\[
X=(x_1,\dots,x_b),\qquad Y=(y_1,\dots,y_b),
\]

\(x_j\) and \(y_j\) must represent the same entity. Ambient dimensions may differ. RTD receives the two within-view dissimilarity matrices

\[
W^X_{ij}=d_X(x_i,x_j),\qquad W^Y_{ij}=d_Y(y_i,y_j),
\]

and does not require a cross-space distance \(d(x_i,y_j)\).

Each adapter must provide symmetric finite dissimilarities, zero diagonal, nonnegative entries, identical entity ordering, and a declared normalization. A joint permutation of both matrices must not change the score.

## Implemented reference convention (frozen 2026-08-13)

The exact, non-differentiable reference lives in
`src/homymoly/metrics/exact_rtd.py`. Its default convention is:

- normalize each within-view dissimilarity matrix independently by its 0.9
  quantile;
- compute that quantile over the **complete \(b \times b\) matrix**, including
  the zero diagonal and both copies of every symmetric off-diagonal entry, as
  in the published RTD implementation;
- report the published scalar RTD convention in homological degree 1;
- report SRTD in one explicitly named homological degree (degree 1 by default),
  or use the `*_by_degree` APIs when several degrees are part of a declared
  analysis; and
- construct one extra simplex/chain degree internally to determine deaths,
  then exclude truncation-frontier generators from the returned degrees.

`max_dim` is a construction and reporting bound. It is not an instruction to
sum persistence across all degrees. Likewise, max normalization remains an
explicit alternative for sensitivity analysis, not the default reference
convention.

The pre-audit Gate-3 corruption reports violated this convention: they used
max normalization and added scores from multiple degrees, including a
truncation frontier. Those historical “exact SRTD” scalars are invalid and
must not be compared with results from the corrected reference.

The differentiable signal in `src/homymoly/metrics/paired_topology.py` is a
separate **H0 RTD-style surrogate**. It is not a cross-barcode implementation
of published RTD or SRTD and must retain that qualified name in tables and
claims.

## Required variants

Initial evaluations should report:

- directional RTD in both directions;
- the original half-sum symmetrization;
- RTD-lite where appropriate;
- SRTD or SRTD-lite;
- Normalized Topological Similarity for comparisons across sample sizes or conditions.

Directional RTD remains valuable because HOMYMOLY cares about asymmetric source-to-target behavior. SRTD is preferable when a single symmetric optimization signal is desired.

## What is reused

- paired-object comparison across different ambient dimensions;
- independently normalized within-view distances;
- directional cross-barcodes and persistence-length aggregation;
- repeated paired subsampling with uncertainty estimates;
- synthetic identity, collapse, cluster, circle, and localized-feature tests;
- the RTD-AE principle that persistence pairings select filtration values through which gradients can flow.

The public RTD-AE implementation must not be copied into this repository until its licensing is explicitly verified. An independent implementation from the published algorithm is the safe default.

## What HOMYMOLY adds

### Representation adapters

Vector, graph, cell, and sheaf experts expose invariant within-batch dissimilarities. The adapter is part of the metric definition and must always be reported.

### Map-aware cone defects

For a declared transition \(F:C\to D\), HOMYMOLY first measures

\[
\sum_n\|d^D_nF_n-F_{n-1}d^C_n\|^2
\]

and then evaluates a statistic derived from \(H_*(\operatorname{Cone}F)\). This directly interrogates the learned map. RTD instead compares the topology and localization induced by two distance representations.

### Routing

RTD contributes to route quality but cannot select a route alone. A topology-preserving representation can be task-irrelevant, semantically destructive, or too expensive. Routing combines task utility, reconstruction, chain/cone defects, RTD, and measured compute.

### Spectral approximations

When exact persistence or discrete ranks are too costly or unstable during training, low-eigenvalue statistics of cone Hodge Laplacians can serve as differentiable proxies. Exact RTD/SRTD and homology calculations remain evaluation standards.

## Mandatory baselines

Representation comparisons:

- linear CKA;
- representational similarity based on distance matrices;
- triplet-ranking agreement;
- Wasserstein or bottleneck distance between ordinary persistence diagrams;
- Topological Autoencoder loss;
- RTD, SRTD, and NTS.

Architectural comparisons:

- vector/graph-only, cell-only, and sheaf-only experts;
- best fixed route;
- random route at matched compute;
- validation-selected oracle route as an upper bound;
- task plus reconstruction/cycle loss;
- chain residual without cone loss;
- cone loss without RTD;
- RTD without direct cone loss;
- the full HOMYMOLY objective.

## Validation tests

The RTD module is not accepted until:

- identical representations give zero up to numerical tolerance;
- rigid isometries and uniform rescaling give numerical zero after normalization;
- jointly permuting entities preserves the result;
- directional scores may disagree while their half-sum is symmetric;
- collapsing a target reproduces the theoretical barcode relationship;
- localized differences missed by ordinary barcode comparison are detected;
- differentiable scores agree with finite differences away from filtration ties;
- repeated subsampling is reproducible and reports uncertainty;
- runtime and peak memory are recorded.

## Claim boundary

Allowed:

> RTD measures discrepancies in multiscale topology and feature localization between paired finite metric representations.

Not allowed:

> RTD directly measures all information loss, exactness of an arbitrary learned transformation, or equivalence between arbitrary mathematical data structures.
