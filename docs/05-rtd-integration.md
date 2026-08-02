# RTD integration

## Decision

HOLYMOLY will use Representation Topology Divergence. RTD is a validated comparison tool, an auxiliary loss for paired metric views, and a baseline for the proposed map-aware cone loss. It is not renamed or presented as a HOLYMOLY contribution.

## Why RTD belongs in the project

RTD already captures three properties central to the original idea:

- it compares representations in different ambient spaces;
- it respects the one-to-one identity of underlying samples;
- its cross-barcode has an exact-sequence/mapping-cone interpretation.

Discarding RTD would remove the closest established operationalization of structural representation mismatch. The correct move is to reproduce it faithfully and identify exactly where HOLYMOLY goes beyond it.

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

## Required variants

Initial evaluations should report:

- directional RTD in both directions;
- the original half-sum symmetrization;
- RTD-lite where appropriate;
- SRTD or SRTD-lite;
- Normalized Topological Similarity for comparisons across sample sizes or conditions.

Directional RTD remains valuable because HOLYMOLY cares about asymmetric source-to-target behavior. SRTD is preferable when a single symmetric optimization signal is desired.

## What is reused

- paired-object comparison across different ambient dimensions;
- independently normalized within-view distances;
- directional cross-barcodes and persistence-length aggregation;
- repeated paired subsampling with uncertainty estimates;
- synthetic identity, collapse, cluster, circle, and localized-feature tests;
- the RTD-AE principle that persistence pairings select filtration values through which gradients can flow.

The public RTD-AE implementation must not be copied into this repository until its licensing is explicitly verified. An independent implementation from the published algorithm is the safe default.

## What HOLYMOLY adds

### Representation adapters

Vector, graph, cell, and sheaf experts expose invariant within-batch dissimilarities. The adapter is part of the metric definition and must always be reported.

### Map-aware cone defects

For a declared transition \(F:C\to D\), HOLYMOLY first measures

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
- the full HOLYMOLY objective.

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
