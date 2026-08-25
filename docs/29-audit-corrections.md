# Post-campaign audit corrections

Status: **preparatory record; corrected numerical fields await the clean
canonical rerun**.

Date: 2026-08-24.

This record preserves every material correction found after the conversion
campaign. Frozen protocol 27 remains unchanged so its pre-run SHA-256 and
provenance remain auditable. Corrections are made in the runner, the corrected
machine result, the manuscript, and this record rather than by rewriting the
historical protocol.

## 1. Mathematical names

The frozen key `exact` was described too strongly. The executed objective

    mean((B1 Wᵀ)²)

penalizes violation of `B1 Wᵀ = 0`, the boundary-of-boundary or
cycle-subspace condition. It does **not** establish exactness
`im Wᵀ = ker B1`; `W = 0` is an immediate counterexample. The publication
therefore uses **boundary compatibility** or **compatibility defect** and retains
`exact` only as a quoted historical machine key.

The frozen keys `cone` and `rtd` also named surrogates:

- `exp(−2·σ_min(W))` is a **singular-value cone surrogate**, not
  mapping-cone homology;
- normalized pairwise-distance MSE is an **RTD-inspired distance surrogate**,
  not the published RTD or SRTD statistic.

Claims about these objectives are now scoped to the implemented surrogates.

The continuous task was also described too broadly. It learns
`W: R^E -> R^F` from edge signals to coordinates in a chosen cycle basis;
`W^T: R^F -> R^E` is a candidate degree-2 differential. This degree-changing
lifting is not a degree-preserving typed chain map or a conversion between
complexes. The annulus selection experiment remains the genuine typed-chain-map
setting.

The compatibility penalty does not directly consume `B2` or response labels,
but `B1` determines the target cycle subspace. With the deterministic generator
algorithm known, the chosen NetworkX basis `B2` is algorithmically recoverable
from the graph. The unpenalized baseline ignores `B1`, and the campaign lacks an
analytic cycle-basis, Hodge-projection, or nullspace oracle. “Not leaked
supervision” has therefore been replaced by the more accurate description
**strong input-derived structural side information**.

The RTD-inspired source-versus-lifted distance term is target-misaligned. It asks
a rank-`F` lifting to preserve full edge-space geometry, whereas the planted
`W = B2^T` intentionally discards cut-space components. Its no-detected-
improvement result is scoped to this surrogate and says nothing negative about
published RTD/SRTD.

## 2. Pearson implementation

The original C1 helper standardized each nine-point vector with PyTorch's sample
standard deviation and then took the mean product. With `n = 9`, that computes
`(n−1)/n = 8/9` of the conventional Pearson coefficient. The former point
estimate `+0.8540517955` and interval
`[+0.8308661882, +0.8772374027]` are withdrawn.

The corrected implementation uses the conventional centered dot product divided
by the product of centered Euclidean norms and includes a regression test against
an independent formula.

The clean canonical rerun gives mean Pearson `+0.9608082699`, unadjusted 95%
interval `[+0.9347244618, +0.9868920780]`, positive in 29/29 eligible seeds.

H4 is Pearson correlation across nine points of
`log10(max(||B1 W^T||_F, 1e-30))` versus
`log10(max(held-out MSE, 1e-300))` within each eligible seed. The Frobenius
defect reported here is not the elementwise mean-square compatibility term used
for training. H4 remains a prespecified secondary analysis with an unadjusted 95% interval; it
is not part of the H1–H3 multiplicity family. It is a deterministic
regularization-path association: within each topology, the nine fits reuse the
same data and initialization and differ in the common driver `lambda`. Defect
and held-out error therefore covary along one path, and the nine fits are not
independent observations. C1 does not establish independent predictive
information or off-path calibration.

## 3. Bonferroni critical value

The original implementation labeled a two-sided 98.333% Student-t interval but
used critical value `2.763262`, corresponding to 99% at 28 degrees of freedom.
The protocol-required two-sided 98.333% critical value is approximately
`2.546465`. All previously published adjusted intervals are withdrawn.

The clean canonical rerun gives Bonferroni-adjusted 98.33% intervals:

- boundary compatibility: `[−2.7491611963, −1.5110005341]`;
- singular-value cone surrogate: `[+0.1088999144, +0.2699881016]`;
- RTD-inspired distance surrogate: `[−0.0015726478, +0.0377992082]`.

The paired observations, means, standard deviations, unadjusted intervals, and
sign-test counts are unchanged. The governing qualitative decisions remain:
boundary compatibility improves, the singular-value surrogate harms, and the
RTD-inspired surrogate has no multiplicity-controlled conclusion.

## 4. Frozen-protocol implementation deviation

Protocol 27 specifies the squared Frobenius sum `‖B1 Wᵀ‖²`. The executed runner
used the elementwise mean `mean((B1 Wᵀ)²)`. Because the number of matrix entries
varies by topology, the two are not related by one campaign-wide scale factor.
The corrected result records the executed objective explicitly. The paper treats
H1–H3 as a prospectively specified primary analysis in a same-generator-family
replication with this deviation disclosed, not as pristine preregistration or
independent confirmation.

Protocol 27 records that the hypotheses, effect directions, endpoint,
training-set size, and weights were chosen after outcome-informed exploration on
the same generator family. Document 26 did not retain the exploratory seed
identities, so possible overlap with frozen seeds 20261001–20261030 cannot be
audited. Prospective locking prevents post-freeze changes but does not erase
this design history. An untouched disjoint generator family or seed block is
still needed for independent confirmation.

The runner optimizes a full `F×E` matrix under a finite-weight penalty. It does
not parameterize `W` inside a nullspace or reduce the parameter count to
`F×F`. Only the penalty's zero set has rows in `ker B1`; the manuscript now
describes off-cycle shrinkage rather than a hard constraint.

## 5. H5 decision-rule flaw

H5 divided the selected route's error by
`min(cell error, graph error)` for the same trial. Every ratio is therefore at
least one and every log-ratio observation is nonnegative, while the decision rule
required an interval below zero. Support was impossible by construction. H5 is
now labeled non-informative rather than a routing null.

The original H5 analysis also pseudoreplicated its unit: 28 evaluation rows are
two weights for each of only 14 topology clusters. The df=27 Student-t interval
treated the rows as independent, and the 25/28 selection count has the same
dependence problem. Both are withdrawn from inferential use. The corrected
canonical record reports the historical row-naive interval
`[−0.1113835925, +0.3816648417]` and the topology-clustered descriptive interval
`[−0.1230973860, +0.3933786352]`; neither is inferential, and H5 has no decision.

## 6. Annulus interpretation

Exact cone acyclicity is constant on the twelve **hard decoded** dihedral maps.
That certificate cannot choose among those vertices. It does not follow that the
differentiable `cone_soft_betti` training objective is constant: convex mixtures
can be non-invertible and the proxy varies.

The annulus RTD-style loss is also not constant. It compares cross-example
pairwise distances of predicted and target typed representations, selectors vary
by example, and soft mixtures are non-orthogonal. It consumes target batch
structure and is not an unsupervised control. The empirical chance accuracies for
`cone_only` and `rtd_only`, and the 6,000/6,000 acyclic hard decoded cones,
remain valid; the former constancy explanation is withdrawn.

All six objectives with task or reconstruction supervision have decoded
transformation and cell-face accuracy 1.000 on all five seeds, but their mean map
MSE is not uniformly `1e-16`: it spans `2.618e-17` to `2.504e-8`. Mean map MSE
for `cone_only` and `rtd_only` is 0.109 and 0.191, respectively, about 7–16
orders above the supervised range depending on the comparison. The 10/10
engineering recovery gate applies only to `task_reconstruction` and `combined`,
five seeds each.

## 7. Evidence and reproducibility boundaries

The reviewer snapshot verifies and rehashes compact evidence, reruns the CPU
edge-to-cycle lifting campaign, runs tests, and regenerates figures and the paper. It
excludes the 8.8 GB raw artifact tree. Full GB10 annulus or trained-checkpoint
reruns require separately supplied raw artifacts/checkpoints and compatible GPU
hardware.

Exploratory ridge, matched-rank random-subspace, and nonlinear-link numbers were
removed from the journal manuscript because no generating script or
machine-readable publication result is retained. Specificity against generic
regularization remains open.

The learned edge-to-cycle lifting parameter count is corrected to median `E×F = 242`
(range 30–770) over 29 eligible topologies. Untracked claims about 300 samples
and 32 defect profiles were removed.

Each topology fits a separate `W`; the campaign is within-topology prediction,
not generalization of a shared converter to unseen topologies. NetworkX
`cycle_basis` supplies noncanonical target coordinates, so the compatibility
penalty identifies a cycle subspace while paired labels determine basis
coordinates. The paper now states both boundaries.

The primary statistical unit is one eligible generator seed, which jointly
instantiates topology, inputs, and training noise (`n = 29`). The estimand is the
mean paired log ratio over eligible seeds; Student-t inference assumes those
seed-level effects are exchangeable. Pairing controls the shared realization but
does not remove topology variance. Training responses add Gaussian noise with
sigma 0.02, while held-out targets are noiseless, so held-out MSE measures
recovery of the planted lifting.

The campaign's large effect is interpreted in its system-identification
geometry. With `N_train = 16`, 21/29 eligible seeds have `E > 16`, 24/29 have
cycle dimension `F <= 16`, and 16/29 move from the former underdetermined
ambient system to the latter potentially identifiable hard-subspace system.
Median dimensions are `E = 23`, `F = 11`; five seeds have `F > 16`. The
executed finite penalty does not hard-reduce the `F x E` parameter count, but it
shrinks toward this lower-dimensional subspace.

Weights were selected after exploratory work and frozen before the campaign, but
the full candidate sweep and exploratory seed identities are not retained in
machine-verifiable publication evidence. Claims that each weight was
demonstrably “best” or “most favorable” were removed, and the run is now labeled
a prospectively locked same-family replication rather than independent
confirmation.

Compute ratios are now reported as means with Student-t 95% intervals across
five seeds, rather than ambiguous “plus/minus” notation: dense/routed 1.532
[1.489, 1.575], and routed/fastest-fixed 2.269 [2.215, 2.322].

## 8. Canonical corrected artifacts

After the clean rerun, the canonical record is:

- `results/campaigns/conversion-campaign-v1-corrected.json`;
- its entry in `results/MANIFEST.json`;
- the regenerated campaign figure and paper PDF.

The superseded schema-v1 result is retained only if its manifest role is marked
historical; it must not be cited for corrected C1 or adjusted intervals.
