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

## 9. v2 lifting replication: pre-seal audit, seal, and post-campaign validation

The untouched-seed v2 replication
([`docs/31`](31-independent-lifting-replication-protocol.md),
[`docs/32`](32-independent-lifting-replication-seal.json),
[`docs/33`](33-lifting-replication-v2-results.md)) underwent its own
adversarial audit cycle. This entry records the findings and their
resolutions; the sealed files themselves are immutable.

### 9.1 Pre-seal adversarial audit findings and resolutions

An adversarial review of the v2 design before sealing found and resolved the
following, each reflected in the sealed protocol, runner, and tests:

- **Naked command-line hash replaced by seal-file parsing.** An early design
  passed the expected runner hash as a CLI argument, which is unenforceable
  and self-referential. The runner instead parses and validates the committed
  machine-readable seal record, requires it at HEAD, and compares its own
  runtime SHA-256 against the seal's `runner_sha256`. There is no
  `--expected-runner-sha256` flag.
- **C1 epsilon floor removed.** A draft added an epsilon to defects before
  log transformation, which would tune the endpoint after outcomes are
  imaginable. The sealed design requires every defect and error to be finite
  and strictly positive and treats any violation as a whole-design failure;
  Fisher clipping uses only the exact `nextafter(1, 0)` endpoint rule.
- **Generator exceptions made fail-closed.** A generation exception in the
  declared block is a campaign failure (`design_failure`), never an exclusion
  or a silent skip; the failing seed is recorded and no seed is deleted alone.
- **Basis orthonormality made a hard assertion.** Cycle-basis membership
  (`||B1 Q_cycle||_F <= 1e-10`), orthonormality
  (`||Q_cycle.T Q_cycle − I||_F <= 1e-10`), observed rank exactly `V − 1`
  under the recorded tolerance, and random-basis orthonormality with a
  nonzero QR diagonal are hard stop conditions, not advisory diagnostics.
- **Information-flow firewall made structural.** The fitting APIs for all
  arms receive only their declared training tensors and, where applicable,
  `B1` or the seeded random basis; `B2`, face-cycle metadata, and held-out
  values are unreachable from fitting code, enforced by API shape rather than
  by convention. The truth-access oracle is segregated from every fitted arm.
- **Rank recording required.** Every gelsd least-squares fit records its
  returned numerical rank and smallest singular value, and the closed-form
  soft solve records its pseudoinverse effective rank; a rank violation stops
  the campaign.
- **Arm names aligned to the frozen handoff identifiers.** Arm and claim
  identifiers match
  [`docs/30`](30-journal-completion-handoff.md) §7 and the seal's primary
  family exactly, so no relabeling can occur between execution and reporting.

### 9.2 Two-commit seal

The design was sealed under the fail-closed two-commit procedure of
[`docs/30`](30-journal-completion-handoff.md) §9.1. Commit A
(`044322c7dc6a6255eec941dbcb76c45288a9666c`) contains the protocol, runner,
and tests; commit B
(`9baae6b8322120724e7f5aff3c47fd7ef343086c`) adds the seal record with the
four verified SHA-256 fingerprints (protocol, runner, generator, lock) and was
pushed to the private remote before any declared seed was instantiated,
creating a remote timestamp ahead of execution. No design file changed between
commit A and execution, and the recorded execution worktree was clean.

### 9.3 Post-campaign independent validation verdict

The completed result
(`results/campaigns/lifting-replication-v2.json`, status `complete`; 33 of 36
declared seeds eligible; six of seven claims supported, H5 not supported) was
validated independently of the runner: every summary, interval, support flag,
and C1 value was recomputed from the retained raw rows, with all checks
passing and maximum absolute discrepancies from `0.0` to `1e-13`. The
validation also verified the deterministic sub-seeds, the cycle-nullspace and
random-basis diagnostics, the `inner_cv_ridge` fold-loss recomputation and
selected penalties, the eligibility accounting, and the audit block.

### 9.4 Retention-gap note

Post-campaign validation found one retention gap: the executed runner did not
retain or assert the per-seed closed-form stationarity residual that protocol
§7.3 lists for `soft_boundary_closed_form_lambda3`. The residual had been
test-verified below `1e-10` pre-seal on hand fixtures and historical seed
`20261001`, and no inferential value depends on it. Because the seed block is
sealed and consumed, rerunning over a retention gap is forbidden; the gap is
recorded transparently here and in
[`docs/33`](33-lifting-replication-v2-results.md) rather than repaired.

**Addendum (2026-08-25 audit).** A later external audit found three further
enforcement gaps in the executed runner beyond this retention gap: the gelsd
rank/positivity rejection (protocol §6) and the raw boundary defect in the C1
positivity guard (§9) were recorded but not asserted, and the seal parser bound
only claim identifiers. Post-hoc verification confirmed none of the omitted
conditions would have fired (minimum gelsd rank 3, minimum smallest singular
value `0.03859306138226727`, minimum C1 boundary defect
`0.03715218399403853`). The full corrected deviation list is in docs/33; the
hardened runner and the automated recomputing validator now enforce every
testable condition.
