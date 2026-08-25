# Edge-to-cycle lifting campaign v1: results

Status: **prospectively locked same-generator-family replication with a
disclosed implementation deviation**. H1–H3 were run once against the protocol
frozen in [`docs/27`](27-conversion-campaign-protocol.md) before execution. This
supports a prospectively specified analysis, not independent confirmation or
pristine preregistration. H4 and H5 are prespecified secondary analyses outside
the multiplicity-controlled family. The frozen protocol remains byte-for-byte unchanged; this results record and
[`docs/29`](29-audit-corrections.md) disclose the corrections found during
post-run audit.

Three frozen keys need precise names. `exact` is an elementwise-mean
**boundary-compatibility penalty**, not exactness of a sequence; `cone` is
`exp(−2·σ_min(W))`, a **singular-value cone surrogate**, not mapping-cone
homology; and `rtd` is a normalized pairwise-distance MSE, an
**RTD-inspired surrogate**, not published RTD/SRTD.

The frozen “conversion” name is historical. The continuous task fits
`W: R^E -> R^F` from edge signals to coordinates in a chosen cycle basis, while
`W^T: R^F -> R^E` is a candidate `d2`. This is a degree-changing lifting, not a
degree-preserving typed chain map between complexes. The annulus experiment is
the separate setting that selects genuine typed chain maps.

The protocol writes `‖B1 Wᵀ‖²`, but the executed runner used
`mean((B1 Wᵀ)²)`. Because matrix size varies across topologies, this is not one
global rescaling. No endpoint, weight, decision rule, or sample size changed,
but this objective mismatch prevents describing the run as a pristine protocol
replication.

The hypotheses, effect directions, endpoint, training-set size, and weights were
chosen after outcome-informed exploration on the same generator family. Document
26 does not retain the exploratory seed identities, so overlap with the frozen
20261001–20261030 seed block cannot be audited. An untouched disjoint generator
family or seed block is required for independent confirmation.

| item | value |
|---|---|
| protocol | `docs/27-conversion-campaign-protocol.md` |
| protocol SHA-256 | `503cc282f40d118ba1739c2afe1bfc77eaf2b1733baaddb91c0c3363e75ae2b8` |
| protocol committed at | `d5d18af` |
| campaign run at | `11644c6`, clean worktree |
| result | `results/campaigns/conversion-campaign-v1-corrected.json` |
| eligible topologies | **29 of 30** (seed 20261025 had fewer than three faces and was skipped, not replaced) |

## Primary contrasts

Endpoint is the paired `log10(held-out MSE with term / held-out MSE with no
term)`, one value per eligible generator seed. Each of the 29 seeds jointly
instantiates a topology, input data, and training noise; penalized and baseline
fits share that realization. Pairing controls the shared realization but does
not remove topology variance. The estimand is the mean paired log ratio over
eligible seeds, and the Student-t analysis assumes these seed-level effects are
exchangeable. Training labels use Gaussian noise with sigma 0.02, while held-out
targets are noiseless `B2^T x`, so MSE measures recovery of the planted lifting.
Negative means the term improves recovery. Decisions are governed by the
Bonferroni-adjusted interval across the family of three; unadjusted intervals are
shown for reference.

| objective | weight | 95% interval | Bonferroni 98.33% | sign test | decision |
|---|---:|---|---|---:|---|
| boundary compatibility (`exact`) | 3.0 | [−2.628, −1.632] | **[−2.749, −1.511]** | <1e-5 | **improves** |
| singular-value cone surrogate (`cone`) | 0.01 | [+0.125, +0.254] | **[+0.109, +0.270]** | <1e-5 | **harms** |
| RTD-inspired distance surrogate (`rtd`) | 0.1 | [+0.002, +0.034] | [−0.002, +0.038] | 0.458 | **no detected improvement** |

**H1 supported for the executed objective, subject to the disclosed
deviation.** Boundary compatibility improves held-out recovery of an
edge-to-cycle lifting, family-wise adjusted. The penalty does not directly use
`B2` or response labels,
but it uses `B1`, which determines the target cycle subspace; the unpenalized
baseline ignores this strong structural side information. With the known
deterministic generator, `B2` is algorithmically recoverable from the graph, and
no analytic cycle-basis, Hodge-projection, or nullspace oracle was included.

This is favorable scarce-probe system-identification geometry. With 16 training
probes, `E > 16` in 21/29 eligible seeds, `F <= 16` in 24/29, and 16/29 cross
from the former underdetermined ambient system to the latter potentially
identifiable hard cycle-subspace system. Median dimensions are `E = 23`,
`F = 11`; five seeds have `F > 16`. The executed finite penalty only shrinks a
full `F x E` matrix toward that subspace; it does not hard-reduce its parameter
dimension.

**H2 supported for the tested surrogate.** The adjusted interval lies entirely
above zero, so the singular-value cone surrogate **harms** the lifting at the
tested weight. This does not test mapping-cone homology.

**H3 shows no detected improvement.** The RTD-inspired distance surrogate's
adjusted interval contains zero and the sign test is 0.458. No equivalence margin
was frozen, so this is not evidence of equivalence or inertness and does not
generalize to published RTD/SRTD. The implemented surrogate asks source and
lifted pairwise distances to match even though the rank-`F` truth intentionally
discards cut-space components. Its target mismatch makes the result especially
inapplicable to published topology divergence.

## H4 — claim C1 (prespecified secondary)

Within each eligible seed, nine prespecified compatibility-penalty weights produce
learned maps of varying quality; the endpoint is the Pearson correlation between
a map's `log10(max(||B1 W^T||_F, 1e-30))` and
`log10(max(held-out MSE, 1e-300))` along that path. This reported Frobenius
defect differs from the elementwise mean-square penalty used for training. Its
95% interval is unadjusted because H4 is outside the H1–H3 multiplicity family.

| quantity | value |
|---|---|
| mean within-seed correlation | **+0.961** |
| 95% interval | **[+0.935, +0.987]** |
| eligible seeds with positive correlation | **29 / 29** |

**Scoped path covariation supported.** Within this synthetic family, defect
covaries with held-out error along the compatibility-penalty path. The nine fits
reuse the same data and initialization and differ in the common driver `lambda`;
they are not independent observations. This does not establish independent
predictive information, off-path or unseen-topology calibration, an isolated
intervention effect, or causality.

## H5 — routing endpoint: non-informative by construction

The prespecified endpoint compares routing against `min(cell error, graph
error)` computed per row, with the threshold fixed on the even-indexed
threshold split and applied unchanged to the odd-indexed evaluation split.

| quantity | value |
|---|---|
| threshold (from 30 threshold-split rows = 15 topology clusters x 2 weights) | 0.0516 |
| evaluation rows | 28 = 14 topology clusters x 2 weights |
| historical naive mean `log10` ratio | +0.135 |
| historical naive df=27 interval | [−0.111, +0.382] — **withdrawn** |
| decision | **none / non-informative** |

**Protocol-design flaw.** The formula's denominator is the per-row minimum of
the same two route errors from which the numerator is selected. The ratio is
therefore at least one for every row and each `log10` observation is
nonnegative. Yet the frozen decision required a confidence interval strictly
below zero. That outcome was mathematically impossible. A finite-sample
Student-t interval can extend below zero, as the reported lower bound does, but
its upper bound cannot be below zero when the sample mean is nonnegative. The
split threshold prevents in-sample tuning; it does not repair the endpoint.

**Pseudoreplication flaw.** The 28 evaluation rows are two weights for each of
only 14 topology clusters. The historical df=27 interval treated those rows as
independent, and the historical 25/28 selection count has the same dependence
problem. Both are withdrawn from inferential use. The corrected canonical record
reports the historical row-naive interval `[−0.111, +0.382]` and the
topology-clustered descriptive interval `[−0.123, +0.393]`; neither is
inferential, and H5 receives no hypothesis decision.

The exploratory version of this in [`docs/26`](26-exactness-as-a-prior.md)
appeared to beat both fixed strategies. It does not survive an out-of-sample
threshold. **That earlier routing result is withdrawn.**

## Summary against the original idea

| component | status after this campaign |
|---|---|
| hold several views | built, works |
| learn an edge-to-cycle-coordinate lifting | works, and is measurable |
| defects as measurements | compatibility defect covaries with error along the fixed penalty path in an unadjusted secondary result, +0.961, 29/29; independent prediction remains untested |
| structural terms to train the lifting | boundary compatibility improves subject to a protocol deviation and strong-side-information baseline gap; singular-value surrogate harms; target-misaligned distance surrogate shows no detected improvement |
| defects to choose the view | frozen endpoint is non-informative by construction |

H1–H3 provide the primary lifting analysis within this same-family
replication; H4 is a prespecified secondary regularization-path association. H5
does not test its intended routing claim because its decision
rule could never succeed.

The historically named conversion campaign did not train with mapping-cone homology or published
RTD. It trained with a singular-value cone surrogate and an RTD-inspired
distance surrogate. The former harmed at the tested weight; the latter showed no
detected improvement. The successful objective was boundary compatibility under
the frozen historical key `exact`, not exactness of a sequence.

## Boundary

- One synthetic lifting family, 29 eligible joint topology/data/noise seed-level
  replicates, one training-set size. Inference assumes seed-level exchangeability.
- The hypotheses, directions, endpoint, training size, and weights were selected
  after same-family exploration. Exploratory seed identities were not retained,
  so overlap with the frozen block is unverifiable. This is not independent
  confirmation; an untouched disjoint family or seed block remains necessary.
- A separate full `F×E` matrix is fitted and evaluated within each topology;
  there is no shared model or unseen-topology generalization. Median `E×F` is
  242 (range 30–770).
- NetworkX `cycle_basis` chooses noncanonical basis coordinates. The
  compatibility penalty favors the cycle subspace; paired targets identify that
  topology's coordinate convention.
- `B1` determines the cycle subspace, and the deterministic generator makes its
  chosen `B2` algorithmically recoverable. The baseline ignores `B1`; no
  analytic cycle-basis, Hodge-projection, or nullspace oracle was included.
- Scarce-probe geometry favors the structural prior: with `N_train = 16`, 21/29
  seeds have `E > 16`, 24/29 have `F <= 16`, and 16/29 cross those thresholds
  (median `E = 23`, `F = 11`; five have `F > 16`). The finite penalty is still
  shrinkage, not hard dimension reduction.
- Weights were chosen after exploratory work and frozen before execution, but
  the full exploratory sweep is not retained as machine-verifiable publication
  evidence.
- C1 summarizes one deterministic nine-weight path per eligible seed. The fits reuse
  data and initialization, and `lambda` jointly drives defect and error; the
  nine points are not independent and do not establish independent predictive
  information or off-path calibration.
- Nothing here concerns the paper's §6.3 result, which is about selection over a
  fixed finite class and is untouched.
- The zero set of `mean((W B1ᵀ)²)` has every row of `W` in the cycle space.
  The executed finite penalty encourages that subspace but leaves all `F×E`
  entries trainable; it does not hard-constrain `W` or literally reduce the
  parameter dimension. The supported claim is that this input-derived bias
  helped under the executed mean-squared penalty.
- Specificity against a tuned generic regularizer or rank-matched random
  subspace remains untested in machine-verifiable publication evidence.
- The RTD-inspired surrogate is structurally target-misaligned because the true
  rank-`F` lifting discards cut-space geometry; its result does not test
  published RTD/SRTD.
- H5's 28 rows comprise 14 topology clusters at two weights. Its naive df=27
  interval and 25/28 count are withdrawn; the corrected record reports only the
  row-naive `[−0.111, +0.382]` and topology-clustered `[−0.123, +0.393]`
  descriptive intervals and no decision.
- In the separate annulus setting, all six supervised objectives have perfect
  decoded transformation/cell accuracy; their mean map MSE spans `2.618e-17` to
  `2.504e-8`, versus 0.109 and 0.191 for the controls. The 10/10 engineering
  gate applies only to `task_reconstruction` and `combined` (five seeds each).
- Separate five-seed compute ratios are means with Student-t 95% intervals:
  dense/routed 1.532 [1.489, 1.575], and routed/fastest-fixed 2.269
  [2.215, 2.322].
- Exploratory work in `docs/26` is superseded by this document wherever the two
  disagree.
