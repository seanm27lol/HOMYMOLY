# Conversion campaign v1: results

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
term)`, one value per topology. Negative means the term improves the model.
Decisions are governed by the Bonferroni-adjusted interval across the family of
three; unadjusted intervals are shown for reference.

| objective | weight | 95% interval | Bonferroni 98.33% | sign test | decision |
|---|---:|---|---|---:|---|
| boundary compatibility (`exact`) | 3.0 | [−2.628, −1.632] | **[−2.802, −1.458]** | <1e-5 | **improves** |
| singular-value cone surrogate (`cone`) | 0.01 | [+0.125, +0.254] | **[+0.102, +0.277]** | <1e-5 | **harms** |
| RTD-inspired distance surrogate (`rtd`) | 0.1 | [+0.002, +0.034] | [−0.003, +0.039] | 0.458 | **no detected improvement** |

**H1 supported for the executed objective, subject to the disclosed
deviation.** Boundary compatibility improves a learned conversion — roughly a
hundredfold to a thousandfold reduction in held-out error, family-wise adjusted.

**H2 supported for the tested surrogate.** The adjusted interval lies entirely
above zero, so the singular-value cone surrogate **harms** the conversion at the
tested weight. This does not test mapping-cone homology.

**H3 shows no detected improvement.** The RTD-inspired distance surrogate's
adjusted interval contains zero and the sign test is 0.458. No equivalence margin
was frozen, so this is not evidence of equivalence or inertness and does not
generalize to published RTD/SRTD.

## H4 — claim C1 (prespecified secondary)

Within each topology, nine prespecified compatibility-penalty weights produce
learned maps of varying quality; the endpoint is the Pearson correlation between
a map's boundary-compatibility defect and its held-out error along that path.
Its 95% interval is unadjusted because H4 is outside the H1–H3 multiplicity
family.

| quantity | value |
|---|---|
| mean within-topology correlation | **+0.854** |
| 95% interval | **[+0.831, +0.877]** |
| topologies with positive correlation | **29 / 29** |

**Scoped path covariation supported.** Within this synthetic family, defect
covaries with held-out error along the compatibility-penalty path. The nine fits
reuse the same data and initialization and differ in the common driver `lambda`;
they are not independent observations. This does not establish independent
predictive information, off-path or unseen-topology calibration, an isolated
intervention effect, or causality.

## H5 — routing endpoint: non-informative by construction

The prespecified endpoint compares routing against `min(cell error, graph
error)` computed per trial, with the threshold fixed on the even-indexed
threshold split and applied unchanged to the odd-indexed evaluation split.

| quantity | value |
|---|---|
| threshold (from 30 threshold-split trials) | 0.0516 |
| evaluation trials | 28 |
| mean `log10` ratio | +0.135 |
| 95% interval | **[−0.111, +0.382]** |
| decision | **invalid / non-informative** |

**Protocol-design flaw.** The formula's denominator is the per-trial minimum of
the same two route errors from which the numerator is selected. The ratio is
therefore at least one for every trial and each `log10` observation is
nonnegative. Yet the frozen decision required a confidence interval strictly
below zero. That outcome was mathematically impossible. A finite-sample
Student-t interval can extend below zero, as the reported lower bound does, but
its upper bound cannot be below zero when the sample mean is nonnegative. The
split threshold prevents in-sample tuning; it does not repair the endpoint.

For completeness, and clearly **post hoc**: routing against the genuinely better
fixed strategy, always-cell, gives mean −0.013 with 95% interval
[−0.282, +0.256]. That descriptive contrast contains zero but was not the frozen
endpoint and is not multiplicity-adjusted.

The router picks the lower-error view on **25 of 28** trials. That descriptive
rate motivates a correctly designed follow-up; it is not inferential evidence
that the defect selects views.

The exploratory version of this in [`docs/26`](26-exactness-as-a-prior.md)
appeared to beat both fixed strategies. It does not survive an out-of-sample
threshold. **That earlier routing result is withdrawn.**

## Summary against the original idea

| component | status after this campaign |
|---|---|
| hold several views | built, works |
| move between them | works, and is measurable |
| defects as measurements | compatibility defect covaries with error along the fixed penalty path in an unadjusted secondary result, +0.854, 29/29; independent prediction remains untested |
| defects to train the maps | boundary compatibility improves subject to a protocol deviation; singular-value surrogate harms; distance surrogate shows no detected improvement |
| defects to choose the view | frozen endpoint is non-informative by construction |

H1–H3 provide the primary conversion analysis within this same-family
replication; H4 is a prespecified secondary regularization-path association. H5
does not test its intended routing claim because its decision
rule could never succeed.

The conversion campaign did not train with mapping-cone homology or published
RTD. It trained with a singular-value cone surrogate and an RTD-inspired
distance surrogate. The former harmed at the tested weight; the latter showed no
detected improvement. The successful objective was boundary compatibility under
the frozen historical key `exact`, not exactness of a sequence.

## Boundary

- One synthetic family, 29 topologies, one training-set size, linear conversions.
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
- Weights were chosen after exploratory work and frozen before execution, but
  the full exploratory sweep is not retained as machine-verifiable publication
  evidence.
- C1 summarizes one deterministic nine-weight path per topology. The fits reuse
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
- Exploratory work in `docs/26` is superseded by this document wherever the two
  disagree.
