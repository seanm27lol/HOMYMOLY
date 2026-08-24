# Frozen protocol: conversion campaign v1

Status: **preregistration**. Committed before the campaign runs. Nothing in this
document may be revised after the first confirmatory run; a revision voids the
freeze and the campaign must be renamed.

Date frozen: 2026-08-23.

## Why this exists

Every conversion result in [`docs/26`](26-exactness-as-a-prior.md) is
exploratory. Endpoints were chosen after seeing the generator behave, weights
were swept and the best reported, and the routing threshold was fitted on the
data it was scored on. Those results are suggestive and none of them may be
published as findings.

This protocol freezes five hypotheses, their endpoints, their decision rules, and
their sample sizes, so the campaign can either confirm or refute them once.

## Setting

Generator: `homymoly.data.conversion.ConversionDataset`, specified in
[`docs/25`](25-conversion-generator-spec.md).

Task: learn `W: R^E -> R^F` mapping an edge cochain to face coefficients. Ground
truth is `W = B2ᵀ`. **The cycle basis `B2` is withheld from the model.** `B1` is
observable, because it is the graph.

Structural terms, all written as conditions on the implied complex `Wᵀ`:

| name | term |
|---|---|
| `exact` | `‖B1 Wᵀ‖²` — the implied complex satisfies `d∘d = 0` |
| `cone` | `exp(−2·σ_min(W))` — no implied 2-cell collapses |
| `rtd` | mean squared mismatch of normalised pairwise distances between `X` and `XWᵀ` |

## Frozen design

| item | value |
|---|---|
| topologies | **30**, from `ConversionDataset(1, seed=s)` for `s` in `20261001 … 20261030` |
| eligibility | a topology enters iff it has at least 3 faces; ineligible topologies are skipped and counted, never replaced |
| training pairs | **16** |
| held-out pairs | 3072 |
| observation noise | 0.02 |
| optimiser | Adam, learning rate 0.05, **2500 steps**, `W` initialised to zeros |
| dtype | float64 |
| term weights | `exact` 3.0, `cone` 0.01, `rtd` 0.1 |

**Weight provenance.** Each weight is the best-performing value for that term in
the exploratory work in `docs/26`, so every term is represented at its most
favourable tested setting. The weights are not comparable across terms because
the terms have different scales; this is deliberate and is not a defect of the
design.

## Hypotheses, endpoints, decision rules

Endpoint for H1–H3 is the **paired** quantity, one value per topology:

    d = log10( held-out MSE with term / held-out MSE with no term )

Negative means the term improves the model. Paired, so topology variance cancels.

| id | hypothesis | prediction |
|---|---|---|
| **H1** | exactness improves a learned conversion | interval **below** zero |
| **H2** | the mapping cone does not improve it | interval **not** below zero |
| **H3** | RTD does not improve it | interval **not** below zero |

Uncertainty: Student-t interval on the mean of `d`, df = (eligible topologies − 1).

**Multiplicity is adjusted.** H1–H3 form one family of three primary contrasts.
The confirmatory interval is Bonferroni-adjusted: each uses a two-sided
98.333% interval, so the family-wise error rate is 0.05. Unadjusted 95%
intervals are also reported, and the adjusted interval governs the decision.

An exact two-sided sign test on `d` is reported as a distribution-free
sensitivity for each contrast. It does not govern any decision.

### H4 — defects predict damage (claim C1)

Within each topology, fit `W` at nine exactness weights
`0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0`, and compute the Pearson
correlation between `log ‖B1 Wᵀ‖` and `log10` held-out error across those nine
fits.

Endpoint: the mean of that within-topology correlation across topologies.

**Decision:** C1 supported iff the 95% interval on that mean lies **above** zero.

### H5 — routing on measured defect beats either fixed view

Downstream task: predict `t = cᵀ(B2ᵀx)` for a fixed random `c`. The *cell route*
answers through the learned `W`; the *graph route* fits a vector directly from
edge features on the same data budget.

**The routing threshold must be chosen out of sample.** Topologies are split by
index parity: even-indexed eligible topologies form the *threshold split*, odd
form the *evaluation split*. The threshold is the median measured defect on the
threshold split, and is then applied unchanged to the evaluation split.

Endpoint on the evaluation split, one value per trial:

    d = log10( routed error / min(always-cell error, always-graph error) )

**Decision:** supported iff the 95% interval lies **below** zero. This compares
routing against the better of the two fixed strategies, not the average, which is
the harder comparison.

## Stopping rules

The campaign runs once, over all 30 declared topologies. There is no interim
analysis, no early stop, and no extension. If the result is null it is reported
as null.

## Stop conditions

Halt and preserve evidence rather than adjust, if any of these occur:

- fewer than 24 of the 30 topologies are eligible;
- any fit produces a non-finite loss or held-out error;
- the recorded generator hash does not match `docs/25`'s generator at the
  campaign commit.

## What this campaign cannot establish

- Anything beyond this one synthetic family.
- Anything about nonlinear conversions; H1–H3 are linear, matching the setting
  the exploratory effect was found in.
- Any claim that the mechanism is surprising. `W B1ᵀ = 0` forces each row of `W`
  into the cycle space and cuts the effective dimension from `F×E` to `F×F`. A
  correct linear constraint helping a linear problem is expected. The hypothesis
  under test is that **exactness is the correct constraint and is available from
  the input**, not that anything mysterious occurs.
- Anything about the paper's §6.3 result, which concerns selection over a fixed
  finite class and is untouched by this.
