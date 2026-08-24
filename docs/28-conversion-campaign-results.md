# Conversion campaign v1: results

Status: **confirmatory**. Run once against the protocol frozen in
[`docs/27`](27-conversion-campaign-protocol.md) before execution. No endpoint,
weight, decision rule, or sample size was changed after the protocol was
committed.

| item | value |
|---|---|
| protocol | `docs/27-conversion-campaign-protocol.md` |
| protocol SHA-256 | `503cc282f40d118ba1739c2afe1bfc77eaf2b1733baaddb91c0c3363e75ae2b8` |
| protocol committed at | `d5d18af` |
| campaign run at | `11644c6`, clean worktree |
| result | `results/campaigns/conversion-campaign-v1.json` |
| eligible topologies | **29 of 30** (seed 20261025 had fewer than three faces and was skipped, not replaced) |

## Primary contrasts

Endpoint is the paired `log10(held-out MSE with term / held-out MSE with no
term)`, one value per topology. Negative means the term improves the model.
Decisions are governed by the Bonferroni-adjusted interval across the family of
three; unadjusted intervals are shown for reference.

| term | weight | 95% interval | Bonferroni 98.33% | sign test | decision |
|---|---:|---|---|---:|---|
| `exact` | 3.0 | [−2.628, −1.632] | **[−2.802, −1.458]** | <1e-5 | **improves** |
| `cone` | 0.01 | [+0.125, +0.254] | **[+0.102, +0.277]** | <1e-5 | **harms** |
| `rtd` | 0.1 | [+0.002, +0.034] | [−0.003, +0.039] | 0.458 | **inert** |

**H1 confirmed.** Exactness improves a learned conversion — roughly a
hundredfold to a thousandfold reduction in held-out error, family-wise adjusted.

**H2 confirmed, and more strongly than predicted.** The protocol predicted the
mapping cone would fail to improve. It does worse than that: the adjusted
interval lies entirely above zero, so the cone term **actively harms** the
conversion.

**H3 confirmed.** RTD is inert. Its unadjusted interval sits marginally above
zero, the adjusted interval contains zero, and the sign test is 0.458.

## H4 — claim C1

Within each topology, nine exactness weights produce learned maps of varying
quality; the endpoint is the correlation between a map's exactness violation and
its held-out error.

| quantity | value |
|---|---|
| mean within-topology correlation | **+0.854** |
| 95% interval | **[+0.831, +0.877]** |
| topologies with positive correlation | **29 / 29** |

**Supported.** A learned conversion's measured structural defect predicts its
task damage. Claim C1 has been marked untested in the claims ledger since the
project began.

## H5 — routing on measured defect: not supported

The preregistered endpoint compares routing against `min(cell error, graph
error)` computed per trial, with the threshold fixed on the even-indexed
threshold split and applied unchanged to the odd-indexed evaluation split.

| quantity | value |
|---|---|
| threshold (from 30 threshold-split trials) | 0.0516 |
| evaluation trials | 28 |
| mean `log10` ratio | +0.135 |
| 95% interval | **[−0.111, +0.382]** |
| decision | **not supported** |

**Honest note on the endpoint.** The protocol prose described this as "the better
of the two fixed strategies", but the formula it specified — a per-trial minimum
— is an **oracle**, not a fixed strategy. The implementation followed the
formula, which is the stricter reading. The mismatch is recorded rather than
resolved by reanalysis.

For completeness, and clearly **post hoc**: routing against the genuinely better
fixed strategy, always-cell, gives mean −0.013 with 95% interval
[−0.282, +0.256]. That also contains zero. Routing is not distinguishable from
simply always using the cell view.

The router picks the better view on **25 of 28** trials, so the defect does carry
real signal about *which* view to use; the three misses are costly enough to
erase the average gain.

The exploratory version of this in [`docs/26`](26-exactness-as-a-prior.md)
appeared to beat both fixed strategies. It does not survive an out-of-sample
threshold. **That earlier routing result is withdrawn.**

## Summary against the original idea

| component | status after this campaign |
|---|---|
| hold several views | built, works |
| move between them | works, and is measurable |
| defects as measurements | **C1 supported**, +0.854, 29/29 |
| defects to train the maps | **exactness improves** (confirmed); cone **harms**; RTD inert |
| defects to choose the view | **not supported** |

Four of five hypotheses resolved as predicted. The fifth, routing, failed under a
stricter test than the exploratory work used, which is what preregistration is
for.

The mapping cone and RTD were the two objects this project built its campaigns
around. Confirmatorily, one harms and the other does nothing. Exactness — present
in the architecture from the beginning, enforced by `ExactChainMapLayer`, and
never itself the thing under test — is what improves the model.

## Boundary

- One synthetic family, 29 topologies, one training-set size, linear conversions.
- Nothing here concerns the paper's §6.3 result, which is about selection over a
  fixed finite class and is untouched.
- The mechanism is understood rather than surprising: `W B1ᵀ = 0` forces each row
  of `W` into the cycle space and cuts the effective dimension from `F×E` to
  `F×F`. The confirmed claim is that **exactness is the correct constraint and is
  available from the input**, not that anything mysterious occurs.
- Exploratory work in `docs/26` is superseded by this document wherever the two
  disagree.
