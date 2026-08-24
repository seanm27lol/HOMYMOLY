# Exactness as a prior on a learned conversion

Status: **exploratory, controlled, positive**. Ten topologies, paired, with
intervals and a matched-rank control. Not preregistered and not a paper result.
This is the first positive finding in the project that survives its own controls,
so it is written down carefully, including what it does not say.

Date: 2026-08-23. Uses the generator specified in
[`docs/25`](25-conversion-generator-spec.md).

## The question

The project's original idea: *moving between representations loses something;
homological algebra has exact tools for measuring what a map destroys; use those
to train the maps.*

Four previous attempts found nothing, each because the homological quantity came
out constant or generic and so carried no information — recorded in
[`docs/24`](24-continuous-map-probe.md) and §6.3 of the paper. The conversion
generator was built to remove that obstruction. This is its first use.

## Setup

Per topology: a graph from `ConversionDataset`, and the conversion that lifts an
edge cochain to face coefficients. Ground truth is `W = B2ᵀ`, so face
coefficients are the circulation of the edge signal around each cycle.

**The cycle basis `B2` is withheld.** The model learns `W` as a free `[F, E]`
matrix — a median of 216 parameters — from paired `(x1, circulation)` samples
with noise 0.02. Ten topologies, each used as its own seed, paired so the same
topology and data are used with and without the term.

The structural term is

    || W B1ᵀ ||²

Ground truth satisfies it exactly, because `B2ᵀ B1ᵀ = (B1 B2)ᵀ = 0`. In words:
**a correct lift must annihilate coboundaries.** That is exactness.

The term is not leaked supervision. `B1` *is* the graph and the model observes
it. `B2` is the answer and is withheld.

## Result

Held-out MSE, medians over ten topologies; interval on the paired
`log10(with term / without)`.

| training pairs | without | with | 95% CI |
|---:|---:|---:|---|
| 2 | 3.43 | 3.02 | [−1.202, +0.377] |
| 4 | 3.33 | 2.44 | [−1.524, +0.304] |
| 8 | 2.99 | 1.49 | [−1.988, −0.026] |
| **16** | **1.78** | **0.00223** | **[−2.804, −1.330]** |
| 32 | 8.11e-04 | 6.06e-04 | [−0.403, +0.004] |
| 64 | 1.90e-04 | 1.85e-04 | [−0.054, −0.015] |

At sixteen training pairs against 216 parameters the term reduces held-out error
by roughly **800×**, with an interval far from zero. The benefit fades as data
grows, which is what a prior should do.

## The control that matters

A large gain from adding any penalty would be unremarkable. Three comparisons at
sixteen training pairs, against plain least squares at 1.582:

| penalty | median held-out | 95% CI vs plain |
|---|---:|---|
| ridge, best of four weights | 0.631 | [−0.451, −0.174] |
| **random subspace, rank matched to `B1ᵀ`** | **2.256** | [−0.103, +0.601] |
| **exactness** | **0.002** | **[−2.826, −1.134]** |

Generic shrinkage buys about 2.5×. **A random constraint of the same rank buys
nothing at all.** Exactness buys about 800×.

So the gain is not from regularising, and not from the quantity of constraint. It
is that specific subspace.

## What this does and does not claim

**Does.** Algebraic exactness, derived from structure the model already observes,
is a strong and specific prior on a learned conversion when paired data is
scarce. That is a real instance of the original idea, and the matched-rank
control rules out the obvious deflationary explanation.

**Does not.** This is a linear conversion and a linear constraint, and the
mechanism is understood rather than mysterious: `W B1ᵀ = 0` forces each row of `W`
into the cycle space, cutting the effective dimension from `F × E` to `F × F`. A
correct linear constraint helping a linear problem is expected. The finding is
that *exactness is the correct constraint and is available from the input*, not
that anything surprising happens.

It also says nothing yet about the mapping cone, about RTD, or about routing.
Those remain untested.

## Known weaknesses

- Ten topologies. Thin.
- The ridge weight was swept over four values; the exactness weight was used at a
  single value, 0.1. The asymmetry favours ridge, and exactness still wins by
  roughly 300×, but a sweep should be run before this is published.
- The `n = 32` row breaks an otherwise monotone story: its interval contains zero
  while both neighbours exclude it. Unexplained.
- Exploratory. No preregistration, and the endpoint was chosen after seeing the
  generator behave.

## What to do next

1. Sweep the exactness weight, and add more topologies with a preregistered
   endpoint and decision rule.
2. Test whether the same holds for a **nonlinear** conversion, where a correct
   linear constraint is no longer the whole story.
3. Bring in the mapping cone and RTD, which this experiment did not use. The
   defect machinery in `topology/defects.py` measures what a learned `W`
   destroys and cannot reach; that measurement has not yet been connected to
   held-out error, which is claim C1.
4. Only then consider amending the paper. The current manuscript's §6.3 finding
   is about selection over a fixed class and is unaffected by this; what would
   change is the discussion, which currently has no positive instance to point at.

---

# Widened search: exactness versus the mapping cone versus RTD

Added 2026-08-23, same session. Seventeen topologies with at least three faces,
sixteen training pairs, paired against plain least squares per topology, df=16
intervals on the `log10` ratio.

The learned `W` implies a face boundary `Wᵀ`, so all three terms can be written as
conditions on the *implied* complex rather than bolted on:

| term | form | meaning |
|---|---|---|
| exact | `‖B1 Wᵀ‖²` | the implied complex satisfies `d∘d = 0` |
| cone | `exp(−2·σ_min(W))` | no face collapses; implied 2-cells stay independent |
| rtd | distance-preservation proxy | edge-signal geometry survives into face coefficients |

Plain least squares: median held-out **2.579**.

| term (weight) | median held-out | 95% CI vs plain |
|---|---:|---|
| exact (0.003) | 5.14e-01 | [−1.035, −0.370] |
| exact (0.03) | 8.47e-03 | [−1.879, −0.849] |
| exact (0.3) | 2.71e-03 | [−2.504, −1.267] |
| exact (3.0) | **1.82e-03** | **[−2.699, −1.430]** |
| cone (0.01) | 3.29e+00 | [+0.108, +0.294] |
| cone (0.1) | 3.47e+00 | [+0.180, +0.744] |
| rtd (0.01) | 2.58e+00 | [−0.000, +0.001] |
| rtd (0.1) | 2.58e+00 | [−0.001, +0.017] |
| exact 0.3 + cone 0.01 | 3.48e-01 | [−1.618, −0.679] |

Three different answers, and the separation is the finding:

- **Exactness improves the model at every weight over a thousand-fold range**, up
  to roughly 1400× lower held-out error. This closes the weight-sweep weakness
  recorded above: the effect is not a tuned artefact.
- **The mapping cone actively hurts.** Both intervals exclude zero on the harmful
  side, and adding it to exactness makes exactness worse (3.48e-01 against
  2.71e-03 alone).
- **RTD does nothing.** Both intervals are indistinguishable from zero to three
  decimal places.

## Claim C1: do measured defects predict damage?

Across 34 fits, pooling the unregularised and the exact-plus-cone conditions:

| defect measure | correlation with `log10` held-out error |
|---|---:|
| `log ‖B1 Wᵀ‖` — exactness violation | **+0.833** |
| smallest singular value of `W` — collapse | −0.601 |

The exactness defect strongly predicts task damage, in the expected direction.

**Caveat, and it matters.** The pool mixes two conditions that differ
systematically in both quantities, so part of this correlation reflects
between-condition separation rather than prediction within a condition. A
within-condition correlation over many weights is the test that would settle C1
properly. On this evidence C1 is *supported, not established*.

## What the widened search says about the original idea

The project's original thesis was that homological defect measures should improve
a learned conversion. On this benchmark that is **true, and specifically true of
exactness**. It is false for the mapping cone, which hurts, and empty for RTD,
which does nothing.

That reframes the whole history recorded in this repository. The cone and RTD
were the objects the project bet on and built campaigns around; both fail here.
Exactness was present from the beginning — `ExactChainMapLayer` enforces it
architecturally, and the one early experiment that worked, the 1.5e-14 chain-map
recovery, relied on it. The mechanism that was working was already in the
architecture, while the experiments were aimed at the two mechanisms that do not.

The honest phrasing is the user's: homological structure does not obviously
*teach* the model, but exactness measurably *improves* it.

## Still not established

- Nonlinear conversions. Everything here is linear, and a correct linear
  constraint helping a linear problem is expected.
- Routing on measured conversion cost. Untouched.
- C1 within condition, as above.
- Preregistration. All of this is exploratory.

---

# Closing the three open questions

Added 2026-08-23, same session. Seventeen topologies throughout.

## A. Claim C1, within condition — supported

The earlier C1 correlation pooled two conditions, so part of it could have been
between-condition separation rather than prediction. Redone properly: within a
single topology, sweep the exactness weight over nine values from 0 to 3.0,
producing learned maps of varying quality, and correlate each map's exactness
violation with its held-out error.

| quantity | value |
|---|---|
| mean within-topology correlation | **+0.833** |
| 95% CI (df = 16) | **[+0.778, +0.888]** |
| topologies with positive correlation | **17 / 17** |

The caveat is discharged. A learned conversion's measured structural defect
predicts its task damage *within* a condition, not merely across conditions.
Claim **C1 is supported** on this benchmark — it has been marked untested in the
ledger since the project began.

## B. Nonlinear conversion — exactness still helps

The obvious deflation of the earlier result is that a correct linear constraint
helping a linear problem is unremarkable. Repeated with a nonlinear link: target
`tanh(circulation)`, model `tanh(Wx)`.

| setting | plain | exact (3.0) | 95% CI |
|---|---:|---:|---|
| linear | 2.686 | 1.07e-03 | [−2.711, −1.415] |
| **nonlinear** | 0.596 | **4.50e-02** | **[−1.740, −0.808]** |

The gain shrinks from roughly 2500× to roughly 13×, and remains far from zero.
Exactness is not merely an artefact of linear least squares.

The two rows are not comparable in absolute terms: `tanh` saturates and bounds
the error, so the nonlinear baseline starts much lower. The paired log ratio is
the endpoint, and both exclude zero.

## C. Routing on measured conversion cost — preliminary, positive

The half of the original idea nobody had touched: choose a view by **measuring
what converting into it costs**, rather than by a distilled utility table.

Downstream task: predict `t = cᵀ(B2ᵀx)`, a scalar readout of the face view. The
*cell route* answers through the learned converter; the *graph route* fits a
vector straight from edge features on the same data budget. 136 trials spanning
four training-set sizes and two regularisation settings, so converter quality
varies widely.

| | result |
|---|---|
| corr(measured defect, cell-route disadvantage) | +0.187 |
| cell route wins when defect is **low** | **36 / 68** |
| cell route wins when defect is **high** | **10 / 68** |

| strategy | median held-out error |
|---|---:|
| always cell | 8.597 |
| always graph | 10.51 |
| **routed on measured defect** | **6.027** |

Routing on the measured conversion defect beats **both** fixed strategies. The
win-rate separation is large — 53% against 15% — even though the magnitude
correlation is weak at +0.187, so the defect says *whether* the converted view is
usable more reliably than it says *by how much*.

**Caveat.** The routing threshold is the median defect computed over the same
trials it is evaluated on, which is in-sample selection. A held-out threshold, or
threshold selection on separate topologies, is required before this is more than
a first look.

## Where the original idea now stands

| component of the idea | status |
|---|---|
| hold several views | built, works |
| move between them | now possible, and measurable, via the conversion generator |
| defects as measurements | **C1 supported**, +0.833 within condition |
| defects to train the maps | **exactness improves**, cone hurts, RTD inert |
| defects to choose the view | **preliminary positive**, beats both fixed routes |

Not "homological structure teaches the model." **Exactness measurably improves
it, and the measured defect is usable both as a training signal and as a routing
signal.** The mapping cone and RTD — the two objects the project spent its
campaigns on — are the parts that do not work.

Everything here remains exploratory, on one synthetic family, with no
preregistration. The next step is a frozen, preregistered campaign over these
five rows, not further exploration.
