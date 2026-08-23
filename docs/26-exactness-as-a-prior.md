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
