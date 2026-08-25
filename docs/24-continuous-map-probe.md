# Continuous learned-map probe: does a homological term help?

Status: **exploratory**. Not preregistered, not a frozen campaign, and not a
paper result. Recorded because the outcome sharpens the mechanism behind the
§6.3 finding in [`docs/18-paper.md`](18-paper.md), and because the code it
produced is now part of the repository.

Date: 2026-08-23.

## Why this was run

The identifiable campaign selects one map from a fixed basis of twelve. Exact
cone acyclicity is constant across the twelve **hard decoded vertices** because
all are chain isomorphisms. The executed `cone_soft_betti` objective is evaluated
on convex mixtures during training and can vary, so the observed chance
performance is empirical; hard-vertex constancy does not explain it by itself.

A fair objection: that is a property of *selection*, not of homology. When a map
is learned as continuous parameters, its cone score changes as the parameters
change, so a cone term has a real gradient. `ExactChainMapLayer` already does
exactly this — it learns coordinates in the nullspace of the chain-map
constraint, between complexes that may differ.

So the probe asks the question the annulus campaign could not: **does a
homological term help a learned continuous map generalise?**

## Setup

- Source: cycle complex on 8 vertices, Betti (1, 1).
- Target: cycle complex on 6 vertices, Betti (1, 1). Genuinely different from
  the source, so the map cannot be a permutation.
- Learned map: `ExactChainMapLayer`, **49 free parameters** out of the 96 numbers
  a raw map would need; the chain-map law removes the other 47.
- Planted map: a random point of that nullspace, held fixed per seed.
- Data: paired signals with additive noise 0.05. Each training pair supplies 12
  constraints, so the fit is underdetermined below `n = 5` and overdetermined at
  or above it.
- Structural term: `exp(-10 * smallest singular value of the induced H1 map)`,
  which penalises a learned map that comes close to collapsing the cycle class.
  This is an ad hoc term written for the probe, not the repository's
  `cone_soft_betti_loss`.
- Design: **paired** — the same seed, planted map, and data are used with the
  term on and off, so the contrast removes seed variance. Twelve seeds, df = 11
  Student-t interval, exact two-sided sign test.

## Result: no effect

Endpoint is `log10(held-out error with term / held-out error without)`. Negative
would mean the term helps.

| n | regime | median | 95% CI (df = 11) | sign test |
|---:|---|---:|---|---:|
| 1 | underdetermined | +0.002 | [−0.014, +0.014] | 0.388 |
| 2 | underdetermined | −0.002 | [−0.030, +0.023] | 0.388 |
| 3 | underdetermined | −0.013 | [−0.036, +0.054] | 0.774 |
| 4 | underdetermined | −0.053 | [−0.111, +0.101] | 0.774 |
| 5 | overdetermined | +0.028 | [−0.104, +0.241] | 0.388 |
| 6 | overdetermined | −0.045 | [−0.339, +0.460] | 0.388 |
| 8 | overdetermined | −0.011 | [−0.421, +0.639] | 0.774 |
| 12 | overdetermined | +0.041 | [+0.063, +1.166] | 0.388 |
| 32 | overdetermined | +0.002 | [−0.130, +0.825] | 0.388 |

Every interval contains zero except `n = 12`, which excludes it on the side where
the term **hurts**. That is one marginal exclusion out of nine uncorrected
comparisons, which is roughly what chance produces; no weight is placed on it.
No sign test falls below 0.388.

An earlier unpaired eight-seed comparison of medians appeared to show a 25%
improvement at `n = 4`. The paired design removes it. That earlier number was
noise and is withdrawn.

## Why there was nothing to find

A separate check explains the null. Sampling 200 random maps from the same
nullspace and measuring each one:

**200 of 200 preserved structure** — every one was a quasi-isomorphism, with
defect signature `(0, 0, 0, 0)` at both degrees.

When both complexes have one-dimensional homology, a random linear chain map is
almost surely an isomorphism on homology. So a prior that prefers
structure-preserving maps is choosing among candidates that already are
structure-preserving. It cannot bind.

This is the same wall as §6.3 reached through a different door:

| setting | mechanism | consequence |
|---|---|---|
| fixed basis of 12 (§6.3) | exact cone acyclicity is **constant on hard decoded vertices**, while the soft proxy varies on mixtures | hard certificates cannot select a vertex; chance training performance remains empirical |
| learned continuous map (here) | structure preservation is **generic** in the map space | prior never binds |

The unified statement, which neither setting alone establishes but both are
consistent with: *for linear chain maps between fixed complexes, preserving
homology is generic whenever the Betti numbers match, so an objective that
rewards preservation has little or nothing to discriminate.*

## What would have to change

For a structural prior to do work, the truth needs a property that generic fits
**lack**. Structure preservation is not that property here. A setting where it
could be would need one of:

- a map space where degenerate maps are common rather than measure-zero;
- source and target whose Betti numbers differ, so a defect is forced — though
  then the defect is constant again, which is the §6.3 failure;
- a nonlinear or heavily constrained parameterisation where overfitting actually
  destroys homology.

None of these is obviously reachable from the current architecture, and this
probe does not claim any of them is impossible.

## What the probe produced that is worth keeping

- `src/homymoly/topology/defects.py`: directional chain-map defects. Per degree
  it reports what a map **destroys** (kernel of the induced homology map) and
  what it **cannot reach** (cokernel), at both chain and homology level. This is
  closer to the project's original framing than a single acyclicity bit.
  `cone_betti_from_defects` recomputes the mapping cone from the pair and matches
  `cone_betti_numbers` exactly on all 24 candidates, which makes concrete that a
  cone bundles the two directions into one number and loses their separation.
- A rank-scale fix: a purely relative rank threshold rescales to its own noise,
  so an induced map that is uniformly at float64 noise level would be reported as
  full rank, hiding a destroyed class entirely. Regression-tested.
- `build_annulus_map_system(include_cycle_killing=True)`: an opt-in 24-candidate
  class pairing each rotation with a cycle-killing twin, and a `cycle_weight`
  dial on the dataset. The v1 path is byte-identical, so the sealed campaign
  stays reproducible.

## Reproducing

The probe scripts were exploratory and are not tracked. The tracked pieces are
`src/homymoly/topology/defects.py`, `tests/test_defects.py`, and the opt-in
extensions in `src/homymoly/experiments/identifiable_maps.py`, all covered by the
test suite.
