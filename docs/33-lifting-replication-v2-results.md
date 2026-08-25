# Edge-to-cycle lifting replication v2: results

Status: **complete, sealed before execution, and independently validated**.
H1–H7 were run once against the protocol frozen in
[`docs/31`](31-independent-lifting-replication-protocol.md) and the
machine-readable seal
[`docs/32`](32-independent-lifting-replication-seal.json) under the
two-commit seal procedure of [`docs/30`](30-journal-completion-handoff.md)
§9.1. Six of the seven prespecified claims are supported; H5 is not. The frozen
protocol, seal, runner, tests, and result remain byte-for-byte unchanged; this
record and the v2 entry in [`docs/29`](29-audit-corrections.md) disclose the
audit history, including one retention gap found after execution.

This is an **untouched-seed, outcome-informed, same-generator-family
replication**. The seed block `20270101..20270136` was never instantiated
before the seal, but the hypotheses, arms, directions, and weights were chosen
after v1 outcomes and a hostile post-campaign diagnostic on the old seeds were
known. It is therefore not an independent-lab or independent-generator
replication and not a pristine preregistration. V1 and v2 estimates are never
pooled, meta-analyzed, or jointly interval-estimated; v2 stands alone.

The main result follows the framing frozen in
[`docs/30`](30-journal-completion-handoff.md) §10: `hard_cycle_ls` beat the
soft penalty, so the finding is that graph-derived cycle-subspace information
is valuable for this scarce-probe system-identification task and that an exact
classical constraint is preferable to soft shrinkage here. The soft penalty is
**not** a new superior method. Nothing in this campaign tests transfer to
unseen topologies, neural nonlinear translators, sheaves, or real data.

## Seal provenance

| item | value |
|---|---|
| protocol | `docs/31-independent-lifting-replication-protocol.md` |
| protocol SHA-256 | `6288eade4755aa188299760303b389ce13acd42659b8b9bc340cb9d4024afec0` |
| runner | `scripts/run_lifting_replication_v2.py` |
| runner SHA-256 | `48d4df774cb1e65385e5ebbd11bce9f8a0e36ee2a5d8e9dec1b18fd29a75e8d7` |
| generator SHA-256 | `c37ab1c725aa2101e88c1a0ad8fa3b279d72330feba35077e23fec930a4df69d` |
| lock SHA-256 | `05c6a5ad02db5b1651d426d157add170a8542634260ce8c265a3ee32693073bf` |
| design commit (A) | `044322c7dc6a6255eec941dbcb76c45288a9666c` |
| seal / execution commit (B) | `9baae6b8322120724e7f5aff3c47fd7ef343086c` |
| seal file SHA-256 | `6bbc77c1d0a0a47bacb40ece0a239cf2a973cec60da4b99d9c5a5bb6ddd46931` |
| result | `results/campaigns/lifting-replication-v2.json` |
| result committed at | `64a1c3bf6b5f824fb5990392f5efb08bf36559b9` |
| status | `complete` |

Commit A contains the protocol, runner, and tests. Commit B adds only the seal
record and was pushed to the private GitHub remote before the first declared
seed was instantiated, creating a remote timestamp (2026-08-24 22:31:13 -0400)
ahead of execution. The runner verified at preflight that the seal record was
committed at HEAD, that every embedded hash matched both its frozen constants
and the actual file bytes, that its own runtime SHA-256 equaled the seal's
`runner_sha256`, and that the worktree was clean (recorded `git status --short`
is empty). No design file changed between commit A and execution; the recorded
execution revision equals commit B. The canonical command was

```bash
env CUDA_VISIBLE_DEVICES=-1 .venv/bin/python \
  scripts/run_lifting_replication_v2.py \
  --output results/campaigns/lifting-replication-v2.json
```

## Environment

The recorded environment matched the frozen requirement exactly
(`matches_expected: true`): Python `3.12.3`, PyTorch base version `2.13.0`
(build string `2.13.0+cu130`), NetworkX `3.6.1`, NumPy `2.5.2`, CPU tensors in
`torch.float64`, `torch` threads fixed at `1`, and CUDA unavailable
(`CUDA_VISIBLE_DEVICES=-1`, `cuda_available: false`). Platform: Linux aarch64.
No dependency was installed or upgraded between seal and run.

## Eligibility accounting

| quantity | value |
|---|---:|
| declared seeds | 36 (`20270101..20270136`) |
| eligible seeds | **33** |
| ineligible seeds | 3: `20270103`, `20270116`, `20270120` |
| generation failures | 0 |
| disconnected topologies | 0 (the generator requires connectivity) |

Each ineligible seed generated `num_faces = 2 < 3` and was retained with its
dimensions (`20270103`: V=14, E=15, F=2; `20270116`: V=9, E=10, F=2;
`20270120`: V=9, E=10, F=2) and never replaced. The eligible count meets the
frozen minimum of 30. One eligible generator seed is the sole unit of
inference; it jointly determines topology and, through the frozen SHA-256
sub-seed schedule, the training predictors, label noise, test predictors, and
random specificity subspace. The Student-t analysis assumes these eligible
seed-level joint realizations are exchangeable.

## Primary seven-claim family

The endpoint of every claim is the paired
`log10(MSE_arm / MSE_reference)` within a seed; the estimand is its mean over
eligible seeds, the conditional synthetic-generator quantity
`E_seed[log10(MSE_arm/MSE_reference) | connected, F>=3]`. Decisions use
one-sided Student-t bounds at Bonferroni `alpha = 0.05/7` with `n = 33`,
`df = 32`, critical value `2.5912722991315227`. Negative estimates mean the
numerator arm improves recovery. Sign tests are direction-neutral sensitivity
analyses and never govern support.

| id | endpoint (numerator / reference) | estimate | SE | one-sided bound | direction / threshold | support |
|---|---|---:|---:|---:|---|---|
| h1-soft-vs-ambient-adam | `soft_boundary_lambda3 / ambient_adam` | −1.8725758600456448 | 0.23992881027454085 | upper −1.2508549802176443 | less / 0.0 | **supported** |
| h2-hard-cycle-vs-ambient-ls | `hard_cycle_ls / ambient_min_norm_ls` | −1.8768977017190436 | 0.2120168770125922 | upper −1.3275042413679385 | less / 0.0 | **supported** |
| h3-hard-cycle-vs-soft-closed-form | `hard_cycle_ls / soft_boundary_closed_form_lambda3` | −0.1273955146649585 | 0.030495144698371748 | upper −0.04837429095006027 | less / 0.0 | **supported** |
| h4-hard-cycle-vs-hard-random | `hard_cycle_ls / hard_random_subspace_ls` | −3.231929907937372 | 0.2546663947665419 | upper −2.572019933659139 | less / 0.0 | **supported** |
| h5-ridge-vs-ambient-ls | `inner_cv_ridge / ambient_min_norm_ls` | +0.015166698168512417 | 0.029731877453991792 | upper +0.09221008861621441 | less / 0.0 | **not supported** |
| h6-singular-surrogate-harm | `singular_value_surrogate / ambient_adam` | +0.19055581717427286 | 0.04042401873457078 | lower +0.0858061772078059 | greater / 0.0 | **supported** |
| h7-rtd-bounded-benefit-futility | `rtd_inspired_distance_surrogate / ambient_adam` | +0.01876868147417076 | 0.0087122550006163 | lower −0.003807143571896345 | greater / −0.045757490560675115 | **supported** |

**H1 supported.** The soft boundary-compatibility penalty at frozen weight
`3.0` improves over the graph-blind ambient Adam reference on untouched seeds
(geometric mean MSE ratio `0.01340985679332218`; descriptive two-sided 95%
interval `[−2.36129485375057, −1.3838568663407198]`; sign test 33 negative of
33, two-sided p = `2.3283064365386963e-10`). This replicates the v1
soft-penalty effect on a disjoint block.

**H2 supported.** Exact least squares restricted to `ker(B1)` improves over
the solver-matched ambient minimum-norm LS reference (geometric mean ratio
`0.013277071629811591`; descriptive 95% interval
`[−2.3087619478822723, −1.4450334555558149]`; sign test 33 of 33).

**H3 supported.** Hard cycle-subspace LS improves over the closed-form
solution of the same frozen soft objective, with optimizer confounding removed
by construction (geometric mean ratio `0.745769272372031`; descriptive 95%
interval `[−0.18951209171471245, −0.06527893761520454]`; sign test 28
negative, 3 positive, 2 ties discarded, p = `4.649162292480469e-06`). This is
the central comparison: under the frozen design the exact classical constraint
is preferable to soft shrinkage of the same information.

**H4 supported.** The true cycle subspace improves over the dimension-matched
seeded random subspace (geometric mean ratio `0.0005862327707089732`;
descriptive 95% interval `[−3.7506683788961146, −2.7131914369786294]`; sign
test 33 of 33). The advantage is specific to the cycle subspace, not to
dimension reduction alone.

**H5 not supported.** Training-only four-fold-selected ridge did not improve
over ambient minimum-norm LS: the governing one-sided upper bound
`+0.09221008861621441` is not below zero (geometric mean ratio
`1.0355395685149351`; descriptive 95% interval
`[−0.04539515438119311, +0.07572855071821795]`; sign test 19 positive, 14
negative, p = `0.48685024166479707`). Generic shrinkage selected without
held-out data shows no detected benefit here. The claim is reported as frozen
and is not reinterpreted.

**H6 supported.** The singular-value surrogate at its frozen weight `0.01`
harms versus ambient Adam: the one-sided lower bound `+0.0858061772078059`
exceeds zero (geometric mean ratio `1.5508000886966786`; descriptive 95%
interval `[+0.10821478553716982, +0.2728968488113759]`; sign test 29 positive,
4 negative, p = `1.0928604751825333e-05`). This replicates the v1 harm
direction. The claim is scoped to the exact implemented formula, not
mapping-cone homology.

**H7 supported as a bounded-benefit/futility statement.** The one-sided lower
bound `−0.003807143571896345` exceeds the prespecified margin
`log10(0.90) = −0.045757490560675115`, ruling out an RTD-inspired benefit of
10% or more in geometric-mean held-out MSE versus ambient Adam (estimate
`+0.01876868147417076`; descriptive 95% interval
`[+0.0010223987666884045, +0.03651496418165312]`). This is **not** a
noninferiority or equivalence conclusion: the direction and estimand semantics
are reversed relative to those procedures, and the result cannot establish
equality or the absence of every benefit. The claim is scoped to this
target-misaligned surrogate, not published RTD/SRTD.

## Descriptive optimizer diagnostics

These are prespecified descriptive audits outside the seven-claim family; they
carry no support decision.

- `ambient_adam / ambient_min_norm_ls`: mean log10 ratio
  `+0.22888064836548128` (median `+0.2811655771640295`). The finite-step Adam
  reference is measurably worse than the closed-form minimum-norm solution of
  the same unpenalized objective; exact sign test 25 positive, 8 negative,
  two-sided p = `0.004551384132355452`. This gap is why the soft penalty's
  historical reference is Adam while the classical estimators are referenced
  to the same-solver LS solution.
- `soft_boundary_lambda3 / soft_boundary_closed_form_lambda3`: mean log10
  ratio `+0.10580697537392178`; per-seed solution gap
  `||A_adam − A_closed_form||_F` mean `0.8556475807071382` (maximum
  `4.005285286298402`). Finite-step Adam does not fully reach the closed-form
  optimum of the penalized objective, so part of the H1 margin is optimizer
  slack rather than the penalty geometry; H3 removes this confound.
- Final full-batch gradient norms of each Adam arm remain small but nonzero
  (maxima: ambient `0.02345710523737225`, soft `0.03134258426804703`,
  singular-value `0.016266304990303224`, RTD-inspired `0.016048508983914404`).

## C1 — off-path association (prespecified secondary, no decision)

C1 replaces the v1 regularization-path correlation with an off-path design:
per eligible topology, 12 independent training-input/label-noise replicates of
`ambient_min_norm_ls` share one independently generated 3,072-row noiseless
test set; replicate 0 reuses the primary training realization. Conventional
Pearson `r` between log10 defect and log10 held-out MSE is clipped just inside
`(−1, 1)`, Fisher-transformed, and aggregated across the 33 topology seeds
with unadjusted two-sided 95% Student-t intervals (`t_(0.975,32) =
2.036933343460102`). C1 is outside the Bonferroni family and receives **no
support decision**.

| quantity | mean Fisher-z | 95% interval (z) | back-transformed mean r | back-transformed interval |
|---|---:|---:|---:|---:|
| cycle-projector defect | +0.7712258159811779 | [+0.43398720808657104, +1.1084644238757848] | +0.6476416755579272 | [+0.4086480828165953, +0.8035189197230416] |
| matched random-subspace defect | −0.3168499119888693 | [−0.5622502040999926, −0.07144961987774598] | −0.30665582410031944 | [−0.5096450828921225, −0.07132828305985571] |
| paired delta-z (cycle − random) | +1.0880757279700473 | [+0.8873984853168706, +1.288752970623224] | — | — |

The cycle-projector defect covaries positively with held-out error across
independent training/noise realizations, and the paired delta-z interval lies
above zero, while the dimension-matched random-projector defect does not show
that positive association. Two positive correlations alone would not establish
specificity; the paired `delta_z = z_cycle − z_random` is the relevant
descriptive contrast. This remains association, not causation: the analysis
does not establish that either defect is independently calibrated, causal, or
useful for routing, and it says nothing about real data. The legacy raw
`boundary_compatibility_defect_frobenius` is retained per replicate as a
descriptive diagnostic only and enters no correlation.

## Generator-cycle-basis oracle

`generator_cycle_basis_oracle` sets `A = B2` from the withheld generator
basis. Its held-out MSE and its error relative to the mean squared test target
are exactly `0.0` on all 33 eligible seeds (minimum, median, and maximum all
`0.0`). It is a deterministic attainability ceiling and numerical-integrity
check: it shows how much generator knowledge the learning problem withholds,
is isolated from every fitted arm and inferential table, and no log ratio is
formed for it. It is not evidence that any learned method discovered the
generator basis.

## Independent validation record

After execution and before this record was written, every primary summary was
recomputed independently from the retained raw rows: all seven estimates,
standard errors, one-sided Bonferroni bounds, critical value, support flags,
geometric mean ratios, descriptive intervals, and sign-test counts; the
descriptive optimizer means; all three C1 Fisher-z summaries, intervals, and
back-transformations; and the audit block's per-claim means. Every check
passed; the maximum absolute discrepancy between recomputation and the
serialized values ranged from `0.0` to at most `1e-13` (summation-order
roundoff). The validator also verified the exact eligible/ineligible
accounting above, the retained SHA-256-derived sub-seeds against the frozen
derivation, the cycle-nullspace and random-basis certificates (orthonormality
and membership residuals at the frozen `1e-10` tolerance, observed rank equal
to `V − 1`, nonzero random-basis diagonal), the `inner_cv_ridge` fold-loss
recomputation and selected alpha on every seed, finite strictly positive MSEs
entering every ratio, paired row identity, the oracle's zero error, and the
audit block's declared recompute scope. No failure rows exist to preserve.

An independent re-derivation performed while writing this record reproduced
every estimate, standard error, bound, interval, support flag, and C1 value
from the raw rows with a maximum absolute discrepancy of `4.45e-16`.

## Known protocol deviations

One retention gap, discovered by post-campaign validation: protocol
[`docs/31`](31-independent-lifting-replication-protocol.md) §7.3 lists a
per-seed closed-form stationarity residual
`||M A_soft_closed − X_train.T Y_train||_F <= 1e-10 * ||X_train.T Y_train||_F`
among the quantities the runner asserts and retains, but the executed runner
did not retain or assert that per-seed residual for
`soft_boundary_closed_form_lambda3` (the retained metadata carry the
pseudoinverse effective rank, minimum singular value, and rank cutoff instead).
The closed-form normal-equation residual was test-verified below `1e-10`
pre-seal on hand fixtures and historical seed `20261001`, and the H3 endpoint
depends only on the retained per-arm held-out MSEs, so no inferential value is
affected. The gap is recorded transparently rather than repaired by rerunning:
the seed block is sealed and consumed, and rerunning a sealed block over a
retention gap is forbidden. No other deviation from the frozen protocol or
seal is known; eligibility, stop rules, and the no-preview declaration were
honored as written.

## Boundary

- One synthetic generator family, 33 eligible seed-level replicates, one
  training-set size (`N_train = 16`), one noise level (`sigma = 0.02`).
  Inference assumes seed-level exchangeability and quantifies Monte Carlo
  variation over the frozen seed mechanism, not uncertainty for real data.
- Outcome-informed design: arms, directions, and weights were chosen with v1
  outcomes and a hostile old-seed diagnostic known. The untouched block
  removes selection on v2 outcomes, not this design history.
- Each topology fits a separate matrix; there is no shared model and no
  unseen-topology generalization. No claim extends to unseen generator
  families, neural nonlinear translators, sheaves, mapping-cone homology,
  published RTD/SRTD, or real data.
- H7 is a bounded-benefit/futility result at one prespecified margin. It is
  not noninferiority, not equivalence, and not evidence of exact inertness.
- The networkx cycle basis is noncanonical; only projectors, constrained
  predictions, and inferential endpoints are basis-invariant.
- The supported H3 contrast favors an exact classical constraint over soft
  shrinkage of the same `B1`-derived information in this setting; it does not
  make constrained least squares a novel algorithm, and the soft penalty must
  not be presented as a superior method.
- V1 and v2 are never pooled. The v1 record
  ([`docs/28`](28-conversion-campaign-results.md)) remains the account of the
  historical campaign with its disclosed deviation; this document supersedes
  no part of it.
