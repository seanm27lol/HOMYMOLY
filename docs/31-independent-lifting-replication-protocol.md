# Frozen-design protocol: independent edge-to-cycle lifting replication v2

Status: **complete and freeze-ready, but not yet sealed or executable**. This
document becomes immutable only when the implementation, tests, and design-seal
record described in §13 have been committed. Until then, the declared seeds must
not be instantiated. After sealing, any substantive change creates a new
campaign with a new untouched seed block; it must not be called v2.

Protocol date: 2026-08-24.

## 1. No-preview declaration and seal

The v2 generator seeds are exactly:

```text
20270101, 20270102, 20270103, 20270104, 20270105, 20270106,
20270107, 20270108, 20270109, 20270110, 20270111, 20270112,
20270113, 20270114, 20270115, 20270116, 20270117, 20270118,
20270119, 20270120, 20270121, 20270122, 20270123, 20270124,
20270125, 20270126, 20270127, 20270128, 20270129, 20270130,
20270131, 20270132, 20270133, 20270134, 20270135, 20270136
```

At the time this protocol was drafted, this block was absent from Git history
and no project run had instantiated or previewed any member. In particular, no
one has generated a topology, counted faces, printed dimensions, performed a
smoke test, tuned a method, or examined an outcome using a seed in the block.

The no-preview rule remains in force until all of the following are committed:

1. this complete protocol;
2. the dedicated v2 runner and all its tests;
3. the exact generator, dependency-lock, protocol, and runner hashes;
4. all estimands, decisions, stop conditions, and failure rules; and
5. the machine-readable design-seal record
   `docs/32-independent-lifting-replication-seal.json` (§1.1), identifying the
   committed implementation revision.

Tests may use hand-built fixtures or historical seed `20261001`; they may not
parameterize over, import, derive data from, or otherwise touch the sealed block.
Source-code constants containing the declared integers are permitted. Merely
constructing `ConversionDataset(1, seed=s)` for a declared seed counts as a
preview even if the resulting sample is discarded.

If any declared seed is instantiated before the design seal is committed, the
entire 36-seed block is void. It will not be partially salvaged. A new
consecutive block must be chosen, verified absent from history, and declared in
a newly committed protocol before any execution.

### 1.1 The design-seal record

The design seal is the machine-readable file
`docs/32-independent-lifting-replication-seal.json`, committed as its own
commit immediately after the design commit. Its contents are frozen:

- `schema`: the literal tag `homymoly-lifting-replication-seal/1`;
- `design_commit`: the full hash of commit A, the commit containing this
  protocol, the runner, and the tests;
- `protocol_sha256`, `runner_sha256`, `generator_sha256`, and `lock_sha256`:
  the SHA-256 of this protocol, `scripts/run_lifting_replication_v2.py`,
  `src/homymoly/data/conversion.py`, and `uv.lock`;
- `seed_interval`: `{first: 20270101, last: 20270136}`;
- `no_preview_declaration`: the renewed attestation that no seed in the
  interval has been instantiated or previewed;
- `primary_family`: the complete seven-claim family of §8 as seven claim
  objects;
- `stop_rules`: every stop condition of §11; and
- `output_path`: exactly `results/campaigns/lifting-replication-v2.json`.

The runner accepts a `--seal` argument defaulting to that path. It parses and
validates this machine-readable file rather than trusting a naked command-line
hash; there is no `--expected-runner-sha256` flag. Validation requires the
schema tag and every field above, equality of each embedded hash with both the
runner's frozen constants and the actual file bytes, equality of the runner's
own runtime SHA-256 with `runner_sha256`, the seal file's presence in the HEAD
commit, a clean worktree, and equality of `output_path` with the `--output`
argument. The runner records `design_commit` (commit A), the HEAD revision at
execution time as the execution revision (commit B), and `sys.argv`.

## 2. Question, prior evidence, and scope

The task is a linear lifting on one synthetic graph-generator family. For each
connected graph, let

- `B1 in R^(V x E)` be its oriented vertex-edge incidence matrix;
- `B2 in R^(E x F)` be the generator's noncanonical NetworkX cycle-basis
  matrix, with `B1 B2 = 0`;
- `X in R^(N x E)` contain Gaussian edge probes; and
- `Y = X B2` be the noiseless target in cycle-basis coordinates.

The fitted operator is `A = W.T in R^(E x F)`, and predictions are `X A`.
`B2` is not supplied to any fitted estimator. It is used only to generate noisy
training responses, noiseless held-out responses, and the explicitly labelled
oracle ceiling. Consequently, the training responses encode the target
coordinate system even though the operator itself is withheld.

This firewall is enforced in code, not by convention. The fitting APIs for the
ambient, ridge, soft, hard-cycle, hard-random, singular-value, and RTD-inspired
arms receive only their declared training tensors and, where applicable, `B1`
or the seeded random basis. They may never receive `B2`, face-cycle metadata,
held-out targets, or held-out losses. `B2` is used outside those APIs only to
form response tensors, to evaluate already-fitted matrices, and to compute the
explicitly segregated truth-access oracle.

Corrected v1 showed that a soft penalty based on the observable `B1` improved
over a graph-blind Adam baseline on the historical seeds. That comparison did
not include a graph-aware hard cycle-space estimator or a solver-matched
least-squares baseline. V2 has three purposes:

1. replicate the v1 soft-penalty effect on an untouched seed block;
2. compare it with exact classical cycle-space, random-subspace, minimum-norm,
   and ridge estimators; and
3. replace v1's regularization-path correlation with an off-path association
   analysis driven by independent training/noise replicates.

This remains a same-generator-family synthetic replication. It cannot establish
a general representation-switching architecture, a typed chain map, sequence
exactness, mapping-cone homology or published RTD/SRTD efficacy, transfer to
unseen topologies by one shared model, sheaf or derived-category results, or
performance on real data.

## 3. Frozen software and environment

Known immutable inputs at protocol drafting are:

| object | path | SHA-256 |
|---|---|---|
| conversion generator | `src/homymoly/data/conversion.py` | `c37ab1c725aa2101e88c1a0ad8fa3b279d72330feba35077e23fec930a4df69d` |
| dependency lock | `uv.lock` | `05c6a5ad02db5b1651d426d157add170a8542634260ce8c265a3ee32693073bf` |
| historical v1 protocol, for lineage only | `docs/27-conversion-campaign-protocol.md` | `503cc282f40d118ba1739c2afe1bfc77eaf2b1733baaddb91c0c3363e75ae2b8` |

The v2 runner will be `scripts/run_lifting_replication_v2.py`. Its SHA-256 is the
only unavailable fingerprint at protocol-draft time because the file has not
yet been finalized: **`PENDING-DESIGN-SEAL`**. It must appear as an actual
64-hex fingerprint in the `runner_sha256` field of the committed design-seal
record (§1.1) before any declared seed is instantiated. The final protocol hash
is likewise recorded externally in that same record rather than inside this
file, and the runner never embeds its own hash: it computes its runtime SHA-256
and requires equality with the seal's `runner_sha256`. This avoids a self-hash
cycle.

The runner must fail before dataset construction unless the hashes above and
the design-seal fingerprints match. It must also require this lock-derived
environment:

| component | required value |
|---|---|
| Python | `3.12.3` |
| PyTorch base version | `2.13.0` |
| NetworkX | `3.6.1` |
| NumPy | `2.5.2` |
| tensor device and dtype | CPU, `torch.float64` |
| PyTorch threads | `1` |

The runner records the full PyTorch build string, operating system, machine,
processor, CPU/thread settings, `CUDA_VISIBLE_DEVICES`, Git revision, initial
`git status --short`, every verified hash, and the canonical command. CUDA must
be unavailable or hidden. No dependency may be installed or upgraded between
the seal and run.

## 4. Sampling, eligibility, and inference unit

For every declared integer `s`, and only after the seal, construct exactly
`ConversionDataset(1, seed=s, dtype=torch.float64)[0]` with the default
`ConversionConfig`.

- A generated topology is eligible if and only if `sample.num_faces >= 3`.
- The generator already requires connected graphs; a disconnected result or
  generation exception is a campaign failure, not an exclusion.
- Ineligible topologies are retained in the result with their seed, dimensions,
  and reason. They are never replaced.
- All 36 declared seeds are assessed. There is no optional stopping, interim
  analysis, or sample-size extension.
- If fewer than 30 seeds are eligible, stop without fitting and report
  `design_failure_insufficient_eligible`.

One eligible generator seed is the sole unit of inference. It jointly determines
the topology and, through the sub-seed schedule below, its training predictors,
label noise, test predictors, and random specificity subspace. Paired contrasts
share these quantities within a seed.

Student-t inference assumes that eligible seed-level joint realizations are
exchangeable draws from this deterministic generator-and-sub-seed scheme. It
does not separate topological heterogeneity from predictor or noise-realization
heterogeneity, and it conditions on the frozen face-count eligibility rule.

## 5. Deterministic random streams

Apart from `ConversionDataset`'s already frozen internal seed derivation, every
stochastic component uses this exact function:

```text
message = f"homymoly-lifting-v2:{topology_seed}:{component}:{replicate}"
digest = SHA256(message encoded as UTF-8)
raw = unsigned big-endian integer represented by digest[0:8]
subseed = raw & ((1 << 63) - 1)
```

The literal domain prefix is `homymoly-lifting-v2:`; the three separators are
literal ASCII colons. Integers use base-10 without padding, `component` is a
nonempty colon-free label from the table below, and the digest slice is the
first eight bytes (Python `digest[:8]`). UTF-8 and ASCII coincide for every
allowed message. The high-bit mask yields a nonnegative 63-bit PyTorch seed.
The subseed initializes a fresh CPU `torch.Generator`; no global or reused
generator is allowed.

| component | replicate field | generated object |
|---|---:|---|
| `primary-train-inputs` | `0` | shared `16 x E` primary training matrix |
| `primary-training-noise` | `0` | shared `16 x F` Gaussian training noise |
| `primary-test-inputs` | `0` | shared `3072 x E` primary held-out matrix |
| `matched-random-subspace` | `0` | shared `E x F` Gaussian matrix used to construct `Q_random` |
| `c1-test-inputs` | `0` | C1's independently generated shared `3072 x E` test matrix |
| `c1-train-inputs` | `1` through `11` | C1 replicate-specific `16 x E` training matrix |
| `c1-training-noise` | `1` through `11` | C1 replicate-specific `16 x F` label noise |

Every matrix draw uses `torch.randn(..., dtype=torch.float64,
generator=fresh_generator)`. The component-separated streams make draw order
irrelevant. The result retains every derived 63-bit subseed and the literal
derivation version string.

## 6. Frozen data and optimization constants

For each eligible topology, primary arms share:

- `N_train = 16` and `N_test = 3072`;
- `X_train` and `X_test` with independent standard-normal entries;
- `Y_train = X_train B2 + epsilon`, where epsilon has independent
  `N(0, 0.02^2)` entries;
- noiseless `Y_test = X_test B2`;
- one separately fitted matrix per topology, with no cross-topology training.

All Adam arms use `W = 0`, learning rate `0.05`, exactly `2500` full-batch
steps, no weight decay, and PyTorch `torch.optim.Adam` defaults for every
unspecified option. The supervised term is

```text
mean((X_train @ W.T - Y_train)^2)
```

where the mean is over all 16-by-`F` elements. No intercept or feature
standardization is used. The test endpoint for every fitted arm is

```text
MSE = mean((X_test @ A - Y_test)^2)
```

with `A = W.T`. Primary test data may not be used for fitting, tuning, stopping,
model selection, or debugging.

The common minimum-norm least-squares primitive used by all unregularized LS
arms is exactly

```text
LS(U, V) = torch.linalg.lstsq(U, V, driver="gelsd", rcond=1e-12).solution
```

This solver and its arguments may not vary by arm. Every call to this primitive
records the returned gelsd numerical rank and the smallest returned singular
value. All returned singular values must be finite and strictly positive and
the returned rank must be at least one; a violation is a campaign failure
(`design_failure`).

## 7. Frozen arms

### 7.1 Graph-blind reference estimators

`ambient_adam` is the historical v1 reference: fit a free `F x E` matrix `W`
with the supervised Adam objective in §6 and no structural term.

`ambient_min_norm_ls` is the solver-matched classical reference:

```text
A_ambient_ls = LS(X_train, Y_train)             # shape E x F
```

Both references are required. Adam is the reference for replication of the
historical learned objectives; minimum-norm LS is the reference for classical
closed-form estimators. Their paired log-MSE ratio is descriptive and is not a
member of the confirmatory family.

### 7.2 Soft boundary compatibility

`soft_boundary_lambda3` uses Adam and the frozen v1 executed objective:

```text
supervised_MSE + 3.0 * mean((B1 @ W.T)^2)
```

This is an elementwise mean-square boundary-compatibility penalty, not the
unnormalized squared Frobenius norm written in the historical v1 protocol and
not sequence exactness.

### 7.3 Solver-matched soft boundary compatibility

`soft_boundary_closed_form_lambda3` minimizes the same mean-normalized
objective as `soft_boundary_lambda3`, but removes finite-step optimizer error.
Writing `A = W.T`, the stationarity equation is

```text
(X_train.T @ X_train + 3.0 * N_train / V * B1.T @ B1) @ A
    = X_train.T @ Y_train.
```

The potentially singular system is solved by

```text
M = X_train.T @ X_train + 3.0 * N_train / V * B1.T @ B1
A_soft_closed = torch.linalg.pinv(M, rcond=1e-12) \
                @ X_train.T @ Y_train
```

The factor `N_train / V` follows exactly from differentiating the supervised
mean over `N_train * F` entries and the compatibility mean over `V * F`
entries. The runner records the pseudoinverse effective rank — the number of
singular values of `M` exceeding the frozen `rcond` cutoff — and asserts the
frozen stationarity residual

```text
||M @ A_soft_closed - X_train.T @ Y_train||_F
    <= 1e-10 * ||X_train.T @ Y_train||_F.
```

The normal equation is consistent by construction, so a violation indicates an
implementation fault and is a whole-design failure (`design_failure`).
`soft_boundary_lambda3 / soft_boundary_closed_form_lambda3` is retained as a
descriptive optimizer audit and has no confirmatory decision.

### 7.4 Exact hard cycle-space least squares

`hard_cycle_ls` receives `B1` but not `B2`. For a connected eligible graph,
`rank(B1) = V - 1`. Compute the full float64 CPU SVD

```text
U, S, Vh = torch.linalg.svd(B1, full_matrices=True)
Q_cycle = Vh[V - 1:, :].T
```

Independently compute the rank tolerance
`tau = max(V,E) * eps_float64 * max(S)` and assert that the observed rank — the
number of singular values exceeding `tau` — is exactly `V - 1`. Assert that
`Q_cycle` has shape `E x F`, that `||Q_cycle.T @ Q_cycle - I_F||_F <= 1e-10`,
and that `||B1 @ Q_cycle||_F <= 1e-10`. These are hard assertions, not merely
recorded diagnostics: any violation is a campaign failure (`design_failure`).
Record the observed rank, the rank tolerance `tau`, and both Frobenius
residuals. For stable serialization, make each column's first
maximum-absolute entry positive; this sign choice does not make the basis
canonical.

Fit and reconstruct:

```text
Z_cycle = X_train @ Q_cycle
C_cycle = LS(Z_cycle, Y_train)
A_cycle = Q_cycle @ C_cycle
```

The SVD basis itself is not canonical and may rotate within the nullspace. Its
projector, constrained least-squares predictions, and inferential endpoint are
basis-invariant; no claim is made about individual basis columns.

### 7.5 Dimension-matched random-subspace least squares

`hard_random_subspace_ls` is a specificity control that uses neither `B1` nor
`B2` to orient its subspace. Draw `G in R^(E x F)` from the frozen
`matched-random-subspace` stream and compute reduced QR:

```text
Q_raw, R = torch.linalg.qr(G, mode="reduced")
sign_j = +1 if R[j,j] >= 0 else -1
Q_random[:,j] = sign_j * Q_raw[:,j]
Z_random = X_train @ Q_random
C_random = LS(Z_random, Y_train)
A_random = Q_random @ C_random
```

An exactly zero diagonal uses sign `+1`. Assert that every diagonal entry of
`R` is finite and nonzero, recording `min |diag R|`, and that
`||Q_random.T @ Q_random - I_F||_F <= 1e-10`; any violation is a campaign
failure (`design_failure`). `Q_random` has the same `E x F` shape and hence
the same fitted `F x F` coefficient count as `hard_cycle_ls`.

### 7.6 Training-only selected ridge

`inner_cv_ridge` is graph-blind. Its candidate penalties are exactly

```text
1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100
```

Use exact deterministic four-fold cross-validation. For fold `k` in
`{0, 1, 2, 3}`, rows whose zero-based index satisfies `i mod 4 == k` are its
four validation rows and the other 12 rows are its fit rows. For each candidate
alpha and fold, minimize the conventional unnormalized objective
`SSE + alpha * ||A||_F^2` without an intercept:

```text
A_alpha,k = solve(X_fit,k.T @ X_fit,k + alpha * I_E,
                  X_fit,k.T @ Y_fit,k)
fold_MSE_alpha,k = mean((X_validation,k @ A_alpha,k - Y_validation,k)^2)
score_alpha = mean(fold_MSE_alpha,0, ..., fold_MSE_alpha,3)
```

Select the smallest mean score; an exact floating-point tie selects the smaller
alpha. Retain all 36 fold losses and all nine mean scores. Refit on all 16
training rows using the selected alpha and the same `SSE + alpha ||A||_F^2`
formula, then evaluate once on the primary held-out set. No v2 held-out value,
`B1`, `B2`, or across-seed result enters selection.

### 7.7 Historical singular-value surrogate

`singular_value_surrogate` uses Adam with the objective

```text
supervised_MSE + 0.01 * exp(-2 * sigma_min(W))
```

This is the historical `cone` arm. It is a singular-value anti-collapse
surrogate, not a computation of mapping-cone homology, acyclicity, or
quasi-isomorphism.

### 7.8 Historical RTD-inspired distance surrogate

`rtd_inspired_distance_surrogate` uses Adam with weight `0.1`. On every step,
including the diagonal entries, define

```text
D_source = torch.cdist(X_train, X_train)
D_mapped = torch.cdist(X_train @ W.T, X_train @ W.T)
R = mean((D_mapped / (mean(D_mapped) + 1e-12)
          - D_source / (mean(D_source) + 1e-12))^2)
objective = supervised_MSE + 0.1 * R
```

This is not published RTD or SRTD and is target-misaligned because it asks the
output coordinates to preserve input-example geometry.

### 7.9 Generator-cycle-basis oracle

`generator_cycle_basis_oracle` sets `A_oracle = B2` and evaluates it without
training. It uses the withheld target operator and is therefore only a
deterministic attainability ceiling and numerical-integrity check. Record both
its raw MSE and its relative error `MSE_oracle / mean(Y_test^2)`. Its error is
reported descriptively, is expected to be zero up to roundoff, and enters no
ratio, hypothesis, tuning rule, or inferential family; no log ratio is ever
formed for it, and it remains isolated from every fitted arm.

## 8. Primary endpoint, seven-claim family, and decisions

For each comparison and each eligible seed, define

```text
d_s(arm, reference) = log10(MSE_arm_s / MSE_reference_s)
```

All constituent MSEs must be finite and strictly positive. The estimand is the
arithmetic mean of `d_s` across eligible seeds. With `n` eligible seeds, let
`SE = sample_standard_deviation(d) / sqrt(n)`.

The inferential target is the conditional synthetic-generator quantity
`E_seed[log10(MSE_arm/MSE_reference) | connected, F>=3]`; the sample mean
`mean(d)` estimates it with one eligible seed as the draw. Intervals quantify
Monte Carlo variation over that seed mechanism, not uncertainty for real data.
V1 and v2 estimates are never pooled, meta-analyzed, or jointly
interval-estimated; v2 stands alone.

The following seven and only these seven claims form the primary confirmatory
family:

| id | paired contrast | prediction and governing decision |
|---|---|---|
| H1 | `soft_boundary_lambda3 / ambient_adam` | improvement; upper one-sided bound `< 0` |
| H2 | `hard_cycle_ls / ambient_min_norm_ls` | improvement; upper one-sided bound `< 0` |
| H3 | `hard_cycle_ls / soft_boundary_closed_form_lambda3` | hard constraint improves; upper one-sided bound `< 0` |
| H4 | `hard_cycle_ls / hard_random_subspace_ls` | cycle specificity; upper one-sided bound `< 0` |
| H5 | `inner_cv_ridge / ambient_min_norm_ls` | ridge improves; upper one-sided bound `< 0` |
| H6 | `singular_value_surrogate / ambient_adam` | replicated harm; lower one-sided bound `> 0` |
| H7 | `rtd_inspired_distance_surrogate / ambient_adam` | lower one-sided bound `> log10(0.90)` rules out a benefit of 10% or more |

H7 is a bounded-benefit/futility statement at one prespecified geometric-mean
MSE-ratio margin. It is not noninferiority, not an equivalence test, and cannot
establish equality or absence of every benefit.

With `theta` denoting each claim's estimand `E_seed[d_s]`, the formal
hypotheses are fixed as:

- H1: `theta = E_seed[d_s(soft_boundary_lambda3, ambient_adam)]`; H0:
  `theta >= 0`; H_A: `theta < 0`; supported iff the one-sided upper bound is
  below `0`.
- H2: `theta = E_seed[d_s(hard_cycle_ls, ambient_min_norm_ls)]`; H0:
  `theta >= 0`; H_A: `theta < 0`; supported iff the one-sided upper bound is
  below `0`.
- H3: `theta = E_seed[d_s(hard_cycle_ls, soft_boundary_closed_form_lambda3)]`;
  H0: `theta >= 0`; H_A: `theta < 0`; supported iff the one-sided upper bound
  is below `0`.
- H4: `theta = E_seed[d_s(hard_cycle_ls, hard_random_subspace_ls)]`; H0:
  `theta >= 0`; H_A: `theta < 0`; supported iff the one-sided upper bound is
  below `0`.
- H5: `theta = E_seed[d_s(inner_cv_ridge, ambient_min_norm_ls)]`; H0:
  `theta >= 0`; H_A: `theta < 0`; supported iff the one-sided upper bound is
  below `0`.
- H6: `theta = E_seed[d_s(singular_value_surrogate, ambient_adam)]`; H0:
  `theta <= 0`; H_A: `theta > 0`; supported iff the one-sided lower bound
  exceeds `0`.
- H7: `theta = E_seed[d_s(rtd_inspired_distance_surrogate, ambient_adam)]`;
  H0: `theta <= log10(0.90)`; H_A: `theta > log10(0.90)`; H0 is rejected —
  ruling out a benefit of 10% or more — when the one-sided lower bound exceeds
  `-0.045757490560675115`.

Control family-wise type-I error at `0.05` by Bonferroni over seven claims:

```text
alpha_per_claim = 0.05 / 7
q = 1 - alpha_per_claim = 0.9928571428571429
upper = mean(d) + t_q,n-1 * SE
lower = mean(d) - t_q,n-1 * SE
log10(0.90) = -0.045757490560675115
```

The hard-coded one-sided critical values, computed once with
`scipy.stats.t.ppf` for design only, are:

| eligible n | df | `t_(0.9928571428571429,df)` |
|---:|---:|---:|
| 30 | 29 | `2.606750672048818` |
| 31 | 30 | `2.601227904110613` |
| 32 | 31 | `2.5960807947257787` |
| 33 | 32 | `2.5912722991315227` |
| 34 | 33 | `2.586770085672467` |
| 35 | 34 | `2.5825458097369376` |
| 36 | 35 | `2.5785745178415116` |

The runner must contain and test this table; SciPy is not a runtime dependency.
The direction-specific bound in the H1--H7 table is the sole support rule. Each
of the seven summaries records: the estimate `mean(d)`; its standard error; the
geometric mean ratio `10^mean(d)`; the unadjusted two-sided 95% Student-t
interval, explicitly marked descriptive; the governing one-sided Bonferroni
bound; the critical value; the direction; the threshold; the support decision;
all raw paired values; and an exact two-sided sign test. Sign tests are
direction-neutral sensitivity analyses and never govern support. No hypothesis
may be added, dropped, reversed, or moved between families after execution.

The following are descriptive only: `ambient_adam / ambient_min_norm_ls`;
`soft_boundary_lambda3 / soft_boundary_closed_form_lambda3`; the per-seed
solution gap `||A_adam - A_closed_form||_F` between those two soft fits and the
final full-objective gradient norm of each Adam-fitted arm; every arm's
absolute MSE; dimensions and parameter counts; boundary defects; the selected
penalties and fold scores of `inner_cv_ridge`; oracle error; runtimes; and all
non-primary pairings.

## 9. Off-path C1 association analysis

C1 is a prespecified **secondary, descriptive** analysis, not an eighth member
of the seven-claim efficacy family. It has no confirmatory support decision and
tests association, not causation.

For each eligible topology, evaluate 12 independent training/noise realizations
of `ambient_min_norm_ls`. Replicate `0` reuses the already fitted primary
`ambient_min_norm_ls`, including its `primary-train-inputs` and
`primary-training-noise` streams. Replicates `1` through `11` use the
corresponding `c1-train-inputs` and `c1-training-noise` streams and

```text
Y_train_r = X_train_r @ B2 + epsilon_r.
A_r = LS(X_train_r, Y_train_r)
```

All 12 fits share one independently generated `c1-test-inputs` matrix and its
noiseless target. They do not use the primary test matrix. Closed-form
minimum-norm LS removes finite-step optimizer error. For each fitted `A_r`,
retain:

```text
error_r = mean((X_c1_test @ A_r - Y_c1_test)^2)
cycle_projector_defect_r = norm((I_E - Q_cycle @ Q_cycle.T) @ A_r,
                                ord="fro")
random_specificity_defect_r = norm((I_E - Q_random @ Q_random.T) @ A_r,
                                   ord="fro")
boundary_defect_r = norm(B1 @ A_r, ord="fro")
```

`boundary_defect_r` is the raw legacy descriptive diagnostic, serialized as
`boundary_compatibility_defect_frobenius`; it enters no correlation and no
decision. `Q_random` is the same topology-specific matched random subspace
frozen in §7.5. Every defect, including the raw boundary defect, and every
error must be finite and strictly positive; otherwise C1 is undefined and the
campaign fails with `design_failure` rather than adding an epsilon after
outcomes.

Within each topology, compute conventional Pearson `r_cycle` between
`log10(cycle_projector_defect_r)` and `log10(error_r)` across the 12 independent
replicates. Compute `r_random` analogously. The projectors make this a
dimension-matched cycle-versus-random specificity analysis. Use centered sums
directly, not the biased v1 standardized-product formula.

Before Fisher transformation, clamp only exact or roundoff-exceeding endpoints
to `[-nextafter(1,0), +nextafter(1,0)]`, then compute `z = atanh(r)`. Separately
summarize seed-level `z_cycle`, `z_random`, and their paired difference
`delta_z = z_cycle - z_random`; report back-transformed mean correlations for
the first two with `tanh`. Each unadjusted two-sided 95% Student-t interval is

```text
mean(value) +/- t_(0.975,n-1) * sd(value) / sqrt(n).
```

The frozen critical values are:

| eligible n | df | `t_(0.975,df)` |
|---:|---:|---:|
| 30 | 29 | `2.0452296421327034` |
| 31 | 30 | `2.0422724563012378` |
| 32 | 31 | `2.039513446396408` |
| 33 | 32 | `2.036933343460102` |
| 34 | 33 | `2.0345152974493383` |
| 35 | 34 | `2.0322445093177186` |
| 36 | 35 | `2.030107928250343` |

All three intervals are descriptive and govern no support decision. Two
positive correlations alone do not establish specificity; the paired
`delta_z = z_cycle - z_random` is the relevant descriptive contrast. Report
every 12-fit row, both correlations, both Fisher-z values, paired `delta_z`, and
all intervals on their stated scale. Do not pool the 12 fits as independent
topologies. No causal, confirmatory, or calibration claim is permitted. If any
defect or error is nonpositive/nonfinite, a correlation is undefined, or the
12-point input is constant, the whole design fails rather than dropping that
seed or modifying the endpoint.

H5 routing from v1 is absent. Its historical endpoint was algebraically
non-informative and is not repaired or replaced here.

## 10. Numerical validation and tolerances

All scientific endpoints are computed from retained raw float64 rows by a
separate aggregation function that is unit-tested against hand calculations.
The runner must retain, at minimum:

- every declared and eligible seed and all topology dimensions;
- all derived sub-seeds;
- every primary-arm MSE and structural diagnostic;
- all candidate scores and the selected alpha of `inner_cv_ridge`;
- cycle and random basis orthogonality and membership residuals, the
  cycle-basis observed rank and rank tolerance, and the random-basis
  `min |diag R|`;
- every returned numerical rank: the gelsd rank and smallest returned singular
  value for each least-squares fit, and the pinv effective rank and
  stationarity residual for `soft_boundary_closed_form_lambda3`;
- every C1 replicate error, both projector defects, and the legacy raw boundary
  defect `boundary_compatibility_defect_frobenius`;
- every paired log ratio, correlation, Fisher-z value, interval, critical value,
  sign-test count, and decision; and
- the oracle's absolute and relative numerical error.

The rank threshold and `1e-10` basis tolerances in §7 are frozen and must be
tested before sealing; they may not be changed after viewing v2 values. A failed
dimension, rank, orthogonality, cycle-membership, stationarity, finite-value, or
positive-endpoint assertion is a campaign failure (`design_failure`). Raw
observations may not be rounded before aggregation.

An independent validator must recompute all summaries solely from retained raw
rows and fail closed on missing rows, duplicate seeds, arm imbalance, incorrect
references, wrong t constants, or any decision inconsistent with §8 or §9.

### 10.1 Minimum result schema

The v2 JSON result contains, at minimum:

- schema/version and the frozen configuration;
- protocol, runner, generator, lock, design-commit, and execution-commit
  hashes;
- environment and clean-worktree provenance, including the recorded `sys.argv`;
- all 36 candidate seed records, including explicit ineligibility and failure
  rows;
- per eligible seed: topology dimensions, all derived sub-seeds, per-arm MSE,
  log-ratios, nullspace and orthogonality diagnostics, the `inner_cv_ridge`
  choice, C1 replicate defects and MSEs — including the legacy raw
  `boundary_compatibility_defect_frobenius` — and the analytic oracle's absolute
  and relative error;
- the seven fixed primary summaries, each with estimate, standard error, the
  governing one-sided adjusted bound, critical value, direction, threshold, and
  support decision, plus the geometric mean ratio and the unadjusted two-sided
  95% interval marked descriptive;
- the descriptive optimizer comparison and exact paired-sign sensitivities;
- C1 Fisher-z summaries and back-transformed correlations; and
- an audit block whose values are recomputable from the raw rows.

All ratios are computed within the same seed and shared data realization, and
base-10 logarithms are used throughout.

## 11. Execution and stopping policy

Before constructing any dataset, the runner must, in this order:

1. resolve and verify the repository root;
2. refuse an existing output or partial-output path;
3. require an empty `git status --short`;
4. parse and validate the design-seal record (§1.1): the seal file is committed
   at HEAD, every embedded protocol, runner, generator, and lock hash matches
   both the runner's frozen constants and the actual file bytes, the runner's
   own runtime SHA-256 equals the seal's `runner_sha256`, and the seal's
   `output_path` equals the `--output` argument; record the seal's
   `design_commit` (commit A) and HEAD as the execution revision (commit B);
5. set and verify exactly one PyTorch thread, verify exact dependency versions,
   CPU `torch.float64` mode, and that CUDA is unavailable or hidden; and
6. record the preflight provenance in memory, including `sys.argv`.

Only then may it instantiate all 36 declared samples and apply eligibility. If
the minimum eligible count is met, it runs every frozen arm and all 12 C1 fits
for every eligible topology. It must not compute or print inferential summaries
until every required raw fit has completed. Progress output may contain only
seed identifiers, arm identifiers, completion state, and timing—not MSEs,
defects, ratios, correlations, or running decisions.

There is no outcome-based early stop. A slow arm, unexpected direction, null
effect, or apparent decisive result does not alter execution. Arm-specific
deletion and complete-case inference are forbidden.

The canonical first-run command is:

```bash
env CUDA_VISIBLE_DEVICES=-1 .venv/bin/python \
  scripts/run_lifting_replication_v2.py \
  --output results/campaigns/lifting-replication-v2.json
```

`--seal` defaults to `docs/32-independent-lifting-replication-seal.json` and
therefore does not appear; no hash is ever passed on the command line.

The runner writes JSON through a temporary file followed by an atomic rename.
Its terminal status is exactly one of `complete`, `design_failure`,
`design_failure_insufficient_eligible`, `execution_failure`, or `interrupted`;
only `complete` permits inference. The mapping is frozen:

- a generation exception, or any rank, dimension, orthogonality, nullspace,
  stationarity, or C1-positivity validation failure, yields `design_failure`:
  the whole campaign stops and the offending seed is never deleted alone;
- fewer than 30 eligible seeds yields `design_failure_insufficient_eligible`;
- any other unexpected exception yields `execution_failure`; and
- `KeyboardInterrupt` yields `interrupted`.

A caught failure preserves every completed raw row, identifies the failing seed
and arm, records the exception type and message and the preflight provenance,
and records a null decision for every one of the seven claims. It must never
quietly skip a failed fit.

The following are mandatory stop conditions:

- any preflight mismatch, seal-validation failure, or dirty worktree;
- an existing canonical or partial result;
- fewer than 30 eligible seeds;
- generator failure or a topology outside the declared eligibility logic;
- dimension, rank, orthogonality, nullspace, or stationarity validation
  failure;
- missing, duplicate, non-finite, zero, or negative required endpoint;
- optimization, linear-solver, or aggregation failure;
- an incomplete arm or C1 replicate set; or
- any manual cancellation or infrastructure interruption.

If infrastructure fails after execution begins, preserve that attempt under a
distinct failure path. An exact-code retry is allowed only for a documented
non-scientific failure, before endpoint summaries are inspected, with identical
hashes and a new non-overwriting attempt record. Every attempt remains in the
evidence bundle. The first complete, fully validated attempt is canonical; no
choice among complete attempts is allowed. Any code, tolerance, dependency, or
analysis change after a partial run invalidates the seed block and requires a
new protocol version and new seeds.

## 12. Result interpretation fixed before outcomes

- If `hard_cycle_ls` outperforms the soft penalty, the main conclusion is that
  graph-derived cycle-subspace information is valuable and that the exact
  classical constraint is preferable in this setting.
- If hard and soft methods are close, the soft penalty may be described as a
  differentiable approximation, but no accuracy advantage over constrained LS
  may be claimed without a prespecified equivalence procedure; none is present.
- If soft outperforms hard under H3, report the frozen result and investigate
  shrinkage or label-noise mechanisms only in clearly post hoc analyses.
- If hard cycle does not beat the dimension-matched random subspace under H4,
  do not attribute an advantage merely to topology-specific cycle geometry.
- `inner_cv_ridge` results distinguish a generic variance-control explanation
  from the structural explanation but do not by themselves prove either
  mechanism.
- The singular-value and RTD-inspired arms support claims only about their exact
  formulas at their frozen weights.
- The oracle is expected to be numerically exact by construction and is never
  evidence that a learned method discovered the generator basis.
- A positive descriptive C1 association does not establish that either defect
  is independently calibrated, causal, or useful for routing.

All nulls and failed predictions are reported. Post hoc analyses must be
labelled, separated from the confirmatory tables, and may not alter the title or
abstract as though they had been frozen.

## 13. Freeze, implementation, and post-run sequence

The following order is mandatory:

1. Commit this protocol without instantiating a declared seed.
2. Implement the new runner; do not modify the historical v1 runner.
3. Add tests using hand fixtures and historical seed `20261001` only. Tests must
   cover sub-seed derivation, no sealed-seed fixture use, SVD nullspace checks,
   QR sign canonicalization, solver matching, ridge selection and tie-breaking,
   all seven decisions, t tables, sign tests, Fisher-z aggregation, provenance,
   design-seal parsing and validation, dirty-worktree refusal, and
   output-exists refusal.
4. Run Ruff and the full test suite without touching the declared block.
5. Commit protocol, runner, validator, and tests. Compute their exact hashes.
6. Create `docs/32-independent-lifting-replication-seal.json` with exactly the
   contents frozen in §1.1, including the actual runner SHA-256 in place of
   `PENDING-DESIGN-SEAL` and an explicit renewed no-preview attestation. Commit
   it as commit B and push commit B before any declared seed is instantiated.
   This avoids circular self-hashing: the runner embeds the protocol,
   generator, and lock hashes but never its own, and instead compares its
   runtime SHA-256 with the seal's `runner_sha256`. Verify the remote contains
   the seal commit before any declared seed is instantiated.
7. Confirm a clean worktree and pushed seal, and only then execute the canonical
   command once.
8. Independently validate the retained raw rows before editing any result prose.
9. Add the immutable JSON, validator output, updated manuscript, figures, PDF,
   evidence manifest, and an outcome-independent audit record.
10. Run full tests, Ruff, manifest verification, reviewer-snapshot construction,
    secret and large-file scans, stale-claim searches, and visual inspection of
    every rendered page.
11. Commit and push every artifact, require green CI, and recheck that the
    GitHub repository is private.

The campaign is not complete, and the manuscript is not journal-ready, until
the untouched comparison is executed, validated, interpreted under §12, and
all evidence is committed and pushed.
