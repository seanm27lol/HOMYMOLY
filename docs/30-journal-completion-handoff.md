# Journal-completion handoff

Status time: 2026-08-24, America/New_York  
Repository: `seanm27lol/HOMYMOLY` (private)  
Working tree: `/home/seanjazm27/HOMYMOLY`  
Branch: `paper/journal-corrections-v2`
Last merged pull request: `https://github.com/seanm27lol/HOMYMOLY/pull/23`
Latest pushed revision at this update: `df15368`

This is the authoritative continuation record for taking the manuscript from an
honest audited case study to the strongest journal-ready result attainable on
the existing synthetic generator. Read it before running or changing anything.

## 0. Progress after the initial handoff commit

The corrected-v1 integration described in §§4–5 is now complete:

- roundoff-tolerant, fail-closed routing validation was committed and pushed at
  `7b0dc32`;
- the evidence exporter and `--verify-only` both pass;
- the bundle contains 50 files totaling 2,989,387 bytes;
- corrected values and completed H5 withdrawal language are propagated through
  the active README, manuscript, result record, and audit ledger;
- campaign and recovery figures plus the 20-page PDF were regenerated;
- all 20 PDF pages and every figure page were visually inspected without clipping
  or overlap;
- the full suite passes: `308 passed, 1` expected CUDA skip;
- full Ruff lint passes, and all Python files touched by this correction pass
  Ruff's format check.

The CI-only frozen-environment test failure was repaired at `df15368` without
weakening the production runner's strict environment guard. The tests now mock
the exact frozen environment and include a mismatch-stop regression. The next
scientific milestone is §6 onward: commit the complete untouched-v2 protocol
and implementation before opening any sealed seed.

PR #23 is already merged at `ea4dc99`. Revision `df15368` was pushed afterward,
so it and the v2 work require a **new** pull request. After a fetch, `origin/main`
was `0a180e3` and contained four subsequent Dependabot merges. Before the v2
design seal, incorporate current `origin/main` non-destructively and retain the
CI-portable test patch. Do not mistake the stale failed check attached to merged
PR #23 for a check of `df15368`.

## 0.1 If this process is interrupted now

Run these commands before doing scientific work:

```bash
cd /home/seanjazm27/HOMYMOLY
git fetch origin
git status --short --branch
git log --oneline --decorate -12
sed -n '1,520p' docs/30-journal-completion-handoff.md
```

Then inspect, review, and finish the three uncommitted v2 deliverables if they
exist:

- `docs/31-independent-lifting-replication-protocol.md`;
- `scripts/run_lifting_replication_v2.py`;
- `tests/test_run_lifting_replication_v2.py`.

No seed in `20270101..20270136` may be instantiated merely to smoke-test those
files. Use hand fixtures and old seed `20261001`. The work must pass the focused
tests, full tests, Ruff, and `git diff --check`; then follow the two-commit seal
procedure in §9.1. If those files are only partially written, their intended
contract is fully specified in §§6–9 below.

## 1. Scientific judgment

HOMYMOLY is now credible research engineering with a real negative-results
story. Its strongest established result is narrower than the original vision:

> Input-derived boundary compatibility improves recovery of a separately fitted
> edge-to-cycle-coordinate lifting under scarce probes on one deterministic
> synthetic generator family.

The project has **not** yet established a general architecture for switching
between graphs, complexes, sheaves, or categories; a typed chain-map conversion;
sequence exactness; a benefit from actual mapping-cone homology or published
RTD/SRTD; generalization to unseen topologies by one shared model; a Langlands,
Fourier--Mukai, or eigensheaf construction; or validation on real data.

The current method claim is also incomplete because the v1 graph-blind baseline
does not receive `B1`, while the structural penalty does. A graph-aware hard
cycle-nullspace least-squares comparator was omitted. A hostile post-campaign
diagnostic found that comparator beat the soft penalty on 27 of the 29 old seeds.
That diagnostic is useful for design but **must not be reported as confirmatory
evidence**. An untouched, frozen follow-up is required.

## 2. Immutable and current revisions

The frozen v1 protocol must remain byte-identical:

- `docs/27-conversion-campaign-protocol.md`
- SHA-256: `503cc282f40d118ba1739c2afe1bfc77eaf2b1733baaddb91c0c3363e75ae2b8`

The frozen dependency lock must remain byte-identical for the v1 correction:

- `uv.lock`
- SHA-256: `05c6a5ad02db5b1651d426d157add170a8542634260ce8c265a3ee32693073bf`

Other pinned v1 provenance:

- generator SHA-256:
  `c37ab1c725aa2101e88c1a0ad8fa3b279d72330feba35077e23fec930a4df69d`
- historical v1 result SHA-256:
  `836914d251db8d381aef9a2dcb0ac14a14562652f3e323dc840108b5f24d5ee1`
- historical v1 runner SHA-256:
  `8a478e5a3906d5bb5cfc3645159f8739cc3e840a50bbce851564533b2ce89fb6`
- original campaign revision:
  `11644c68ec0b8c28416a14ce4d8799e4c9ca0860`

Audit and integration commits already made on the working branch:

- `a4c193856c604f93b36f3b90820b81653eff701d` — Pearson/t-critical and
  first claim corrections;
- `0c6fa574cb1f7d93e3382ad20e27d69520f9a14c` — second journal audit,
  evidence guards, terminology, information-flow, and H5 withdrawal.
- `7b0dc32` — guarded tolerance for irrelevant rerun roundoff in the withdrawn
  routing audit.
- `ea4dc99` — corrected v1 artifact, evidence, figures, manuscript, and PDF.
- `df15368` — CI-portable tests for the unchanged strict environment guard.

The corrected artifact is tracked at
`results/campaigns/conversion-campaign-v1-corrected.json`. Do not overwrite it.
Its internal provenance records an empty Git status, CPU-only execution, the
exact environment, and revision `0c6fa57`.

## 3. Corrected v1 result

The canonical command already completed successfully:

```bash
env CUDA_VISIBLE_DEVICES=-1 .venv/bin/python \
  scripts/run_conversion_campaign.py \
  --output results/campaigns/conversion-campaign-v1-corrected.json
```

Exact corrected results from the artifact:

| endpoint | estimate | unadjusted 95% interval | Bonferroni 98.33% interval | conclusion |
|---|---:|---:|---:|---|
| boundary compatibility (`exact`, historical key) | mean log10 ratio `-2.130080865244953` | `[-2.6280764680617494, -1.6320852624281563]` | `[-2.7491611963412117, -1.5110005341486945]` | improves versus graph-blind ambient baseline |
| singular-value cone surrogate (`cone`) | `+0.18944400798360397` | `[+0.12465337837039316, +0.2542346375968148]` | `[+0.10889991440927267, +0.2699881015579353]` | harms at tested weight |
| RTD-inspired normalized-distance surrogate (`rtd`) | `+0.018113280176106524` | `[+0.002277684864924124, +0.03394887548728892]` | `[-0.0015726478379199105, +0.03779920819013296]` | no multiplicity-controlled conclusion |

Corrected C1:

- conventional Pearson mean: `0.9608082698917015`;
- unadjusted Student-t 95% interval:
  `[0.93472446177427, 0.986892078009133]`;
- positive in `29/29` eligible seeds;
- claim is only a positive within-seed regularization-path association;
- it does not establish independent predictive information, off-path
  calibration, or a causal effect because the common `lambda` drives both axes
  and the nine within-seed fits are not independent.

Corrected H5 status:

- `28` rows are `14` topology clusters times two weights;
- old row-naive interval:
  `[-0.11138359254255889, 0.38166484169833903]`;
- topology-clustered descriptive interval:
  `[-0.12309738604162501, 0.3933786351974051]`;
- `supported: null` and `decision: withdrawn-non-informative`;
- the routed numerator is one of the two values in its per-row oracle
  denominator, making every endpoint nonnegative while the frozen support rule
  required an upper bound below zero. No inferential H5 conclusion is possible.

## 4. Resolved v1 integration blocker (historical audit context)

The first publication export after the canonical run failed with:

```text
publication evidence export failed: routing raw trials changed during the H5 audit correction
```

This was resolved at commit `7b0dc32` with the guarded `1e-12` comparison and a
regression test. No action remains in this section. The following explanation is
retained as audit context. The rerun differs from the historical routing
rows only at roughly `1e-14` in a few repeated `graph_error` floats. Examples:

- historical `1.3286839475087884`, rerun `1.3286839475087906`;
- historical `36.96853289528931`, rerun `36.9685328952893`.

Aggregate values are unchanged except one `3.6e-15` median drift. The exporter at
`scripts/export_publication_evidence.py::_validate_withdrawn_routing` now does
the following:

1. Require identical list length, row order, key sets, seeds, split labels, and
   term weights.
2. Compare floating fields with `math.isclose(rel_tol=0.0, abs_tol=1e-12)`.
3. Compare the five retained routing aggregates with the same tolerance.
4. Add a regression test showing a `1e-14` drift is accepted and a `1e-6` drift
   is rejected.
The historical values were not copied into the corrected artifact, and the
tolerance was not loosened beyond deterministic floating-point rerun noise.

## 5. Finish v1 evidence and manuscript

These steps have been completed and remain the reproduction checklist:

1. Regenerate and verify the compact evidence:

   ```bash
   .venv/bin/python scripts/export_publication_evidence.py
   .venv/bin/python scripts/export_publication_evidence.py --verify-only
   ```

2. Replace every withdrawn number with artifact-derived values in:

   - `README.md`;
   - `docs/00-original-idea.md`;
   - `docs/18-paper.md`;
   - `docs/28-conversion-campaign-results.md`;
   - `docs/29-audit-corrections.md`.

   Search before and after:

   ```bash
   rg -n "0\.854|0\.831|0\.877|-2\.802|-1\.458|0\.102|0\.277|-0\.003|0\.039|PENDING|will aggregate" \
     README.md docs
   ```

3. Change H5 future tense to completed withdrawal language. Never restore an H5
   `interval_95` or support decision.

4. Regenerate all tracked SVGs. The renderer already uses safe labels for the
   edge-to-cycle campaign and the annulus cone/RTD-style surrogates:

   ```bash
   .venv/bin/python scripts/render_figures.py
   ```

   Confirm `fig-campaign.svg` no longer says “learned conversion,”
   “preregistered,” or “one value per topology,” and `fig-recovery.svg` no
   longer says “structure-only control.”

5. Regenerate the paper PDF with the repository renderer; inspect its CLI if
   needed:

   ```bash
   .venv/bin/python scripts/render_paper.py --help
   ```

6. Visually inspect every PDF page and the architecture/campaign/recovery
   figures. Figure 1 has longer corrected target-information text and may need
   layout adjustment. Do not accept clipped or overlapping labels.

7. Re-export the evidence if any tracked result input changed, verify the
   manifest, and update every manuscript bundle/file-count statement from the
   resulting manifest rather than guessing.

8. Run:

   ```bash
   .venv/bin/ruff format <Python files changed by the milestone>
   .venv/bin/ruff check scripts src tests
   .venv/bin/python -m pytest -q
   git diff --check
   ```

   Before the artifact landed, the full suite was `306 passed, 3 skipped`; the
   two artifact-related skips should become active tests and pass. The remaining
   CUDA skip is expected on this host.

9. Commit all corrected v1 evidence, figures, PDF, documentation, and manifest.
   Confirm the worktree is clean.

## 6. Untouched v2 seal

The proposed v2 generator seeds are exactly:

```text
20270101 through 20270136 inclusive
```

They were checked for absence from Git history and **have never been
instantiated or previewed**. Preserve that seal. Do not construct a dataset,
count eligibility, print dimensions, smoke-test, or debug with any seed in this
block until the complete protocol, runner, tests, generator hash, lock hash,
decision rules, and stop conditions are committed from a clean worktree.

If even one sealed seed is accidentally instantiated before the design commit,
discard the entire block and choose a new consecutive block whose absence is
verified before use. Use old seed `20261001` or synthetic hand fixtures for tests.

## 7. v2 scientific design

Purpose: replicate the soft boundary-compatibility effect on an untouched seed
block and compare it fairly with graph-aware classical estimators, using one
eligible generator seed as the unit of inference. Call this an untouched-seed,
outcome-informed, same-generator-family replication—not an independent-lab or
independent-generator replication and not a pristine preregistration.

Frozen data/training constants should retain comparability with v1:

- same `ConversionDataset` implementation and recorded SHA-256;
- all connected generated cases with `F >= 3`, no cherry-picking or replacement;
- seed block `20270101..20270136`;
- stop and report design failure if fewer than `30` seeds are eligible;
- `N_train = 16`;
- `N_test = 3072`;
- training-label Gaussian noise SD `0.02`;
- test targets noiseless;
- `W = 0` initialization, float64;
- Adam learning rate `0.05`, `2500` steps for learned arms;
- paired arms share topology, train inputs, label noise, test inputs, and learned
  initialization;
- explicit SHA-256-derived sub-seeds for every stochastic component;
- output path must not exist; runner must refuse a dirty worktree and mismatched
  environment/lock/generator/runner fingerprints.

Enforce an information-flow firewall in code. Fitting APIs for ambient, ridge,
soft, hard-cycle, hard-random, singular-value, and RTD-inspired arms may receive
only their declared training tensors and, where applicable, `B1` or the seeded
random basis. They may not receive `B2`, face-cycle metadata, held-out targets,
or held-out losses. `B2` is used outside those APIs only to form response tensors,
evaluate the already fitted matrices, and compute the explicitly segregated
truth-access oracle.

Primary and essential comparator arms:

1. `ambient_adam`: graph-blind unpenalized full `F x E` matrix, v1 baseline.
2. `ambient_min_norm_ls`: the graph-blind full-space minimum-norm least-squares
   solution from `torch.linalg.lstsq(X_train, Y_train).solution`. This is the
   essential solver-matched reference for hard constrained least squares.
3. `soft_boundary_lambda3`: v1 boundary-compatibility penalty at frozen weight
   `3.0`.
4. `soft_boundary_closed_form_lambda3`: the exact minimum-norm solution of the
   same mean-normalized convex objective as arm 3. In `A = W.T` orientation its
   normal equation is
   `(X.T X + 3*N_train/V * B1.T B1) A = X.T Y`; solve by a frozen SVD
   pseudoinverse convention. This removes finite-step optimizer confounding.
5. `hard_cycle_ls`: graph-aware least squares restricted exactly to
   `ker(B1)`, the missing classical comparator.
6. `hard_random_subspace_ls`: deterministic Haar-like random `E x F`
   orthonormal subspace, dimension matched to the cycle subspace, fitted by least
   squares; specificity control that does not use `B2`.
7. `inner_cv_ridge`: choose from
   `{1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100}` using deterministic four-fold
   training-only CV. Fold `k` contains row indices satisfying `i mod 4 = k`;
   fit on 12, validate on four, average the four multi-output validation MSEs,
   and retain every fold loss. There is no intercept. Each fit solves
   `(X.T X + alpha I) A = X.T Y`. Exact ties choose the smaller alpha; refit on
   all 16. Never use the 3,072 held-out rows for tuning.
8. `singular_value_surrogate`: historical `cone` arm at `0.01`, named precisely.
9. `rtd_inspired_distance_surrogate`: historical `rtd` arm at `0.1`, named
   precisely.
10. `generator_cycle_basis_oracle`: use truth-access `B2` as the analytic output-
   coordinate/generator ceiling. It is not an estimator and must be isolated
   from every fitted arm and inferential table. Store raw MSE only; do not form a
   log ratio when numerical zero is possible.

The oracle can also reconstruct the generator's noncanonical
   NetworkX cycle basis and use its analytic mapping. This is a deterministic
   attainability ceiling, not a learned method and not part of efficacy
   inference.

Implement the hard estimators without `B2`:

- Require connectivity, `rank(B1) = V-1`, and `F = E-V+1`. Compute a full float64
  SVD and take `Q_cycle = Vh[V-1:].T`, shape `E x F`. Validate `B1 Q_cycle = 0`
  and `Q_cycle.T Q_cycle = I` at frozen tolerances. The basis is not itself
  cross-platform canonical under repeated singular values; only its projector
  and the fitted predictions are basis-invariant.
- Let `Z = X_train @ Q_cycle`.
- Solve `C = lstsq(Z, Y_train, driver="gelsd", rcond=1e-12)` on CPU.
- Set `W.T = Q_cycle @ C` and evaluate `X_test @ W.T`.
- For the matched random control, form a deterministic Gaussian `E x F` matrix
  from a SHA-256 sub-seed, take reduced QR with a documented sign convention,
  and use the identical least-squares procedure.

Use the same `driver="gelsd", rcond=1e-12` convention for ambient minimum-norm
LS. Freeze the pseudoinverse cutoff for the closed-form soft solution. Record
all returned numerical ranks and stop the entire design on a rank, dimension,
orthogonality, nullspace, or stationarity validation failure; never delete only
the offending seed.

Primary estimand for every pair is the mean across eligible seeds of
`log10(MSE_arm / MSE_reference)`. The seed jointly determines topology, data,
and noise and is the only inference unit. State the exchangeability assumption.

Intended confirmatory family (copy exactly into the protocol and freeze before
opening seeds):

1. soft boundary versus ambient Adam: mean log ratio `< 0`;
2. hard cycle versus ambient minimum-norm least squares: `< 0`;
3. hard cycle versus the closed-form soft-boundary solution: `< 0`;
4. hard cycle versus hard random subspace: `< 0` (specificity);
5. frozen ridge versus ambient minimum-norm least squares: `< 0`;
6. singular-value surrogate versus ambient Adam: `> 0` (replicated harm);
7. RTD-inspired surrogate bounded-benefit/futility test: one-sided lower bound on
   `log10(MSE_rtd/MSE_ambient_adam)` greater than `log10(0.90)` rules out a
   benefit of 10% or more. This is **not** an equivalence test and must use that
   prespecified margin, whose exact log threshold is
   `-0.045757490560675115`. Do not call this noninferiority: the usual direction
   and estimand semantics are reversed.

`ambient_adam` versus `ambient_min_norm_ls` is a descriptive optimizer
diagnostic, not an eighth confirmatory claim. This split is essential: the soft
penalty is compared with its historical Adam baseline, while hard constrained
LS and ridge are compared with the same-solver full-space LS reference. It
prevents a solver difference from masquerading as a structural-prior effect.
The Adam-versus-closed-form soft endpoint and solution/stationarity gaps are
also descriptive optimization audits, not primary claims.

Use Bonferroni `alpha = 0.05/7` with one-sided Student-t bounds and hard-code
verified critical values for every possible eligible `n = 30..36`. Compute them
once with SciPy for design, record the values in the protocol and runner, and
test them; SciPy need not become a runtime dependency. Report exact paired sign
tests only as direction-neutral sensitivity analyses.

The independently computed quantiles `t.ppf(1 - 0.05/7, n - 1)` are:

| eligible n | df | critical value |
|---:|---:|---:|
| 30 | 29 | 2.606750672048818 |
| 31 | 30 | 2.601227904110613 |
| 32 | 31 | 2.5960807947257787 |
| 33 | 32 | 2.5912722991315227 |
| 34 | 33 | 2.586770085672467 |
| 35 | 34 | 2.5825458097369376 |
| 36 | 35 | 2.5785745178415116 |

Do not add, drop, or reinterpret claims after inspecting v2 outcomes. For every
claim, the protocol must spell out `theta`, null, alternative, reference arm,
bound direction, threshold, and support rule. The target is the conditional
synthetic-generator quantity
`E_seed[log10(MSE_arm/MSE_reference) | connected, F>=3]`. Intervals quantify
Monte Carlo variation over that seed mechanism, not uncertainty for real data.
Do not pool v1 and v2.

## 8. Off-path C1 replacement (prespecified secondary/descriptive)

The v1 C1 path association is confounded by `lambda`. V2 should test a genuinely
off-path version:

- use the unregularized `ambient_min_norm_ls` estimator only, rather than a
  finite-step optimizer, so optimizer error cannot drive the association;
- for each topology, run `12` independent training-input/label-noise replicates;
- share one independently generated `3072`-example noiseless test set across
  those replicates;
- use deterministic SHA-256 sub-seeds;
- compute conventional Pearson correlation between log10 cycle-projector defect
  `||(I-Q_cycle Q_cycle.T)A||_F` and log10 held-out MSE across the 12 replicates;
- clip each `r` just inside `(-1, 1)`, apply Fisher `atanh`, average Fisher-z over
  topology seeds, and form the prespecified seed-level interval;
- report raw `||B1 A||_F` only as a legacy descriptive diagnostic;
- use `||(I-Q_random Q_random.T)A||_F`, with the same deterministic
  topology-specific `Q_random` as the hard-random arm, as the dimension-matched
  negative control; reuse that basis across all 12 replicates;
- report the separate Fisher-z summaries and the paired seed-level difference
  `z_cycle - z_random`, each with an unadjusted two-sided 95% interval.

For each defect, clip each correlation using an exact rule such as
`nextafter(1, 0)`, transform by Fisher `atanh`, and use the eligible topology
seeds as the inference units. Use ordinary unadjusted two-sided 95% Student-t
intervals on Fisher-z and back-transform the individual summaries for
interpretation. **Make no support decision:** C1 is outside the seven-test
Bonferroni family. Two positive correlations alone do not establish specificity;
the paired delta-z is the relevant descriptive contrast. This remains
association, not causality. A constant/undefined vector is a whole-design
failure, never grounds for deleting one topology.

This design varies training/noise independently rather than moving a common
regularization weight. It still supports association, not causation.

Remove H5 entirely from v2. The frozen v1 endpoint is algebraically
non-informative and should not be rehabilitated by changing its decision rule
after the fact.

## 9. v2 execution sequence

1. Finish and commit corrected v1 evidence first.
2. Write `docs/31-independent-lifting-replication-protocol.md` with every choice
   above, hashes, stop rules, estimands, critical values, and explicit
   no-preview declaration.
3. Implement a new runner rather than mutating the v1 historical runner. Suggested
   path: `scripts/run_lifting_replication_v2.py`.
4. Write unit tests using hand fixtures and old seed `20261001` only. Test QR sign
   canonicalization, exact cycle-nullspace membership, LS formulas, deterministic
   sub-seeds, multiplicity decisions, Fisher-z aggregation, clean-worktree
   refusal, output-exists refusal, and provenance.
5. Run Ruff/full tests without touching sealed seeds.
6. Commit protocol, runner, and tests. Record the full commit and runner SHA-256
   in a signed/sealed design note. Confirm a clean worktree.
7. Only then run the sealed block on CPU or the free GB10. The workload is small
   linear algebra; CPU is likely more reproducible. Hide CUDA if using CPU.
8. Do not stop early based on outcomes. Preserve failures and ineligible seeds.
9. Validate the result independently from retained raw rows, export it, update
   figures and manuscript, and label it an untouched same-generator-family
   replication—not a new generator-family or real-data validation.
10. Run the final reviewer snapshot, full tests, Ruff, manifest verification,
    PDF visual inspection, secret/large-file scan, and stale-claim search.

### 9.1 Fail-closed two-commit seal

A runner cannot safely contain its own byte hash: adding that hash changes the
file. Do not invent a circular self-hash check. Use this auditable two-commit
sequence instead:

1. Incorporate `origin/main` and resolve changes before the seal. Review the
   resulting diff; retain `df15368`'s strict-environment test coverage.
2. Finish protocol, runner, and old-seed/fixture tests. The runner may embed the
   finalized protocol SHA-256 plus immutable generator and lock hashes. At
   runtime it computes and records its own hash rather than comparing against a
   constant inside itself.
3. Prove that no sealed seed has been instantiated. A text occurrence in the
   protocol or an inert constant is allowed; a dataset construction, result,
   cache, debug print, or generated artifact is not. Search all tracked and
   untracked files and inspect shell history if available.
4. Run all pre-seal gates without a sealed seed:

   ```bash
   .venv/bin/ruff format scripts/run_lifting_replication_v2.py \
     tests/test_run_lifting_replication_v2.py
   .venv/bin/ruff check scripts src tests
   .venv/bin/python -m pytest -q
   git diff --check
   ```

5. Commit the design files as commit A. Record the full commit-A hash and the
   SHA-256 values of protocol, runner, generator, and lock.
6. Create `docs/32-independent-lifting-replication-seal.json` containing those
   exact hashes, the seed interval, the no-preview declaration, the complete
   primary family, all stop rules, and the exact repository-relative output
   path. The runner must parse and validate this machine-readable file rather
   than trusting a naked command-line hash. Commit it as commit B and push commit
   B to the private GitHub remote before the first seed is opened, creating a
   remote timestamp. Do not modify the design files between A and execution.
7. Require a clean worktree at commit B. The runner must verify the protocol,
   generator, and lock hashes, ensure its runtime hash equals the runner hash
   recorded in the external seal note, confirm that its current code content is
   still commit-A content, refuse an existing output, and record commit A,
   commit B/current execution revision, environment, CLI, and all hashes.
8. Only after commit B is clean may the first sealed seed be instantiated. Run
   all 36 without outcome-dependent stopping:

   ```bash
   env CUDA_VISIBLE_DEVICES=-1 .venv/bin/python \
     scripts/run_lifting_replication_v2.py \
     --output results/campaigns/lifting-replication-v2.json
   ```

   CPU is preferred because this workload is small float64 linear algebra and
   reproducibility matters more than accelerator utilization. If the runner's
   frozen environment requires the GB10 image, use that exact environment and
   record the deviation from this suggested command; never relax a guard merely
   to make the campaign start.
9. Independently recompute every summary, interval, support flag, and C1 result
   from retained raw rows. Verify dimensions, paired row identity, finite
   positive MSEs, exact eligible/ineligible accounting, deterministic sub-seeds,
   cycle-nullspace residuals, random-basis orthogonality, selected ridge alpha,
   and oracle numerical tolerance. Preserve all failures.
10. Commit the immutable result before interpretive manuscript edits. Never
    rerun with another seed block because the result is surprising or weak.

If fewer than 30 seeds are eligible, the campaign is a frozen design failure:
write and commit the failure artifact, make no confirmatory claims, and design a
new prospective protocol. If a numerical or implementation fault occurs, retain
the failed artifact/log, diagnose without looking selectively at successful
outcomes, amend the protocol transparently, and use a wholly new untouched seed
block.

### 9.2 Expected result schema and audit invariants

The v2 JSON should contain, at minimum:

- schema/version and frozen configuration;
- protocol, runner, generator, lock, design-commit, and execution-commit hashes;
- environment and clean-worktree provenance;
- all 36 candidate seed records, including explicit ineligibility/failure rows;
- per eligible seed: topology dimensions, all derived sub-seeds, per-arm MSE,
  log-ratios, nullspace/orthogonality diagnostics, ridge choice, C1 replicate
  defects and MSEs, and analytic-oracle error;
- the seven fixed primary summaries with estimate, standard error, one-sided
  adjusted bound, critical value, direction, threshold, and support decision;
- descriptive optimizer comparison and exact paired-sign sensitivities;
- C1 Fisher-z summaries and back-transformed correlations;
- an audit block whose values are recomputable from raw rows.

All ratios must be computed within the same seed and shared data realization.
Use base-10 logarithms throughout. Never treat 36 training examples or 12 C1
replicates as independent inferential units; topology/generator seed is the
unit. Never substitute a 95% two-sided interval for a one-sided Bonferroni bound.

### 9.3 Mathematical interpretation to preserve

Let `A_* = W_*^T = B2`, let `S = ker(B1)`, and let `P_S` be the Euclidean
orthogonal projector onto `S`. Every column of `A_*` lies in `S`. Therefore, for
any estimated `A`, Pythagoras gives the exact identity

```text
||P_S A - A_*||_F^2
  = ||A - A_*||_F^2 - ||(I - P_S) A||_F^2.
```

For isotropic noiseless test predictors this is also the reduction in expected
prediction squared error, up to the output-averaging convention. This identity
explains the useful information supplied by `B1`; it does not make the classical
projection or constrained least squares a novel algorithm.

The observed boundary defect is a spectrally weighted version of the removable
off-cycle component. If `sigma_min+` is the smallest nonzero singular value of
`B1`, then

```text
sigma_min+^2 ||(I-P_S)A||_F^2
  <= ||B1 A||_F^2
  <= sigma_max(B1)^2 ||(I-P_S)A||_F^2.
```

Consequently the soft objective is graph-Laplacian/Tikhonov shrinkage of
cut-space directions; hard-cycle LS removes those directions; the random
subspace control tests whether dimension reduction alone is enough; and ridge
tests generic shrinkage. This is the clean mechanistic spine for the final
paper. It also explains why raw defect magnitudes are comparable within a fixed
topology but need spectral normalization before strong cross-topology claims.

If this argument is added as a proposition, prove it in an appendix and verify
the matrix orientations carefully. State that it is a standard orthogonal
projection consequence, not a priority claim. Do not imply that hard-cycle LS
is literally the Euclidean projection of ambient LS for an arbitrary finite
design matrix; those estimators coincide only under additional geometry.

## 10. Manuscript interpretation after v2

Let the result determine the title and claim:

- If `hard_cycle_ls` beats the soft penalty, the main result is a system
  identification study showing that graph-derived cycle-subspace information is
  valuable and that an exact classical constraint is preferable to soft
  shrinkage. Do not sell the soft penalty as a new superior method.
- If soft and hard approaches tie within a prespecified practical margin, frame
  soft compatibility as a differentiable approximation with no established
  accuracy advantage over classical constrained LS.
- If soft unexpectedly beats the hard estimator under the frozen design, report
  that honestly but analyze whether label noise/implicit shrinkage explains it;
  do not infer general superiority.
- The analytic generator-basis oracle will have zero or numerical-zero error by
  construction. Present it only as an attainable ceiling demonstrating how much
  generator knowledge the learning problem withholds.
- No result on this same generator establishes transfer to unseen topologies,
  neural nonlinear translators, sheaves, or real data.

The likely strongest journal paper is therefore about **scarce-probe system
identification with graph-derived cycle-subspace priors**, accompanied by an
audited account of why cone- and RTD-inspired surrogates were misaligned or
uninformative in these specific experiments.

## 11. Final repository and GitHub gates

Before calling the project complete:

- `git status --short` is empty;
- all intended files, including JSON, SVG, PDF, manifest, protocol, and handoff,
  are committed;
- `uv.lock` and frozen docs 27 hashes are unchanged;
- full pytest and Ruff pass;
- `scripts/export_publication_evidence.py --verify-only` passes;
- reviewer snapshot builds from the exact committed revision and correctly says
  compact evidence can be verified while full GB10 reruns require separately
  supplied large artifacts/checkpoints;
- the paper contains no stale `+0.854`, old adjusted intervals, inferential H5,
  “preregistered” wording for v1, class-wide claims from finite screening, or
  conflation of sequence exactness with boundary compatibility;
- all figures and every PDF page are visually inspected;
- repository visibility is rechecked as private;
- branch is pushed, a pull request exists, and CI is green.

Suggested Git commands after each evidence milestone:

```bash
git status --short
git diff --check
git add <explicit files>
git commit -m "<specific evidence milestone>"
git push -u origin paper/journal-corrections-v2
```

Do not use destructive Git commands, do not rewrite the frozen protocol, and do
not claim “journal ready” merely because engineering checks pass. Journal-ready
means the corrected evidence is integrated, the missing graph-aware comparison
has been run on untouched seeds, the claims match the result, and every artifact
is committed and pushed.
