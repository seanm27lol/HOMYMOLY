# Journal-completion handoff

Status time: 2026-08-24, America/New_York  
Repository: `seanm27lol/HOMYMOLY` (private)  
Working tree: `/home/seanjazm27/HOMYMOLY`  
Branch: `paper/journal-corrections-v2`

This is the authoritative continuation record for taking the manuscript from an
honest audited case study to the strongest journal-ready result attainable on
the existing synthetic generator. Read it before running or changing anything.

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

Audit commits already made on the working branch:

- `a4c193856c604f93b36f3b90820b81653eff701d` — Pearson/t-critical and
  first claim corrections;
- `0c6fa574cb1f7d93e3382ad20e27d69520f9a14c` — second journal audit,
  evidence guards, terminology, information-flow, and H5 withdrawal.

At handoff time, the corrected v1 artifact has been generated from clean commit
`0c6fa57` but has not yet been committed:

- `results/campaigns/conversion-campaign-v1-corrected.json`

Do not overwrite it. Its internal provenance records an empty Git status,
CPU-only execution, the exact environment, and revision `0c6fa57`.

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

## 4. Immediate v1 integration blocker

The first publication export after the canonical run failed with:

```text
publication evidence export failed: routing raw trials changed during the H5 audit correction
```

This is not a scientific change. The rerun differs from the historical routing
rows only at roughly `1e-14` in a few repeated `graph_error` floats. Examples:

- historical `1.3286839475087884`, rerun `1.3286839475087906`;
- historical `36.96853289528931`, rerun `36.9685328952893`.

Aggregate values are unchanged except one `3.6e-15` median drift. The exporter at
`scripts/export_publication_evidence.py::_validate_withdrawn_routing` currently
compares dictionaries and aggregates with exact equality. Correct it as follows:

1. Require identical list length, row order, key sets, seeds, split labels, and
   term weights.
2. Compare floating fields with `math.isclose(rel_tol=0.0, abs_tol=1e-12)`.
3. Compare the five retained routing aggregates with the same tolerance.
4. Add a regression test showing a `1e-14` drift is accepted and a `1e-6` drift
   is rejected.
5. Run Ruff and the focused exporter tests.
6. Commit that code change without adding the generated artifact yet.

Do not copy historical values into the corrected artifact and do not loosen the
tolerance beyond what is needed for deterministic floating-point reruns.

## 5. Finish v1 evidence and manuscript

After fixing the tolerance:

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
   .venv/bin/ruff format scripts src tests
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

Purpose: independently replicate the soft boundary-compatibility effect and
compare it fairly with graph-aware classical estimators, using one eligible
generator seed as the unit of inference.

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

Primary and essential comparator arms:

1. `ambient_adam`: graph-blind unpenalized full `F x E` matrix, v1 baseline.
2. `soft_boundary_lambda3`: v1 boundary-compatibility penalty at frozen weight
   `3.0`.
3. `hard_cycle_ls`: graph-aware least squares restricted exactly to
   `ker(B1)`, the missing classical comparator.
4. `hard_random_subspace_ls`: deterministic Haar-like random `E x F`
   orthonormal subspace, dimension matched to the cycle subspace, fitted by least
   squares; specificity control that does not use `B2`.
5. `ridge`: a tuning rule frozen without v2 outcome access. Prefer a small
   nested training-only validation split or a prespecified grid and deterministic
   inner selection; never select using the held-out v2 endpoint.
6. `singular_value_surrogate`: historical `cone` arm at `0.01`, named precisely.
7. `rtd_inspired_distance_surrogate`: historical `rtd` arm at `0.1`, named
   precisely.
8. `generator_cycle_basis_oracle`: reconstruct the generator's noncanonical
   NetworkX cycle basis and use its analytic mapping. This is a deterministic
   attainability ceiling, not a learned method and not part of efficacy
   inference.

Implement the hard estimators without `B2`:

- Compute an orthonormal `Q_cycle` spanning `ker(B1)` using a deterministic SVD;
  shape `E x F` for a connected graph.
- Let `Z = X_train @ Q_cycle`.
- Solve `C = lstsq(Z, Y_train)`.
- Set `W.T = Q_cycle @ C` and evaluate `X_test @ W.T`.
- For the matched random control, form a deterministic Gaussian `E x F` matrix
  from a SHA-256 sub-seed, take reduced QR with a documented sign convention,
  and use the identical least-squares procedure.

Primary estimand for every pair is the mean across eligible seeds of
`log10(MSE_arm / MSE_reference)`. The seed jointly determines topology, data,
and noise and is the only inference unit. State the exchangeability assumption.

Recommended confirmatory family (freeze the exact family before opening seeds):

1. soft boundary versus ambient: mean log ratio `< 0`;
2. hard cycle versus ambient: `< 0`;
3. hard cycle versus soft boundary: `< 0`;
4. hard cycle versus hard random subspace: `< 0` (specificity);
5. frozen ridge versus ambient: `< 0`;
6. singular-value surrogate versus ambient: `> 0` (replicated harm);
7. RTD-inspired surrogate noninferiority test: one-sided lower bound on
   `log10(MSE_rtd/MSE_ambient)` greater than `log10(0.90)` rules out a benefit of
   10% or more. This is **not** an equivalence test and must use that prespecified
   margin.

Use Bonferroni `alpha = 0.05/7` with one-sided Student-t bounds and hard-code
verified critical values for every possible eligible `n = 30..36`. Compute them
once with SciPy for design, record the values in the protocol and runner, and
test them; SciPy need not become a runtime dependency. Report exact paired sign
tests only as direction-neutral sensitivity analyses.

If seven primary claims make the design too diffuse, remove claims before the
protocol commit rather than silently changing multiplicity later. Never add or
drop claims after inspecting v2 outcomes.

## 8. Independent C1 replacement

The v1 C1 path association is confounded by `lambda`. V2 should test a genuinely
off-path version:

- use the unregularized ambient model only;
- for each topology, run `12` independent training-input/label-noise replicates;
- share one independently generated `3072`-example noiseless test set across
  those replicates;
- use deterministic SHA-256 sub-seeds;
- compute conventional Pearson correlation between log10 Frobenius boundary
  defect and log10 held-out MSE across the 12 replicates;
- clip each `r` just inside `(-1, 1)`, apply Fisher `atanh`, average Fisher-z over
  topology seeds, and form the prespecified seed-level interval/test;
- report the boundary-defect result as primary C1 and a matched random-subspace
  defect as a specificity sensitivity analysis.

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
