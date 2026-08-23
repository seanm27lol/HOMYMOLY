# HOMYMOLY journal-release handoff

Last updated: 2026-08-22. Repository: `/home/seanjazm27/HOMYMOLY`.

This is the authoritative continuation point if the active Codex session ends.
The large training campaign is finished. The remaining work is bounded:
aggregate the final diagnostics, export compact evidence, finish the manuscript
and documentation, validate the release, and commit and push every publishable
file. Do not rerun the 40-run campaign unless a provenance audit finds a real
mismatch.

## Definition of done

Do not call the project finished until all of the following are true:

1. Every final result is represented by a strict, checksummed, machine-readable
   summary under tracked `results/`; large raw artifacts and checkpoints remain
   outside Git.
2. `docs/18-paper.md` and `docs/18-paper.pdf` are journal-ready: technically
   correct, readable by a mathematically literate reader who did not build the
   repository, complete, visually inspected, and free of placeholders.
3. The README and technical documentation explain the architecture, experiment,
   reproduction path, results, and limitations in clear language.
4. All tests and release checks pass, the worktree is clean, every source,
   compact result, document, and PDF is committed and pushed, the current
   release pull request is green, and `seanm27lol/HOMYMOLY` is still private.

## Completed and independently audited

- Evidence-generating commit:
  `8021292e97abfec91768f1b5437c883a42c29c60` on
  `agent/research-remediation`.
- Historical remediation pull request #3 was merged with both CI jobs green:
  <https://github.com/seanm27lol/HOMYMOLY/pull/3>. Commits made after that merge
  are being collected in draft release pull request #8:
  <https://github.com/seanm27lol/HOMYMOLY/pull/8>.
- Repository visibility: `PRIVATE`. The repository was briefly public earlier;
  making it private cannot retract that historical exposure.
- Audited GB10 campaign: 56/56 scheduler steps completed—40 training runs, one
  strict summary, ten identifiable-map checkpoint benchmarks, and five routing
  benchmarks. The managed HOMYMOLY cron entry was removed after completion.
- Launch fingerprint:
  `44408d7adf8467e594879b46e25a1cb7fd89a7e7a5d5f3446548bcbf3ed1096e`.
- Frozen full-config SHA-256:
  `22abb205e8a89586b38799d7f7b8d53f0c24cef45f872453533ddf34e20fad73`.
- Strict summary:
  `artifacts/identifiable-maps/campaign-summary.json`, SHA-256
  `0cd0defb0b0d41b5f7563c364cfdda62cb72c5e5a845bdd5d1ab76a2e1cb953c`.
  It includes 40/40 runs, no missing/replaced/excluded runs, verified checkpoints,
  verified hashes, clean committed provenance, and paired sample identities.
- Engineering recovery gate: 10/10 applicable runs passed
  (`task_reconstruction` and `combined`, five seeds each). Both accuracies were
  1.0 in every applicable run; map errors were at numerical precision, chain-map
  residuals met the fixed tolerance, and hard cones were acyclic.
- Structural-loss ablations: no benefit was established. Accuracy was saturated,
  every declared continuous contrast interval contained zero, and `cone_only`
  and `rtd_only` remained near chance despite acyclic decoded cones.
- Routing v2: the frozen five-seed hard-minus-best-fixed margin was
  `0.1098039`, with t-based 95% CI `[0.0952866, 0.1243213]`. The exact sign-test
  sensitivity value is `p=0.0625`; privileged regime-distillation and the
  pre-freeze procedural deviation remain disclosed.
- Trained GB10 routing benchmark: the descriptive dense-to-routed
  median-latency ratio was `1.532 ± 0.035`, and routed evaluation had lower peak
  allocated memory under this runner. It was about `1.863 ± 0.071x` slower than
  the fastest fixed route. The files contain p95, not p90, and the result is not
  a preregistered matched-compute Pareto claim.
- Trained identifiable-map benchmark: `task_reconstruction` and `combined`
  used the same inference graph, both saturated accuracy, and had statistically
  indistinguishable forward timing. Mean median latency was approximately
  `0.2762 ms` and `0.2753 ms`, respectively, at batch 192 on GB10; both used
  35,069,440 peak allocated bytes under the benchmark runner.
- Corrected Gate-3 base evaluation: four final reports and the paired base
  comparison are complete under `artifacts/gate3/`. All nine candidate-kind
  paired intervals include zero.
- Corrected gauge evaluation: all 16 final reports and all eight seed-matched
  paired comparisons are complete under `artifacts/gate3g/`. These are
  fixed-expert embedding diagnostics, not translator or conversion evaluations.
- PR CI previously passed both supported Python versions. Re-run CI after the
  release commit.

No further large training run is planned. Only analysis, packaging, writing,
and verification remain unless a strict check fails.

## Scientific claim boundary

The strongest supported new result is deliberately narrow: the implementation
recovers a finite dihedral family of exact three-term maps on one synthetic
six-sector cellular annulus. The model uses explicit identifying markers and a
flattened MLP to select from a hard-coded finite group-action basis. This is a
controlled implementation and exactness study, not a general graph neural
network or universal representation translator.

The work does not currently establish any of the following:

- superiority of cone or RTD training losses;
- conversion quality on real data or out-of-distribution structures;
- general equivalence between graphs, cellular complexes, and sheaves;
- a learned quasi-isomorphism or exact sequence—the verified identity is the
  chain-map law up to a fixed numerical tolerance;
- a Langlands, eigensheaf, Fourier–Mukai, or category-theoretic ML result;
- a translator-based Gate-3 claim—the corruption script evaluates fixed expert
  embeddings only.

Report null and negative results prominently. Acyclic mapping cones show the
decoded maps are invertible in the synthetic template family; they do not prove
that the selected map is the correct target map. All structural contrasts are
descriptive and unadjusted, with five training seeds and an accuracy ceiling.

## Start the next session here

```bash
cd /home/seanjazm27/HOMYMOLY
git fetch origin
git status --short
git log -3 --oneline --decorate
gh repo view seanm27lol/HOMYMOLY \
  --json nameWithOwner,visibility,isPrivate,url
gh pr view 3 --repo seanm27lol/HOMYMOLY \
  --json state,mergedAt,headRefName,baseRefName,statusCheckRollup,url
gh pr list --repo seanm27lol/HOMYMOLY \
  --head agent/research-remediation --state open
```

The evidence was generated from commit `8021292`. Later documentation-only or
evidence-export commits are expected, but every compact result must retain the
generating commit and source hashes. If the worktree is unexpectedly dirty,
inspect it before doing anything; never discard another session's edits.

Quick artifact checks:

```bash
test "$(find artifacts/gate3g -maxdepth 2 \
  -name corruption_report_final.json | wc -l)" -eq 16
test "$(find artifacts/gate3g -maxdepth 2 \
  -name paired_comparison_final.json | wc -l)" -eq 8
sha256sum artifacts/identifiable-maps/campaign-summary.json
```

## Remaining work, in order

### 1. Build strict compact summaries

Add tests before trusting either new summarizer.

1. Create `scripts/summarize_gauge_corruption_campaign.py` and a focused test.
   It must validate all eight matched baseline/candidate pairs, report exact
   seeds and report/checkpoint/config/evaluator hashes, verify block and draw
   pairing, and aggregate each candidate-minus-baseline adjusted statistic
   across training seeds. Report mean, sample SD, df=7 t interval, exact sign
   test, and the absence of multiplicity adjustment. Its scope must remain
   “fixed-expert embedding diagnostic.”
2. Create `scripts/summarize_compute_campaign.py` and a focused test. It must
   validate the ten identifiable and five routing JSON files against their
   configs, checkpoints, runner hashes, hardware, and the sealed scheduler
   completion receipt. Preserve the routing p95/identifiable p90 distinction.
   State benchmark order, batch, dtype, timing method, exclusions, and whether
   raw iteration timings were retained.
3. Independently recompute the published aggregates from the generated summaries
   once. Any mismatch is a stop condition, not an invitation to choose a more
   favorable statistic.

### 2. Export publication evidence

Extend `scripts/export_artifact_bundle.py` with explicit include patterns, or
add an equivalently strict publication exporter. Write a curated bundle under
tracked `results/` with a manifest containing path, byte count, SHA-256, source
commit, generating command, and schema version.

Include compact summaries, endpoint tables, gate decisions, final corruption
reports or lossless compact derivatives, and benchmark summaries. Exclude
checkpoints, prediction JSONL, histories, scheduler logs, caches, environments,
and other large raw artifacts. `/artifacts/` is intentionally ignored and is
not durable journal evidence by itself. Never force-add multi-gigabyte files.

### 3. Make the paper journal-ready

Rewrite `docs/18-paper.md`; do not merely append raw experiment logs. The paper
must be a coherent technical manuscript that a reader outside this project can
understand without reading the source code. It must contain:

- a precise research question and restrained novelty statement grounded in the
  cited primary literature;
- plain-language definitions before notation for RTD/SRTD, filtered mapping
  cones, typed maps, chain maps, exactness defects, and the cellular annulus;
- an architecture/data-flow figure and a table defining every loss and ablation;
- the frozen design, seeds, denominators, hardware, software, hashes, stopping
  rules, estimands, uncertainty procedures, and multiplicity limits;
- the 10/10 recovery-gate result, analytic decoder and chance baselines, all
  structural null/negative results, and the exact claim boundary;
- the frozen routing result, trained compute results, and corrected Gate-3
  diagnostic results with compatible units and clearly labeled p90 versus p95;
- limitations covering ceiling effects, five seeds, synthetic data, privileged
  historical routing supervision, target-view historical translators,
  fixed-expert Gate-3 scope, timing order, and artifact boundaries;
- code/data availability, reproducibility, ethics, and declaration sections.
  Do not invent affiliation, ORCID, funding, conflicts, license, or other author
  metadata—mark those as author-supplied submission fields if still unknown.

Remove every `PENDING`, `TODO`, stale pre-remediation result, unsupported “first”
claim, and implication of Langlands/Fourier–Mukai validation. Every number in a
paper table should be traceable to a tracked machine-readable result.

Render the PDF and inspect it, not just the Markdown:

```bash
.venv/bin/python scripts/render_paper.py \
  --input docs/18-paper.md \
  --output docs/18-paper.pdf
pdfinfo docs/18-paper.pdf
pdftotext docs/18-paper.pdf - | rg -n \
  'PENDING|TODO|FIXME|placeholder|untrained benchmark'
```

Render every page to images or contact sheets and visually check clipping,
tables, equations, citations, captions, page breaks, font sizes, and link text.
The committed PDF must match the committed Markdown.

### 4. Make the technical documentation understandable

Update, at minimum:

- `README.md`: current outcome, a five-minute smoke path, evidence locations,
  and a short “what this does not show” section;
- `docs/08-claims-ledger.md`: record the annulus experiment as completed while
  keeping structural superiority unsupported and Gate-3 conversion untested;
- `docs/20-audit-remediation.md`: mark completed remediations and remaining
  evidence/publication boundaries;
- `docs/21-identifiable-typed-map-protocol.md`: add the frozen result without
  changing the historical protocol;
- new `docs/23-identifiable-results.md`: a standalone, readable methods/results
  record with an evidence map and exact provenance;
- this handoff: move superseded launch instructions into a short historical
  appendix if useful, but keep the current resume path first.

Use consistent terms across code, paper, and docs. Define each abbreviation on
first use; state tensor/matrix shapes where they clarify the architecture; give
units, denominators, seed counts, uncertainty type, and scope beside results.

### 5. Run the release checks

```bash
uvx ruff check .
.venv/bin/python -m pytest -q
git diff --check
uv lock --check
docker compose config --quiet
bash -n scripts/train_gate2.sh scripts/run_gate2_cron.sh
```

Also validate JSON schemas/hashes, scan tracked history and the staged diff for
private keys or high-confidence credentials, check for accidental large files,
and confirm the PDF contains no stale language. A repository-wide formatter may
rewrite unrelated legacy files; format only changed Python files unless the
scope is intentionally expanded.

### 6. Commit and push everything publishable

Inspect every change and stage exact paths only—never use `git add .` or
`git add -A`. A sensible sequence is:

1. strict summarizers, tests, and tracked compact evidence;
2. paper, PDF, README, claims ledger, results record, and this handoff;
3. any narrowly scoped release correction found by final audit.

For each commit:

```bash
git status --short
git diff --check
git diff --cached --check
git diff --cached --stat
git commit -m "<specific message>"
git push origin agent/research-remediation
```

PR #8 is the current draft release PR. Keep it draft until the paper and
evidence are complete. After every pushed commit, wait for every check and
re-run:

```bash
gh pr checks 8 --repo seanm27lol/HOMYMOLY --watch
gh repo view seanm27lol/HOMYMOLY \
  --json visibility,isPrivate,url
git status --short
```

Verify or attempt `main` branch protection if account/repository policy permits.
Do not merge merely to make the tree look finished; the manuscript, compact
evidence, PDF inspection, and CI must all be complete first.

## Safe delegation if another session takes over

These tasks can proceed in parallel after assigning non-overlapping files:

- Evidence agent: both strict summarizers, tests, compact export, and independent
  arithmetic/hashing audit.
- Manuscript agent: `docs/18-paper.md`, citations, tables, and author-readable
  narrative using only frozen evidence.
- Documentation/release agent: README, claims/remediation/results docs, PDF visual
  audit, secret/size scan, GitHub privacy/PR/CI checks.
- Primary agent: reconcile all claim boundaries, run the integrated suite, stage
  exact paths, commit, push, and perform the final requirement-by-requirement
  audit.

No agent should edit the same file concurrently. The primary agent must inspect
and validate every delegated result before committing it.

## Stop conditions

Stop and preserve evidence if any expected checkpoint, config, script, commit,
sample count, seed, pairing key, schema, hash, or clean-provenance field differs.
Do not average mismatched runs, overwrite historical artifacts, silently replace
failed seeds, selectively report favorable endpoints, or broaden the claim to
match an attractive result.

## Continuation prompt

Use this prompt in a fresh high-quality Codex session:

> Continue HOMYMOLY from `docs/22-overnight-handoff.md`. The 56-step GB10
> campaign, corrected Gate-3 base reports, all 16 gauge reports, and eight gauge
> paired comparisons are complete. Follow the remaining work in order. Produce
> strict tracked evidence, make the paper journal-ready and the technical docs
> clear to an outside reader, visually inspect the PDF, run the complete release
> audit, and commit and push every publishable file to the private GitHub branch.
> Never commit checkpoints or broaden claims beyond the frozen evidence.
