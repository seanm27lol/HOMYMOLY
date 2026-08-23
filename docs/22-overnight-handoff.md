# HOMYMOLY journal-release handoff

Last updated: 2026-08-23. Repository: `/home/seanjazm27/HOMYMOLY`.

**The journal release is complete.** The training campaign, the strict compact
evidence, the manuscript and PDF, and the technical documentation are all done
and committed. This document is now a release record plus a short list of what a
next session would actually work on. Superseded launch instructions are in the
[historical appendix](#historical-appendix).

Do not rerun the 40-run campaign. It is sealed, and every provenance check
passes.

## Start the next session here

```bash
cd /home/seanjazm27/HOMYMOLY
git fetch origin
git status --short
git log -3 --oneline --decorate
python scripts/export_publication_evidence.py --verify-only
gh repo view seanm27lol/HOMYMOLY --json nameWithOwner,visibility,isPrivate,url
```

Read, in order: [`README.md`](../README.md) for the outcome,
[`docs/23`](23-identifiable-results.md) for the results record, and
[`docs/18-paper.md`](18-paper.md) for the manuscript.

## What was delivered

| item | state |
|---|---|
| 40-run identifiable campaign, 56/56 scheduler steps | complete and sealed |
| strict compact summarizers (gauge, compute) plus tests | `scripts/summarize_{gauge_corruption,compute}_campaign.py`, 18 tests |
| tracked publication evidence | `results/`, 48 files, 2.85 MB, manifest verified |
| journal manuscript | `docs/18-paper.md`, 12 sections |
| rendered PDF | `docs/18-paper.pdf`, 15 pages, every page visually inspected |
| results record | `docs/23-identifiable-results.md` |
| README, claims ledger, remediation record, protocol appendix | updated |

### Verified provenance

- All **296** files in the sealed scheduler receipt verify by byte count and
  SHA-256 against the on-disk artifacts.
- All **48** gauge report, checkpoint, and config hashes verify.
- Launch fingerprint
  `44408d7adf8467e594879b46e25a1cb7fd89a7e7a5d5f3446548bcbf3ed1096e`; campaign
  commit `8021292e97abfec91768f1b5437c883a42c29c60`; strict summary SHA-256
  `0cd0defb0b0d41b5f7563c364cfdda62cb72c5e5a845bdd5d1ab76a2e1cb953c`.
- No stop condition fired.

### Headline results

- **Exact recovery.** Every objective with task or reconstruction supervision
  reached transformation and cell-face accuracy 1.000 on all five seeds, map MSE
  at 1e-16. Engineering recovery gate 10/10 applicable runs.
- **Structural nulls.** All 21 declared continuous contrast intervals contain
  zero, under a saturated ceiling that a closed-form analytic decoder also
  reaches.
- **Acyclicity is not correctness.** `cone_only` (0.0815) and `rtd_only`
  (0.0833) identify at chance (0.0833) while producing acyclic cones in 6,000 of
  6,000 evaluated examples.
- **Routing.** Frozen five-seed margin +0.1098, 95% CI [+0.0953, +0.1243], under
  privileged regime distillation and the disclosed pre-freeze deviation.
- **Corruption diagnostics.** Nine Gate-3 base intervals and three eight-seed
  gauge intervals all contain zero.
- **Trained compute.** Dense-to-routed 1.532 ± 0.035; routed-to-fastest-fixed
  2.269 ± 0.043; identifiable p90 and routing p95 never pooled.

## Correction issued in this release

An earlier revision of this handoff recorded the trained routing benchmark as
"about `1.863 ± 0.071x` slower than the fastest fixed route." **That figure is
wrong.** It is not reproducible from any artifact in this repository under any
ratio definition we could construct — median, mean, p95, throughput, or
best/mean/max fixed route — and it appears in no machine-readable result.

The recomputed value from the five sealed trained benchmarks is
**2.269 ± 0.043**, which is *less* favorable to routing than the figure it
replaces. The correction is recorded in `docs/18-paper.md` §6.5,
`docs/23-identifiable-results.md` §5, and `docs/20-audit-remediation.md`.

Separately, two `artifacts/benchmarks/compute-remediation*.json` files record
`checkpoint: null`. They timed an untrained model. They are excluded from every
reported result, and `scripts/summarize_compute_campaign.py` now refuses any
routing benchmark without a checkpoint so an untrained measurement cannot
re-enter the record.

## Scientific claim boundary

The strongest supported new result is deliberately narrow: the implementation
recovers a finite dihedral family of exact three-term maps on one synthetic
six-sector cellular annulus, using explicit identifying markers and a flattened
MLP to select from a hard-coded finite group-action basis. This is a controlled
implementation and exactness study, not a general graph neural network or
universal representation translator.

The work does not establish superiority of cone or RTD training losses;
conversion quality on real or out-of-distribution data; general equivalence
between graphs, cellular complexes, and sheaves; a learned quasi-isomorphism or
exact sequence; any Langlands, eigensheaf, Fourier–Mukai, or category-theoretic
result; or a translator-based Gate-3 claim.

Report null and negative results prominently. Acyclic mapping cones show the
decoded maps are invertible in the synthetic template family; they do not prove
the selected map is the correct target map.

## What a next session should actually do

Ranked by scientific value, not by effort.

1. **Build a harder benchmark.** The current annulus saturates — an analytic
   marker decoder reaches 1.000 — so its structural nulls cannot distinguish
   "cone and RTD terms are useless" from "this task is too easy to reveal their
   value." A template family where the correct map is *not* analytically
   attainable is the single highest-value next step. More seeds on the current
   benchmark would add nothing.
2. **Run the canonical target-held-out Gate-2 configuration.** Still pending:
   `.venv/bin/python -m homymoly train --config configs/gate2.yaml`. Treat it as
   an integration/null diagnostic, not conversion evidence, because the held-out
   targets are unidentifiable from graph inputs in the current generator.
3. **Design a real C1/C2 test.** A task where homological exactness is causally
   relevant, with genuine translator reconstruction measured on held-out
   examples. C4 is already answered on the identifiable benchmark.
4. **Complete the author-supplied submission fields** in `docs/18-paper.md` §12
   — affiliation, ORCID, funding, competing interests, data and code license.
   These are deliberately unset rather than guessed and must be filled before
   submission.

## Release checks

```bash
uvx ruff check .
.venv/bin/python -m pytest -q
git diff --check
uv lock --check
docker compose config --quiet
bash -n scripts/train_gate2.sh scripts/run_gate2_cron.sh
python scripts/export_publication_evidence.py --verify-only
```

For the paper, re-render and re-inspect rather than trusting the Markdown alone:

```bash
.venv/bin/python scripts/render_paper.py \
  --input docs/18-paper.md --output docs/18-paper.pdf
pdfinfo docs/18-paper.pdf
pdftotext docs/18-paper.pdf - | rg -n 'PENDING|TODO|FIXME|placeholder|untrained benchmark'
pdftoppm -png -r 100 docs/18-paper.pdf /tmp/page && ls /tmp/page*
```

`scripts/render_paper.py` now stages its HTML beside the output rather than in
`/tmp`, because a snap-confined Chromium has a private `/tmp` and would silently
render its own "file not found" page into a one-page PDF while exiting zero. The
renderer fails loudly if the page count is implausible for the source length.

## Stop conditions

Stop and preserve evidence if any expected checkpoint, config, script, commit,
sample count, seed, pairing key, schema, hash, or clean-provenance field differs.
Do not average mismatched runs, overwrite historical artifacts, silently replace
failed seeds, selectively report favorable endpoints, or broaden a claim to match
an attractive result.

Both summarizers and the exporter enforce this in code: they fail closed rather
than degrading to a partial result.

## Historical appendix

Superseded. Retained for provenance only.

- **Draft release PR #8** was the collection point for post-remediation commits
  and has been merged into `main`. Historical remediation PR #3 merged earlier
  with both CI jobs green.
- **Campaign launch instructions.** The managed HOMYMOLY cron entry that drove
  the 56-step scheduler was removed after completion. Reinstalling it is not
  part of any remaining work; the campaign must not be relaunched absent a real
  provenance mismatch.
- **Repository visibility.** `seanm27lol/HOMYMOLY` is `PRIVATE`. The repository
  was briefly public earlier, and making it private cannot retract that
  historical exposure. GitHub secret scanning is unavailable for this
  repository/account configuration; only a local tracked-text
  credential-pattern scan was run.
- **Withdrawn figures.** The architecture-only 1.77× routed/dense ratio and the
  `1.863 ± 0.071` routed-to-fastest-fixed ratio are both withdrawn. See the
  correction section above.
