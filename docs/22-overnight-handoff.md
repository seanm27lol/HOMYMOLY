# Overnight execution handoff

Last updated: 2026-08-22. Repository: `/home/seanjazm27/HOMYMOLY`.

This document is the fail-safe continuation path if the active Codex session
ends. It is intentionally conservative: never launch the scientific campaign
from a dirty worktree, and never enable a provisional manifest merely to make
progress.

## Current boundary

The audited experiment is a five-seed, eight-mode pilot on a six-sector
cellular annulus. The campaign contains 40 training runs, ten trained
identifiable-checkpoint benchmarks, five trained routing-v2 benchmarks, and a
strict summary gate between training and benchmarking: 56 steps in total. The
scheduler is resumable and checks the physical GPU before every step.

Until the pre-freeze checks below pass, the manifest must contain
`"execution_enabled": false` and no HOMYMOLY campaign cron entry should exist.
The resident Ollama and UI CUDA contexts may remain present: the intended
policy permits at most two contexts and 49,152 MiB aggregate memory, but still
requires three utilization samples at or below 10 percent plus an immediate
recheck before every step. The scheduler never terminates another process.

## Freeze checklist

From the repository root:

```bash
cd /home/seanjazm27/HOMYMOLY
uvx ruff check .
.venv/bin/python -m pytest -q
git diff --check
uv lock --check
docker compose config --quiet
bash -n scripts/train_gate2.sh scripts/run_gate2_cron.sh
```

The remediation Python files were checked with Ruff's formatter explicitly.
Do not run a repository-wide format rewrite during freeze: unrelated legacy
files are intentionally outside this change.

Before activation, additionally require:

1. `docs/21-identifiable-typed-map-protocol.md`, the full config, training
   runner, model module, strict summarizer, scheduler, and campaign manifest are
   final and mutually consistent.
2. The strict summarizer tests use safe valid checkpoint fixtures, actually
   invoke checkpoint validation, and reject dirty provenance.
3. Scheduler tests cover input-bound downstream markers, aggregate completion
   revalidation, effective-policy/environment fingerprinting, bounded retries,
   symlink-safe artifact confinement, and global crontab locking.
4. The campaign manifest is regenerated for `data.sectors: 6`, topology counts
   12/18/6, Betti numbers `[1,1,0]`, RTD prefix 48, and cone weight 0.1.
5. The branch is committed and `git status --short` is empty. Record the commit,
   config SHA-256, manifest SHA-256, protocol SHA-256, and launch fingerprint.

First commit and push the entire frozen tree while the regenerated manifest is
still disabled. Then make a second commit whose only change is
`"execution_enabled": false` to `true`, validate it, push it, and require a
clean worktree before launch. Do not reuse an enabled pre-audit manifest.

## Commit and publish the freeze

Inspect the entire worktree first. Stage exact paths only; never use
`git add .` or `git add -A`. Commit and push the disabled freeze, then commit
and push the single manifest enablement described above. Open a draft pull
request from `agent/research-remediation` against `main`. The GitHub repository
must remain private. After the enablement commit, do not edit tracked files
until all 40 training provenance records have been written.

## Install and start the resumable campaign

Use the policy stored in the manifest; do not pass ad-hoc threshold overrides:

```bash
cd /home/seanjazm27/HOMYMOLY
.venv/bin/python scripts/gpu_idle_train.py \
  --project-root . \
  --campaign-manifest configs/identifiable-maps/gb10-campaign.json \
  --print-fingerprint

.venv/bin/python scripts/install_training_cron.py \
  --project-root . \
  --campaign-manifest configs/identifiable-maps/gb10-campaign.json \
  --interval-minutes 5

crontab -l
```

Invoke the scheduler once manually so an already-idle GPU does not wait for the
next cron tick:

```bash
.venv/bin/python scripts/gpu_idle_train.py \
  --project-root . \
  --campaign-manifest configs/identifiable-maps/gb10-campaign.json
```

Exit code 75 means the GPU became busy and is a normal pause; cron resumes at
the next incomplete step. A latched deterministic failure must be investigated
from its attempt-numbered log and explicitly reset with the runner's documented
reset option—do not delete scheduler state or checkpoints.

Scheduler state lives under:

```text
artifacts/scheduler/identifiable-gb10-factorial-v1/
```

Training artifacts live under:

```text
artifacts/identifiable-maps/campaign/seed-2026082{1..5}/<mode>/
```

## Validate and summarize results

The scheduler runs the strict summarizer as step 41, immediately after all 40
training runs and before any benchmark. After all steps report complete,
inspect its output. The following command is an optional idempotent manual
revalidation (use `--help` first if its final CLI differs from this frozen
intent):

```bash
.venv/bin/python scripts/summarize_identifiable_campaign.py \
  --campaign-root artifacts/identifiable-maps/campaign \
  --source-config configs/identifiable-maps/gb10-full.yaml \
  --output artifacts/identifiable-maps/campaign-summary.json
```

The primary outcome is only the development-informed recovery/exactness gate
for every `task_reconstruction` and `combined` seed. All `+cone`, `+RTD`, and
`combined` contrasts are descriptive and unadjusted because the perfect
analytic marker decoder creates an accuracy ceiling. Never relabel those
contrasts as confirmatory superiority evidence.

## Remaining release work

1. Rerun the four corrected Gate-3 reports from the clean commit and regenerate
   their paired comparison so final script hashes and checkpoint-load metadata
   match.
2. Export compact checksummed evidence bundles from ignored `artifacts/` into
   tracked `results/`; never commit multi-gigabyte checkpoints.
3. Update `docs/18-paper.md` with the actual annulus and trained-compute results,
   including negative/null results, analytic baseline, ceiling limitation,
   n=5 caveats, and the scheduler/provenance boundary.
4. Render and visually inspect `docs/18-paper.pdf`.
5. Run the full release checks again, make a second exact-path results commit,
   push it, wait for draft-PR CI, and attempt `main` branch protection.
6. Scan current history for high-signal credential filenames/tokens. The repo is
   private now, but it was briefly public; privacy cannot retract prior exposure.

If any expected hash, config, commit, clean-status field, sample count, or
checkpoint identity differs, stop and preserve the artifacts. Do not average
in a mismatched run.
