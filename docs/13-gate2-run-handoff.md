# Gate-2 run handoff (2026-08-03)

**OUTCOME: the first full run finished with status `gate-failed` — see
"Run outcome" below. This is a recorded Gate-2 result, not a crash.**

This note records exactly what was done in the session that launched the first
full Gate-2 training run, so any person or tool picking this up has full
context. It supplements [`12-gate2-training.md`](12-gate2-training.md), which
describes the stack itself.

## Run outcome (first full run, 2026-08-03)

The run completed 8 epochs / 544 steps of the `fixed_experts` phase and then
stopped at the enforced phase gate `fixed_expert_specialization`:

- **cell route: PASSED** — improvement 0.4968 over the graph route;
  `expert_cell_on_cell_accuracy` = 1.0 on test (all cross-regime accuracies
  0.5, as intended).
- **sheaf route: FAILED to specialize** — improvement 0.0; every sheaf-expert
  accuracy stayed at chance (0.5) including on its own regime.
- Gate requires ≥2 routes improving over the graph route → `passed: false`,
  so translator/router phases never ran (by design; see the Gate-2 criterion
  in [`10-gb10-experimental-plan.md`](10-gb10-experimental-plan.md): the
  benchmark/model design is revised before routing work begins).
- Also notable: the **graph expert itself stayed at chance on its own regime**
  (0.5), so "improvement over graph" was measured against a non-learning
  baseline.

Full numbers: `artifacts/gate2/summary.json`, `artifacts/gate2/status.json`,
`artifacts/gate2/metrics/history.jsonl`, `artifacts/gate2/metrics/test_predictions.json`.
Checkpoints: `artifacts/gate2/checkpoints/best-fixed_experts.pt` and `last.pt`.

Per the plan's long-run discipline, this null result is recorded as evidence
about the current mechanism, not relabeled. Open questions for review: why the
sheaf expert did not learn (capacity? supervision signal? the hardcoded
"last two node channels = stalk vectors" contract at `models/experts.py:294`
not matching the confirmatory generator?), and whether the graph expert at
chance indicates a signal/optimization issue shared by the graph and sheaf
routes. Do not weaken the gate to make it pass.

## What was done in this session

1. **Twelve-scope audit of the entire repo** (parallel review agents, read-only):
   confirmatory data, core data pipeline, experts, model assembly, metrics,
   training engine, config/launch path, baselines, topology foundation,
   CLI/packaging, full test+lint health, infra scripts. **Zero launch blockers
   found.** Every scope was verified live on the GB10 (dry-run, four-phase
   smoke, crash-resume determinism: all reproducible metrics bit-identical
   after a simulated mid-epoch crash).
2. **Fixed the one red test** — `tests/test_training_engine.py:82` compared
   resumed vs original test metrics including the wall-clock key
   `hard_milliseconds_per_example`, which can never reproduce. Timing keys are
   now excluded from the comparison. This was a test bug, not a resume bug.
3. **`ruff check --fix`** — six cosmetic errors (import order, unused `noqa`,
   `typing.Callable`), all auto-fixed.
4. **Full gate green**: 99 passed, 1 environment-conditional skip; ruff clean;
   `git diff --check` clean.
5. **Committed and pushed** the complete Gate-2 branch:
   `agent/gate2-training` → `origin` (github.com/seanm27lol/HOMYMOLY).
6. **Launched the full run directly** (no cron, per instruction), detached with
   `setsid nohup` so it survives shell/session exit:

   ```bash
   cd ~/HOMYMOLY
   setsid nohup .venv/bin/homymoly train --config configs/gate2.yaml --resume \
     > artifacts/gate2/training.log 2>&1 < /dev/null &
   ```

## How to check on the run

- `cat artifacts/gate2/status.json` — phase, epoch, status, latest validation
  metrics (updated every epoch).
- `tail artifacts/gate2/metrics/history.jsonl` — per-epoch train/validation
  metrics.
- `tail artifacts/gate2/training.log` — stdout/stderr of the process.
- `pgrep -af "homymoly train"` — process liveness; `nvidia-smi` shows it under
  compute apps (~1 GiB).
- When finished: `artifacts/gate2/summary.json` holds the final report and
  `artifacts/gate2/checkpoints/` the per-phase `best.pt` plus `last.pt`.

## Critical caveats (read before touching anything)

- **Do not edit `src/homymoly/**`, `pyproject.toml`, or
  `scripts/train_gate2.sh` while the run is live or before any `--resume`.**
  The resume guard hashes these files; any change makes resume hard-fail.
  Docs/tests/configs under `configs/` are not hashed — but treat the tree as
  frozen anyway.
- **A gate-failed run still exits 0.** Judge the outcome from
  `summary.json` (`status: "completed"` vs `"gate-failed"`), not the exit code.
- **A non-finite loss/grad aborts the whole run loudly** (uncaught
  `FloatingPointError` traceback in `training.log`, `status.json` marked
  failed). Recovery: `homymoly train --config configs/gate2.yaml --resume`
  restarts from the last atomic checkpoint — but a *deterministic* bad batch
  would crash again at the same place and needs investigation first.
- Resume from a completed run re-evaluates rather than exiting instantly; that
  is expected.
- Do not run the pytest suite concurrently with training (one transient CUDA
  OOM was observed when the suite overlapped GPU work).

## Known non-blocking issues found by the audit (for later review)

- `training/engine.py:415-427` — phase trainability references nonexistent
  components `router_context`/`cheap_router` (silent `getattr` miss; the router
  itself is still trained). Cosmetic but misleading.
- `data/types.py:469` and `models/system.py:71` — docstrings say
  "without supervision metadata" / "label-independent" while the cell view
  includes ground-truth `face_active`. Safe for the confirmatory dataset (the
  active-face *count* is label/regime-independent by construction) but the
  docs should be aligned.
- `models/translators.py:248` — sheaf `structure_logits` are residual norms,
  not logits; nothing consumes them as logits.
- `training/io.py:86-89` — MetricLogger raises on re-logging an older step;
  harmless with `checkpoint_every: 1` but fragile for sparser intervals.
- `models/ops.py` — `scatter_add_` under CUDA is not bit-deterministic even
  with `deterministic: true` (torch 2.13); soften any bit-exact GPU
  reproducibility claim. CPU/exact-oracle paths are FP64-deterministic.
- `pin_memory: true` is silently a no-op with the custom batch type, and
  `StructuredBatch.to()` revalidates on every device transfer — perf only.
- `pip` metadata still says 0.1.0 (stale editable-install dist-info; code is
  0.2.0). Refresh with `.venv/bin/pip install -e .` when convenient.
- The idle-GPU cron scripts (`scripts/gpu_idle_train.py`,
  `scripts/install_training_cron.py`) are shipped and tested but **not
  installed** — the run was launched directly per instruction.
