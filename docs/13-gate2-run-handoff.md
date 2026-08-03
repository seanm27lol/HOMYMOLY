# Gate-2 run handoff (2026-08-03)

This note records exactly what was done in the session that launched the first
full Gate-2 training run, so any person or tool picking this up has full
context. It supplements [`12-gate2-training.md`](12-gate2-training.md), which
describes the stack itself.

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
- `tail artifacts/gate2/metrics/metrics.jsonl` — per-epoch train/validation
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

## Early run state (epoch 3, phase `fixed_experts`)

Everything finite and logging correctly. Notable: the cell expert already
shows intended-regime specialization (`expert_cell_on_cell_accuracy ≈ 0.997`
with all other routes at chance 0.5). Most other metrics are still at chance —
expected this early; the four phases (fixed experts → router warmup →
translators → router joint) run sequentially per `configs/gate2.yaml`.

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
