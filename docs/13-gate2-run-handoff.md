# Gate-2 run handoff (2026-08-03)

**STATUS: run 3 is in progress** (launched after two design fixes — see
"Second session" below). Run 1 ended `gate-failed`; run 2 passed the
fixed-expert gate but hit a latent NaN crash in the translators phase, now
fixed. Watch `artifacts/gate2/status.json`; final judgment comes from
`artifacts/gate2/summary.json`, not the process exit code.

This note records exactly what was done so any person or tool picking this
up has full context. It supplements
[`12-gate2-training.md`](12-gate2-training.md), which describes the stack
itself (note: the sheaf expert description there predates the holonomy
pathway added in the second session).

## Second session: sheaf-expert fix, translator NaN fix, run 3

Run 1's gate failure was traced to a design defect, not noise:

1. **Root cause of the sheaf failure.** The confirmatory sheaf label is
   *cycle holonomy*: a defect rotation is composed onto one face edge. The
   expert's only sheaf-specific input was the per-edge residual between node
   stalk vectors and transports — but node-field angles and connection frame
   angles are drawn independently, so that residual is label-independent
   noise. Verified empirically: per-edge residual range 1.83–2.51 for both
   labels, while the per-face holonomy defect is exactly 0.0 (label 0) vs
   2.0 (label 1). No per-edge computation can see the signal.
2. **The fix** (commit `40fbcbb`). New `ops.face_holonomy` computes every
   face's transport holonomy exactly (oriented boundary coefficients;
   complex product, valid because transports are planar rotations and hence
   commute; FP32; padding-safe; 5e-8 vs brute force). `ConnectionSheafExpert`
   encodes each face's `H − I`, scatters face messages to nodes, and reads
   out mean **and max** over faces — the max is essential because the defect
   is a single-face event and mean pooling dilutes it (0.60 → 1.00 held-out
   accuracy). The sheaf route view now receives `face_index`/`face_mask`
   (a cellular sheaf is defined over the complex; faces are observation-level
   structure, not supervision metadata). The expert remains provably
   insensitive to `face_active`. Parameter counts stay matched:
   graph 817k, cell 933k, sheaf 917k.
3. **Run 2 outcome.** The fixed-expert gate **passed**: sheaf 1.0 and cell
   ~0.85 on their own regimes (graph route still at chance on its own —
   see open questions). The run then crashed in translators epoch 1 with a
   non-finite gradient. Anomaly-mode reproduction named `SqrtBackward0` at
   the sheaf translator's `residual_norm`: the consistency loss drives
   residuals toward zero, where `sqrt` has infinite derivative. Latent
   until now because run 1 never reached the phase and smoke schedules
   never push residuals near zero.
4. **The NaN fix** (commit `f1b9554`). `clamp_min(eps)` on the residual
   norm in both `GraphToSheafTranslator` and `ConnectionSheafExpert`
   (zero derivative below eps). Stress test: residual driven to exactly
   0.0 over 300 steps, all gradients finite. Suite 99 passed, ruff clean,
   four-phase smoke completed.
5. **Run 3 launched** from a clean slate (old runs archived to
   `artifacts/gate2-run1-gate-failed/` and `artifacts/gate2-run2-nan-translators/`;
   the src edits invalidate checkpoint fingerprints by design, so resume
   from run 2 was impossible anyway).

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
about the current mechanism, not relabeled. The sheaf half of the question is
now answered (see "Second session" above: the expert had no pathway that
could observe cycle holonomy). Still open: the **graph expert stays at
chance on its own regime** in both runs — its label is which signs meet
across two vertex-disjoint anchor edges with no marker of which vertices are
anchors, so mean-pooled readout may be unable to locate the signal; that is
the next design question if a future gate depends on the graph route. Do not
weaken the gate to make it pass.

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
