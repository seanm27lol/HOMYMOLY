# Gate-2 run handoff (2026-08-03)

**STATUS: run 9 is in progress** (confirming run 8's Gate-4 pass at the
configured LR). Run 8 was the first run to clear the Gate-4 utility bar —
see "Run 8 outcome". Earlier history: runs 4–7 below.
gates but its router collapsed to always-cell (graph expert broken). Run 5
fixed the graph expert; run 6 gave the router regime-informative features;
both still failed the Gate-4 utility bar for reasons traced to the routing
oracle. See the run sections below.

This note records exactly what was done so any person or tool picking this
up has full context. It supplements
[`12-gate2-training.md`](12-gate2-training.md), which describes the stack
itself (note: the sheaf expert description there predates the holonomy
pathway added in the second session).

## Run 4 outcome (2026-08-03, commit `3e464ff` tree)

Status `completed`: 40 epochs / 2720 steps, all four phases.

- **fixed-expert gate: passed** — cell +0.351, sheaf +0.500 over the graph
  route. Sheaf held 1.000 test accuracy on its regime through the full run.
- **translator gate: passed** — 14.4% relative improvement in
  reconstruction+consistency over its own baseline (minimum 2%). The gate
  carries its own claim boundary: it does not replace the Gate-3
  predictive-value criterion. Translated task accuracy stayed at chance
  (~0.50) — recorded for the Gate-3 ablation work.
- **router: collapsed** — `route_utilization_cell = 1.0`, graph/sheaf 0.0,
  `regime_route_mutual_information = 0.0`, `route_accuracy` 0.505,
  `oracle_regret` 0.269. Hard accuracy 0.675 = the cell expert's accuracy;
  soft accuracy 0.772; utility oracle 0.950.
- **Root cause of the collapse (diagnosed, not speculative):** the graph
  expert scores 0.5 on its own regime in every run, so always-cell is
  near-optimal for 2/3 of regimes. The graph label is sign agreement across
  two *unmarked* anchor edges — a per-edge statistic that the expert's
  mean-pooled readout dilutes below the noise floor (identical failure class
  to the pre-fix sheaf expert: the signal exists but no pathway preserves it
  through aggregation).
- **Prescription (same class as the sheaf fix):** add a masked max-pooled
  edge readout to `GraphExpert` so one informative edge survives pooling,
  then rerun. Only after all three routes work can Gate-4 routing metrics
  (utilization, regime-route MI, oracle regret) be interpreted.

Artifacts archived at `artifacts/gate2-run4-completed-router-collapsed/`
(summary, history, test predictions, checkpoints for all four phases).

## Run 7 outcome (2026-08-03, table oracle shipped)

Status `completed`; both engineering gates passed. The oracle was now
regime-aligned and the supervision learnable (offline probe 0.556 vs
marginal 0.335), yet the router converged to *exactly* uniform predictions
(`oracle_route_ce` → ln 3, route accuracy 1/3, MI ~ 0).

- **Root cause (isolated by offline replication with the actual router
  module and the engine's exact loss):** the single 40-epoch cosine
  schedule leaves router warmup at LR ~1e-4 and joint finetune at ~1e-6.
  At those rates the router learns nothing (0.324); the same setup at
  3e-4–1e-3 learns readily (0.54–0.59). The entropy/cost/balance loss
  terms were measured and exonerated. Classification: simple optimization
  scheduling, not architecture.

Artifacts archived at `artifacts/gate2-run7-starved-router-lr/`.

## Run 8 outcome (2026-08-03): Gate-4 utility bar passed, with a caveat

Status `completed`; both engineering gates passed; **the router cleared
the Gate-4 criterion for the first time**:

- hard accuracy **0.788** vs best fixed route 0.667, random 0.672, dense
  ensemble 0.734 (the dense ensemble also runs ~3× the compute);
  `oracle_accuracy` 0.999, route accuracy 0.574 (at the linear probe
  ceiling), regime-route MI 0.124 (was ≤0.012), utilization
  293/312/313 — non-collapsed with per-regime accuracies 0.78–0.79.
- **Caveat that requires a confirmatory run:** the intended per-phase LR
  restart never fired (the hook keyed on `phase_index !=
  state.phase_index`, but `state.phase_index` is advanced at each phase
  end, making the condition always false). Run 8's router phases ran on
  the stale first-phase schedule mirror-rising past `T_max`, landing at an
  unintended ~2–3e-3 effective LR (~10× configured 3e-4). The router
  learned well at that rate — consistent with the offline LR probe that
  showed 1e-3 ≫ 3e-4 — but the record must state that the result was
  obtained at an unplanned LR.
- **Fix shipped:** restart now keys on `first_epoch == 0` (verified
  in-engine: one clean cosine per phase). Run 9 repeats the experiment at
  the configured LR with the corrected mechanism. If run 9 is weaker, the
  LR sensitivity itself is the finding and a deliberate router-LR change
  becomes a pilot-campaign decision.

Artifacts archived at `artifacts/gate2-run8-gate4-passed-lr-anomaly/`.

## Run 6 outcome (2026-08-03, regime-informative router features)

Status `completed`; both engineering gates passed. Experts: graph 0.997,
cell 0.726, sheaf 1.000 on their regimes; oracle accuracy 0.914.

- Router improved but still failed the Gate-4 utility bar: hard accuracy
  0.672 vs best fixed route 0.682 (random 0.675, dense 0.741, oracle
  0.914); route accuracy 0.47, MI 0.012, and the graph route was never
  selected despite having the strongest amplitude cue (F=37).
- **Root cause, chased through four measured estimators.** The routing
  supervision (oracle route) was the bottleneck, not the features:
  - *raw logP utility (shipped)*: the graph expert is 0.997 accurate but
    underconfident (logP ~ -0.3) next to a confidently correct cell expert
    (logP ~ -0.05), so cell won the graph regime 62% of the time;
    supervision probe ceiling 0.536 vs marginal 0.484.
  - *per-route temperature scaling*: rejected — miscalibration is
    regime-conditional, not global (fitted T ≈ [0.98, 1.22, 1.22];
    picks-own 42%, probe 0.529).
  - *correctness-first utility*: rejected — lucky cross-regime guesses
    dominate (marginal 0.693 vs probe 0.712).
  - *regime-conditional accuracy table (shipped in run 7)*: oracle becomes
    perfectly regime-aligned, probe 0.556 vs marginal 0.335 — learnable
    well beyond chance, matching the benchmark's design intent.
- The table is fitted on the validation split at router-warmup entry,
  persisted in checkpoint buffers, and is used only as a supervision
  target. **Comparability note:** route-accuracy/oracle metrics in runs 7+
  use the table oracle; runs 4-6 used the logP oracle. A side effect of the
  table oracle: route accuracy and regime-route accuracy now measure the
  same thing.

Artifacts archived at `artifacts/gate2-run6-biased-oracle/`.

## Run 7 hypothesis (in progress)

With regime-aligned, learnable supervision, the router should show:
utilization tracking regime frequencies (~1/3 each), route accuracy toward
the probe ceiling (≥0.55 linear, more for the MLP), hard accuracy clearly
above the best fixed route (0.682) and toward regime-routing utility
(0.914), and nonzero regime-route MI. If utilization still collapses, the
remaining suspects are router optimization dynamics (entropy/load-balance
weights, straight-through temperature), not supervision or features.

## Run 5 outcome (2026-08-03, post-graph-fix tree)

Status `completed`; both engineering gates passed again.

- **All three experts specialized**: graph 0.997, cell 0.726, sheaf 1.000 on
  their own regimes (test). Oracle accuracy jumped 0.748 → 0.907; utility
  oracle 0.960. The three-route system now works as the benchmark intends.
- **Router still regime-blind**: mutual information 0.002, route accuracy
  0.441, hard accuracy 0.672 (random route 0.673, dense 0.741). Utilization
  split cell/sheaf but not conditionally on regime.
- **Root cause (measured, not speculative):** the shipped routing
  diagnostics are regime-blind by construction — one-way F-statistics
  across regimes are 0.00 (densities, active-face fraction), 0.56–0.70
  (sheaf residual/deviation), 3.74 (edge energy). The generator exposes
  route reliability through *overlapping amplitude ranges* (the intended
  label-independent cue, capped at ~0.80 identifiability by the
  anti-shortcut design), and every shipped diagnostic is a mean or count
  that dilutes exactly those amplitude cues — the same dilution failure
  class as the sheaf and graph experts, one level up.
- **Gate-3 prep finding:** both translators sit at chance task accuracy
  (damage rate 0.500), and their structural diagnostics show zero
  correlation with conversion damage (|ρ| ≤ 0.01). The Gate-3
  predictive-value criterion is untestable until translators are
  task-competent. `GraphToSheafTranslator` has the same missing-holonomy
  defect the sheaf expert had (per-edge residuals only); that is the first
  translator fix when Gate-3 work begins.

Artifacts archived at `artifacts/gate2-run5-regime-blind-router/`.

## Run 6 hypothesis (in progress)

The router context was extended with label-independent amplitude cues:
per-channel max-abs for node/edge features plus mean stalk norm (context
dim 6 → 13). Measured F-statistics: node ch0 max-abs 37.1 (graph), edge
ch1 max-abs 8.8 (cell), stalk norm 2.3 (sheaf, weak by design). A linear
regime probe reaches 0.573 held-out (chance 0.333); the router MLP has
more capacity. Success looks like: regime-route MI substantially above
zero, utilization tracking regime frequencies, hard accuracy approaching
oracle (0.907) rather than the best fixed route, and route accuracy toward
the ~0.80 identifiability ceiling. If MI stays near zero, the next suspect
is the router training dynamics (entropy/load-balance terms vs
supervision), not the features — the probe already shows the information
is present.

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
4. **The NaN fix, take one — wrong** (commit `f1b9554`). `clamp_min(eps)` on
   the residual norm. Stress test passed and run 3 was launched, but it
   crashed in translators epoch 1 identically: clamping *after* the sqrt
   still chains `0 * inf = NaN` in backward (the clamp gate contributes 0,
   the sqrt derivative contributes inf). Take one contained the padded-edge
   case only.
5. **The NaN fix, take two** (current HEAD). Eps moved *inside* the sqrt
   (`.sum(-1).add(eps).sqrt()`) in both `GraphToSheafTranslator` and
   `ConnectionSheafExpert`, so the derivative at zero residual is finite
   (`0.5/sqrt(eps)`). Verified in the exact crash regime — node stalks
   forced to exact zero at every valid edge with all loss terms active,
   gradients finite — and locked in by a permanent regression test
   (`test_sheaf_translator_backward_is_finite_at_zero_residual`; suite now
   100 passed). **Run 4 launched** from a clean slate (runs archived to
   `artifacts/gate2-run1-gate-failed/`, `artifacts/gate2-run2-nan-translators/`,
   `artifacts/gate2-run3-nan-translators/`).

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
