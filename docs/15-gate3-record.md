# Gate-3 record (2026-08-03)

Gate 3 asks whether translator and structural-loss machinery adds value:
either a structural diagnostic adds predictive value for conversion damage
after controlling for reconstruction error, or a structural regularizer
improves a matched-compute downstream result (criterion in
[the plan](10-gb10-experimental-plan.md)). This document records the two
experiments and their outcomes. Run detail is in
[the handoff log](13-gate2-run-handoff.md); claims in
[the ledger](08-claims-ledger.md).

## Verdict

**Gate 3 is not passed, and the structural-loss mechanism is not supported
on this benchmark — recorded as evidence, not relabeled.** The routing
result from Gate 2 is unaffected: learned representation routing beats
fixed routes and the dense ensemble at matched compute regardless of the
structural terms, which are measured to be inert here.

## Experiment 1: corruption suite (C1, correlation question)

Design: `src/homymoly/data/corruptions.py` (three graded, per-sample,
deterministic channels: transport rotations, edge-cochain noise,
node-anchor noise); `scripts/eval_corruption.py` applies them to the
held-out split of the run-10 checkpoint at five severities and measures
task damage of the affected route's expert, reconstruction displacement,
and the exact SRTD between clean and corrupted embeddings (subsampled to
`rtd_max_points = 24`).

Results (60 batches per kind, 1530 examples; full numbers in
`artifacts/gate3/corruption_report_full.json`):

- Exact SRTD tracks damage strongly: Spearman ρ = 0.92 (sheaf), 0.82
  (cell), 0.81 (graph).
- **But it adds nothing beyond plain embedding displacement**: partial
  correlation controlling for reconstruction is ≈ 0 (−0.16 / +0.16 /
  +0.24), and displacement alone tracks damage equally (ρ 0.81–0.96).
- Per-example expert diagnostics are uninformative (|ρ| ≤ 0.16).

C1: **not supported** — structural defects predict damage, but not beyond
reconstruction error.

## Experiment 2: structural-loss ablation ladder (C2, intervention question)

Design: four full 40-epoch runs with identical seed and schedule (expert
phases verified bit-identical), differing only in translator-phase loss
weights: task-only → +reconstruction → +chain → full (configs in
`configs/gate3/`, runner `scripts/run_gate3_ablations.sh`). Each variant
also ran the corruption suite.

| variant | hard | dense | best fixed | oracle | g2c | g2s | route acc | MI | sheaf dmg | cell dmg | graph dmg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| task-only | 0.746 | 0.741 | 0.673 | 0.983 | 0.668 | 0.667 | 0.505 | 0.069 | 0.161 | 0.048 | 0.046 |
| +recon | 0.733 | 0.734 | 0.670 | 0.965 | 0.670 | 0.667 | 0.496 | 0.062 | 0.152 | 0.048 | 0.054 |
| +chain | 0.751 | 0.742 | 0.671 | 0.989 | 0.670 | 0.667 | 0.510 | 0.073 | 0.167 | 0.058 | 0.045 |
| full | 0.746 | 0.742 | 0.674 | 0.987 | 0.671 | 0.667 | 0.505 | 0.068 | 0.151 | 0.049 | 0.048 |

Every variant trains competent translators (each solves its intended
regime) and reproduces the Gate-4 routing result; differences are within
±0.01 accuracy (~9 examples), with no monotonic trend on any axis, and
corruption damage is flat across variants.

C2: **not supported** — task supervision alone matches the full objective;
reconstruction, chain-consistency, and RTD-surrogate terms are
individually and jointly inert on this benchmark.

## Why the structural terms are inert here (mechanistic notes)

- The sheaf cochain-consistency surrogate (per-edge residual/stalk energy
  ratio) has an irreducible noise floor: node field angles and connection
  frame angles are drawn independently, so per-edge residuals cannot go
  below ≈ 2 × quality² regardless of translation quality. It is pinned at
  ~2.0 in every run — it cannot distinguish good from bad conversions.
- The differentiable RTD surrogate is H0-only; none of the three synthetic
  tasks is connectivity-driven, so the term carries no task-relevant
  gradient.
- The topological defect is real and measurable (experiment 1) — the
  current benchmark simply offers it nothing to improve.

## Addendum: the gauge-tier revision test (2026-08-03)

The mechanistic note above said the cochain surrogate has an irreducible
noise floor because the generator's node field is not a section of the
connection. The gauge tier removes that objection: `stalk_mode = "gauge"`
makes clean samples approximate global sections (sentinel: max per-edge
residual 1e-7 at zero noise) while preserving the holonomy label signal
(0.0 vs 2.0). On this tier the consistency surrogate has real dynamic
range, and the chain term visibly works on its target:

| variant | consistency | hard | sheaf corruption damage | topo partial |
|---|---|---|---|---|
| gauge task-only | 1.398 (drifts up unopposed) | 0.757 | 0.164 | 0.140 |
| gauge +chain | 0.182 (held low) | 0.766 | 0.177 | 0.311 |

Verdict: the structural term controls its target quantity for the first
time — and it still does not improve task accuracy or corruption
robustness (damage flat to slightly worse). **The Gate-3 null is therefore
robust across both data designs: structural regularization of the
translators does not help even where it is measurable and controllable.**

One weak non-null signal is recorded for the follow-up: with translations
held structurally consistent, the topological defect's partial correlation
with damage roughly doubles (0.311 vs 0.140) — consistent with the theory
that cone/holonomy defects become informative precisely when maps are
constrained to be consistent. A multi-seed pilot is the honest next step
if this thread is pursued; the project's resources otherwise move to Gate
5 with the routing contribution.

## Seed-robustness pilot on the gauge signal (2026-08-03, three pairs)

Three task-only/+chain pairs on independent data seeds (s03–s05), each
followed by the corruption suite:

| pair | task-only consistency | task-only partial ρ | +chain consistency | +chain partial ρ | Δ |
|---|---|---|---|---|---|
| s03 | 1.398 | 0.140 | 0.182 | 0.311 | +0.17 |
| s04 | 1.682 | −0.054 | 0.187 | 0.435 | +0.49 |
| s05 | 1.974 | 0.275 | 0.182 | 0.127 | −0.15 |

- The chain term holds consistency near the gauge floor (~0.18) in every
  +chain run and lets it drift (1.4–2.0) in every task-only run — the
  manipulation is reliable.
- The utility null is robust: corruption damage is flat across all six
  runs (0.131–0.177), confirming the Gate-3 verdict on three seeds.
- The topological signal effect is **real but not robust**: two of three
  pairs show the lift (mean Δ +0.17 across pairs, +0.29 vs +0.12 pooled
  means) and one pair reverses it. Recorded as weak evidence consistent
  with the chain-contract-before-interpretation principle; a larger seed
  campaign (5–10 pairs) would be required to confirm or kill it.

- **C4 is moot on this benchmark**: with both cone-style and RTD-style
  terms inert, "cone beyond RTD" has no operating regime to distinguish.
- The translator phase gate stays an *engineering* gate (finite held-out
  structural loss with relative improvement), explicitly not evidence for
  the mechanism — its own label already says so.
- Per the plan, the mechanism needs a revised benchmark/model design
  before structural claims advance: a task family where exactness damage
  is the bottleneck (e.g., tasks whose labels are homology-determined with
  continuous corruption of the chain data, so cone/holonomy defects are
  the natural sufficient statistic), and a consistency surrogate without
  an irreducible noise floor.
- Gate 5 (molecular transfer) remains meaningful for the *routing*
  contribution, which does not depend on the inert terms; if pursued, it
  should test the routed architecture with task/reconstruction objectives
  only, and report structural terms as a null-ablation from the start.
