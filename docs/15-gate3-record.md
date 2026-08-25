# Gate-3 record (2026-08-03)

> **Validity correction (2026-08-13).** The historical corruption inference
> and “exact SRTD” scalars in this record are invalidated. The evaluator reused
> five sample blocks at five severities, counted repeated observations as
> independent, omitted severity/block adjustment, generated draws through
> process-salted Python `hash`, and used a nonstandard rank-residual
> correlation. More importantly, the scalar summed multiple homological
> degrees—including truncation-frontier degrees—and used max normalization
> instead of the published full-matrix q0.9 convention. Corrected reports were
> completed on 2026-08-22. They evaluate clean versus corrupted embeddings from
> a **fixed expert**; they do not call either typed translator. Therefore this
> is an embedding-diagnostic repair, not a test of conversion exactness or
> translator damage. The ablation ladder still descriptively shows that the
> *implemented* reconstruction/local-consistency/H0 surrogates did not improve
> task accuracy, but it never tested a direct learned-map cone objective.

Gate 3 asks whether translator and structural-loss machinery adds value:
either a structural diagnostic adds predictive value for conversion damage
after controlling for reconstruction error, or a structural regularizer
improves a matched-compute downstream result (criterion in
[the plan](10-gb10-experimental-plan.md)). This document records the two
experiments and their outcomes. Run detail is in
[the handoff log](13-gate2-run-handoff.md); claims in
[the ledger](08-claims-ledger.md).

## Verdict

**Gate 3 is not passed. C1 and the direct C2/C4 claims are untested.** Historical
ablations provide a scoped null for reconstruction, local-consistency, and
H0-distance surrogates. They do not justify a comprehensive null for map-aware
cone losses, and the old corruption correlations cannot be used. The
corrected fixed-expert diagnostic cannot close C1 because no learned typed map
is evaluated.

## Experiment 1: corruption suite (C1, correlation question)

Design: `src/homymoly/data/corruptions.py` (three graded, per-sample,
deterministic channels: transport rotations, edge-cochain noise,
node-anchor noise); `scripts/eval_corruption.py` applies them to the
held-out split at five severities and measures prediction damage from the
affected **fixed expert**, clean/corrupted expert-embedding displacement, and
the degree-1 exact SRTD between those two embedding clouds (subsampled to
`rtd_max_points = 24`). Its evaluation path invokes the selected fixed expert
and does not invoke a translator or learned chain map. The word “conversion”
in the historical description was therefore incorrect.

Historical report (invalid for inference): there were 306 unique examples per
kind observed at five severities (1530 repeated example observations) and five
unique batch blocks observed at five severities (25 batch observations), not
60 independent batches per kind. The old report is retained at
`artifacts/gate3/corruption_report_full.json` for provenance.

- The previously printed ρ values (0.81–0.92) are descriptive outputs of the
  invalid scalar and pseudoreplicated analysis, not evidence for C1.
- The historical variable called `reconstruction` was clean/corrupted
  **expert-embedding displacement**, not translator reconstruction error. The
  corrected schema names it `mean_embedding_displacement`.
- The corrected evaluator reports SRTD degree 1, deterministic paired draws,
  unique block/sample counts, severity and block controls, a complete-block
  bootstrap interval, and within-block residual permutation p-value.
- Its `mean_embedding_displacement` quantity is fixed-expert embedding MSE
  between clean and corrupted inputs. It is a displacement control, not
  graph-to-cell or graph-to-sheaf reconstruction error.

### Corrected fixed-expert result (2026-08-22)

The four schema-v3 reports are
`artifacts/gate3/{task-only,plus-recon,plus-chain,full}/corruption_report_v2.json`.
Every report uses the same deterministic pairing contract, five severities,
13 complete blocks, 65 batch observations, and 306 unique examples **per
corruption kind**. The statistic is the correlation of rank residuals for
degree-1 SRTD and damage, adjusted for ranked severity, ranked embedding
displacement, and block fixed effects. Intervals resample complete blocks; the
p-values permute residuals within blocks.

| checkpoint | corruption kind | adjusted statistic | block-bootstrap 95% CI | permutation p |
|---|---|---:|---:|---:|
| task-only | edge-cochain noise | 0.157 | [−0.114, 0.426] | 0.270 |
| task-only | node-anchor noise | 0.102 | [−0.158, 0.351] | 0.495 |
| task-only | transport rotation | 0.026 | [−0.201, 0.261] | 0.865 |
| +recon | edge-cochain noise | 0.175 | [−0.076, 0.375] | 0.165 |
| +recon | node-anchor noise | 0.122 | [−0.034, 0.261] | 0.374 |
| +recon | transport rotation | 0.014 | [−0.145, 0.214] | 0.925 |
| +chain | edge-cochain noise | 0.135 | [−0.070, 0.453] | 0.335 |
| +chain | node-anchor noise | 0.132 | [−0.121, 0.335] | 0.363 |
| +chain | transport rotation | 0.005 | [−0.166, 0.182] | 0.971 |
| full | edge-cochain noise | 0.214 | [0.002, 0.488] | 0.115 |
| full | node-anchor noise | 0.091 | [−0.150, 0.344] | 0.536 |
| full | transport rotation | −0.047 | [−0.218, 0.150] | 0.764 |

Eleven of 12 within-checkpoint intervals include zero. The full/edge-cochain
bootstrap interval narrowly excludes zero, while its separately calibrated
within-block permutation test gives p=0.115. No multiplicity adjustment was
made across four checkpoints and three corruption kinds, so this isolated
interval is not treated as robust evidence.

The paired analysis in `artifacts/gate3/paired_comparison_v2.json` compares
each added-loss checkpoint with task-only on identical complete blocks:

| contrast vs task-only | corruption kind | Δ adjusted statistic | paired block-bootstrap 95% CI | exact randomization p |
|---|---|---:|---:|---:|
| +recon | edge-cochain noise | 0.018 | [−0.301, 0.355] | 0.903 |
| +recon | node-anchor noise | 0.020 | [−0.234, 0.278] | 0.896 |
| +recon | transport rotation | −0.011 | [−0.169, 0.120] | 0.894 |
| +chain | edge-cochain noise | −0.022 | [−0.318, 0.313] | 0.878 |
| +chain | node-anchor noise | 0.030 | [−0.122, 0.154] | 0.671 |
| +chain | transport rotation | −0.021 | [−0.181, 0.088] | 0.854 |
| full | edge-cochain noise | 0.057 | [−0.185, 0.332] | 0.701 |
| full | node-anchor noise | −0.011 | [−0.219, 0.233] | 0.936 |
| full | transport rotation | −0.072 | [−0.210, 0.021] | 0.324 |

All nine paired intervals include zero and all p-values are at least 0.32397.
These comparisons also have no multiplicity adjustment and are conditional on
the four fixed trained checkpoints and sampled held-out blocks; they do not
estimate variation across training seeds.

C1 remains **untested**. The corrected diagnostic finds no robust incremental
degree-1 SRTD signal in this fixed-expert analysis, but it cannot establish or
refute predictiveness of damage during typed conversion because no translator,
learned map, chain-map residual, or mapping cone is evaluated.

## Experiment 2: structural-loss ablation ladder (C2, intervention question)

Design: four full 40-epoch runs with identical seed and schedule (expert
phases verified bit-identical), differing only in translator-phase loss
weights: task-only → +reconstruction → +chain → full (configs in
`configs/gate3/`, runner `scripts/run_gate3_ablations.sh`). Each variant
also ran the corruption suite.

| variant | hard | dense | best fixed | oracle | g2c | g2s | route acc | MI | historical sheaf dmg† | historical cell dmg† | historical graph dmg† |
|---|---|---|---|---|---|---|---|---|---|---|---|
| task-only | 0.746 | 0.741 | 0.673 | 0.983 | 0.668 | 0.667 | 0.505 | 0.069 | 0.161 | 0.048 | 0.046 |
| +recon | 0.733 | 0.734 | 0.670 | 0.965 | 0.670 | 0.667 | 0.496 | 0.062 | 0.152 | 0.048 | 0.054 |
| +chain | 0.751 | 0.742 | 0.671 | 0.989 | 0.670 | 0.667 | 0.510 | 0.073 | 0.167 | 0.058 | 0.045 |
| full | 0.746 | 0.742 | 0.674 | 0.987 | 0.671 | 0.667 | 0.505 | 0.068 | 0.151 | 0.049 | 0.048 |

† Retained only to identify the old artifact. These corruption-damage columns
come from the invalid/pseudoreplicated protocol and are not inferential
evidence. The corrected fixed-expert results are reported above rather than
silently substituted into this historical task table.

Every variant's target-view encoder solves its intended task and reproduces
the development routing behavior; differences are within ±0.01 accuracy
(~9 examples), with no monotonic task trend. These encoders read target cell
activity or sheaf transports and therefore are not learned graph-only
conversions.

The audited target-held-out mode removes those values from translator inputs,
but this generator supplies no graph observable that identifies the selected
active face or the sheaf defect/transport target. It is therefore an
implemented structure-reconstruction objective awaiting a redesigned,
identifiable conversion benchmark—not a competent graph-only translator.

C2: **the implemented surrogates show a descriptive null.** Task supervision
matches reconstruction + local consistency + H0-distance terms here. No
mapping-cone loss of a learned chain map was present, so the direct C2 claim
was not tested.

## Why the structural terms are inert here (mechanistic notes)

- The sheaf cochain-consistency surrogate (per-edge residual/stalk energy
  ratio) has an irreducible noise floor: node field angles and connection
  frame angles are drawn independently, so per-edge residuals cannot go
  below ≈ 2 × quality² regardless of translation quality. It is pinned at
  ~2.0 in every run — it cannot distinguish good from bad conversions.
- The differentiable RTD surrogate is H0-only; none of the three synthetic
  tasks is connectivity-driven, so the term carries no task-relevant
  gradient.
- The corrected topological-defect diagnostic is reported above. Because it
  evaluates fixed experts rather than learned typed maps, it does not support a
  mechanistic interpretation about conversion.

## Addendum: the gauge-tier revision test (2026-08-03)

The mechanistic note above said the cochain surrogate has an irreducible
noise floor because the generator's node field is not a section of the
connection. The gauge tier removes that objection: `stalk_mode = "gauge"`
makes clean samples approximate global sections (sentinel: max per-edge
residual 1e-7 at zero noise) while preserving the holonomy label signal
(0.0 vs 2.0). On this tier the consistency surrogate has real dynamic
range, and the chain term visibly works on its target:

| variant | consistency | hard | historical sheaf corruption damage† | historical topo partial† |
|---|---|---|---|---|
| gauge task-only | 1.398 (drifts up unopposed) | 0.757 | 0.164 | 0.140 |
| gauge +chain | 0.182 (held low) | 0.766 | 0.177 | 0.311 |

Verdict: the local-consistency term controls its target quantity for the first
time and does not improve task accuracy. This is a scoped null for that
surrogate across both data designs. Historical corruption-damage and partial-ρ
values below are withdrawn; these gauge-tier and later seed-campaign
checkpoints were not regenerated under the corrected protocol. The corrected
four-checkpoint base ladder is reported in Experiment 1 above.

† Invalid/pseudoreplicated corruption outputs, shown only for provenance.

The previously noted partial-correlation increase (0.311 vs 0.140) inherits
the invalid repeated-measures analysis and is not evidence for a signal.

## Historical seed-robustness pilot (2026-08-03, three pairs)

Three task-only/+chain pairs on independent data seeds (s03–s05), each
followed by the corruption suite:

| pair | task-only consistency | historical task-only partial ρ† | +chain consistency | historical +chain partial ρ† | historical Δ† |
|---|---|---|---|---|---|
| s03 | 1.398 | 0.140 | 0.182 | 0.311 | +0.17 |
| s04 | 1.682 | −0.054 | 0.187 | 0.435 | +0.49 |
| s05 | 1.974 | 0.275 | 0.182 | 0.127 | −0.15 |

† Invalid/pseudoreplicated corruption inference, retained for provenance only.

- The chain term holds consistency near the gauge floor (~0.18) in every
  +chain run and lets it drift (1.4–2.0) in every task-only run — the
  manipulation is reliable.
- Task accuracy remains flat across these runs. Corruption-damage and
  topological-signal interpretations are withdrawn until regenerated under
  the corrected protocol.

## Historical eight-pair campaign (2026-08-03)

The pilot was extended to eight independent task-only/+chain pairs
(s03–s10). It historically reported these per-pair partial-correlation
deltas, all of which inherit the invalid/pseudoreplicated analysis:

`+0.171, +0.489, −0.148, +0.576, +0.302, −0.182, +0.059, −0.500`
— **5/8 positive, mean +0.096 ± 0.364**.

- Manipulation reliability held in all 16 runs (consistency 0.182–0.187
  with the chain term, 1.40–1.99 drifting without).
- The historically pooled partial correlations (+chain mean 0.191 versus
  task-only mean 0.095) are withdrawn, as is the early “doubling” narrative.
- The historical 0.127–0.177 corruption-damage range is a descriptive
  fixed-expert output from the invalid protocol. It does not establish a
  conversion-utility null.

**Corrected verdict on the historical thread:** across eight paired runs, the local
consistency term reliably controls its own surrogate and shows no task
benefit. The partial-correlation deltas inherit the invalid corruption
analysis and are withdrawn. No historical translator enforced an exact chain
map, and no direct mapping-cone objective was trained.

- **C4 is untested**: the historical “cone-style” label referred to local
  reconstruction/consistency surrogates, not a learned-map mapping cone.
- The translator phase gate stays an *engineering* gate (finite held-out
  structural loss with relative improvement), explicitly not evidence for
  the mechanism — its own label already says so.
- Per the plan, the mechanism needs a revised benchmark/model design
  before structural claims advance: a task family where chain-map or induced-homology damage
  is the bottleneck (e.g., tasks whose labels are homology-determined with
  continuous corruption of the chain data, so cone/holonomy defects are
  the natural sufficient statistic), and a consistency surrogate without
  an irreducible noise floor.
- Gate 5 (molecular transfer) remains meaningful for the *routing*
  contribution, which does not depend on the inert terms; if pursued, it
  should test the routed architecture with task/reconstruction objectives
  only, and report structural terms as a null-ablation from the start.
