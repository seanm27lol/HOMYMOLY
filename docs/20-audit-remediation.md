# Audit remediation and continuation record

- Status date: 2026-08-22
- Source remediation commit: `e69b07707950b6abe332366c51fe8c94254899f3`
- Working branch: `agent/research-remediation`

This document records what the audit changed, what evidence now exists, and
what work remains before release. A `pending` item is deliberately not a
result. Do not infer an outcome from a partial artifact or from an experiment
being in progress.

## Remediation matrix

| Finding | Remediation | Verification/evidence | State and claim boundary |
|---|---|---|---|
| Exact RTD/SRTD used max normalization and summed unrelated homological degrees, including a truncation frontier. | The independent GF(2) reference now defaults to full-matrix q0.9 normalization and degree-specific scoring; scalar RTD/SRTD defaults to degree 1. An extra chain degree is built only to determine deaths and is filtered from reported degrees. | `src/homymoly/metrics/exact_rtd.py`, `src/homymoly/metrics/distances.py`, and hand/collapse/normalization/frontier tests in `tests/test_exact_rtd.py`. | **Completed.** This is hand-validated compatibility, not an official-code reproduction. All pre-audit “exact SRTD” corruption scalars are invalid. |
| The corruption report treated five blocks repeated over five severities as independent rows, omitted block/severity adjustment, used process-salted draws, and used a nonstandard residual correlation. | Stable SHA-256 paired draws; explicit unique-observation counts; rank-residual partial Spearman with severity and block fixed effects; complete-block bootstrap; within-block residual permutation. | `scripts/eval_corruption.py`, `tests/test_eval_corruption_stats.py`, four `artifacts/gate3/*/corruption_report_v2.json` reports, and `artifacts/gate3/paired_comparison_v2.json`. Each kind has 13 complete blocks, 65 batch observations, and 306 unique examples. | **Corrected fixed-expert diagnostic completed.** Eleven of 12 within-checkpoint intervals include zero; full/edge-cochain gives CI [0.002, 0.488] but permutation p=0.115. All nine paired contrasts against task-only include zero with p≥0.32397. There is no multiplicity adjustment. Historical correlations remain invalid. |
| The corruption experiment was described as typed-conversion damage and its displacement control as reconstruction loss. | Code and documentation now follow execution: the evaluator calls a fixed expert on clean/corrupted inputs, compares expert embeddings, never invokes a translator, and emits `mean_embedding_displacement`. | Fixed-expert lookup and forward calls in `scripts/eval_corruption.py`, claim-boundary tests, and the schema-v3 reports above. | **C1 remains untested.** The corrected result is conditional on four fixed checkpoints and sampled held-out blocks; it does not include training-seed variation and is not evidence about conversion exactness or translator reconstruction. |
| Historical “translator” results consumed target cell/sheaf structure, so they were target-view encoders rather than graph-only typed conversions. | Translator model defaults deny target values; canonical `configs/gate2.yaml` sets `translator_target_structure_access: false`, enables structure-reconstruction supervision, and uses per-example router supervision. Candidate cell incidence remains supplied. | Model/config tests plus `test_counterfactual_pairs_match_structure_and_unary_marginals`: cell/sheaf label pairs have identical graph observations but different held-out activity/transports. | **Objective implementation completed; competent conversion remains untested.** A canonical run is an integration/null diagnostic only. An identifiable benchmark is required. |
| No historical translator satisfied an exact chain-map contract or trained a direct mapping-cone objective. | Added an exact chain-map layer, chain-map residual, bidirectional/cycle training path, and differentiable soft cone-acyclicity proxy. | `src/homymoly/models/chain_map.py`, `tests/test_chain_map_model.py`, and `artifacts/chain-map-exact/summary.json`. The completed synthetic run reports forward/reverse residual maxima \(1.19 \times 10^{-7}\)/\(2.38 \times 10^{-7}\), cone Betti numbers `[0, 0, 0]`, and test MSE \(1.54 \times 10^{-14}\). | **Synthetic machinery validation completed. C2/C4 downstream claims remain untested.** This paired permutation-complex result is not a task-value ablation. |
| Routing evidence mixed development selection, post-selection reruns, and an inconclusive first comparison. | Added five immutable configs and a frozen core analysis protocol with a single primary margin and confidence-interval decision rule. | `configs/routing-confirmatory-v2-s1.yaml` through `s5.yaml`, `docs/19-routing-confirmatory-v2-protocol.md`, `scripts/summarize_routing_campaign.py`, and `artifacts/routing-confirmatory-v2-summary.json`. All five valid runs share commit `e69b077`, executable fingerprint `473fb0f…1f1d8`, and the recorded GB10 environment. | **Scoped endpoint supported under the v2 decision rule:** mean hard-minus-best-fixed +0.1098039216, SD 0.0116918587, t95 CI [0.0952865615, 0.1243212816]. This is historical regime-distilled, structured-view routing, not graph-only conversion. An aborted pre-commit seed-20260906 attempt exposed validation metrics under different code, so the campaign is protocol-aligned with a disclosed deviation, not pristine preregistration. Exact two-sided sign-test sensitivity is p=0.0625. |
| Compute-saving language lacked a direct execution benchmark. | Added a synchronized GB10 BF16 execution benchmark for routed, fixed, and dense paths. | `artifacts/benchmarks/compute-remediation-e69b077.json`: untrained architecture-only median latency is 38.34 ms routed versus 67.81 ms dense at batch 64 (1.77× ratio). | **Timing completed, accuracy/compute claim still open.** No checkpoint was loaded, so this cannot establish a matched-accuracy Pareto improvement. |
| Environments, CI, artifacts, and idle scheduling were insufficiently reproducible or guarded. | Added a lockfile, direct GB10 constraints, molecular dependency extra, pinned CI action, Dependabot, bounded text-artifact exporter, atomic scheduler state, process/memory allowances, locking, and completion markers. | `uv.lock`, `constraints/gb10-ngc-26.06-direct.txt`, `.github/workflows/ci.yml`, `.github/dependabot.yml`, `scripts/export_artifact_bundle.py`, `scripts/gpu_idle_train.py`, and their tests. | **Code completed; final CI, bundle export, and cron-install verification pending.** |
| The requested repository was not private. | Repository visibility was changed and rechecked through GitHub. | `seanm27lol/HOMYMOLY` reports `PRIVATE`; a local high-signal tracked-text credential-pattern scan returned no matches. | **Current privacy completed.** This cannot retract prior public exposure. GitHub secret scanning was unavailable for this repository/account configuration, and the local scan is not a comprehensive secret audit. |

## Completed evidence that may be cited now

- The exact RTD/SRTD implementation has a declared, degree-specific reference
  convention: independent q0.9 scaling over every entry of each complete
  dissimilarity matrix, with scalar degree 1 by default.
- An isolated synthetic paired permutation-complex experiment learned forward
  and reverse exact chain maps with acyclic evaluated cones at the recorded
  tolerance. This validates that implementation path only.
- The architecture-only GB10 benchmark measured a routed/dense execution
  ratio of 1.77× in its recorded setup. Because it loaded no trained
  checkpoint, it is not an accuracy-versus-compute result.
- The protocol-aligned routing-v2 campaign's hard-minus-best-fixed endpoint was
  +0.1098039216 across five valid seeds (SD 0.0116918587; Student-t 95% CI
  [0.0952865615, 0.1243212816]), meeting its frozen numerical decision rule.
  This is evidence only for the regime-distilled router over available
  structured views, with the procedural-deviation and small-n qualifications
  in `docs/19`.
- The corrected Gate-3 reports provide a fixed-expert embedding diagnostic.
  All nine paired added-loss-versus-task-only intervals include zero and their
  exact whole-block randomization p-values are at least 0.32397. This is a
  checkpoint-conditional result without multiplicity adjustment, not a test of
  a translator or learned map.
- The repository currently reports private visibility. No claim is made that
  earlier exposure was reversible or that GitHub secret scanning ran.

## Invalidated or unsupported statements

- Do not cite any historical Gate-3 corruption ρ, partial-ρ, confidence, or
  “60 batches” statement. The true design reused five blocks over five
  severity levels and the metric convention was wrong.
- Do not call clean/corrupted expert-embedding MSE translator reconstruction
  loss.
- Do not describe the corrected corruption evaluator as a graph-to-cell,
  graph-to-sheaf, chain-map, cone-map, or typed-conversion experiment.
- Do not describe target-view encoder accuracy as evidence that a graph-only
  translator reconstructed cell or sheaf structure.
- Do not treat the exact-chain-map synthetic result as evidence that a direct
  cone loss improves a downstream task.
- Do not claim the routed model has a matched-accuracy compute advantage from
  the architecture-only benchmark.
- Do not call routing-v2 graph-only, input-only, genuinely untouched, or a
  pristine preregistration. Training used privileged regime labels, inference
  used structured-view summaries, and the pre-commit seed-20260906 deviation
  is recorded in `docs/19`.

## Open work and handoff

Execute and report these items in order; preserve failed and null outcomes.

1. **Run the canonical target-held-out configuration — pending.** Execute
   `.venv/bin/python -m homymoly train --config configs/gate2.yaml`. Report
   translator gates, structure-reconstruction losses, routing baselines, and
   failures. Because the held-out targets are unidentifiable from graph inputs,
   treat it as an integration/null diagnostic, not conversion evidence.
2. **Design a real C1/C2/C4 test — open.** Execute a declared learned typed map
   on held-out examples; measure map/translator damage and genuine translator
   reconstruction; compare cone-only, RTD-only, combined, and matched-compute
   controls. A task where homological exactness is causally relevant is needed
   before a null or positive mechanism claim is defensible.
3. **Complete release checks — pending.** Run the full test and lint suites,
   export bounded textual evidence bundles after experiments finish, install
   and verify only the HOMYMOLY-managed cron block if unattended training is
   still needed, push the branch, open the review PR, and require its CI before
   merge. Record any platform feature that remains unavailable.

If work must stop before these items finish, update each `pending` marker with
the exact command, process/session state, artifact path, commit/config hash,
and last verified checkpoint. Never replace a pending marker with a numerical
claim until the complete artifact and its provenance checks exist.
