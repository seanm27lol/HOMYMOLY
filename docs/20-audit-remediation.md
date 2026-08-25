# Audit remediation and continuation record

> **Historical audit record, superseded for current manuscript claims by
> [`docs/29`](29-audit-corrections.md).** Counts and completion statements below
> describe the earlier release named in this document, not the present worktree.

- Status date: 2026-08-23
- Source remediation commit: `e69b07707950b6abe332366c51fe8c94254899f3`
- Campaign commit: `8021292e97abfec91768f1b5437c883a42c29c60`
- Working branch: `agent/journal-release`
- Results record: [`docs/23`](23-identifiable-results.md)

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
| Compute-saving language lacked a direct execution benchmark. | Added a synchronized GB10 BF16 execution benchmark for routed, fixed, and dense paths, then re-ran it against trained checkpoints across five seeds. | `results/summaries/compute-campaign.json` and `results/benchmarks/`: trained routed inference has dense/routed ratio 1.532 ± 0.035 and routed/fastest-fixed ratio 2.269 ± 0.043 at batch 64, where ± is the sample SD over five seeds. | **Trained timing completed; accuracy/compute claim still open.** The superseded `artifacts/benchmarks/compute-remediation*.json` files record `checkpoint: null` and are excluded from all reported results; `scripts/summarize_compute_campaign.py` refuses any routing benchmark without a checkpoint. Timing is descriptive from one runner with paths timed in fixed order, so it is still not a matched-accuracy Pareto result. |
| Environments, CI, artifacts, and idle scheduling were insufficiently reproducible or guarded. | Added a lockfile, direct GB10 constraints, molecular dependency extra, pinned CI action, Dependabot, bounded text-artifact exporter, atomic scheduler state, process/memory allowances, locking, and completion markers. | `uv.lock`, `constraints/gb10-ngc-26.06-direct.txt`, `.github/workflows/ci.yml`, `.github/dependabot.yml`, `scripts/export_artifact_bundle.py`, `scripts/gpu_idle_train.py`, and their tests. | **Completed.** The tracked publication bundle now exists at `results/` (48 files, 2.85 MB) with a full manifest, produced by the allowlist-based `scripts/export_publication_evidence.py`. The managed HOMYMOLY cron entry was removed after campaign completion. |
| Large evidence was untracked, so reproducibility claims rested on local paths. | Added a strict publication exporter with an explicit allowlist, a denylist that refuses checkpoints and prediction dumps, per-batch lossless corruption derivatives, and a timestamp-free manifest. | `scripts/export_publication_evidence.py`, `tests/test_export_publication_evidence.py`, `results/MANIFEST.json`. Verify with `--verify-only`. | **Completed.** Every exported file records its generating commit; 47 come from `8021292` and the earlier-frozen routing endpoint table from `e69b077`. Dropping `per_example` is lossless for every published statistic, verified by recomputing the adjusted partial Spearman from retained rows to 1e-15. |
| Two new campaign families had no strict machine-readable summary. | Added `scripts/summarize_gauge_corruption_campaign.py` and `scripts/summarize_compute_campaign.py` with focused tests. | `results/summaries/gauge-corruption-campaign.json`, `results/summaries/compute-campaign.json`, and 18 tests. | **Completed.** Both fail closed on provenance mismatch. The gauge summarizer validates all eight seed-matched pairs and reports df=7 intervals with exact sign tests; the compute summarizer validates against the sealed 56-step receipt and refuses to pool the identifiable p90 and routing p95 tails. |
| The requested repository was not private. | Repository visibility was changed and rechecked through GitHub. | `seanm27lol/HOMYMOLY` reports `PRIVATE`; a local high-signal tracked-text credential-pattern scan returned no matches. | **Current privacy completed.** This cannot retract prior public exposure. GitHub secret scanning was unavailable for this repository/account configuration, and the local scan is not a comprehensive secret audit. |

## Completed evidence that may be cited now

- The exact RTD/SRTD implementation has a declared, degree-specific reference
  convention: independent q0.9 scaling over every entry of each complete
  dissimilarity matrix, with scalar degree 1 by default.
- An isolated synthetic paired permutation-complex experiment learned forward
  and reverse exact chain maps with acyclic evaluated cones at the recorded
  tolerance. This validates that implementation path only.
- The trained GB10 benchmarks measured a dense-to-routed median-latency ratio of
  1.532 ± 0.035 and a routed-to-fastest-fixed ratio of 2.269 ± 0.043 across five
  seeds, where ± is the sample SD, with routed peak allocated memory below dense
  in every seed. This is
  descriptive timing, not an accuracy-versus-compute result. The earlier
  architecture-only 1.77× figure loaded no checkpoint and is withdrawn.
- The identifiable typed-map campaign completed all 40 frozen runs and recovered
  the planted map exactly, passing its engineering recovery gate in 10 of 10
  applicable runs. Its structural contrasts are null: all 21 declared continuous
  contrast intervals contain zero, and `cone_only`/`rtd_only` identify at chance
  while producing acyclic cones in 6,000 of 6,000 evaluated examples. Full
  record in `docs/23-identifiable-results.md`.
- The eight-seed gauge corruption family raises the unit of analysis to the
  training seed. All three across-seed intervals contain zero with exact sign
  tests p ≥ 0.727. It remains a fixed-expert embedding diagnostic.
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
  any benchmark in this repository, trained or untrained.
- Do not cite the withdrawn 1.77× architecture-only routed/dense ratio, or the
  `1.863 ± 0.071` routed-to-fastest-fixed figure that appeared in an earlier
  handoff. The latter is not reproducible from any artifact under any ratio
  definition; the recomputed value is 2.269 ± 0.043.
- Do not treat cone acyclicity as evidence that a decoded map is correct. The
  `cone_only` and `rtd_only` controls produce acyclic cones in every evaluated
  example while identifying the planted map at chance.
- Do not compare the identifiable p90 latency tail with the routing p95 tail.
  The two runners record different statistics and are never pooled.
- Do not call routing-v2 graph-only, input-only, genuinely untouched, or a
  pristine preregistration. Training used privileged regime labels, inference
  used structured-view summaries, and the pre-commit seed-20260906 deviation
  is recorded in `docs/19`.

## Open work and handoff

Execute and report these items in order; preserve failed and null outcomes.

1. **Run the canonical target-held-out configuration — still pending.** Execute
   `.venv/bin/python -m homymoly train --config configs/gate2.yaml`. Report
   translator gates, structure-reconstruction losses, routing baselines, and
   failures. Because the held-out targets are unidentifiable from graph inputs,
   treat it as an integration/null diagnostic, not conversion evidence.
2. **Design a real C1/C2 test — open, and now better specified.** C4 has been
   answered on the identifiable benchmark: cone-only, RTD-only, and combined
   ablations exist on identical paired data and are all null or at chance. What
   remains is a task where chain-map or induced-homology defects are *causally relevant* and the
   correct map is **not** analytically attainable. The current annulus benchmark
   saturates — an analytic marker decoder reaches 1.000 — so its nulls cannot
   distinguish "these terms are useless" from "this task is too easy." A harder
   template family, not more seeds, is the informative next step.
3. **Complete release checks — completed for this release.** The full test and
   lint suites pass, the tracked publication bundle is exported and verified,
   the paper and PDF are regenerated and visually inspected, and the managed
   HOMYMOLY cron block was removed after campaign completion. Remaining
   platform limitation: GitHub secret scanning is unavailable for this
   repository/account configuration, so only a local tracked-text
   credential-pattern scan was run.

If work must stop before these items finish, update each `pending` marker with
the exact command, process/session state, artifact path, commit/config hash,
and last verified checkpoint. Never replace a pending marker with a numerical
claim until the complete artifact and its provenance checks exist.
