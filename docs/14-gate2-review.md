# Gate-2 review (2026-08-03)

> **Superseded verdict (audit 2026-08-13).** This document is retained as the
> historical engineering review. It does not establish a confirmatory routing
> win: the original untouched five-seed margin was +0.036 with 95% CI
> [−0.018, +0.090], while the later stabilized +0.108 result reused the seeds
> that selected the fix. The router was trained by privileged
> latent-regime-to-utility distillation, and the historical “translators” read
> target-view structure. See `docs/17`, `docs/19`, and the updated claims
> ledger.

This is the consolidated review of Gate 2 (RTD reproduction, fixed experts,
observable specialization, and routing) against the acceptance criteria in
[the experimental plan](10-gb10-experimental-plan.md). Run-level detail is in
[the handoff log](13-gate2-run-handoff.md); claims status is in
[the ledger](08-claims-ledger.md). Gate 3 is tracked separately there.

## Verdict

**Engineering integration passed; the scientific routing verdict is
pending.** All three fixed experts specialize and the anti-shortcut controls
hold. Historical runs show that routing can outperform fixed alternatives,
but the untouched confidence interval was inconclusive. Measured GB10 latency
was added only in the audit: 38.3 ms routed versus 67.8 ms dense at batch 64
(1.77×), replacing the old declared-cost proxy.

## Criteria, point by point

| Plan requirement | Status | Evidence |
|---|---|---|
| Reproduce RTD/SRTD; freeze conventions before use on learned representations | **Corrected in 2026-08-13 audit** | The old scalar incorrectly summed degrees and used max normalization. The current reference defaults to published degree 1 and full-matrix q0.9 normalization, returns per-degree scores explicitly, filters truncation-frontier degrees, and includes hand fixtures plus cutoff-invariance tests. Historical corruption scores must be recomputed. |
| Three parameter-matched experts, common embedding | **Done** | graph 851k, cell 933k, sheaf 917k params; common `[B, 64]` embedding contract |
| Each specialized route has an advantage in its intended regime without the hidden identifier as input | **Done** | run 9/10 test: graph 0.997, cell ~0.73, sheaf 1.000 on their own regimes, all cross-regime ~0.5 |
| Confirmatory tier with overlapping reliability + preregistered shortcut baselines | **Done** | `ConfirmatoryStructuredSignal` counterfactual groups; shortcut baselines at chance (0.50–0.53), relational oracles at 1.0; group-disjoint splits |
| Gate: ≥2 specialized routes improve over the graph route | **Passed** | enforced phase gate; all three routes specialize |
| Learned routing beats best fixed route at matched measured compute, non-collapsed | **Confirmatory result pending** | Run 9/10 are development evidence. Original untouched n=5: +0.036 [−0.018, +0.090]. Stabilized same-seed rerun: +0.108 descriptively. The GB10 execution benchmark measures a 1.77× routed/dense speedup; it does not convert post-selected accuracy into confirmatory evidence. |
| Long-run discipline (checkpoints, resume, aborts, leakage guards) | **Done** | atomic checkpoints with config+code fingerprints; deterministic resume verified by crash-injection; aborts on non-finite loss; reserved metadata keys |

## The debugging arc (why the first five runs failed)

Every barrier was identified by a targeted measurement rather than by
sweeping; details are in the handoff log.

1. **Run 1**: sheaf expert at chance — the sheaf label is cycle holonomy and
   the expert only saw per-edge residuals (verified label-independent).
   Fix: exact face-holonomy pathway + mean/max face readout.
2. **Runs 2–3**: latent NaN in the translators phase — `sqrt` at zero
   residual (`0 * inf` in backward; first fix attempt with post-clamp was
   itself wrong). Fix: eps inside the sqrt + a permanent regression test.
3. **Run 4**: router collapsed to always-cell — graph expert at chance on
   its own regime (pairwise sign statistic diluted by pooling). Fix: raw
   endpoint-pair pathway with max-pooled readout.
4. **Run 5**: router regime-blind (MI 0.002) — routing diagnostics were
   means/counts that dilute exactly the amplitude cues that distinguish
   regimes (F ≈ 0 measured). Fix: per-channel max-abs amplitude features
   (F 9–37; linear probe 0.573 vs 0.333 chance).
5. **Run 6**: routing lost to best fixed; graph never selected — the
   confidence-based oracle was regime-conditionally miscalibrated
   (underconfident-but-accurate graph expert lost its own regime 62% of
   the time). Three alternative estimators were measured and rejected
   (temperature scaling, correctness-first) before shipping the
   regime-conditional accuracy table (validation-fitted supervision).
6. **Run 7**: router converged to exactly uniform predictions — the single
   40-epoch cosine starved the router phases (LR ~1e-4 → 1e-6; offline
   replication showed 0.32 vs 0.54–0.59 at 3e-4–1e-3). Fix: per-phase LR
   restart.
7. **Run 8**: Gate-4 passed but at an unplanned ~10× LR — the restart hook
   keyed on a condition that never fired (`state.phase_index` is advanced
   at each phase end). Fix: key on `first_epoch == 0`, verified in-engine.
8. **Run 9**: confirmatory — Gate-4 passed at the configured LR with clean
   provenance (0.743).
9. **Run 10**: target-view encoders were made task-competent (sheaf pathway
   read observed transports; cell pathway read `face_active`). This reproduced
   routing at 0.746 but did not demonstrate graph-to-cell/sheaf conversion.

## Remaining qualifications

- Historical route accuracy near 0.5 is compared descriptively with finite
  probe scores of ~0.55–0.58; those probes do not establish an identifiability
  ceiling. Utility, not regime identification, is the plan's Gate-4 axis.
- The translator phase gate is an engineering check (finite held-out
  structural loss with ≥2% relative improvement); the substantive
  structural-value question belongs to Gate 3 (see the corruption-suite
  result and C1/C2 in the claims ledger).
- Historical routing supervision uses the hidden regime only during training
  to index a validation accuracy table. This is not test-input leakage, but it
  is privileged regime distillation and must be named as such.
- At inference the router receives no label or regime tensor, but its
  diagnostics summarize the available active-face and sheaf-transport views
  in addition to graph features. This is structured-view routing, not
  graph-only routing.
- GPU `scatter_add_` is not bit-deterministic even under
  `deterministic: true`; CPU/exact-oracle paths are FP64-deterministic.
