# Gate-2 review (2026-08-03)

This is the consolidated review of Gate 2 (RTD reproduction, fixed experts,
observable specialization, and routing) against the acceptance criteria in
[the experimental plan](10-gb10-experimental-plan.md). Run-level detail is in
[the handoff log](13-gate2-run-handoff.md); claims status is in
[the ledger](08-claims-ledger.md). Gate 3 is tracked separately there.

## Verdict

**Gate 2 is passed at the confirmatory synthetic benchmark.** All three
fixed experts specialize in their intended regimes, the anti-shortcut
controls hold, learned routing beats the best fixed route and the dense
ensemble at matched measured compute while remaining non-collapsed, and the
exact RTD/SRTD evaluation reference exists alongside the differentiable
surrogate with an explicit claim boundary.

## Criteria, point by point

| Plan requirement | Status | Evidence |
|---|---|---|
| Reproduce RTD/SRTD; freeze conventions before use on learned representations | **Done (exact reference)** | `src/homymoly/metrics/exact_rtd.py` + acceptance tests (`tests/test_exact_rtd.py`): identity zeros, isometry/rescaling invariance after normalization, permutation invariance, directional asymmetry with swap consistency and half-sum symmetry, collapse structure, localized detection, per-interval stability bound. The differentiable H0 surrogate is retained for training with qualified naming; measured directional *ordering* can disagree with the exact reference, so the two are never reported interchangeably |
| Three parameter-matched experts, common embedding | **Done** | graph 851k, cell 933k, sheaf 917k params; common `[B, 64]` embedding contract |
| Each specialized route has an advantage in its intended regime without the hidden identifier as input | **Done** | run 9/10 test: graph 0.997, cell ~0.73, sheaf 1.000 on their own regimes, all cross-regime ~0.5 |
| Confirmatory tier with overlapping reliability + preregistered shortcut baselines | **Done** | `ConfirmatoryStructuredSignal` counterfactual groups; shortcut baselines at chance (0.50–0.53), relational oracles at 1.0; group-disjoint splits |
| Gate: ≥2 specialized routes improve over the graph route | **Passed** | enforced phase gate; all three routes specialize |
| Learned routing beats best fixed route at matched measured compute, non-collapsed | **Passed (run 9; reproduced run 10)** | hard 0.743/0.746 vs best fixed 0.667/0.674, random 0.669/0.670, dense 0.736/0.742; expected cost 1.31 vs ~3.9 dense; utilization ~0.26–0.38 per route; regime-route MI 0.067–0.068; per-regime native selection 0.46–0.54 |
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
9. **Run 10**: translators made task-competent (sheaf translator gained the
   holonomy pathway; cell translator gained `face_active` as input — its
   task is structurally impossible without it) — Gate-4 reproduced (0.746).

## Remaining qualifications

- Route accuracy ~0.5 is the expected operating point, not a defect: the
  benchmark's anti-shortcut design caps regime identifiability from
  label-independent cues near the measured probe ceiling (~0.55–0.58).
  Utility, not regime identification, is the plan's Gate-4 axis.
- The translator phase gate is an engineering check (finite held-out
  structural loss with ≥2% relative improvement); the substantive
  structural-value question belongs to Gate 3 (see the corruption-suite
  result and C1/C2 in the claims ledger).
- GPU `scatter_add_` is not bit-deterministic even under
  `deterministic: true`; CPU/exact-oracle paths are FP64-deterministic.
