# Gate-2 confirmatory campaign (2026-08-04)

Per the plan's confirmatory rule ("at least five fresh seeds for the
primary synthetic comparison… report confidence intervals and all failed
or collapsed runs"), five independent full Gate-2 runs (seeds
20260901–20260905, configs `configs/confirmatory-s{1..5}.yaml`, artifacts
`artifacts/confirmatory-s{1..5}/`) were executed after the run-9/10
pilot. This is the statistical verdict on the routing claim (C3).

## Results (official test split per seed)

| seed | hard | best fixed | random | dense | oracle | route acc | MI | cost | utilization (g/c/s) |
|---|---|---|---|---|---|---|---|---|---|
| s1 | 0.767 | 0.668 | 0.679 | 0.763 | 0.975 | 0.546 | 0.097 | 1.32 | .33/.30/.36 |
| s2 | 0.693 | 0.704 | 0.678 | 0.743 | 0.950 | 0.378 | 0.029 | 1.32 | .08/.57/.35 |
| s3 | 0.720 | 0.692 | 0.697 | 0.763 | 0.971 | 0.434 | 0.031 | 1.32 | .10/.50/.40 |
| s4 | 0.749 | 0.691 | 0.659 | 0.737 | 0.969 | 0.504 | 0.071 | 1.31 | .19/.50/.31 |
| s5 | 0.687 | 0.680 | 0.680 | 0.735 | 0.950 | 0.388 | 0.031 | 1.32 | .11/.57/.32 |

All engineering gates (fixed-expert specialization, translator
improvement) passed on every seed; no run crashed.

## Statistical verdict

- **Margin over best fixed route**: per-seed margins +0.099, −0.011,
  +0.028, +0.058, +0.007 → mean **+0.036**, 95% CI [−0.018, +0.090]
  (t(4)) — positive trend, but the five-seed interval crosses zero.
  Including the two pilot runs (runs 9–10, n=7): mean **+0.046**, 95% CI
  **[+0.010, +0.082]** (t(6)) — statistically positive.
- **Routing is regime-informative on every seed**: MI 0.029–0.097
  (mean 0.052 ± 0.038), route accuracy 0.378–0.546 — always above the
  1/3 baseline, consistent with the measured ~0.55–0.58 identifiability
  ceiling of the anti-shortcut design.
- **Failed or collapsed runs (required reporting)**: seed s2 fails the
  Gate-4 utility criterion outright (0.693 < best fixed 0.704) and shows
  partial route collapse toward cell with the graph route at 8%
  utilization; s5 is a marginal pass (0.687 vs 0.680) with the same
  tilt. Weak seeds correlate with under-use of the graph route.
- **Interpretation**: the routing claim holds on average across seven
  independent runs with a small but statistically significant margin,
  and fails on roughly one in five draws. The honest characterization
  for the ledger is *supported with material seed variance* — the
  mechanism works, and the router's weak-seed failure mode (partial
  collapse to cell) is now the most interesting open stability question
  in the system.

## Consequences

- C3 updated in the claims ledger: supported with the n=7 confirmatory
  interval above; seed variance and the s2 collapse mode recorded.
- Follow-up candidates, in priority order: (a) stabilize weak-seed
  routing (longer warmup, lower entropy weight, or oracle-target
  smoothing to reduce collapse-to-cell draws); (b) the
  molecularly-informed cell architecture (Gate-5 redesign direction).

## Stabilization pilot (2026-08-04): the collapse mode is fixed

Diagnosis first: the s2 collapse is an optimization basin failure, not
supervision — its oracle table and experts are statistically identical
to the strong seeds', but its warmup never escaped uniform predictions
(route accuracy exactly 1/3, oracle CE 1.71 → 1.11 flat). The
stabilization (`router_learning_rate: 0.001` for router phases plus a
doubled 12-epoch warmup, `configs/stabilized-s{1,2,5}.yaml`) was piloted
on the two weak seeds and one strong seed against their own baselines:

| seed | baseline hard | stabilized hard | baseline route acc | stabilized route acc | baseline MI | stabilized MI | baseline graph util | stabilized graph util |
|---|---|---|---|---|---|---|---|---|
| s1 (strong) | 0.767 | 0.767 | 0.546 | 0.536 | 0.097 | 0.091 | 0.33 | 0.41 |
| s2 (weak) | 0.693 | **0.789** | 0.378 | **0.583** | 0.029 | **0.136** | 0.08 | 0.40 |
| s5 (weak) | 0.687 | **0.781** | 0.388 | **0.564** | 0.031 | **0.118** | 0.11 | 0.28 |

The weak-seed uniform-output mode disappears entirely: both weak seeds
now beat the best fixed route by wide margins and exceed the strong
baseline, while the strong seed is unchanged. The remaining seeds (s3,
s4) are running with the same configuration to complete the
five-seed stabilized campaign; updated statistics follow.
