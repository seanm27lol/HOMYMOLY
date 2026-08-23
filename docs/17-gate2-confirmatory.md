# Gate-2 confirmatory campaign (2026-08-04)

> **Validity correction (updated 2026-08-22).** The first five-seed campaign
> had mean margin +0.036, 95% CI [−0.018, +0.090], and was inconclusive. The
> stabilization was designed after inspecting weak seeds s2/s5 and then
> evaluated on those same seeds; its +0.108 interval is post-selection
> development evidence. The protocol-aligned v2 campaign below met its frozen
> numerical decision rule, with a disclosed procedural deviation that prevents
> calling it pristine preregistration. All of these historical routers distilled
> targets from validation latent-regime labels. Regime was not an inference
> input, but this is privileged training supervision. At inference, the router
> also summarizes available cell and sheaf observations, including active-face
> and transport statistics; it is not graph-only.

## Protocol-aligned routing-v2 result

The five valid runs used seeds 20260906–20260910 and the immutable config
hashes recorded in [the v2 protocol](19-routing-confirmatory-v2-protocol.md).
All completed with no failed gate. Their environment records agree on commit
`e69b07707950b6abe332366c51fe8c94254899f3`, executable fingerprint
`473fb0f6714798274c38949107221df3bd941e89273a6eef76e54394d6c1f1d8`,
PyTorch 2.13.0+cu130, CUDA 13.0, and NVIDIA GB10. Later runs started from a
dirty worktree, but the recorded executable fingerprint was identical.

| seed | hard | best fixed | margin | dense | hard−dense | route acc | MI | utilization (g/c/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s1 | 0.8028 | 0.6743 | +0.1285 | 0.7723 | +0.0305 | 0.6100 | 0.1622 | .327/.344/.329 |
| s2 | 0.7745 | 0.6754 | +0.0991 | 0.7571 | +0.0174 | 0.5468 | 0.0972 | .305/.344/.351 |
| s3 | 0.7680 | 0.6667 | +0.1013 | 0.7582 | +0.0098 | 0.5381 | 0.0934 | .318/.390/.292 |
| s4 | 0.7789 | 0.6667 | +0.1122 | 0.7342 | +0.0447 | 0.5752 | 0.1264 | .381/.329/.290 |
| s5 | 0.7745 | 0.6667 | +0.1078 | 0.7505 | +0.0240 | 0.5501 | 0.1006 | .350/.340/.310 |

The primary hard-minus-best-fixed margin is **+0.1098039216** across n=5
(sample SD 0.0116918587; two-sided Student-t 95% CI
[0.0952865615, 0.1243212816]). Its lower endpoint is above zero, so the
campaign **supports the frozen, scoped endpoint under its decision rule**.
All five margins are positive; an exact two-sided sign-test sensitivity gives
p=0.0625. With only five seeds, normality, independence, and
representativeness assumptions cannot be checked empirically.

This is not an unrestricted claim that routing is graph-only, input-only, or
necessary across tasks. It concerns the historical regime-distilled router
over available structured views on this synthetic benchmark. It does not test
a graph-to-cell/sheaf translator, learned chain map, mapping cone, or
representation conversion.

There was one protocol-integrity deviation: an aborted pre-commit attempt for
seed 20260906 exposed validation metrics under a different executable before
the valid frozen-config rerun. The core endpoint, seed list, decision rule, and
config hashes were committed before the valid campaign, but the earlier
observation means “genuinely untouched” and “preregistered” overstate the
evidence. The appropriate label is **protocol-aligned with a disclosed
procedural deviation**.

Per the plan's confirmatory rule ("at least five fresh seeds for the
primary synthetic comparison… report confidence intervals and all failed
or collapsed runs"), five independent full Gate-2 runs (seeds
20260901–20260905, configs `configs/confirmatory-s{1..5}.yaml`, artifacts
`artifacts/confirmatory-s{1..5}/`) were executed after the run-9/10
pilot. This is the original statistical verdict on the routing claim (C3).

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
  Pooling the two earlier pilot runs gives +0.046 [ +0.010, +0.082 ], but
  mixes development and confirmatory evidence and is not a valid
  confirmatory interval.
- **Routing is regime-informative on every seed**: MI 0.029–0.097
  (mean 0.052 ± 0.038), route accuracy 0.378–0.546 — always above the
  1/3 baseline, consistent with the measured ~0.55–0.58 identifiability
  ceiling of the anti-shortcut design.
- **Failed or collapsed runs (required reporting)**: seed s2 fails the
  Gate-4 utility criterion outright (0.693 < best fixed 0.704) and shows
  partial route collapse toward cell with the graph route at 8%
  utilization; s5 is a marginal pass (0.687 vs 0.680) with the same
  tilt. Weak seeds correlate with under-use of the graph route.
- **Interpretation**: the point estimate is positive, but the prespecified
  five-seed interval crosses zero and one seed loses to the best fixed route.
  The confirmatory characterization is *inconclusive with material seed
  variance*.

## Consequences of the original campaign

- The original campaign is inconclusive. Pilot runs remain development history
  and are not added to its confidence interval. The later v2 campaign supports
  its separately frozen, narrowly scoped endpoint with the qualification above.
- Follow-up candidates, in priority order: (a) stabilize weak-seed
  routing (longer warmup, lower entropy weight, or oracle-target
  smoothing to reduce collapse-to-cell draws); (b) the
  molecularly-informed cell architecture (Gate-5 redesign direction).

## Adaptive stabilization study (2026-08-04; exploratory)

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
baseline, while the strong seed is unchanged.

## Stabilized same-seed rerun (2026-08-04; post-selection)

With the stabilization applied to all five seeds (`configs/stabilized-s{1..5}.yaml`):

| seed | hard | best fixed | dense | oracle | route acc | MI | utilization (g/c/s) |
|---|---|---|---|---|---|---|---|
| s1 | 0.767 | 0.667 | 0.766 | 0.999 | 0.536 | 0.091 | .41/.25/.34 |
| s2 | 0.789 | 0.669 | 0.749 | 0.998 | 0.583 | 0.136 | .40/.32/.28 |
| s3 | 0.766 | 0.667 | 0.748 | 0.999 | 0.534 | 0.093 | .38/.34/.27 |
| s4 | 0.782 | 0.678 | 0.736 | 0.976 | 0.576 | 0.126 | .36/.34/.30 |
| s5 | 0.781 | 0.667 | 0.764 | 0.999 | 0.564 | 0.118 | .28/.42/.29 |

- **Descriptive margin over best fixed: mean +0.108, interval
  [+0.096, +0.119]** (the t calculation is descriptive only because the
  configuration was chosen using these seeds)
  — against +0.036 [−0.018, +0.090] unstabilized. All five seeds beat
  the best fixed route *and* the dense ensemble.
- **Observed variance fell 4×**: hard accuracy 0.777 ± 0.010 (was
  0.723 ± 0.043); no seed exhibits the uniform-output or
  collapse-to-cell mode; utilization is balanced on every seed.
- Routing is strongly regime-informative on these reused seeds: MI mean 0.113
  (was 0.052), route accuracy mean 0.559. Prior finite benchmark classifiers
  scored approximately 0.55–0.58; that is a descriptive range, not a proven
  identifiability ceiling, so the residual gap cannot be attributed entirely
  to intentional overlap.
- **Config change**: `router_learning_rate: 0.001` and
  `router_warmup_epochs: 12` are now the canonical settings in
  `configs/gate2.yaml` (measured better on every axis: same-or-better
  accuracy on every reused seed, no failed seeds, tighter descriptive
  interval). Its generalization is supported by the protocol-aligned v2
  campaign only within the scope and procedural qualifications stated above.
