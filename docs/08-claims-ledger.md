# Claims ledger

Every proposed claim must identify its closest prior art, supporting experiment, falsifier, and current status.

| ID | Proposed claim | Closest prior art | Required evidence | Falsifier | Status |
|---|---|---|---|---|---|
| C1 | Structural defects predict task damage during typed conversions | RTD, RTD-AE, SRTD | Held-out correlation and intervention tests beyond reconstruction baselines | No incremental predictive value | **Not supported (corruption suite, run 10 ckpt)** — exact SRTD tracks damage (ρ 0.81–0.96) but partial correlation controlling for reconstruction is ≈ 0 (−0.16 to +0.24); reconstruction displacement alone predicts damage equally well (ρ 0.81–0.96). The correlation question is answered; the intervention question is C2's ablation |
| C2 | Map-aware cone loss improves a learned graph/cell/sheaf translator | RTD-AE, differentiable liftings | Matched-compute ablations across synthetic regimes and a real task | Task/cycle-only model matches it | **Not supported (comprehensively)** — task-only matches +recon/+chain/full on the independent tier; on the gauge tier the chain term reliably controls consistency (0.18 vs ~1.4–2.0) across 16 runs without improving task accuracy or robustness; and an eight-pair seed campaign finds no robust independent damage signal from the topological defect (5/8 positive, mean Δ +0.096 ± 0.364) |
| C3 | A cost-aware router specializes by structural regime | MoE routing, learned topological liftings | Regime-conditioned route analysis and fixed/random/oracle comparisons | Route collapse or shortcut behavior | **Supported (stabilized confirmatory, n=5)** — margin over best fixed +0.108, 95% CI [+0.096, +0.119] (t(4)); all seeds beat best fixed and dense; variance 0.777 ± 0.010; MI mean 0.113, balanced utilization everywhere; weak-seed collapse mode eliminated by router-phase LR + longer warmup (`docs/17-gate2-confirmatory.md`) |
| C4 | Direct cone loss supplies information not contained in RTD | RTD/SRTD | Cone-only, RTD-only, and combined ablations on identical paired data | Metrics are redundant in prediction and intervention | **Moot on this benchmark** — cone-style and RTD-style terms are both inert, so no operating regime distinguishes them (see `docs/15-gate3-record.md`) |
| C5 | Sheaf routes are selected only when local compatibility matters | Neural Sheaf Diffusion, Knowledge Sheaves | Controlled local-consistency benchmark | Sheaf selection unrelated to regime or utility | **Partially supported (run 9)** — sheaf selected on 51% of sheaf-regime vs 17–26% elsewhere; informative but not exclusive, matching the design's overlapping-reliability ceiling |
| C6 | Chemically valid ring/2-cell lifts improve molecular property prediction over the graph route | Differentiable liftings, TopoBench | OGBG-MOLHIV official scaffold split and evaluator, matched configs, multiple seeds | Graph route matches or beats the ring-lift route | **Not supported (3 seeds)** — cell 0.723 ± 0.017 vs graph 0.771 ± 0.014 test AUROC on the official split; consistent across seeds, a generalization gap rather than optimization (valid within 0.012); official test is ~100% ring-bearing so the cell route had every opportunity |

## Claim discipline

- Diagnostic correlation is not causal evidence that a loss improves training.
- Improvement over one fixed graph model is not evidence that dynamic routing is necessary.
- A lower topological defect is not evidence of greater semantic information.
- A synthetic result is not evidence of universal representation efficiency.
- “First” claims require a refreshed systematic search immediately before submission.
