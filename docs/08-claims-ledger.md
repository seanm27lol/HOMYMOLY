# Claims ledger

Every proposed claim must identify its closest prior art, supporting experiment, falsifier, and current status.

| ID | Proposed claim | Closest prior art | Required evidence | Falsifier | Status |
|---|---|---|---|---|---|
| C1 | Structural defects predict task damage during typed conversions | RTD, RTD-AE, SRTD | Held-out correlation and intervention tests beyond reconstruction baselines | No incremental predictive value | **Not supported (corruption suite, run 10 ckpt)** — exact SRTD tracks damage (ρ 0.81–0.96) but partial correlation controlling for reconstruction is ≈ 0 (−0.16 to +0.24); reconstruction displacement alone predicts damage equally well (ρ 0.81–0.96). The correlation question is answered; the intervention question is C2's ablation |
| C2 | Map-aware cone loss improves a learned graph/cell/sheaf translator | RTD-AE, differentiable liftings | Matched-compute ablations across synthetic regimes and a real task | Task/cycle-only model matches it | **Not supported (two data designs)** — task-only matches +recon/+chain/full on the independent tier; on the gauge tier (clean samples are sections, surrogate has real range and is held at 0.18 vs 1.40) the chain term still does not improve task accuracy or corruption robustness. Weak follow-up signal across three seed pairs: topological-defect partial correlation with damage rises under consistency constraint in 2/3 pairs (mean Δ +0.17, high variance) — needs a larger campaign to confirm or kill |
| C3 | A cost-aware router specializes by structural regime | MoE routing, learned topological liftings | Regime-conditioned route analysis and fixed/random/oracle comparisons | Route collapse or shortcut behavior | **Supported (synthetic, run 9)** — MI 0.067, route acc 0.503 vs 0.335 marginal, native selection 0.46–0.54 per regime, hard 0.743 > best fixed 0.667, non-collapsed; capped by the benchmark's anti-shortcut identifiability ceiling |
| C4 | Direct cone loss supplies information not contained in RTD | RTD/SRTD | Cone-only, RTD-only, and combined ablations on identical paired data | Metrics are redundant in prediction and intervention | **Moot on this benchmark** — cone-style and RTD-style terms are both inert, so no operating regime distinguishes them (see `docs/15-gate3-record.md`) |
| C5 | Sheaf routes are selected only when local compatibility matters | Neural Sheaf Diffusion, Knowledge Sheaves | Controlled local-consistency benchmark | Sheaf selection unrelated to regime or utility | **Partially supported (run 9)** — sheaf selected on 51% of sheaf-regime vs 17–26% elsewhere; informative but not exclusive, matching the design's overlapping-reliability ceiling |
| C6 | Chemically valid ring/2-cell lifts improve molecular property prediction over the graph route | Differentiable liftings, TopoBench | OGBG-MOLHIV official scaffold split and evaluator, matched configs, multiple seeds | Graph route matches or beats the ring-lift route | **Not supported (3 seeds)** — cell 0.723 ± 0.017 vs graph 0.771 ± 0.014 test AUROC on the official split; consistent across seeds, a generalization gap rather than optimization (valid within 0.012); official test is ~100% ring-bearing so the cell route had every opportunity |

## Claim discipline

- Diagnostic correlation is not causal evidence that a loss improves training.
- Improvement over one fixed graph model is not evidence that dynamic routing is necessary.
- A lower topological defect is not evidence of greater semantic information.
- A synthetic result is not evidence of universal representation efficiency.
- “First” claims require a refreshed systematic search immediately before submission.
