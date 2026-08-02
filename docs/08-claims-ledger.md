# Claims ledger

Every proposed claim must identify its closest prior art, supporting experiment, falsifier, and current status.

| ID | Proposed claim | Closest prior art | Required evidence | Falsifier | Status |
|---|---|---|---|---|---|
| C1 | Structural defects predict task damage during typed conversions | RTD, RTD-AE, SRTD | Held-out correlation and intervention tests beyond reconstruction baselines | No incremental predictive value | Proposed |
| C2 | Map-aware cone loss improves a learned graph/cell/sheaf translator | RTD-AE, differentiable liftings | Matched-compute ablations across synthetic regimes and a real task | Task/cycle-only model matches it | Proposed |
| C3 | A cost-aware router specializes by structural regime | MoE routing, learned topological liftings | Regime-conditioned route analysis and fixed/random/oracle comparisons | Route collapse or shortcut behavior | Proposed |
| C4 | Direct cone loss supplies information not contained in RTD | RTD/SRTD | Cone-only, RTD-only, and combined ablations on identical paired data | Metrics are redundant in prediction and intervention | Proposed |
| C5 | Sheaf routes are selected only when local compatibility matters | Neural Sheaf Diffusion, Knowledge Sheaves | Controlled local-consistency benchmark | Sheaf selection unrelated to regime or utility | Proposed |

## Claim discipline

- Diagnostic correlation is not causal evidence that a loss improves training.
- Improvement over one fixed graph model is not evidence that dynamic routing is necessary.
- A lower topological defect is not evidence of greater semantic information.
- A synthetic result is not evidence of universal representation efficiency.
- “First” claims require a refreshed systematic search immediately before submission.
