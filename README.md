# HOMYMOLY

**Homological routing between structured representations.**

HOMYMOLY is a research project investigating whether a machine-learning system can choose among vector, graph, cell-complex, and cellular-sheaf representations while explicitly measuring the structural damage caused by each conversion.

The working thesis is:

> A task- and compute-aware mixture of structured experts can learn typed representation changes, constrain those changes to behave like chain maps, and use mapping-cone defects to quantify topology lost or introduced during routing.

This repository contains the research specification, the executable Stage 1
foundation, three structured experts, graph-to-cell/sheaf translators, a
cost-aware router, degree-specific RTD/SRTD references, exact finite chain-map
layers, a corruption suite, and resumable GB10 experiments.

Current status after the validity audit and protocol-aligned routing-v2
campaign: the historical regime-distilled hard router beat the best fixed
expert on all five valid seeds. The mean margin was **+0.1098** (sample SD
0.0117; Student-t 95% CI [0.0953, 0.1243]), meeting the frozen numerical
decision rule. This is a narrow result for routing over available structured
views: training used privileged regime labels to distill utility targets, and
inference used graph, active-face, and sheaf-transport summaries. It is not a
graph-only conversion result. The campaign is protocol-aligned rather than a
pristine preregistration because an aborted pre-commit seed-20260906 attempt
exposed validation metrics under different code before the valid rerun; see
[`docs/19`](docs/19-routing-confirmatory-v2-protocol.md). Corrected Gate-3
reports now provide fixed-expert embedding diagnostics only and do not test a
translator, learned map, or conversion claim. Each corruption kind has 13
complete blocks, 65 batch observations, and 306 unique examples. Eleven of 12
within-checkpoint bootstrap intervals include zero; the sole exception has
within-block permutation p=0.115. All nine paired added-loss-versus-task-only
intervals include zero (p≥0.32397). These checkpoint-conditional analyses have
no multiplicity adjustment.

## Should HOMYMOLY use RTD?

**Yes.** Representation Topology Divergence (RTD) is one of the strongest foundations for the project, but it is a component and baseline rather than the novelty claim.

HOMYMOLY will use RTD in four roles:

1. as an independently implemented, hand-validated diagnostic for paired
   metric representations;
2. as an auxiliary topology-preservation loss when two views have one-to-one sample correspondence;
3. as a baseline against which a map-aware mapping-cone loss is tested;
4. as a source of established constructions, tests, and differentiation strategies.

RTD compares the Vietoris–Rips filtrations induced by two paired point-cloud representations. HOMYMOLY's proposed extension operates on explicit typed transformations between graph, cell, and sheaf complexes and combines structural defects with task utility and compute-aware routing. Directional RTD remains useful for asymmetric diagnosis; SRTD is a natural symmetric comparison.

## Repository map

- [Original idea and reconstruction](docs/00-original-idea.md)
- [Research brief](docs/01-research-brief.md)
- [Literature review](docs/02-literature-review.md)
- [Mathematical contract](docs/03-mathematical-contract.md)
- [Proposed method](docs/04-method.md)
- [RTD integration](docs/05-rtd-integration.md)
- [Experimental protocol](docs/06-experimental-protocol.md)
- [Derived and Langlands guardrails](docs/07-derived-langlands-scope.md)
- [Claims ledger](docs/08-claims-ledger.md)
- [Stage 1 runtime build](docs/09-stage1-build.md)
- [GB10 experimental plan](docs/10-gb10-experimental-plan.md)
- [Stage 1 validation record](docs/11-stage1-validation.md)
- [Gate 2 training and automatic GB10 launch](docs/12-gate2-training.md)
- [Gate 2 run handoff](docs/13-gate2-run-handoff.md)
- [Gate 2 review](docs/14-gate2-review.md)
- [Gate 3 record](docs/15-gate3-record.md)
- [Gate 5 record: molecular transfer](docs/16-gate5-record.md)
- [Gate 2 confirmatory campaign](docs/17-gate2-confirmatory.md)
- [Paper: Typed Representation Routing with Homological Structure](docs/18-paper.md)
- [Routing confirmatory v2 protocol](docs/19-routing-confirmatory-v2-protocol.md)
- [Audit remediation and continuation record](docs/20-audit-remediation.md)
- [Bibliography](references.bib)

## Stage 1 foundation

For GB10, use the pinned NGC base described in the [Stage 1 runtime build](docs/09-stage1-build.md); a fresh PyPI environment may resolve a different CUDA stack. In an existing compatible PyTorch environment, install the development package and run the exact-oracle gate:

```bash
python -m pip install -e '.[dev]'
python -m pytest
homymoly validate-foundation --config configs/stage1.yaml
```

The gate verifies deterministic balanced data, canonical oriented structures, the boundary law, graph-to-cell chain maps, mapping-cone chain laws, connection-sheaf operators, and hand-checkable Betti and holonomy sentinels. These are implementation invariants, not evidence that a structural loss improves learning.

## Current architecture and experiments

The system keeps two claims separate:

- The routing experiment asks whether a cheap router over the available
  structured observations can select one expert per example. Its summaries
  include graph features, candidate/active-face statistics, and sheaf
  transport statistics; no label or regime tensor is an inference input.
  The routing-v2 campaign used privileged latent-regime supervision during
  training to distill a regime-by-expert utility table. The canonical config
  exposes a per-example utility mode that removes that training privilege,
  but the reported v2 result does not use that mode.
- The conversion experiment holds target cell activity and sheaf transports
  out of translator inputs and predicts them from the graph view, while
  supplying candidate face incidence. In the current synthetic generator,
  those held-out targets are not identifiable from the graph inputs, so this
  is an implemented reconstruction objective awaiting an identifiable
  benchmark—not conversion evidence. A separate exact-chain-map layer
  parameterizes degree maps in the nullspace of the chain-map equations and
  evaluates their mapping cones exactly.

On the GB10, a 100-iteration architectural benchmark measured 38.3 ms median
for selective hard routing and 67.8 ms for the dense three-expert path at batch
64 (1.77× speedup). This is measured latency, not the earlier declared-cost
proxy, and it does not by itself establish an accuracy/compute Pareto win.

## Contribution boundary

HOMYMOLY does **not** claim to introduce:

- topological losses for machine learning;
- mapping-cone comparison of neural representations;
- learned graph liftings;
- neural cellular sheaves;
- categorical descriptions of neural architectures; or
- mixture-of-experts routing.

The candidate contribution is their disciplined intersection: dynamic, cost-aware routing among genuinely different structured representations, using typed transformations, categorical coherence, and map-aware homological defects.

## Status

Stage 1 and the fixed experts are implemented. The five-seed routing-v2 result
supports only the scoped historical regime-distilled, structured-view routing
endpoint described above; n=5 leaves distributional assumptions uncheckable,
and the exact two-sided sign-test sensitivity is p=0.0625. The published scalar
RTD convention is now degree 1 with full-matrix 0.9-quantile normalization;
multi-degree results are returned explicitly rather than summed. A synthetic
exact-chain-map recovery run reached test MSE 1.5e−14 with zero mapping-cone
Betti numbers. Molecular results are exploratory because the official test
split was consulted across architecture iterations. Literature and novelty
conclusions are research judgments, not a patent search or guarantee of
priority.
