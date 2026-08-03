# HOMYMOLY

**Homological routing between structured representations.**

HOMYMOLY is a research project investigating whether a machine-learning system can choose among vector, graph, cell-complex, and cellular-sheaf representations while explicitly measuring the structural damage caused by each conversion.

The working thesis is:

> A task- and compute-aware mixture of structured experts can learn typed representation changes, constrain those changes to behave like chain maps, and use mapping-cone defects to quantify topology lost or introduced during routing.

This repository contains the research specification and the executable Stage 1 foundation: deterministic graph/cell/sheaf-regime data, oriented incidences, finite chain complexes and maps, mapping-cone oracles, Hodge projection, a rank-2 connection-sheaf convention, a GB10 runtime profile, and a machine-readable integration gate. It does not yet claim a trained or experimentally validated routing method.

## Should HOMYMOLY use RTD?

**Yes.** Representation Topology Divergence (RTD) is one of the strongest foundations for the project, but it is a component and baseline rather than the novelty claim.

HOMYMOLY will use RTD in four roles:

1. as a reproduced diagnostic for paired metric representations;
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
- [Bibliography](references.bib)

## Stage 1 foundation

For GB10, use the pinned NGC base described in the [Stage 1 runtime build](docs/09-stage1-build.md); a fresh PyPI environment may resolve a different CUDA stack. In an existing compatible PyTorch environment, install the development package and run the exact-oracle gate:

```bash
python -m pip install -e '.[dev]'
python -m pytest
homymoly validate-foundation --config configs/stage1.yaml
```

The gate verifies deterministic balanced data, canonical oriented structures, the boundary law, graph-to-cell chain maps, mapping-cone chain laws, connection-sheaf operators, and hand-checkable Betti and holonomy sentinels. These are implementation invariants, not evidence that a structural loss improves learning.

## Next milestone

Implement a small three-expert system:

```text
node/vector view <-> graph cochain view <-> cellular-sheaf view
                          |
                          v
                   cell-complex view
```

The first milestone must establish whether mapping-cone or RTD regularization predicts and improves downstream behavior beyond ordinary task, reconstruction, and cycle-consistency losses. A dynamic router is added only after the two-domain conversions are validated.

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

Stage 1 mathematical/data/runtime foundation implemented and tested; learned experts, RTD reproduction, training ablations, and routing remain future gates. Literature checked through **2026-08-02**. Novelty conclusions are evidence-based research judgments, not a patent search or guarantee of priority.
