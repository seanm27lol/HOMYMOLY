# HOLYMOLY

**Homological routing between structured representations.**

HOLYMOLY is a research project investigating whether a machine-learning system can choose among vector, graph, cell-complex, and cellular-sheaf representations while explicitly measuring the structural damage caused by each conversion.

The working thesis is:

> A task- and compute-aware mixture of structured experts can learn typed representation changes, constrain those changes to behave like chain maps, and use mapping-cone defects to quantify topology lost or introduced during routing.

This repository currently contains the research brief, literature review, mathematical contract, proposed architecture, RTD integration plan, experimental protocol, and guardrails for derived-category and Langlands terminology. It does not yet claim an implemented or validated method.

## Should HOLYMOLY use RTD?

**Yes.** Representation Topology Divergence (RTD) is one of the strongest foundations for the project, but it is a component and baseline rather than the novelty claim.

HOLYMOLY will use RTD in four roles:

1. as a reproduced diagnostic for paired metric representations;
2. as an auxiliary topology-preservation loss when two views have one-to-one sample correspondence;
3. as a baseline against which a map-aware mapping-cone loss is tested;
4. as a source of established constructions, tests, and differentiation strategies.

RTD compares the Vietoris–Rips filtrations induced by two paired point-cloud representations. HOLYMOLY's proposed extension operates on explicit typed transformations between graph, cell, and sheaf complexes and combines structural defects with task utility and compute-aware routing. Directional RTD remains useful for asymmetric diagnosis; SRTD is a natural symmetric comparison.

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
- [Bibliography](references.bib)

## Proposed first milestone

Implement a small three-expert system:

```text
node/vector view <-> graph cochain view <-> cellular-sheaf view
                          |
                          v
                   cell-complex view
```

The first milestone must establish whether mapping-cone or RTD regularization predicts and improves downstream behavior beyond ordinary task, reconstruction, and cycle-consistency losses. A dynamic router is added only after the two-domain conversions are validated.

## Contribution boundary

HOLYMOLY does **not** claim to introduce:

- topological losses for machine learning;
- mapping-cone comparison of neural representations;
- learned graph liftings;
- neural cellular sheaves;
- categorical descriptions of neural architectures; or
- mixture-of-experts routing.

The candidate contribution is their disciplined intersection: dynamic, cost-aware routing among genuinely different structured representations, using typed transformations, categorical coherence, and map-aware homological defects.

## Status

Research specification. Literature checked through **2026-08-02**. Novelty conclusions are evidence-based research judgments, not a patent search or guarantee of priority.
