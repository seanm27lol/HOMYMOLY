# Research brief

## Problem

Most neural architectures commit to one representation family: tensors, graphs, simplicial complexes, cellular sheaves, or another fixed domain. Even systems that learn topology usually learn one lifting into one target class. This may be inefficient for heterogeneous data: pairwise structure can favor a graph, higher-order interactions can favor a cell complex, and locally varying compatibility can favor a sheaf.

HOMYMOLY asks whether representation type can become a routed computational choice while conversions remain mathematically inspectable.

## Core hypothesis

A model with multiple structured experts can outperform a fixed-representation model at a matched compute budget when:

- examples genuinely require different structural inductive biases;
- conversions are constrained to respect boundary or coboundary maps;
- structural conversion defects supplement, rather than replace, task and reconstruction losses; and
- a router trades expected utility against measured latency, memory, and sparsity.

## Candidate contribution

The proposed contribution is a **cost-aware typed representation router** with:

- graph, cell-complex, and cellular-sheaf experts;
- learned lift/projection maps;
- chain-map and path-coherence constraints;
- map-aware mapping-cone defects;
- RTD/SRTD adapters for paired metric views;
- forward/reverse cycle or unit–counit defects; and
- explicit compute-aware gating.

## Falsifiable hypotheses

**H1 — Routing utility.** On mixed-regime data, the routed model matches or exceeds the best fixed expert under a matched compute budget.

**H2 — Defect predictiveness.** Mapping-cone and RTD defects predict downstream degradation caused by a representation conversion better than reconstruction loss alone.

**H3 — Regularization value.** Adding chain/cone regularization improves robustness or transfer beyond task loss and ordinary cycle consistency.

**H4 — Specialization.** Learned routes correspond reproducibly to known structural regimes rather than collapsing to one expert or tracking an irrelevant shortcut.

## Explicit nonclaims

- Exactness is not identical to statistical information preservation.
- Quasi-isomorphism does not imply preservation of metric geometry, semantics, conditioning, or compute cost.
- Sheaves are not inherently more efficient than graphs or vectors.
- RTD is not a direct test of an arbitrary learned map's exactness.
- Fourier–Mukai transforms are not generic data converters.
- Geometric Langlands is not part of the initial implementation.

## First scope

The first implementation is deliberately restricted to finite-dimensional real chain/cochain complexes derived from graphs, cell complexes, and cellular sheaves. Arbitrary groups, logical structures, and algebraic varieties remain future work until canonical, computationally viable interfaces are specified.
