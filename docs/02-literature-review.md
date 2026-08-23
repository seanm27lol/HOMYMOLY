# Literature review

## Summary verdict

The broad intuition is supported, but its ingredients have different novelty statuses. Mapping-cone topology discrepancies, trainable persistent-topology losses, learned higher-order liftings, learned sheaf maps, categorical coherence, and dynamic routing all exist independently. The open intersection is a dynamically routed system whose experts inhabit genuinely different structured domains and whose translators are assessed both categorically and homologically.

This is a targeted review of primary literature checked through 2026-08-13. It is not a systematic review, patent search, or guarantee of priority. The search emphasized combinations of representation topology, mapping cones, differentiable graph lifting, cellular/sheaf learning, graph mixtures of experts, multi-view routing, and conditional computation.

## Review method and scope

The audit searched primary conference/journal pages and author papers across
PMLR, OpenReview, NeurIPS, ICLR, ICML, and TMLR. Query families combined
“representation topology divergence,” “mapping cone neural representation,”
“differentiable graph lifting cell complex,” “learned cellular sheaf,” “graph
mixture of experts,” “multi-view graph routing,” and “per-example conditional
computation.” Backward and forward citation links around RTD, differentiable
lifting, and graph MoE papers were used for snowballing. Secondary surveys
helped discover terms but are not the basis of the novelty boundary below.
The inclusion criterion was a primary method that overlaps at least one of:
paired-representation topology, learned change of structured domain,
chain/coherence constraints, or dynamic expert selection.

| HOMYMOLY ingredient | closest established precedent | remaining empirical question |
|---|---|---|
| paired topology discrepancy | RTD, RTD-AE, SRTD | does a degree-specific diagnostic add value for an actual typed learned map? |
| learned graph-to-higher-order structure | DCM, DiffLift | can held-out cell/sheaf targets be identified and reconstructed without target-view shortcuts? |
| local transport/sheaf learning | Neural Sheaf Diffusion, Knowledge Sheaves | when does a sheaf route beat graph/cell alternatives under one controlled task? |
| categorical path/coherence constraints | Learning Functors, Categorical Deep Learning | does an enforced diagram law improve task or conversion fidelity? |
| conditional structured computation | DeepMoE, GMoE, GraphMETRO, MvCGE | can routing among genuinely different domains beat fixed experts at measured matched accuracy/compute? |

## Direct precedent: representation topology divergence

[Representation Topology Divergence (RTD)](https://proceedings.mlr.press/v162/barannikov22a.html) is the closest literal predecessor. It compares two equal-size point clouds representing the same samples, with known one-to-one correspondence but potentially different ambient spaces. Pairwise dissimilarities generate Vietoris–Rips filtrations. Its cross-barcode records failures of induced homology maps to be isomorphisms, and the auxiliary construction is homotopy equivalent to a mapping cone.

Consequently, HOMYMOLY must not claim that it first uses exact sequences or mapping cones to measure representation mismatch.

[Learning Topology-Preserving Data Representations](https://arxiv.org/abs/2302.00136) differentiates RTD and uses it as an autoencoder objective. This already realizes “optimize a representation transformation using topological loss.” Its domain remains an input/latent pair of metric point clouds rather than typed graph, cell, and sheaf objects.

[Symmetric Divergence and Normalized Similarity](https://openreview.net/forum?id=pGgJ9qB2Io), accepted by TMLR in 2026, introduces SRTD, SRTD-lite, and Normalized Topological Similarity. SRTD uses a single symmetric construction whose chain complex is homotopy equivalent to the mapping cone of an inclusion from an intersection Rips complex to a union Rips complex. It makes the proximity to HOMYMOLY's “forward and reverse loss” intuition especially clear.

[A Quotient Homology Theory of Representation in Neural Networks](https://arxiv.org/abs/2502.01360), TMLR 2026, studies equivalence classes induced by non-injective ReLU representations and the homology of the resulting quotient. It supports the intuition that quotient topology can characterize which distinctions a representation glues together, but it is an analysis method for piecewise-linear networks rather than a routed architecture.

## Topological representations and learned liftings

The [ICML Topological Deep Learning Challenge 2024](https://arxiv.org/abs/2409.05211) explicitly studies liftings between point clouds, graphs, hypergraphs, and simplicial, cellular, or combinatorial complexes. Its 52 qualifying submissions demonstrate both the breadth of available transformations and the unresolved question of which lifting is appropriate for a dataset and task.

[TopoBench](https://arxiv.org/abs/2406.06642) modularizes data loading, transformations, liftings, models, training, and evaluation across topological domains. It is a natural experimental substrate for HOMYMOLY.

[From Latent Graph to Latent Topology Inference](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6b97236d90d945be7c58268207a14f4f-Abstract-Conference.html) introduces a Differentiable Cell Complex Module that learns higher-order cell probabilities end to end. [Differentiable Lifting for Topological Neural Networks](https://openreview.net/forum?id=eC89CbINIw), ICLR 2026, learns graph liftings into hypergraph, simplicial, cellular, and combinatorial complexes and reports substantial improvements over static liftings. Neither routes among several representation categories using map-aware homological loss.

## Sheaf and local-to-global learning

[Neural Sheaf Diffusion](https://proceedings.neurips.cc/paper_files/paper/2022/hash/75c45fca2aa416ada062b26cc4fb7641-Abstract-Conference.html) learns cellular-sheaf restriction maps and uses sheaf Laplacians for graph learning. It establishes that local transport geometry can be learned as part of a neural architecture.

[Knowledge Sheaves](https://proceedings.mlr.press/v206/gebhart23a.html) describes a knowledge-graph embedding as an approximate global section satisfying schema-induced local constraints. It provides a strong model of heterogeneous local vector spaces and consistency, but does not dynamically switch among representation types or optimize higher exactness defects.

[Copresheaf Topological Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2025/hash/dc62cd4ec77f3bbec8e6245d0bd91d08-Abstract-Conference.html) supplies a broad local-space framework spanning images, point clouds, graphs, meshes, and manifolds. It still fixes the relevant substrate rather than routing graph ↔ cell ↔ sheaf transformations.

## Categorical learning and two-way consistency

[Learning Functors Using Gradient Descent](https://arxiv.org/abs/2009.06837) expresses CycleGAN-like domains and transformations as a category presented by generators and relations. Training learns functors while path-equivalence losses enforce composition laws. It is the closest categorical predecessor to HOMYMOLY's diagram-coherence loss.

[Categorical Deep Learning](https://proceedings.mlr.press/v235/gavranovic24a.html) argues for category theory as a language unifying architectural constraints and implementations. It supports the typed-diagram viewpoint but does not provide adaptive switching with homological conversion defects.

[Cycle-Consistent Probability Divergences Across Different Spaces](https://proceedings.mlr.press/v151/zhang22d.html) develops two-way objectives across genuinely different probability spaces. It is an important non-homological baseline: reverse consistency and heterogeneous-space comparison do not require homology.

## Routing literature

Mixture-of-experts systems such as [Switch Transformers](https://www.jmlr.org/papers/v23/21-0998.html) learn input-conditioned gates over computational experts. Modular and routing networks similarly select computation paths. Their modules generally expose compatible tensor interfaces; they do not treat a graph-to-sheaf transition as a typed functor, enforce unit/counit or naturality constraints, or compute an obstruction object for structural loss.

The graph-routing neighborhood is considerably closer than that broad comparison suggests. [DeepMoE](https://proceedings.mlr.press/v115/wang20d.html) uses per-example dynamic routing to adaptively sparsify a deep network and reduce computation. [Graph Mixture of Experts](https://papers.nips.cc/paper_files/paper/2023/hash/9f4064d145bad5e361206c3303bda7b8-Abstract-Conference.html) routes graph nodes among specialized aggregation experts. [GraphMETRO](https://papers.nips.cc/paper_files/paper/2024/hash/11c892a9fcc430cc0f4c7d457e5d60ea-Abstract-Conference.html) composes aligned experts for graph distribution shifts. [Multi-view Collaborative Graph Expert Learning](https://openreview.net/forum?id=dsp8dUlZFq) dynamically activates graph-aware experts across multiple graph views with discrepancy and load-balancing objectives. These works substantially overlap HOMYMOLY's routing and multi-view system design; they do not, in their stated methods, evaluate graph/cell/sheaf conversions with degree-specific mapping-cone defects.

## Novelty boundary

In this targeted search, no primary source was located that jointly:

1. exposes several mathematical representation types;
2. learns typed translators and reverse or adjoint-like mates;
3. routes each datum or layer based on task utility and compute;
4. enforces chain, path, or unit–counit coherence; and
5. quantifies irreversible structural loss using map-aware kernels, cokernels, or derived obstruction objects.

This is a narrow candidate intersection, not a priority claim. The current implementation must still demonstrate that its branches are genuine typed conversions and that its homological diagnostic is valid before the intersection can be claimed as an empirical contribution.
