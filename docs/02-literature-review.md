# Literature review

## Summary verdict

The broad intuition is supported, but its ingredients have different novelty statuses. Mapping-cone topology discrepancies, trainable persistent-topology losses, learned higher-order liftings, learned sheaf maps, categorical coherence, and dynamic routing all exist independently. The open intersection is a dynamically routed system whose experts inhabit genuinely different structured domains and whose translators are assessed both categorically and homologically.

This is a targeted review of primary literature checked through 2026-08-02. It is not a systematic review, patent search, or guarantee of priority.

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

## Novelty boundary

No primary source was found that jointly:

1. exposes several mathematical representation types;
2. learns typed translators and reverse or adjoint-like mates;
3. routes each datum or layer based on task utility and compute;
4. enforces chain, path, or unit–counit coherence; and
5. quantifies irreversible structural loss using map-aware kernels, cokernels, or derived obstruction objects.

This intersection—not any individual component—is HOMYMOLY's candidate contribution.
