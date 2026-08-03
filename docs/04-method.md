# Proposed method

## System view

HOMYMOLY is a small typed computation graph rather than a collection of arbitrary interchangeable structures.

```text
                        +------------------+
                        | cost-aware router |
                        +---------+--------+
                                  |
                 +----------------+----------------+
                 |                |                |
          vector/graph        cell-complex    cellular-sheaf
             expert              expert             expert
                 |                |                |
                 +------ learned typed maps -------+
                         + reverse mates
```

Each expert owns its native state and operations. Every edge declares a translator, optional reverse/adjoint-like mate, chain/cochain interface, dissimilarity adapter, and measured compute cost.

## Components

### Structured experts

The MVP contains:

- a node/vector or graph-message-passing expert;
- a cell-complex expert for cycles and higher-order interactions;
- a cellular-sheaf expert for locally varying feature spaces and restriction maps.

### Translators

Translators implement graph liftings, cell projections, or sheaf feature/restriction construction. Wherever the declared complexes make it meaningful, degree-wise maps are constrained as chain maps.

### Router

The router selects an expert or composable path per example or layer. Its features may include:

- input statistics;
- current task uncertainty;
- estimated structural defect;
- predicted latency and memory;
- accumulated route cost.

Discrete routing can use a straight-through, Gumbel-softmax, policy-gradient, or differentiable mixture relaxation. The choice is an implementation decision, not part of the mathematical novelty.

### Diagnostics

The system reports task loss, reconstruction/cycle loss, chain residual, cone statistic, RTD/SRTD where applicable, path disagreement, route distribution, latency, peak memory, and expert utilization.

## Objective

The initial objective is

\[
\begin{aligned}
\mathcal L={}&\mathcal L_{\mathrm{task}}
+\lambda_{\mathrm{recon}}\mathcal L_{\mathrm{recon}}
+\lambda_{\mathrm{chain}}\mathcal L_{\mathrm{chain}}\\
&+\lambda_{\mathrm{cone}}\mathcal L_{\mathrm{cone}}
+\lambda_{\mathrm{cycle}}\mathcal L_{\mathrm{cycle}}
+\lambda_{\mathrm{path}}\mathcal L_{\mathrm{path}}\\
&+\lambda_{\mathrm{RTD}}\mathcal L_{\mathrm{RTD}}
+\lambda_{\mathrm{cost}}\mathbb E[\operatorname{Cost(route)}]
+\lambda_{\mathrm{balance}}\mathcal L_{\mathrm{balance}}.
\end{aligned}
\]

RTD is evaluated only for representations with paired entities and valid within-representation dissimilarities. Cone loss is evaluated only for declared chain maps. Neither term is applied universally merely because it is available.

Here, \(\mathcal L_{\mathrm{cone}}\) is route-conditioned; it is not the sum of all cone Betti numbers. A graph-to-cell inclusion that intentionally fills a cycle has a legitimate degree-two cone class, so forcing every cone to be acyclic would erase the feature the cell route was introduced to represent. Acyclicity is an objective only for a conversion declared to seek a quasi-isomorphism. Other routes use preregistered expected cone signatures, predictive diagnostics, or task-conditioned comparisons.

Exact cone homology is evaluation-only unless the translator is parameterized as an exact chain map. For an approximate learned map, train on the chain residual or a declared differentiable proxy and do not interpret its mapping-cone homology until the chain-map tolerance is met.

## Concrete v0.1 graph-hub design

Every sample begins as an attributed graph with candidate 2-cells and optional local-frame information. Three legal routes produce a common graph-level embedding:

```text
G -> GraphExpert -> pool
G -> CellLift -> CellExpert -> CellProject -> pool
G -> SheafLift -> SheafExpert -> SheafProject -> pool
```

The graph hub avoids six underidentified pairwise translators. Direct cell-to-sheaf conversion and arbitrary multihop routing are deferred until the three primary routes work.

For the Stage-1/2 cell route, enumerate triangles as candidate faces and learn soft inclusion gates. The topology core accepts arbitrary simple-cycle faces, but the current batched data contract is explicitly triangle-only. Before molecular rings or short chordless cycles are enabled, replace the `[3, F]` interface with sparse \(B_2\) or padded oriented boundary-edge indices and coefficients. Every candidate face column must be an oriented cycle so that \(B_1B_2=0\) remains exact. The projection back to edge space can be tested against the Hodge projection

\[
P_1=I-B_2(B_2^\top B_2)^+B_2^\top,
\]

which makes discarded filled-cycle components explicit.

For the sheaf route, lift node channels into finite stalks and infer incidence restriction maps. Learn compatible degree-zero and degree-one maps so that

\[
\delta_{\mathrm{sheaf}}F_0\approx F_1\delta_{\mathrm{graph}}.
\]

Start with low stalk ranks and tied-transpose or pseudoinverse projections. More general adjoint-like translators are a later experiment.

The first router should be deliberately small. It observes cheap, label-independent summaries such as graph size, degree moments, candidate-face count, feature norms, and estimated local inconsistency. Top-1 routing can use a straight-through Gumbel-softmax relaxation. Training proceeds in three stages: pretrain experts and translators, train the router against per-sample oracle routes with experts frozen, then jointly fine-tune while annealing soft to hard routing and monitoring load balance.

## Important anti-collapse constraints

Nilpotency or chain-map penalties alone can favor trivial zero maps. Every experiment therefore requires task supervision plus at least one non-collapse control such as reconstruction, variance preservation, contrastive separation, or an explicit rank constraint.

Homology is coarse: two representations can have identical homology while differing drastically in geometry and semantics. Homological losses supplement rather than replace task, metric, and information-preservation objectives.
