# Experimental protocol

## Experimental order

1. Reproduce RTD/SRTD behavior independently.
2. Validate one graph-to-cell or graph-to-sheaf translator without routing.
3. Test whether chain and cone statistics predict conversion-induced degradation.
4. Test whether those losses improve training beyond reconstruction and cycle consistency.
5. Introduce a two-expert router.
6. Expand to the three-expert graph/cell/sheaf system.

This order prevents router complexity from hiding a nonfunctional structural loss.

## Synthetic mixed-regime benchmark

Construct examples from at least three regimes:

- **pairwise regime:** labels depend on edges or local graph neighborhoods;
- **higher-order regime:** labels depend on triangles, rings, cavities, or multiway interactions;
- **local-consistency regime:** labels depend on transport or compatibility rules that vary across nodes and edges.

Mix regimes within the same training and evaluation splits so that no fixed representation is uniformly optimal. Include controlled perturbations that preserve labels while changing geometry, topology, correspondence quality, or representation cost.

A concrete generator, provisionally named `MixedStructuredSignal`, should use oriented complexes with approximately 24–96 vertices and balance the latent regimes. Shape, class, and nuisance distributions must be matched so that padding, size, or density does not trivially reveal the route:

- In the graph regime, targets depend on community or pairwise-diffusion signals.
- In the cell regime, paired examples share the same 1-skeleton but differ in filled 2-cells; targets depend on curl/harmonic energy or whether a cycle is filled.
- In the sheaf regime, examples share graph topology but differ in local frames, holonomy, or approximate-global-section consistency.

The router needs observable reliability or noise cues that are independent of the label; otherwise selection of a hidden regime can be statistically impossible. Out-of-distribution splits should vary complex size, cycle length, gauge rotations, and corruption level.

## Candidate real domains

- molecular graphs where rings and higher-order motifs matter;
- mesh or trajectory data with node, edge, and face quantities;
- heterogeneous or knowledge graphs with typed local constraints;
- flow datasets naturally represented as graph or cellular cochains.

The first real benchmark should be selected only after the synthetic experiment demonstrates identifiable specialization.

## Baselines

- fixed graph neural network;
- fixed simplicial/cell network;
- fixed neural sheaf model;
- differentiable lifting model;
- standard mixture of experts with tensor-compatible experts;
- fixed, random, and oracle routes;
- RTD-regularized autoencoder or representation model;
- task/reconstruction/cycle-only translator;
- matched-parameter and matched-compute ensembles.

## Ablation ladder

Evaluate cumulative additions:

\[
\mathcal L_{\mathrm{task}}
\rightarrow +\mathcal L_{\mathrm{recon}}
\rightarrow +\mathcal L_{\mathrm{chain}}
\rightarrow +\mathcal L_{\mathrm{cone}}
\rightarrow +\mathcal L_{\mathrm{cycle}}
\rightarrow +\mathcal L_{\mathrm{RTD}}
\rightarrow +\mathcal L_{\mathrm{cost}}.
\]

Also ablate:

- directional RTD versus half-sum RTD versus SRTD;
- evaluation-only versus training-time RTD;
- exact persistence versus spectral proxy;
- homological degree range;
- coefficient field;
- distance adapter and normalization;
- fixed versus resampled paired minibatches;
- correct versus corrupted sample correspondences;
- router temperature and load balancing;
- parameter-matched versus compute-matched comparisons.

## Metrics

- downstream accuracy, AUROC, regression error, or calibration as appropriate;
- measured latency, FLOPs, peak memory, and sparsity;
- reconstruction and round-trip error;
- chain-map residual;
- exact and approximate cone statistics;
- directional RTD, SRTD, and NTS;
- route entropy, expert utilization, and regime-route mutual information;
- regret relative to the per-sample oracle route;
- accuracy–compute Pareto area;
- performance under topology, geometry, noise, and correspondence perturbations.

Report multiple seeds, uncertainty intervals, hardware, batch/subsample sizes, numerical tolerances, and all normalization choices.

## Success criteria

The homological component is supported only if at least one preregistered result holds:

- cone/RTD defects predict downstream conversion damage after controlling for reconstruction error;
- cone regularization improves robustness, transfer, or task performance at matched compute;
- routing informed by structural defects outperforms task/cost-only routing;
- learned routes recover known synthetic regimes reproducibly.

## Falsification criteria

The central mechanism is not supported if ordinary task plus reconstruction/cycle losses match the full method, if the structural scores fail to predict degradation, or if routing collapses to one expert without a defensible compute/accuracy reason. Such results should be reported rather than reclassified as success.

A strong v0.1 go/no-go target is that the routed model approaches the oracle or dense-ensemble result while reducing measured compute, generalizes its routing out of distribution, and shows that cone defect adds predictive or intervention value beyond RTD and cycle error. Numerical thresholds should be preregistered after pilot measurements rather than chosen after observing final test results.
