# GB10 experimental plan

This plan turns the HOMYMOLY research specification into a sequence of falsifiable experiments on the local NVIDIA GB10 system. It deliberately separates mathematical correctness, fixed-route learning, and dynamic routing so a successful router cannot hide an invalid translator.

## Execution principles

- Use FP64 for incidence validation, numerical rank, Betti numbers, chain-map residuals, and exact mapping-cone oracles.
- Use BF16 autocast for neural experts and translators after a short FP32 smoke run. Keep reductions used by structural losses in FP32 or FP64.
- Treat the 128 GB unified memory as a shared CPU/GPU resource. Choose batch size from measured peak memory and throughput rather than allocating against a nominal GPU-memory value.
- Record the git commit, configuration, seed, package/container versions, device profile, wall time, peak memory, and metrics for every run.
- Keep artifacts local. TensorBoard may be exposed only to the private tailnet with Tailscale Serve; do not enable Funnel.

## Stage gates

### Gate 1: mathematical and data foundation

Deliver deterministic incidence matrices, valid chain complexes, chain maps, mapping cones, Hodge projections, and the `MixedStructuredSignal` generator.

Acceptance checks:

- every generated face is an oriented cycle and satisfies `B1 @ B2 == 0` within the declared FP64 tolerance;
- graph-to-cell inclusion satisfies the chain-map equation;
- the cone of an identity map is acyclic;
- triangle and filled-triangle Betti numbers match hand-computed values;
- dataset generation is deterministic by seed and index;
- graph, cell, and sheaf routes receive the same raw observations;
- labels and latent regimes are jointly balanced and are absent from model inputs;
- all tests pass on CPU and the device profiler records the GB10 environment.

No learning claim is permitted at this gate.

### Gate 2: RTD reproduction, fixed experts, and observable specialization

First reproduce RTD/SRTD behavior on published or independently hand-checkable paired-metric examples. Freeze this implementation and its distance-normalization conventions before it is used on learned representations.

Implement three parameter-matched experts that emit a common graph-level embedding:

1. graph message passing;
2. graph-to-cell lift, cell processing, and projection;
3. graph-to-rank-2 cellular-sheaf lift, sheaf processing, and projection.

Train each route independently on identical splits. Include a shared MLP baseline and a dense three-expert ensemble. Verify that each specialized route has an advantage in its intended synthetic regime without using the hidden regime identifier as an input.

Gate criterion: across preregistered seeds, at least two specialized routes must improve on the graph route in their intended regimes, or the benchmark/model design is revised before routing work begins.

The existing `MixedStructuredSignal` mode is only the easy bring-up tier because its reliability amplitudes intentionally expose the active regime. Confirmatory evidence requires overlapping reliability distributions or structurally defined corruptions, together with amplitude-only, graph-only, local-residual, edge-pooling, and structure-shuffle baselines.

### Gate 3: translator and structural-loss ablations

Start with the graph-to-cell inclusion because it has an explicit chain-map contract. Compare:

- task only;
- task plus reconstruction/cycle consistency;
- plus chain residual;
- plus mapping-cone statistic;
- RTD/SRTD evaluation only;
- RTD regularization where paired metric representations make it valid;
- combined structural terms.

For the sheaf route, report the cochain-map residual. Do not call its diagnostic a direct mapping-cone loss until the source and target are represented in a single declared chain-complex convention with a valid map.

Do not minimize total cone homology indiscriminately. The graph-to-cell route is allowed—and sometimes intended—to create or fill classes. Use a route-conditioned expected cone signature, or an acyclicity objective only where the declared conversion should be a quasi-isomorphism. Exact cone homology remains an evaluation oracle until a learned translator satisfies the exact chain-map contract.

Gate criterion: on held-out corruptions, a structural diagnostic must add predictive value for conversion damage after controlling for reconstruction error, or a structural regularizer must improve a matched-compute downstream result. Otherwise the central mechanism is not yet supported.

### Gate 4: routing

Pretrain the fixed experts and translators, freeze them, and produce per-example route utilities. Train a small router from cheap label-independent features against the oracle route, then jointly fine-tune with straight-through top-1 routing.

Compare fixed, random, learned, oracle, and dense-ensemble routing. Report task metrics, measured latency, route utilization, route/regime mutual information, oracle regret, calibration, and accuracy-compute Pareto area.

Gate criterion: learned routing must beat the best fixed route at matched measured compute and remain non-collapsed for a defensible reason. Approaching a dense ensemble without compute savings is not sufficient.

### Gate 5: molecular transfer

Move to a molecular graph benchmark only after the synthetic gates pass. Begin with OGBG-MOLHIV and preserve official splits and evaluator behavior. Compare the graph route with chemically valid ring/2-cell lifts; introduce the sheaf route only with an explicit molecular interpretation for its local frames and restrictions.

The current batch format stores triangular faces only. Before molecular rings are represented as cells, migrate to sparse boundary matrices or padded oriented boundary-edge lists with coefficients; do not encode a long ring as a nonexistent triangle.

Use scaffold-aware reporting, matched parameter counts, measured compute, and multiple seeds. Treat Peptides-func as a secondary stress test only if the primary real-data result is inconclusive.

## GB10 run schedule

### Bring-up runs

- Run all unit tests on CPU in FP64.
- Run one tiny forward/backward batch per expert in FP32 with deterministic algorithms enabled.
- Run the same batch in BF16 and compare loss, gradients, and structural diagnostics against the FP32 reference.
- Profile representative small, medium, and large synthetic examples before selecting batch and worker counts.

### Pilot campaign

Use three development seeds and shortened schedules. Sweep only decisions that can change the conclusion: structural-loss weights, route temperature, projection choice, and a small set of capacity-matched widths. Select settings on validation metrics and freeze the analysis before full runs.

### Confirmatory campaign

Use at least five fresh seeds for the primary synthetic comparison. Save per-example predictions, routes, defects, measured costs, and corruption levels so the predictive-value and oracle-regret analyses can be reproduced. Report confidence intervals and all failed or collapsed runs.

### Long-run discipline

- Give each run an immutable configuration and output directory.
- Checkpoint model, optimizer, scheduler, scaler, RNG, and sampler state.
- Resume only from checkpoints with matching commit and config hashes.
- Keep a small numerical sentinel batch whose structural invariants are checked during training.
- Abort on non-finite losses, invalid chain laws, or leakage of regime/label metadata into model inputs.

## Initial resource policy

The profiler, rather than a hard-coded GB10 batch size, determines the launch configuration. For each candidate batch, record examples/second, step-time percentiles, unified-memory pressure, and host responsiveness. Choose the largest batch before throughput plateaus while retaining headroom for validation and checkpointing. Gradient accumulation is preferred to operating near the unified-memory limit.

Use a single process for the initial campaign. Add data-worker concurrency only after confirming that generation or collation is a bottleneck, and add compilation only after eager-mode correctness and stable shapes are established.

## Required result tables

1. Mathematical invariant and numerical-tolerance table.
2. Dataset balance, leakage, and shortcut-audit table.
3. Fixed-expert performance by latent regime and corruption.
4. Translator ablation with reconstruction, chain, cone, RTD/SRTD, and task metrics.
5. Router performance versus fixed, random, oracle, and dense routes at matched compute.
6. GB10 throughput, latency, peak unified-memory pressure, and energy when measurable.
7. Molecular benchmark results with official metrics and split protocol.

The project advances only when the relevant gate is passed. A null result is recorded as evidence against the current mechanism rather than relabeled as architectural success.
