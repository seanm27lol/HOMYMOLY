# Typed Representation Routing with Homological Structure: What Works, What Doesn't, and What We Measured

**Sean Mahdavian** — HOMYMOLY project
*2026-08-04 · Computational report; all experiments on a local NVIDIA GB10 (128 GB unified memory)*

---

## Abstract

We study whether a learning system can treat *representation type* — graph, cell complex, or cellular sheaf — as a routed computational choice, while conversions between representations remain mathematically inspectable. We build HOMYMOLY: three parameter-matched experts (graph message passing, a triangular-cell route, and a rank-2 connection-sheaf route with an exact cycle-holonomy pathway), graph-hub translators, an exact RTD/SRTD cone-barcode evaluation reference, and a small cost-aware router trained against a regime-conditional accuracy oracle.

On a preregistered anti-shortcut synthetic benchmark, learned routing **beats the best fixed expert by +0.108 accuracy (95% CI [+0.096, +0.119], n=5 seeds)** and the dense ensemble at ~3× less compute, with route–regime mutual information 0.113 and balanced utilization on every seed. This required fixing a cascade of measured failure modes — a sheaf expert architecturally blind to holonomy, a NaN in the translator phase, pooling dilution of pairwise statistics, regime-blind router features, a miscalibrated routing oracle, and router-phase learning-rate starvation — each diagnosed by targeted measurement rather than sweeping.

In contrast, the project's structural-loss mechanism is **measurably inert at these scales**: chain-consistency, reconstruction, and RTD-surrogate terms show no benefit in a four-variant ablation ladder, a corruption suite finds topological defects track conversion damage strongly (ρ 0.81–0.96) but never beyond plain reconstruction displacement (eight-pair seed campaign: mean Δ +0.096 ± 0.364), and on OGBG-MOLHIV the ring-lifted cell route trails the graph route (0.723 vs 0.771 test AUROC) — though a molecularly-informed redesign (strongest-bond readout + ring size) recovers +0.034 of the deficit with the lowest variance of any route (0.757 ± 0.002).

We report both outcomes with equal rigor: the routing result stands as a supported claim with confirmatory statistics; the structural-loss mechanism is recorded as a well-instrumented null. All code, data generators, run artifacts, and acceptance tests are in the repository; every number below traces to a recorded run.

---

## 1. Introduction

Most neural architectures commit to one representation family: tensors, graphs, simplicial or cell complexes, or cellular sheaves. Yet heterogeneous data plausibly favors different structural inductive biases per example: pairwise structure suits graphs, higher-order interactions suit complexes, and locally varying compatibility rules suit sheaves. The HOMYMOLY project asks whether representation type can become a *routed, cost-aware computational choice* — and whether conversions between representations can be kept mathematically inspectable by measuring their structural defects (chain-map residuals, mapping-cone homology, transport holonomy, and representation-topology divergence).

Two hypotheses were preregistered in the research specification:

- **Routing utility (H1/C3):** on mixed-regime data, a routed model matches or exceeds the best fixed expert at matched measured compute, with routes that correspond to latent structural regimes.
- **Structural regularization (C1/C2):** mapping-cone and RTD-style defects predict, or regularizing them improves, downstream conversion behavior beyond task and reconstruction losses.

The experimental program proceeds through falsification gates (mathematical foundation → fixed experts → translator/structural ablations → routing → molecular transfer), each with an explicit acceptance criterion and the instruction that a null result is recorded as evidence rather than relabeled.

## 2. Related work

Representation Topology Divergence (RTD, Barannikov et al., ICML 2022) compares paired point clouds across ambient spaces via cross-barcodes homotopy-equivalent to mapping cones; RTD-AE (Trofimov et al., 2023) differentiates it as an autoencoder objective; SRTD (TMLR 2026) gives a symmetric union/intersection construction. Topological lifting libraries (TopoBench; the ICML TDL Challenge 2024) and learned liftings (Differentiable Cell Complex Module, ICLR 2024; Differentiable Lifting, ICLR 2026) explore which higher-order domain to use but do not route per example. Neural Sheaf Diffusion (NeurIPS 2022) learns sheaf restriction maps; Copresheaf TNNs (NeurIPS 2025) generalize local-space models. Mixture-of-experts routing (Switch Transformers) selects computation but not among typed mathematical domains with homological diagnostics. Learning Functors (2020) enforces categorical composition laws via path-equivalence losses. To our knowledge no prior system jointly exposes several typed representations, learns typed translators, routes per example on task utility and measured compute, and evaluates conversions with exact cone/holonomy defects. Our RTD reproduction is independent of the public RTD-AE code (license-unverified) and validated against the published acceptance properties.

## 3. Method

### 3.1 Benchmark: a counterfactual anti-shortcut design

`ConfirmatoryStructuredSignal` generates groups of six samples sharing one canonical oriented complex (24–64 vertices, triangles as 2-cells), one per (regime, label) pair, with group-disjoint splits. Labels are relational by construction:

- **graph regime:** which signs meet across two unmarked vertex-disjoint anchor edges;
- **cell regime:** whether an energized probe face is active while the edge cochain is fixed;
- **sheaf regime:** cycle holonomy — a defect rotation composed onto one face's transport edge (verified: per-face holonomy defect exactly 0.0 vs 2.0).

Route reliability is observable but label-independent through *overlapping* amplitude intervals, so cheap scalars sit at chance for regime/label identification by design (measured ceiling ~0.55–0.58 for any label-independent regime classifier). A `gauge` stalk mode makes clean samples exact global sections (max per-edge residual 1e-7), giving consistency losses a zero noise floor where the default tier deliberately decouples fields from frames.

### 3.2 Architecture

Three experts share a `[B, 64]` embedding contract at ~0.85–0.96M parameters each:

- **GraphExpert:** edge-conditioned message passing plus a raw endpoint-pair pathway with masked-max readout (the label is a per-edge statistic; mean pooling dilutes it below noise).
- **CellExpert:** graph backbone plus active-face aggregation (oriented boundary sums, vertex means, face messages to nodes).
- **ConnectionSheafExpert:** transport-aware message layers plus an **exact face-holonomy pathway** — per-face transport products evaluated as complex unit products (valid because rank-2 connections are planar rotations, hence abelian), encoded and read out by masked mean *and* max (the defect is a single-face event).

Faces are stored as padded oriented boundary-edge lists with coefficients (migrating off triangle-only storage so molecular rings are never encoded as nonexistent triangles; `B1 @ B2 == 0` holds exactly). Graph-hub translators lift graph observations to cell/sheaf latents with reconstruction, consistency, and gate-supervision surrogates. The router is a small MLP over label-independent features (per-channel max-abs amplitudes carry the regime signal; measured one-way F-statistics 9–37 vs ~0 for the shipped mean/count diagnostics).

### 3.3 Routing supervision and training

Experts, then translators, then the router are trained in phases with atomic checkpoints and deterministic resume (config+code fingerprint guards; crash-resume verified bit-exact). The routing oracle is a **regime-conditional accuracy table** fitted on validation: earlier estimators (per-example confidence utility, temperature scaling, correctness-first) were measured and rejected — the confidence oracle is regime-conditionally miscalibrated, letting a confidently-correct non-native route beat an accurate-but-underconfident native route 62% of the time on its regime. Router phases use a per-phase learning-rate restart with a router-specific rate (1e-3): a single global cosine starves the router (~1e-6 by the router phases; offline replication: route accuracy 0.32 vs 0.54–0.59 at 1e-3–3e-4 restart).

### 3.4 Exact RTD/SRTD evaluation reference

`metrics/exact_rtd.py` computes the ordinary persistent homology of the filtered mapping cone over GF(2) in float64: directional R-Cross-Barcode semantics (source into union), the published half-sum, and the symmetric union/intersection construction. Acceptance tests cover identity zeros, isometry/rescaling invariance after normalization, permutation invariance, directional asymmetry with exact swap consistency, structured collapse, localized-difference detection, and the per-interval stability bound. The differentiable H0 surrogate used in training is provably *not* the cross-barcode (measured: directional ordering can disagree) and is reported only as a surrogate.

### 3.5 Corruption suite and Gate-5 protocol

Three graded per-sample corruption channels (transport rotations, edge-cochain noise, node-anchor noise; deterministic by seed/sample) create continuous conversion damage on the held-out split; damage is compared against reconstruction displacement and the exact SRTD between clean and corrupted expert embeddings. The molecular gate uses OGBG-MOLHIV with the official scaffold split and evaluator; rings enter as rdkit `AtomRings` boundary lists (verified against independent graph-cycle detection; ~0.2% rdkit-unparseable complexes excluded).

## 4. Results

### 4.1 Gate 2: routing works, confirmatory-statistics grade

Five fresh-seed full runs (stabilized configuration):

| seed | hard | best fixed | random | dense | oracle | route acc | MI | utilization |
|---|---|---|---|---|---|---|---|---|
| s1 | 0.767 | 0.667 | 0.679 | 0.766 | 0.999 | 0.536 | 0.091 | .41/.25/.34 |
| s2 | 0.789 | 0.669 | 0.678 | 0.749 | 0.998 | 0.583 | 0.136 | .40/.32/.28 |
| s3 | 0.766 | 0.667 | 0.697 | 0.748 | 0.999 | 0.534 | 0.093 | .38/.34/.27 |
| s4 | 0.782 | 0.678 | 0.659 | 0.736 | 0.976 | 0.576 | 0.126 | .36/.34/.30 |
| s5 | 0.781 | 0.667 | 0.680 | 0.764 | 0.999 | 0.564 | 0.118 | .28/.42/.29 |

**Margin over best fixed: +0.108, 95% CI [+0.096, +0.119] (t(4)).** All seeds beat best fixed and the dense ensemble (0.74 ± 0.01 at ~3× compute); expected route cost 1.31–1.32 vs 3.9 for always evaluating all experts. Experts specialize: graph 0.997, cell ~0.73–0.93, sheaf 1.000 on their regimes. Route accuracy (0.559 mean) sits at the benchmark's designed identifiability ceiling; the residual gap to oracle (~0.999) is the intentional reliability overlap, not router error.

**The debugging arc is part of the result.** The first pilot seed failed the utility criterion with a collapsed router; each failure was traced by measurement: (i) the sheaf label is holonomy, invisible to per-edge residuals (residual ranges 1.83–2.51 for both labels) — fixed by the holonomy pathway; (ii) a NaN in the translator phase was `sqrt` at zero residual with an incorrect first fix (post-clamp still chains 0·inf) — fixed inside the sqrt with a regression test; (iii) the graph label is an endpoint product diluted by mean pooling — fixed by the endpoint-pair pathway; (iv) router features were regime-blind means (F ≈ 0) — fixed by max-abs amplitude cues; (v) the confidence oracle was regime-conditionally miscalibrated — fixed by the conditional-accuracy table; (vi) the router was LR-starved and, in one of five draws, stalled at uniform output — fixed by the per-phase restart with a router rate; (vii) translators could not see their tasks at all (gate collapse; structural impossibility without `face_active` as input).

### 4.2 Gate 3: the structural-loss mechanism is inert at these scales

- **Ablation ladder** (task-only → +reconstruction → +chain → full; expert phases bit-identical): every metric flat within ±0.01 — task accuracy, routing, corruption robustness. Task supervision alone matches the full objective.
- **Corruption suite:** exact SRTD between clean and corrupted embeddings tracks damage strongly (ρ 0.92/0.82/0.81 for sheaf/cell/graph channels) — but partial correlation controlling for reconstruction displacement is ≈ 0 across an eight-pair seed campaign (mean Δ between chain-constrained and unconstrained runs: **+0.096 ± 0.364**, 5/8 pairs positive). The consistency term reliably controls its target (0.18 vs 1.4–2.0 drifting) without changing any downstream metric.
- Mechanistic note: on the default tier the cochain surrogate has an irreducible noise floor (fields and frames are independent draws); the gauge tier removes that objection and the null persists. The current benchmark offers structural terms no purchase; data where exactness damage *is* the bottleneck is the identified redesign direction.

### 4.3 Gate 5: molecular transfer (OGBG-MOLHIV, official split/evaluator)

| route | params | test AUROC (3 seeds) | mean ± std | valid mean |
|---|---|---|---|---|
| graph | 917,954 | 0.777 / 0.780 / 0.755 | 0.771 ± 0.014 | 0.794 |
| cell v1 | 955,842 | 0.713 / 0.742 / 0.713 | 0.723 ± 0.017 | 0.782 |
| cell molecular v2 | 972,354 | 0.756 / 0.756 / 0.759 | **0.757 ± 0.002** | 0.771 |
| cell molecular v3 (bond-type histograms) | 972,994 | 0.733 / 0.752 / 0.702 | 0.729 ± 0.025 | 0.761 |

The vanilla ring-lift loses by 0.048 — a generalization gap (valid within 0.012), not optimization. The molecularly-informed redesign (per-face strongest-bond max + ring size) recovers +0.034 with the tightest variance of any route, but still trails the graph route by 0.014. Bond-type ring histograms are a *negative ablation*: bond distributions are scaffold-specific, injecting exactly the variance the scaffold split shifts. Recorded scaffold property: the official test split is ~100% ring-bearing, so ring-free transfer is not evaluable on it.

## 5. Discussion

**What the positive result means.** The routed system demonstrates the benchmark's intended behavior: three typed experts specialize without regime leakage into inputs, and a cheap router learns regime-conditional routing from label-independent amplitude cues, beating every fixed alternative at a third of the dense ensemble's compute. The confirmatory statistics are tight because the failure modes were eliminated by measurement rather than averaged over.

**What the null means.** The structural-loss mechanism — the project's most distinctive bet — shows no task value in any tested configuration, across independent and gauge data designs, correlation and intervention questions, and 16+ recorded runs. We read this as a benchmark-relevant finding, not merely an absence: the machinery (exact cone barcodes, holonomy, consistency surrogates) is correct and measurably responsive; what is missing is a task family where exactness damage is the binding constraint. Candidate revision: labels that are homology-determined with continuous corruption of chain data, so cone/holonomy defects are the natural sufficient statistic rather than a proxy for displacement.

**Limitations.** Synthetic results are one benchmark with one generator family; the routing margin, while tight, is modest (0.108 absolute); the molecular deficit may be architecture- or scale-specific (single ~1M-param recipe); ring-free transfer is unevaluable on the official MOLHIV test split; GPU scatter-add is not bit-deterministic (CPU/exact paths are FP64-exact); the exact RTD module is bounded to 64-point subsamples per comparison.

## 6. Reproducibility

Everything is in the repository: generators and anti-shortcut audits (`data/`, `tests/`), the phased trainer (`training/engine.py`), exact RTD (`metrics/exact_rtd.py`), corruption suite (`data/corruptions.py`, `scripts/eval_corruption.py`), molecular builder (`data/molecular.py`, `scripts/train_molhiv.py`), all run artifacts under `artifacts/`, and the run-by-run record in `docs/13`–`docs/17` with the claims ledger in `docs/08`. Test suite: 123 passed at tip; BF16/FP32 discipline per the experimental plan.

## References

1. Barannikov, Trofimov, Balabin, Burnaev. *Representation Topology Divergence: A Method for Comparing Neural Network Representations.* ICML 2022.
2. Trofimov, Cherniavskii, Tulchinskii, Balabin, Burnaev, Barannikov. *Learning Topology-Preserving Data Representations.* arXiv:2302.00136, 2023.
3. *Symmetric Divergence and Normalized Similarity* (SRTD/SRTD-lite/NTS). TMLR 2026.
4. *A Quotient Homology Theory of Representation in Neural Networks.* TMLR 2026. arXiv:2502.01360.
5. Bodnar et al. *Neural Sheaf Diffusion.* NeurIPS 2022.
6. Gebhart et al. *Knowledge Sheaves.* ICML 2023.
7. *Copresheaf Topological Neural Networks.* NeurIPS 2025.
8. *ICML Topological Deep Learning Challenge 2024.* arXiv:2409.05211.
9. *TopoBench.* arXiv:2406.06642.
10. *From Latent Graph to Latent Topology Inference* (Differentiable Cell Complex Module). ICLR 2024.
11. *Differentiable Lifting for Topological Neural Networks.* ICLR 2026.
12. *Learning Functors Using Gradient Descent.* arXiv:2009.06837.
13. Gavranović et al. *Categorical Deep Learning.* ICML 2024.
14. Zhang et al. *Cycle-Consistent Probability Divergences Across Different Spaces.* AISTATS 2022.
15. Fedus, Zoph, Shazeer. *Switch Transformers.* JMLR 2022.
16. Moor et al. *Topological Autoencoders.* ICML 2020.

---

*Claim boundary (per the project's own discipline): the routing result is supported on one synthetic benchmark family; the structural-loss nulls are measured at ~1M-parameter scale; nothing here is claimed for molecular data beyond the recorded MOLHIV numbers, and no Langlands- or Fourier–Mukai-level construction is implemented or claimed.*
