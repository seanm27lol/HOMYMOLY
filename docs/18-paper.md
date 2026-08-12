# Representation Routing and Homological Losses: What Works, What Doesn't, and What We Measured

**Sean Mahdavian** — HOMYMOLY project
*2026-08-04 · Computational report; every experiment ran on a local NVIDIA GB10 (128 GB unified memory)*

---

## Abstract

Here's the question: can a model treat *representation type* — graph, cell complex, cellular sheaf — as a per-example routing decision, and can we keep the conversions between those representations mathematically honest while it does it? To find out, we built HOMYMOLY: three parameter-matched experts (graph message passing, a triangular-cell route, and a rank-2 connection-sheaf route with an exact cycle-holonomy pathway), graph-hub translators, an exact RTD/SRTD cone-barcode evaluation reference, and a small cost-aware router.

The short answer is yes for routing, no for structural losses. On a preregistered anti-shortcut synthetic benchmark, learned routing **beats the best fixed expert by +0.108 accuracy (95% CI [+0.096, +0.119], n=5 seeds)** — every seed, every time, at roughly a third of the dense ensemble's compute. Getting there meant finding and fixing seven distinct failure modes, each one measured before it was fixed: a sheaf expert that literally could not see holonomy, a NaN hiding in a square root, pooling that diluted the graph signal to nothing, router features with zero regime information (F ≈ 0), an oracle that systematically preferred the wrong expert, a router starved of learning rate, and translators that couldn't see their own tasks.

The structural-loss side of the project — chain consistency, cone statistics, RTD surrogates — is a different story: a careful, instrumented **null**. A four-variant ablation ladder shows task-only training matching the full objective on every axis. A corruption suite shows topological defects tracking conversion damage strongly (ρ 0.81–0.96) but never beyond plain reconstruction displacement, across an eight-pair seed campaign. On OGBG-MOLHIV, the ring-lifted cell route trails the plain graph route (0.723 vs 0.771 test AUROC); a molecularly-informed redesign recovers +0.034 of that gap with the lowest variance of any route (0.757 ± 0.002).

Both outcomes get the same rigor. Every number traces to a recorded run; every claim cites its artifacts. All of it is in the repo.

---

## 1. Introduction

Neural architectures are monogamous: they commit to one representation family and stay there. Tensors, graphs, simplicial complexes, cellular sheaves — pick one. But real data is heterogenous, and the right inductive bias plausibly varies per example: pairwise structure wants a graph, higher-order interactions want a complex, locally-varying compatibility rules want a sheaf.

HOMYMOLY asks two questions at once. First, can representation type be a *routed* choice — a per-example, cost-aware decision? Second, can the conversions between types be kept inspectable: can we measure, homologically, what a conversion destroys?

The preregistered hypotheses:

- **Routing utility (C3):** on mixed-regime data, a routed model beats the best fixed expert at matched measured compute, with routes that track latent structural regimes.
- **Structural regularization (C1/C2):** mapping-cone and RTD-style defects predict — or regularizing them improves — conversion behavior, beyond task and reconstruction losses.

The program runs through falsification gates: mathematical foundation, fixed experts, structural ablations, routing, molecular transfer. Each gate has an acceptance criterion, and the house rule is that a null result is recorded as evidence, not relabeled as success. This paper is what came out the other end.

## 2. Related work

The closest literal predecessor is Representation Topology Divergence (RTD, Barannikov et al., ICML 2022), which compares paired point clouds across ambient spaces via cross-barcodes that are homotopy-equivalent to mapping cones. RTD-AE (Trofimov et al., 2023) differentiates it into an autoencoder objective; SRTD (TMLR 2026) gives a symmetric union/intersection construction. On the lifting side, TopoBench and the ICML TDL Challenge 2024 map out which higher-order domain to use; the Differentiable Cell Complex Module (ICLR 2024) and Differentiable Lifting (ICLR 2026) learn liftings end-to-end. Neural Sheaf Diffusion (NeurIPS 2022) learns sheaf restriction maps; Copresheaf TNNs (NeurIPS 2025) push local-space models further. Switch Transformers and the MoE lineage route computation, but not among typed mathematical domains with homological diagnostics. Learning Functors (2020) is the closest categorical ancestor to our path-coherence thinking.

Nobody we found does all of it at once: several typed representations, learned typed translators, per-example routing on task utility and measured compute, and exact homological defect evaluation of the conversions. That intersection is HOMYMOLY's candidate contribution. One process note: our RTD reproduction is deliberately independent of the public RTD-AE code (its license was never verified), validated instead against the published acceptance properties.

## 3. Method

### 3.1 The benchmark: counterfactuals with no shortcuts

`ConfirmatoryStructuredSignal` generates groups of six samples on one shared oriented complex (24–64 vertices, triangles as 2-cells) — one sample per (regime, label) pair, so every group is a tiny controlled experiment. Splits are group-disjoint. The labels are relational on purpose:

- **graph regime:** which signs meet across two unmarked, vertex-disjoint anchor edges;
- **cell regime:** whether an energized probe face is active, edge cochain held fixed;
- **sheaf regime:** cycle holonomy — a defect rotation composed onto one face edge. Verified exactly: per-face holonomy defect is 0.0 for label 0, 2.0 for label 1.

Route reliability is observable but label-independent, through deliberately *overlapping* amplitude intervals. Cheap scalars therefore sit at chance for regime identification by design — we measured the ceiling at ~0.55–0.58 for any label-independent classifier. A second `gauge` tier makes clean samples exact global sections (max per-edge residual 1e-7), giving consistency losses a zero noise floor where the default tier deliberately decouples fields from frames. Both exist because the difference between them turned out to matter (§4.2).

### 3.2 Architecture

Three experts, one shared `[B, 64]` embedding contract, ~0.85–0.96M parameters each:

- **GraphExpert** — edge-conditioned message passing plus a raw endpoint-pair pathway with masked-max readout. The graph label is a per-edge statistic; mean pooling dilutes it below the noise floor, so one informative edge needs to survive aggregation.
- **CellExpert** — graph backbone plus active-face aggregation: oriented boundary sums, vertex means, face messages back to nodes.
- **ConnectionSheafExpert** — transport-aware message layers plus an **exact face-holonomy pathway**: per-face transport products computed as complex unit products (valid because rank-2 connections are planar rotations, hence abelian), read out by masked mean *and* max, because the defect is a single-face event.

Faces are stored as padded oriented boundary-edge lists with coefficients — the migration off triangle-only storage that lets molecular rings exist without pretending to be triangles. `B1 @ B2 == 0` holds exactly. Graph-hub translators lift graph observations into cell/sheaf latents with reconstruction, consistency, and gate-supervision surrogates. The router is a small MLP over label-independent features; the per-channel max-abs amplitudes are what carry the regime signal (measured one-way F-statistics 9–37, vs ~0 for the mean/count diagnostics we shipped first).

### 3.3 Routing supervision and training

Training is phased — experts, translators, router — with atomic checkpoints and deterministic resume guarded by config+code fingerprints (crash-resume verified bit-exact). Two supervision decisions turned out to matter more than the architecture:

The routing oracle is a **regime-conditional accuracy table** fitted on validation. We measured and rejected three alternatives first: per-example confidence utility, temperature scaling, and correctness-first ordering. The confidence oracle is regime-conditionally miscalibrated — on graph-regime examples it let a confidently-correct cell expert outrank the accurate-but-underconfident graph expert 62% of the time, and the graph route starved.

Router phases get a **per-phase learning-rate restart with a router-specific rate** (1e-3). A single global cosine leaves the router at ~1e-6 by its phases; offline replication put route accuracy at 0.32 under the old schedule vs 0.54–0.59 with the restart. One in five pilot seeds stalled at uniform output before this fix; none after.

### 3.4 Exact RTD/SRTD as the evaluation reference

`metrics/exact_rtd.py` computes ordinary persistent homology of the filtered mapping cone over GF(2) in float64 — directional R-Cross-Barcode semantics in both directions, the published half-sum, and the symmetric union/intersection construction. Acceptance tests: identity zeros, isometry/rescaling invariance after normalization, permutation invariance, directional asymmetry with exact swap consistency, structured collapse, localized-difference detection, per-interval stability bound. The differentiable H0 surrogate used during training is provably not the cross-barcode (directional ordering can disagree, measured), so it is only ever reported as a surrogate.

### 3.5 Corruption suite and the molecular protocol

Three graded per-sample corruption channels (transport rotations, edge-cochain noise, node-anchor noise — deterministic by seed and sample) create continuous conversion damage on the held-out split; damage is compared against reconstruction displacement and exact SRTD between clean and corrupted embeddings. The molecular gate is OGBG-MOLHIV with the official scaffold split and official evaluator; rings enter as rdkit `AtomRings` boundary lists, cross-checked against independent graph-cycle detection (~0.2% rdkit-unparseable metal complexes excluded).

## 4. Results

### 4.1 Routing works — confirmatory-statistics grade

Five fresh-seed full runs, stabilized configuration:

| seed | hard | best fixed | random | dense | oracle | route acc | MI | utilization |
|---|---|---|---|---|---|---|---|---|
| s1 | 0.767 | 0.667 | 0.679 | 0.766 | 0.999 | 0.536 | 0.091 | .41/.25/.34 |
| s2 | 0.789 | 0.669 | 0.678 | 0.749 | 0.998 | 0.583 | 0.136 | .40/.32/.28 |
| s3 | 0.766 | 0.667 | 0.697 | 0.748 | 0.999 | 0.534 | 0.093 | .38/.34/.27 |
| s4 | 0.782 | 0.678 | 0.659 | 0.736 | 0.976 | 0.576 | 0.126 | .36/.34/.30 |
| s5 | 0.781 | 0.667 | 0.680 | 0.764 | 0.999 | 0.564 | 0.118 | .28/.42/.29 |

**Margin over best fixed: +0.108, 95% CI [+0.096, +0.119] (t(4)).** Every seed beats the best fixed route and the dense ensemble (0.74 ± 0.01 at ~3× the compute). Expected route cost 1.31–1.32 vs 3.9 for running everything. Experts specialize cleanly: graph 0.997, cell ~0.73–0.93, sheaf 1.000 on their own regimes. Route accuracy (0.559 mean) sits exactly at the benchmark's designed identifiability ceiling — the remaining gap to oracle is the intentional reliability overlap, not router error.

**The debugging arc is part of the result.** The first pilot seed failed the utility criterion with a collapsed router. Each failure got a measurement before it got a fix: (i) the sheaf label is holonomy, and per-edge residuals are provably label-independent (1.83–2.51 for both labels) — fixed by the holonomy pathway; (ii) a NaN in the translator phase was `sqrt` at zero residual, and the first fix (post-clamp) still chained 0·inf in backward — fixed by eps *inside* the sqrt, with a regression test; (iii) the graph label is an endpoint product diluted by pooling — fixed by the endpoint-pair pathway; (iv) the shipped router features were regime-blind means (F ≈ 0) — fixed by amplitude cues; (v) the confidence oracle was regime-conditionally miscalibrated — fixed by the conditional-accuracy table; (vi) the router was LR-starved, and one in five draws stalled at uniform — fixed by the per-phase restart with a router rate; (vii) the translators couldn't see their tasks at all — fixed by the holonomy pathway (sheaf) and `face_active` as an input (cell; without it the cell translation task is structurally impossible, gate collapse at precision/recall 0.0).

### 4.2 The structural-loss mechanism is inert at these scales

This is the part we wanted to be wrong about.

- **Ablation ladder** (task-only → +reconstruction → +chain → full, expert phases bit-identical): everything flat within ±0.01 — task accuracy, routing, corruption robustness. Task supervision alone matches the full objective.
- **Corruption suite:** exact SRTD between clean and corrupted embeddings tracks damage strongly (ρ 0.92/0.82/0.81 for sheaf/cell/graph channels) — and adds nothing beyond plain embedding displacement. Across an eight-pair seed campaign, the mean partial-correlation delta between chain-constrained and unconstrained runs is **+0.096 ± 0.364** (5/8 pairs positive). The consistency term reliably controls its target (0.18 held vs 1.4–2.0 drifting) while changing nothing downstream.
- **Mechanistic note:** the default tier's cochain surrogate has an irreducible noise floor (fields and frames are independent draws). The gauge tier removes that objection — clean samples are exact sections — and the null persists anyway. This benchmark simply gives structural terms nothing to grab.

### 4.3 Molecular transfer (OGBG-MOLHIV, official split/evaluator)

| route | params | test AUROC (3 seeds) | mean ± std | valid mean |
|---|---|---|---|---|
| graph | 917,954 | 0.777 / 0.780 / 0.755 | 0.771 ± 0.014 | 0.794 |
| cell v1 | 955,842 | 0.713 / 0.742 / 0.713 | 0.723 ± 0.017 | 0.782 |
| cell molecular v2 | 972,354 | 0.756 / 0.756 / 0.759 | **0.757 ± 0.002** | 0.771 |
| cell molecular v3 (bond-type histograms) | 972,994 | 0.733 / 0.752 / 0.702 | 0.729 ± 0.025 | 0.761 |

The vanilla ring-lift loses by 0.048 — and it's a generalization gap, not optimization (valid AUROC within 0.012). The molecularly-informed redesign (per-face strongest-bond max + ring size) recovers +0.034 with the tightest variance of any route, but still trails the graph route by 0.014. Bond-type ring histograms are a clean *negative ablation*: bond-type distributions are scaffold-specific, so they inject exactly the variance the scaffold split shifts. One recorded property of the official split: test is ~100% ring-bearing, so ring-free transfer is unevaluable on it.

## 5. Discussion

**What the positive result means.** The system does the thing it was built to do: three typed experts specialize with no regime leakage into inputs, and a cheap router learns regime-conditional routing from label-independent amplitude cues — beating every fixed alternative at a third of the dense ensemble's compute. The confidence interval is tight because the failure modes were eliminated by measurement, not averaged over.

**What the null means.** The structural-loss mechanism — the project's most distinctive bet — shows no task value in any configuration we tested, across two data designs, correlation and intervention questions, and sixteen-plus recorded runs. We read this as a finding about the benchmark, not just an absence: the machinery is correct and measurably responsive, but nothing in the current task family makes exactness damage the binding constraint. The redesign we can now articulate: labels that are homology-determined, with continuous corruption of the chain data, so cone/holonomy defects are the natural sufficient statistic instead of a proxy for displacement.

**Limitations.** One benchmark family; the routing margin is tight but modest in absolute terms (0.108); the molecular gap may be recipe- or scale-specific (a single ~1M-parameter configuration); ring-free transfer can't be evaluated on the official MOLHIV test split; GPU scatter-add is not bit-deterministic (CPU and exact-oracle paths are FP64-exact); the exact RTD module is bounded to 64-point subsamples per comparison.

## 6. Reproducibility

All of it: generators and anti-shortcut audits (`data/`, `tests/`), the phased trainer (`training/engine.py`), exact RTD (`metrics/exact_rtd.py`), the corruption suite (`data/corruptions.py`, `scripts/eval_corruption.py`), the molecular builder (`data/molecular.py`, `scripts/train_molhiv.py`), every run artifact under `artifacts/`, and the run-by-run record in `docs/13`–`docs/17` with the claims ledger in `docs/08`. Suite at tip: 123 passed, ruff clean.

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

*Claim boundary: the routing result is supported on one synthetic benchmark family; the structural-loss nulls are measured at ~1M-parameter scale; nothing here is claimed for molecular data beyond the recorded MOLHIV numbers; and no Langlands- or Fourier–Mukai-level construction is implemented or claimed.*
