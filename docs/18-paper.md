# Exact Recovery of a Finite Typed Map Family on a Synthetic Cellular Annulus, with Representation Routing and Homological Diagnostics

**Sean Mahdavian**

**Revision: 2026-08-23**

Affiliation, ORCID, funding, competing interests, and license are author-supplied
submission fields and are deliberately left unset in this draft. See
[Declarations](#12-declarations).

## Abstract

This report asks a narrow, answerable question: can a neural implementation
recover an exact, typed, degree-wise map between two structured representations
when the correct map is guaranteed to lie in a small known family, and does
adding homological structure to the training objective help it do so?

We build a synthetic six-sector cellular annulus — 12 vertices, 18 edges, 6
faces, Betti numbers (1, 1, 0) — and a finite dihedral family of twelve exact
three-term maps acting on it. A model observes explicit identifying markers and
uses a flattened multilayer perceptron to select one member of a hard-coded
group-action basis. Over a frozen 40-run campaign (8 objectives × 5 seeds,
4,800/1,200/1,200 examples per run) the implementation recovers the planted map
exactly: transformation accuracy 1.000 and cell-face accuracy 1.000 in every run
whose objective contains either task or reconstruction supervision, with map
mean-squared error at the 1e-16 level and chain-map residuals inside a fixed 1e-5
tolerance. A prespecified engineering recovery gate passed in 10 of 10 applicable
runs.

The structural results are negative, and we report them as the main scientific
content rather than as a caveat. Adding a mapping-cone term, a
representation-topology-divergence term, or both to a working objective changed
nothing measurable: all 21 declared continuous contrast intervals contain zero,
against a saturated accuracy ceiling. Trained on structural losses *alone*, the
model sits at chance — transformation accuracy 0.0815 (cone-only) and 0.0833
(RTD-only) against a 0.0833 chance baseline — even though the decoded cones are
acyclic in every evaluated example. This is not an optimization failure: every
candidate map is a signed permutation, hence an invertible isometry, so cone
acyclicity and RTD are both *constant* on the hypothesis space and carry exactly
zero information about which element was planted. Acyclicity certifies that the
selected map is invertible; it does not certify that it is the correct map, and
the same degeneracy afflicts any hypothesis class of invertible maps — which is
the setting where a cone objective looks most attractive.

Two further audited results are reported with their own boundaries. A frozen
five-seed routing campaign gives a hard-minus-best-fixed-expert margin of +0.1098
(95% Student-t interval [+0.0953, +0.1243]), under privileged latent-regime
distillation and with target structured views available at inference. Corrected
fixed-expert corruption diagnostics — nine paired contrasts in the Gate-3 base
family and three across eight seed-matched gauge pairs — produce no interval
excluding zero.

We do not claim a general graph neural network, a universal representation
translator, a learned quasi-isomorphism, or any categorical, Langlands,
eigensheaf, or Fourier–Mukai result. The verified identity is the chain-map law
up to a fixed numerical tolerance on one synthetic template family.

## 1. Research question and novelty statement

### 1.1 The question

Given two structured representations of the same underlying object, a *typed map*
is a family of linear maps — one per degree — that is required to commute with
the two boundary operators. Such a map is the discrete analogue of a chain map.
The practical question is whether a learned system can produce one that is
*exact* rather than approximate, and whether homological objectives help.

That question is only answerable if the correct answer is known. We therefore do
not attempt open-ended conversion. We plant a specific map drawn from a finite
group action, give the model the information needed in principle to identify
which member was planted, and measure whether it recovers that member exactly.

This is an implementation and exactness study. It establishes that the machinery
works on a case where ground truth exists, and it measures whether homological
loss terms contribute. It does not establish that the machinery generalizes.

### 1.2 What is and is not new

Learning higher-order structure from graphs is an established direction:
Differentiable Cell Complex Module [4] learns cell probabilities jointly with a
task, Differentiable Lifting [5] learns liftings into cellular, simplicial, and
combinatorial complexes, and Neural Sheaf Diffusion [6] learns sheaf restriction
maps. Comparing paired representations through cross-barcodes is likewise
established: Representation Topology Divergence [1], its differentiable
autoencoder form [2], and the symmetric variant SRTD [3] are the direct basis for
our divergence reference. Per-example dynamic routing among experts has an
established literature as well [7–10].

Against that background our contribution is deliberately small and is stated as a
methodological one: a controlled benchmark in which an exact typed map is planted
and recoverable, a nullspace parameterization under which every parameter value
satisfies the chain-map equation by construction, and an audited factorial
measurement showing that mapping-cone and RTD objectives add nothing on this
benchmark and cannot identify the map on their own — together with the
structural reason why (§6.3), which applies to any hypothesis class of
invertible maps and not only to ours. We make no claim of priority or of
superiority over any cited system, and the comparison in §3 is positioning, not
systematic review.

## 2. Background and definitions

Each concept is stated in plain language first, then in notation.

### 2.1 Graphs, cell complexes, and cellular sheaves

A **graph** records which pairs of objects are related. A **cell complex** adds
higher-dimensional pieces: on top of vertices (degree 0) and edges (degree 1) it
attaches faces (degree 2) glued along closed edge loops. A **cellular sheaf**
attaches a small vector space to each cell and a linear *transport* map to each
incidence, so that data can be moved along the complex; the sheaf records not
just what is connected but how quantities rotate as they move.

Formally, a two-dimensional cell complex has boundary operators `d1` (edges to
vertices) and `d2` (faces to edges) satisfying `d1 d2 = 0`. A cellular sheaf
assigns a stalk to each cell and restriction maps to each incident pair; our
sheaf expert uses rank-2 planar rotations as edge transports.

### 2.2 The cellular annulus

An **annulus** is a ring — a disc with a hole. Our synthetic template is a ring
divided into six sectors. The hole is what makes it interesting: it gives the
complex a one-dimensional cycle that no amount of local information can remove,
so a map that preserves structure must preserve that cycle.

Concretely the template has 12 vertices, 18 edges, and 6 faces, with Betti
numbers `(b0, b1, b2) = (1, 1, 0)`: one connected component, one independent
cycle, no enclosed volume. All 40 campaign runs use this single template.

### 2.3 Typed maps, chain maps, and exactness defects

A **typed map** is a collection of linear maps, one for each degree, that carries
degree-0 data to degree-0 data, degree-1 to degree-1, and so on. It is a **chain
map** when it commutes with the boundary operators — that is, when taking the
boundary and then applying the map gives the same answer as applying the map and
then taking the boundary. Informally: the map does not tear the complex.

For a source complex with boundary `dC` and a target with boundary `dD`, the
degree maps `F0, F1` form a chain map when

    dD F1 - F0 dC = 0.

The left-hand side is the **exactness defect**: the amount by which the square
fails to commute. Our `ExactChainMapLayer` parameterizes `(F0, F1)` inside the
nullspace of that expression, so every parameter value satisfies the equation up
to floating-point roundoff. The **chain-map residual** we report is the largest
observed magnitude of that defect. Because the zero map trivially satisfies the
equation, paired-signal or task supervision remains necessary.

### 2.4 Filtered mapping cones

A **mapping cone** is a construction that packages "what the map fails to
preserve" into a single object. If the cone has no homology — if it is
**acyclic** — the map is an isomorphism on homology. Acyclicity is thus a
certificate of invertibility.

A **filtered** mapping cone applies this at a sequence of thresholds rather than
once, so the certificate can be read as a function of scale. We compute both a
differentiable soft-nullity proxy, usable as a training signal, and an exact
integer-rank cone homology evaluation, used only for reporting.

The crucial interpretive point, which our results make concrete: acyclicity says
the decoded map is invertible. It does not say the decoded map is the *planted*
one. A wrong invertible map has an acyclic cone too.

### 2.5 RTD and SRTD

**Representation Topology Divergence** compares two point clouds that describe
the same items — possibly in different ambient dimensions — by tracking when
topological features appear and disappear as a distance threshold grows, and
scoring the mismatch. **SRTD** is a symmetric variant that replaces an ad hoc
directional combination with a union/intersection construction and reports
degree-specific totals.

Our corrected reference implementation accepts paired dissimilarity matrices with
a one-to-one entity correspondence, normalizes each matrix by its full-matrix 0.9
quantile, reports persistence separately by homological degree, uses degree 1 for
the published scalar convention, constructs one additional simplex degree
internally for deaths while excluding truncation-frontier generators, and caps
exact enumeration at 64 entities. The campaign uses 48 RTD training entities.

A separate training-time H0 distance surrogate used in historical runs is *not*
an exact cross-barcode and may disagree with exact RTD even in directional
ordering; it is labeled a surrogate throughout the code and its historical
outputs are withdrawn.

## 3. Related work

Sections 2.5 and 1.2 cite the primary sources this work builds on. In summary:
RTD [1], RTD-AE [2], and SRTD [3] define the representation-divergence reference;
DCM [4] and DiffLift [5] are the closest precedents for learning higher-order
topological structure from graph data; Neural Sheaf Diffusion [6] is the
precedent for learned sheaf transports; DeepMoE [7], GMoE [8], GraphMETRO [9],
and MvCGE [10] establish per-example dynamic expert routing on graph data.

HOMYMOLY differs in combining fixed representation families, conversion
diagnostics, and routing in one codebase. The present experiments do not
establish that this combination is new or better than any of the above.

## 4. System and method

### 4.1 Architecture and data flow

```
                    synthetic cellular annulus template
                    12 vertices | 18 edges | 6 faces
                    Betti (1, 1, 0) | 6 sectors
                                  |
                    plant one of 12 dihedral group elements
                                  |
                                  v
   +---------------------------------------------------------------+
   |  SOURCE COMPLEX C                     TARGET COMPLEX D        |
   |  signals x0 (vertices)                signals y0 (vertices)   |
   |         x1 (edges)                            y1 (edges)      |
   |         x2 (faces)                            y2 (faces)      |
   |  boundaries dC1, dC2                  boundaries dD1, dD2     |
   +---------------------------------------------------------------+
                     |                              ^
                     |  identifying markers         |
                     v                              |
        +--------------------------+                |
        |  flattened MLP selector  |                |
        |  (observes markers only) |                |
        +--------------------------+                |
                     |                              |
                     v                              |
        +--------------------------+                |
        |  hard-coded basis of 12  |                |
        |  group-action matrices   |                |
        +--------------------------+                |
                     |                              |
                     v                              |
        +--------------------------------+          |
        |  ExactChainMapLayer            |  applies |
        |  (F0, F1, F2) constrained to   |----------+
        |  nullspace of dD F - F dC = 0  |
        +--------------------------------+
                     |
        +------------+-------------+---------------+
        v            v             v               v
   transformation  typed        chain-map      filtered
   accuracy        recon. MSE   residual       mapping cone
   (1 of 12)       per degree   (tol 1e-5)     -> exact Betti
                                               -> soft nullity
```

Read the diagram top to bottom: a group element is planted, the selector sees
only identifying markers, it chooses a member of a fixed basis, and the resulting
typed map is scored four ways. The chain-map constraint is structural — it holds
for every parameter value — so the residual column is a numerical check, not a
learned objective.

### 4.2 Losses and ablations

Every objective is a weighted sum of the terms below. The eight ablations are the
frozen factorial cells of the campaign.

| term | what it penalizes | role |
|---|---|---|
| `task` | error in selecting the planted group element (12-way) | identification supervision |
| `reconstruction` | typed signal MSE at degrees 0, 1, 2 after applying the decoded map | paired-signal supervision |
| `cone` | differentiable soft-nullity proxy for mapping-cone acyclicity | structural (homological) |
| `rtd` | degree-1 SRTD between source and mapped point clouds, 48 entities | structural (homological) |

| ablation | task | reconstruction | cone | rtd | purpose |
|---|:--:|:--:|:--:|:--:|---|
| `task_only` | yes | – | – | – | identification supervision alone |
| `reconstruction_only` | – | yes | – | – | paired-signal supervision alone |
| `task_reconstruction` | yes | yes | – | – | working control for structural contrasts |
| `task_reconstruction_cone` | yes | yes | yes | – | does a cone term add anything? |
| `task_reconstruction_rtd` | yes | yes | – | yes | does an RTD term add anything? |
| `combined` | yes | yes | yes | yes | do both together add anything? |
| `cone_only` | – | – | yes | – | can a cone term identify the map alone? |
| `rtd_only` | – | – | – | yes | can an RTD term identify the map alone? |

Three contrasts were declared before the campaign, each against
`task_reconstruction`: `combined`, `task_reconstruction_cone`, and
`task_reconstruction_rtd`. The two `*_only` cells are identifiability controls,
not contrasts.

### 4.3 Endpoints and baselines

| endpoint | direction | family |
|---|---|---|
| `transformation_accuracy` | higher is better | identification |
| `cell_face_accuracy` | higher is better | cell recovery |
| `map_mse` | lower is better | map error |
| `degree_zero_mse`, `degree_one_mse`, `degree_two_mse` | lower is better | typed reconstruction error |
| `sheaf_transport_frobenius_mse` | lower is better | prespecified descriptive reconstruction error |

Two baselines anchor the accuracy endpoints. **Chance** is 1/12 = 0.0833 for
transformation accuracy and 1/6 = 0.1667 for cell-face accuracy. An **analytic
marker decoder** — a closed-form reader of the identifying markers, with no
learning — achieves 1.000. The analytic decoder matters: it shows the task is
exactly solvable from the provided markers, so a learned accuracy of 1.000 is
recovery of a known-attainable ceiling, not evidence of a powerful model.

### 4.4 Routing experiment

Three experts (graph, cell-complex, connection-sheaf) each return a
64-dimensional embedding and binary logits. A hard router selects one expert per
example from observation summaries that omit label and regime tensors but include
graph features, candidate and active-face statistics, and sheaf-transport
statistics. The synthetic benchmark builds counterfactual groups containing one
example per regime-label pair on a shared oriented complex, with group-disjoint
splits and deliberately relational labels (a sign relation across unmarked anchor
edges; probe-face activity with the edge cochain held fixed; a cycle-holonomy
defect).

Routing training used privileged latent-regime distillation: validation regime
labels indexed a regime-by-expert accuracy table that supplied the router's
utility targets. The regime was never an inference input, but this is a
supervision privilege and is part of the claim boundary throughout.

### 4.5 Fixed-expert corruption diagnostic

Two families of corruption diagnostic are reported. Both compare clean and
corrupted **fixed-expert embeddings**. Neither invokes a translator or a learned
map, so neither can support or refute any conversion claim.

The statistic is the Pearson correlation of rank residuals between a topological
defect diagnostic and the damage rate, after controlling for ranked severity,
ranked mean embedding displacement, and block fixed effects. Draws are
deterministic under a `sha256-block-and-sample-v1` protocol, so a baseline and a
candidate with matching seeds receive identical corrupted examples. Each
corruption kind contributes 13 complete blocks across 5 severity levels
(0.05, 0.1, 0.2, 0.4, 0.8), giving 65 paired batch observations per kind.

- **Gate-3 base family**: four trained runs (`task-only` baseline;
  `plus-recon`, `plus-chain`, `full` candidates), three corruption kinds, nine
  paired contrasts. Uncertainty is a paired complete-block bootstrap; inference
  is whole-block model-label randomization. Conditional on the fixed checkpoint
  pair — it does not estimate training-seed variation.
- **Gauge family**: eight seed-matched pairs (`gauge-task-only` versus
  `gauge-plus-chain`, seeds 20260803–20260810) differing only in
  `translator_weight` (0.0 vs 0.1) and `chain_weight` (0.0 vs 0.05). Here the
  unit of analysis *is* the training seed, so an across-seed Student-t interval
  with df = 7 and an exact two-sided sign test are available.

## 5. Experimental design

### 5.1 Freeze, seeds, denominators

The identifiable campaign was frozen before execution. Source config
`configs/identifiable-maps/gb10-full.yaml`, SHA-256
`22abb205e8a89586b38799d7f7b8d53f0c24cef45f872453533ddf34e20fad73`. Eight
ablations × five seeds (20260821–20260825) = 40 runs, all executed, none missing,
replaced, or excluded. Each run uses 4,800 training, 1,200 validation, and 1,200
test examples on the single annulus template. The map tolerance is fixed at 1e-5.
Cone-only and RTD-only aggregate Betti histograms cover 6,000 test examples each
(5 seeds × 1,200).

The routing campaign froze five configs and its primary endpoint before its valid
runs; seeds 20260906–20260910; the endpoint is hard-routed accuracy minus the
maximum fixed-expert accuracy.

### 5.2 Hardware, software, provenance

All campaign runs and all benchmarks executed on one NVIDIA GB10,
Linux 6.17.0-1026-nvidia aarch64, glibc 2.39, Python 3.12.3, PyTorch 2.13.0+cu130,
CUDA 13.0, NumPy 2.5.2, PyYAML 6.0.3, deterministic algorithms enabled with
`CUBLAS_WORKSPACE_CONFIG=:4096:8`.

| item | value |
|---|---|
| identifiable campaign commit | `8021292e97abfec91768f1b5437c883a42c29c60` |
| routing campaign commit | `e69b07707950b6abe332366c51fe8c94254899f3` |
| campaign launch fingerprint | `44408d7adf8467e594879b46e25a1cb7fd89a7e7a5d5f3446548bcbf3ed1096e` |
| frozen full-config SHA-256 | `22abb205e8a89586b38799d7f7b8d53f0c24cef45f872453533ddf34e20fad73` |
| strict campaign summary SHA-256 | `0cd0defb0b0d41b5f7563c364cfdda62cb72c5e5a845bdd5d1ab76a2e1cb953c` |
| identifiable code fingerprint | `5908adf7d445524c52797d779478945a184b4e1f10056c1d21bcde044bedb360` |
| routing code fingerprint | `473fb0f6714798274c38949107221df3bd941e89273a6eef76e54394d6c1f1d8` |
| scheduler steps completed | 56 of 56 |

The scheduler receipt is sealed and lists all 296 produced files with byte counts
and SHA-256 digests; all 296 verify against the on-disk artifacts. The 56 steps
comprise 40 training runs, one strict summary, ten identifiable-map checkpoint
benchmarks, and five routing benchmarks.

### 5.3 Estimands, uncertainty, multiplicity, stopping rules

The identifiable campaign's estimand for each declared contrast is the mean
across five paired seeds of (candidate − control) on a registered endpoint.
Uncertainty is a Student-t interval with df = 4; a distribution-free exact
two-sided sign test is reported as sensitivity. **No multiplicity adjustment is
applied anywhere in this report.** With five untied paired seeds the exact
two-sided sign test cannot fall below p = 0.0625, so it can never be decisive at
conventional thresholds; with eight untied seeds in the gauge family the floor is
p = 0.0078125.

The gauge estimand is the mean across eight paired training seeds of the
candidate-minus-baseline adjusted statistic, with a df = 7 Student-t interval and
an exact two-sided sign test.

Decision rules were fixed in advance: a contrast is called a difference only if
its interval excludes zero. The engineering recovery gate is an absolute
threshold gate, not a comparison: a run passes when cell-face accuracy ≥ 0.95,
transformation accuracy ≥ 0.95, map MSE ≤ 1e-3, chain residual ≤ 1e-5, and the
hard-cone acyclic fraction = 1.0. Stopping was by fixed epoch budget, not by
monitoring an endpoint.

A stop condition was declared for provenance mismatch: any disagreement in
checkpoint, config, script, commit, sample count, seed, pairing key, schema, or
hash halts analysis rather than triggering a re-run or a substitution. No such
mismatch was found.

## 6. Results

### 6.1 Exact recovery of the planted typed map

Six of the eight objectives — every objective containing task or reconstruction
supervision — recover the planted map exactly on all five seeds.

| ablation | transformation accuracy | cell-face accuracy | map MSE | degree-1 MSE |
|---|---:|---:|---:|---:|
| `task_only` | 1.000 | 1.000 | 2.6e-17 | 4.2e-16 |
| `reconstruction_only` | 1.000 | 1.000 | 2.5e-08 | 3.2e-07 |
| `task_reconstruction` | 1.000 | 1.000 | 1.7e-16 | 2.4e-15 |
| `task_reconstruction_cone` | 1.000 | 1.000 | 1.7e-16 | 2.4e-15 |
| `task_reconstruction_rtd` | 1.000 | 1.000 | 1.6e-16 | 2.3e-15 |
| `combined` | 1.000 | 1.000 | 1.7e-16 | 2.3e-15 |
| `cone_only` | 0.0815 | 0.1697 | 1.09e-01 | 1.07 |
| `rtd_only` | 0.0833 | 0.1703 | 1.91e-01 | 1.87 |

Chance is 0.0833 for transformation accuracy and 0.1667 for cell-face accuracy.
All standard deviations across the five seeds are zero for the saturated
accuracies.

The **engineering recovery gate passed in 10 of 10 applicable runs** — the
`task_reconstruction` and `combined` objectives, five seeds each. In every
applicable run both accuracies were exactly 1.0, map errors were at numerical
precision, chain-map residuals met the fixed 1e-5 tolerance (largest observed
1.42e-14), and hard cones were acyclic. Zero runs failed.

This is an implementation gate. It establishes that the parameterization,
training path, decoder, and exact cone oracle agree on a case where the answer is
known. Because the analytic marker decoder also attains 1.000, the ceiling was
attainable without learning.

### 6.2 Structural losses add nothing measurable

All 21 declared continuous contrast endpoints — three contrasts against
`task_reconstruction`, seven registered endpoints each — have Student-t intervals
containing zero. Accuracy endpoints are exactly tied at 1.000, so their
differences are identically zero and their sign tests are fully tied.

This is a null result under a hard ceiling, and the ceiling is the reason it is
weak evidence rather than strong evidence of absence. When the control already
scores 1.000, no candidate can improve on it, and a null is guaranteed by
construction on the identification endpoints. The continuous reconstruction
endpoints are more informative — they had room to move and did not — but they sit
at 1e-16, within floating-point noise of exact.

The honest reading: on this benchmark, mapping-cone and RTD terms neither help
nor hurt. The benchmark cannot distinguish "these terms are useless" from "this
task is too easy to reveal their value".

### 6.3 Structural losses alone cannot identify the map, and provably cannot

The two identifiability controls are the sharpest structural finding.

| control | transformation accuracy | chance | cell-face accuracy | chance | hard-cone Betti over 6,000 examples |
|---|---:|---:|---:|---:|---|
| `cone_only` | 0.0815 | 0.0833 | 0.1697 | 0.1667 | `[0,0,0,0]` in 6,000 / 6,000 |
| `rtd_only` | 0.0833 | 0.0833 | 0.1703 | 0.1667 | `[0,0,0,0]` in 6,000 / 6,000 |

Both controls sit at chance on both accuracy endpoints, and their map MSEs
(1.09e-01, 1.91e-01) are fifteen orders of magnitude worse than the supervised
objectives. Yet **every decoded cone is acyclic in every one of the 6,000
evaluated examples for both controls.**

That combination is the point. A model trained only to make its mapping cone
acyclic succeeds completely at making its mapping cone acyclic, and learns
nothing about which map was planted.

**This is not an optimization failure. Both signals are constant on the
hypothesis space, so their information content about the planted element is
exactly zero.** The construction makes this checkable rather than conjectural.
Each of the twelve basis maps is built as a signed permutation in every degree:
`F0` permutes vertices, `F1` permutes edges with an orientation sign, and `F2`
is fixed by matching the mapped cellular boundary against a unique oriented
face. We verified numerically that all twelve satisfy `Fᵀ F = I` at degrees 0,
1, and 2, and that each is a signed permutation with exactly one unit-magnitude
entry per row and column.

Two consequences follow directly.

- **The cone signal is constant.** A mapping cone is acyclic exactly when its
  chain map is a quasi-isomorphism. A signed permutation is invertible, so every
  one of the twelve candidates is an isomorphism of chain complexes, hence a
  quasi-isomorphism, hence has an acyclic cone. Cone acyclicity takes the same
  value on all twelve hypotheses. The observed `[0,0,0,0]` in 6,000 of 6,000
  examples is the empirical signature of that constant, not a learned outcome.
- **The RTD signal is constant.** A signed permutation is orthogonal, hence an
  isometry, so the mapped point cloud has the same pairwise dissimilarity matrix
  as the source under every candidate (maximum observed distance change
  1.5e-07, at float precision). The paired matrices RTD consumes are therefore
  identical across the hypothesis space, and the divergence is likewise
  constant.

So both `*_only` objectives are fully satisfiable by any of the twelve
candidates, and chance-level identification is the only attainable outcome. The
generalization is broader than this annulus: **any hypothesis class whose
candidate maps are all invertible has a constant cone-acyclicity signal**, and
that is precisely the setting in which a cone objective looks most attractive.

The obvious objection — that this is circular, because the hypothesis class was
chosen to consist of isomorphisms — is the finding restated rather than a
rebuttal. The property that motivates reaching for a cone objective in the first
place, namely wanting the learned map to be structure-preserving, is the same
property that makes acyclicity useless for choosing among structure-preserving
candidates. What makes it a trap rather than a triviality is the view from
inside the training loop: the structural objective is driven to complete
satisfaction, the diagnostic reports a clean certificate on every example, and
identification never rises above chance. Nothing in the training signal reveals
the problem.

Acyclicity certifies invertibility within the template family; it does not
certify correctness. Any claim that a cone objective supplies a useful training
signal for identification is refuted here.

### 6.4 Frozen routing result

| seed | hard routed | best fixed | margin | dense | route accuracy | route MI (nats) |
|---|---:|---:|---:|---:|---:|---:|
| 20260906 | 0.8028 | 0.6743 | +0.1285 | 0.7723 | 0.6100 | 0.1622 |
| 20260907 | 0.7745 | 0.6754 | +0.0991 | 0.7571 | 0.5468 | 0.0972 |
| 20260908 | 0.7680 | 0.6667 | +0.1013 | 0.7582 | 0.5381 | 0.0934 |
| 20260909 | 0.7789 | 0.6667 | +0.1122 | 0.7342 | 0.5752 | 0.1264 |
| 20260910 | 0.7745 | 0.6667 | +0.1078 | 0.7505 | 0.5501 | 0.1006 |

Mean hard-minus-best-fixed margin **+0.1098** (sample SD 0.0117), two-sided 95%
Student-t interval **[+0.0953, +0.1243]**. All five margins are positive and the
frozen interval rule labels the result supported. The exact sign-test sensitivity
value is p = 0.0625, its floor at five untied seeds. Mean hard-minus-dense
accuracy is +0.0253; dense logits are an unweighted mean of trained expert logits,
not an independently optimized ensemble.

Two disclosures travel with this number permanently. Training used privileged
latent-regime distillation, and an aborted pre-freeze seed-20260906 attempt
recorded validation metrics under different executable code before the committed
seed was rerun. The core endpoint, decision rule, and config hashes were committed
before the valid runs, but a provenance-safeguard paragraph was added during seed
five. We therefore call this protocol-aligned under the committed freeze, not
pristine preregistration.

### 6.5 Trained compute benchmarks

Both benchmark families measure a warmed, synchronized CUDA forward pass on the
same GB10. **They report different tail statistics and are never pooled**: the
identifiable runner records p10/p90 and the routing runner records p95. Neither
is a preregistered matched-compute Pareto claim.

**Identifiable-map checkpoints** — batch 192, 20 warmup and 100 timed iterations,
five seeds per ablation, model forward only (data loading, training losses, exact
RTD, and exact cone oracles are outside the timed region):

| ablation | median latency (mean ± SD over 5 seeds) | p90 latency (mean) | peak allocated bytes |
|---|---:|---:|---:|
| `combined` | 0.2753 ± 0.0037 ms | 0.2899 ms | 35,069,440 |
| `task_reconstruction` | 0.2762 ± 0.0022 ms | 0.2979 ms | 35,069,440 |

The paired difference is −0.00089 ms with a 95% interval of [−0.0075, +0.0057] ms
— indistinguishable, as expected: **the two ablations execute the same inference
graph**, so this contrast is a runner-noise check, not an architectural
comparison. Peak allocated memory is byte-identical across all ten runs.

**Routing checkpoints** — batch 64, bfloat16, 100 timed iterations, five seeds:

| quantity | mean ± SD over 5 seeds |
|---|---|
| dense-to-routed median-latency ratio | 1.532 ± 0.035 |
| routed-to-fastest-fixed median-latency ratio | 2.269 ± 0.043 |

Routed inference is about 1.53× faster than dense three-expert evaluation and
about 2.27× slower than the fastest single fixed route, which was `fixed_graph`
in all five seeds. Routed evaluation also had lower peak allocated memory than
dense (119,415,296 versus 169,401,344 bytes) in every seed.

> **Correction.** An earlier internal handoff recorded the routed-to-fastest-fixed
> ratio as `1.863 ± 0.071`. That figure is not reproducible from any artifact in
> this repository under any ratio definition we could construct, and it appears in
> no machine-readable result. The value above, 2.269 ± 0.043, is recomputed from
> the five sealed trained benchmarks and is *less* favorable to routing than the
> figure it replaces. Two earlier `compute-remediation*.json` benchmarks record
> `checkpoint: null` — they timed an untrained model and are excluded from all
> reported compute results.

### 6.6 Corrected Gate-3 base diagnostic

Nine paired contrasts, all with 13 complete blocks and 65 paired batch
observations per kind:

| candidate − `task-only` | corruption kind | adjusted difference | 95% paired block-bootstrap interval |
|---|---|---:|---|
| `plus-recon` | edge cochain | +0.018 | [−0.311, +0.347] |
| `plus-recon` | node anchor | +0.020 | [−0.235, +0.284] |
| `plus-recon` | transport rotation | −0.011 | [−0.167, +0.120] |
| `plus-chain` | edge cochain | −0.022 | [−0.295, +0.334] |
| `plus-chain` | node anchor | +0.030 | [−0.118, +0.147] |
| `plus-chain` | transport rotation | −0.021 | [−0.164, +0.088] |
| `full` | edge cochain | +0.057 | [−0.185, +0.364] |
| `full` | node anchor | −0.011 | [−0.217, +0.241] |
| `full` | transport rotation | −0.072 | [−0.214, +0.019] |

**All nine intervals contain zero.** No multiplicity adjustment was applied. These
statistics are conditional on each fixed checkpoint pair and the sampled blocks;
they do not estimate training-seed variation.

### 6.7 Gauge diagnostic across eight training seeds

The gauge family upgrades the unit of analysis from the checkpoint pair to the
training seed, giving eight paired seeds and df = 7:

| corruption kind | mean difference | sample SD | 95% Student-t interval (df = 7) | exact sign test |
|---|---:|---:|---|---|
| edge cochain noise | −0.0968 | 0.1595 | [−0.2301, +0.0366] | p = 1.000 (4+/4−) |
| node anchor noise | −0.0549 | 0.1017 | [−0.1400, +0.0301] | p = 0.727 (3+/5−) |
| transport rotation | +0.0506 | 0.1241 | [−0.0531, +0.1543] | p = 0.727 (5+/3−) |

**All three intervals contain zero** and no sign test approaches significance,
against a floor of p = 0.0078125. Adding translator and chain terms to the gauge
objective produces no detectable change in the fixed-expert corruption
diagnostic. Scope is unchanged: this is a fixed-expert embedding diagnostic and
evaluates no conversion.

### 6.8 Historical molecular transfer

Reported for completeness; superseded in importance by the synthetic results
above and unchanged by this revision. Official OGBG-MOLHIV scaffold split
(32,901/4,113/4,113), early stopping on validation AUROC, official evaluator. The
three values behind each mean are initialization seeds on one fixed split; their
SD is not uncertainty over molecule sampling.

| model | status | valid mean | official-test AUROC (mean ± seed SD) |
|---|---|---:|---:|
| graph | initial comparison | 0.794 | 0.771 ± 0.014 |
| cell v1, AtomRing faces | initial comparison | 0.782 | 0.723 ± 0.017 |
| cell v2, boundary-edge max + ring size | post-test development | 0.771 | 0.757 ± 0.002 |
| cell v3, plus bond-type counts | post-test development | 0.761 | 0.729 ± 0.025 |

In the initial v1 read, graph exceeded cell by 0.0481 AUROC and won all three
paired seeds. V2 was designed after inspecting v1 on the official test and v3
after inspecting v2, so v2 and v3 test scores are exploratory development
observations, not clean comparisons. The test split contains no acyclic graphs
(a cycle-rank audit finds 4,113/4,113 cyclic), so acyclic molecular transfer is
unevaluable here.

## 7. Discussion

**Exactness is achievable and was achieved; it was also easy.** The
implementation recovers the planted map to 1e-16 whenever it receives either form
of supervision, and an analytic decoder does the same with no learning. The right
conclusion is that the machinery is correct, not that the model is strong.

**Acyclicity is not correctness.** This is the result we would most want other
work to carry forward, and §6.3 shows it is structural rather than incidental.
Because every candidate map is a signed permutation — invertible and
distance-preserving — cone acyclicity and RTD are constant across the hypothesis
space, so neither can carry information about which element was planted. Chance
identification is the only attainable outcome, and the acyclic cones in 6,000 of
6,000 examples are that constant made visible.

The scope of the warning is what matters. Any hypothesis class of invertible
maps inherits the same degeneracy, and that is exactly the class a practitioner
has in mind when a cone objective seems appealing. Treating acyclicity as
evidence of learned correspondence would be a mistake, and our controls show
precisely how that mistake looks from the inside: the structural objective is
driven to full satisfaction, every example returns a clean certificate, and the
science is absent. A cone term is a legitimate *constraint* and a useless
*discriminator* among structure-preserving candidates.

**Routing and conversion remain separate claims.** A router can choose among
fixed experts without learning any map between their domains. The +0.1098 margin
supports the former in a privileged synthetic setting and provides no evidence
for the latter.

**A responsive diagnostic is not incremental predictive value.** Corrected
RTD/SRTD measures degree-specific representation differences, but across nine
Gate-3 contrasts and three eight-seed gauge contrasts no interval excludes zero.

**Ceiling effects limit what this design can ever show.** The most useful next
step is not more seeds on this benchmark but a harder one: a template family
where the correct map is not attainable analytically, so that structural terms
have room to contribute and a null becomes informative.

## 8. Limitations

- **Ceiling effects.** Six of eight objectives saturate both accuracy endpoints
  at exactly 1.000. Contrasts against a saturated control cannot detect
  improvement, so the structural null is weak evidence of absence.
- **Five seeds.** The identifiable campaign has five paired seeds; the exact
  two-sided sign test floor is p = 0.0625, so it can never be decisive. Five
  seeds are also too few to check Student-t assumptions.
- **Synthetic data, one template.** All 40 runs use a single six-sector annulus
  with a hard-coded finite group action. Nothing here transfers automatically to
  real data, other complexes, or larger map families.
- **Privileged historical routing supervision.** The routing result used
  latent-regime distillation from validation regime labels and had target
  structured views available at inference, plus the disclosed pre-freeze
  procedural deviation.
- **Target-view historical translators.** The historical translator modules
  consumed target face activity and target edge transports. They are target-view
  encoders and do not demonstrate conversion from graph observations. Current
  cell and sheaf targets are additionally unidentifiable from graph inputs in
  this generator.
- **Fixed-expert Gate-3 scope.** Both corruption families compare clean and
  corrupted fixed-expert embeddings. They never invoke a translator or learned
  map and cannot support or refute a typed-conversion claim. Gate-3 base
  statistics are conditional on fixed checkpoints; only the gauge family
  estimates across-seed variation.
- **Timing order and scope.** Within each routing benchmark process all paths
  were timed in a fixed order (routed, fixed_graph, fixed_cell, fixed_sheaf,
  dense), so residual thermal or allocator drift is confounded with path order.
  Raw per-iteration timings were not retained — only the summary quantiles. All
  timings come from one runner, one machine, one batch shape, and one precision
  per family, and are not matched-accuracy comparisons.
- **Mixed tail statistics.** The identifiable runner reports p90 and the routing
  runner reports p95. These are never pooled and must not be compared directly.
- **No multiplicity control.** No adjustment is applied to any interval or
  p-value in this report.
- **Artifact boundaries.** The 8.8 GB `artifacts/` tree is untracked. Tracked
  evidence is the curated `results/` bundle (48 files, 2.85 MB); checkpoints and
  per-example dumps are pinned by SHA-256 but not distributed.
- **Numerical determinism.** GPU scatter-add is not guaranteed bit-deterministic
  even with deterministic flags enabled. Exact cross-barcode enumeration is
  exponential and capped at 64 entities.
- **Molecular results.** The MOLHIV test split was adaptively reused after v1 and
  contains no acyclic graphs.

## 9. Claim boundary

The strongest supported new result is deliberately narrow: **the implementation
recovers a finite dihedral family of exact three-term maps on one synthetic
six-sector cellular annulus, using explicit identifying markers and a flattened
MLP to select from a hard-coded finite group-action basis.** This is a controlled
implementation and exactness study.

This work does **not** establish:

- superiority of cone or RTD training losses — both are null here, and neither
  can identify the map alone, provably so within a hypothesis class of
  invertible maps (§6.3);
- conversion quality on real data or out-of-distribution structures;
- general equivalence between graphs, cellular complexes, and sheaves;
- a learned quasi-isomorphism or exact sequence — the verified identity is the
  chain-map law up to a fixed 1e-5 numerical tolerance;
- any Langlands, eigensheaf, Fourier–Mukai, or category-theoretic machine
  learning result; those ideas are motivation, not results, and nothing in this
  repository validates them;
- a translator-based Gate-3 claim — the corruption programs evaluate fixed expert
  embeddings only.

## 10. Code and data availability

All code, configurations, and compact evidence are in the project repository. The
tracked evidence bundle under `results/` contains 48 files totaling 2.85 MB with a
`MANIFEST.json` recording, for every file, its path, byte count, SHA-256,
generating commit, and generating command:

| location | contents |
|---|---|
| `results/summaries/` | strict compact summaries for the identifiable, gauge, compute, and routing campaigns |
| `results/gate3/`, `results/gate3g/` | gate decisions and per-batch corruption-report derivatives |
| `results/benchmarks/` | ten identifiable and five routing trained benchmark records |

Corruption reports are exported as **per-batch derivatives**: the `per_example`
array is dropped and the `per_batch` array — the unit of analysis — is retained.
This is lossless for every published statistic, verified by recomputing the
adjusted partial Spearman correlation from the retained rows alone and matching
the published value to 1e-15. Each derivative records the SHA-256 of the
untruncated source report, so the dropped detail remains pinned.

The 8.8 GB `artifacts/` tree — checkpoints, per-example prediction dumps,
training histories, scheduler logs — is intentionally untracked and is not
distributed. It is not durable journal evidence by itself.

The synthetic generators are deterministic given the committed configs and seeds;
no external dataset is required to reproduce the identifiable campaign. OGBG-MOLHIV
is obtained through the standard OGB distribution.

## 11. Reproducibility

Regenerating the tracked evidence from the artifact tree is three commands:

```bash
python scripts/summarize_gauge_corruption_campaign.py \
  --output results/summaries/gauge-corruption-campaign.json
python scripts/summarize_compute_campaign.py \
  --output results/summaries/compute-campaign.json
python scripts/export_publication_evidence.py --output-root results
```

Each summarizer revalidates provenance before it aggregates and refuses to
proceed on any hash, seed, pairing, or schema mismatch. The exporter works from
an explicit allowlist and refuses checkpoints, prediction dumps, histories, logs,
and caches even when asked for them. Its manifest carries no timestamp, so
re-exporting unchanged evidence reproduces the bundle byte for byte; verify an
existing bundle with:

```bash
python scripts/export_publication_evidence.py --verify-only
```

The paper PDF is rendered from this Markdown file with
`python scripts/render_paper.py --input docs/18-paper.md --output docs/18-paper.pdf`.

Reproducing the 40-run campaign itself requires an NVIDIA GB10 or equivalent and
the pinned software stack in §5.2; the campaign is not re-executed as part of
release validation.

## 12. Declarations

**Author contributions.** Sean Mahdavian designed and implemented the system,
ran the campaigns, and wrote the manuscript.

**Affiliation, ORCID, funding, competing interests, data license, and code
license** are author-supplied submission fields and are intentionally unset in
this draft rather than guessed. They must be completed before submission.

**Ethics.** This work uses synthetic data generated by committed deterministic
code and one public molecular benchmark (OGBG-MOLHIV) under its published terms.
No human subjects, personal data, or personally identifiable information are
involved. The molecular component is a methodological benchmark comparison and is
not a claim about any specific compound's biological activity; the reported
result there is negative in its initial clean comparison. We see no dual-use
concern arising from this work. Compute costs are reported in §6.5; the total
campaign is small — 40 short training runs and 15 benchmark runs on a single
workstation-class device.

**Reporting.** Null and negative results are reported in the body of this paper
rather than in an appendix, and every number in every table is traceable to a
tracked machine-readable file under `results/`.

## References

1. Barannikov, Trofimov, Balabin, and Burnaev.
   [Representation Topology Divergence: A Method for Comparing Neural Network
   Representations](https://proceedings.mlr.press/v162/barannikov22a.html).
   ICML, 2022.
2. Trofimov, Cherniavskii, Tulchinskii, Balabin, Burnaev, and Barannikov.
   [Learning Topology-Preserving Data
   Representations](https://openreview.net/forum?id=lIu-ixf-Tzf).
   ICLR, 2023.
3. Wang and Hu.
   [Symmetric Divergence and Normalized Similarity: A Unified Topological
   Framework for Representation
   Analysis](https://openreview.net/forum?id=pGgJ9qB2Io).
   TMLR, 2026.
4. Battiloro, Spinelli, Telyatnikov, Bronstein, Scardapane, and Di Lorenzo.
   [From Latent Graph to Latent Topology Inference: Differentiable Cell Complex
   Module](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6b97236d90d945be7c58268207a14f4f-Abstract-Conference.html).
   ICLR, 2024.
5. Franco, Duarte, Nikitin, Ponti, Mesquita, and Souza.
   [Differentiable Lifting for Topological Neural
   Networks](https://openreview.net/forum?id=eC89CbINIw).
   ICLR, 2026.
6. Bodnar, Di Giovanni, Chamberlain, Liò, and Bronstein.
   [Neural Sheaf Diffusion](https://arxiv.org/abs/2202.04579).
   NeurIPS, 2022.
7. Wang, Yu, Dunlap, Ma, Wang, Mirhoseini, Darrell, and Gonzalez.
   [Deep Mixture of Experts via Shallow
   Embedding](https://proceedings.mlr.press/v115/wang20d.html).
   UAI, 2020.
8. H. Wang, Jiang, You, Han, Liu, Srinivasa, Kompella, and Z. Wang.
   [Graph Mixture of Experts: Learning on Large-Scale Graphs with Explicit
   Diversity
   Modeling](https://papers.nips.cc/paper_files/paper/2023/hash/9f4064d145bad5e361206c3303bda7b8-Abstract-Conference.html).
   NeurIPS, 2023.
9. S. Wu, Cao, Ribeiro, Zou, and Leskovec.
   [GraphMETRO: Mitigating Complex Graph Distribution Shifts via Mixture of
   Aligned
   Experts](https://papers.nips.cc/paper_files/paper/2024/hash/11c892a9fcc430cc0f4c7d457e5d60ea-Abstract-Conference.html).
   NeurIPS, 2024.
10. Z. Wu, Cai, Zhang, Lu, Chen, Zhuang, and Wang.
    [Where Graph Meets Heterogeneity: Multi-View Collaborative Graph
    Experts](https://openreview.net/forum?id=dsp8dUlZFq).
    NeurIPS, 2025.
