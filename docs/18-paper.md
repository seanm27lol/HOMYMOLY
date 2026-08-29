# Graph-Derived Cycle-Subspace Information for Scarce-Probe Identification of an Edge-to-Cycle Lifting: An Exact Constraint Outperforms Soft Shrinkage

**Sean Mahdavian**

Phillips Exeter Academy · ORCID
[0009-0000-8432-7825](https://orcid.org/0009-0000-8432-7825)

**Revision: 2026-08-25**

## Abstract

Moving data between structured representations — graphs, cell complexes, sheaves
— can lose information, and homological algebra supplies exact tools for
measuring what a map destroys. We ask whether related constraints can also
*improve* a learned transformation, studying a continuous edge-to-cycle lifting
`W: R^E -> R^F`; a secondary chain-map selection task on an annulus serves as a
contrast case, where the exact cone-acyclicity certificate is constant across
the candidate family and cannot distinguish its twelve maps. The primary
evidence is a **sealed untouched-seed, outcome-informed, same-generator-family
replication**: the protocol and a machine-readable design seal were committed
before any of the 36 declared seeds was instantiated; 33 seeds were eligible,
each a joint topology/data/noise realization with one map fitted per topology.
Six of seven prespecified claims are supported at one-sided Bonferroni
`alpha = 0.05/7`. Exact least squares restricted to the graph's cycle kernel
improves held-out recovery over graph-blind ambient minimum-norm least squares
(one-sided upper bound −1.328 on the paired `log10` ratio) and over the
closed-form solution of the soft boundary-penalty objective at its frozen weight
(upper bound −0.048; geometric-mean MSE ratio 0.746); it also outperformed the
prespecified dimension-matched random-subspace control (upper bound −2.572).
The soft penalty's earlier improvement over graph-blind Adam replicates on this
disjoint block (upper bound −1.251). Training-only cross-validated ridge shows
no detected improvement and its frozen claim is not supported; a singular-value
surrogate replicates harm (lower bound +0.086); and any benefit of an
RTD-inspired surrogate is bounded below 10% of geometric-mean held-out MSE — a
bounded-benefit/futility result at the prespecified margin `log10(0.90)`, not
noninferiority or equivalence. In a prespecified descriptive off-path analysis,
the cycle-projector defect covaries positively with held-out error across
independent training/noise realizations while the random-subspace control defect
does not (paired Fisher-z contrast +1.088, unadjusted 95% interval
[+0.887, +1.289]); this is association, not causation. The design is
outcome-informed and single-family — not an independent-lab or
independent-generator replication — and the evidence comes from one synthetic
generator, one training-set size, one noise level, and one soft-penalty weight.

## 1. Introduction

### 1.1 The question

Two mathematically different learning tasks must be separated. A conversion
between chain complexes is a **typed map**: one degree-preserving component per
degree, satisfying the chain-map equations. The annulus experiment selects such
a map from a finite family. Homological algebra measures what a chain map
destroys (the kernel of its induced homology map), cannot reach (the cokernel),
and whether it is a quasi-isomorphism (equivalently, whether its mapping cone is
acyclic).

The primary continuous experiment is narrower. It learns
`W: R^E -> R^F`, from edge signals to coordinates in a chosen cycle basis;
`W^T: R^F -> R^E` is a candidate degree-2 differential. This degree-changing
lifting is not itself a typed chain map or a conversion between two complexes.
The practical question is whether structural penalties on its implied complex
improve recovery of that lifting.

### 1.2 What is and is not new

Learning higher-order structure from graphs is established: Differentiable Cell
Complex Module [5] learns cell probabilities jointly with a task, Differentiable
Lifting [6] learns liftings into cellular, simplicial, and combinatorial
complexes, and Neural Sheaf Diffusion [7] and Knowledge Sheaves [8] learn sheaf
restriction maps. Comparing paired representations through cross-barcodes is
likewise established: Representation Topology Divergence [1], its differentiable
autoencoder form [2], and the symmetric variant SRTD [3] are the direct basis for
our divergence reference, and a quotient-homology account of neural
representation [4] develops an adjacent algebraic view. Categorical deep
learning [9] and functor learning by gradient descent [10] ask what algebraic
structure a learned map should respect.

Our contribution is narrow and largely confirmatory-with-a-mechanism:

1. A sealed **untouched-seed, outcome-informed, same-generator-family
   replication** (33 eligible of 36 declared seeds) showing that graph-derived
   **cycle-subspace information** improves scarce-probe identification of an
   edge-to-cycle lifting: exact least squares restricted to `ker B1` improves
   over a graph-blind minimum-norm baseline and over the closed-form soft
   boundary penalty at its frozen weight, while the soft penalty's improvement
   over a graph-blind Adam reference replicates from the historical campaign.
2. A prespecified **off-path** secondary analysis, reported with unadjusted
   intervals and no support decision, in which the cycle-projector defect
   covaries with held-out damage across independent training/noise replicates
   and the paired Fisher-z contrast against a dimension-matched random-subspace
   defect is positive — association, not independent prediction or causation.
3. Multiplicity-controlled confirmatory results replicating harm from a
   **singular-value cone surrogate**, bounding any benefit of an
   **RTD-inspired normalized pairwise-distance surrogate** below a prespecified
   futility margin, establishing specificity of the cycle subspace against a
   dimension-matched random subspace, and reporting a frozen, unsupported claim
   for training-only cross-validated ridge.
4. Two **retrospective no-fit diagnostic heuristics**, proposed as prospective
   screens for future work, for detecting constant or target-misaligned signals,
   and a benchmark generator in which homological defects genuinely vary.

This is not a new theorem or algorithm for equality-constrained least squares or
Hodge decomposition. Its narrow contribution is an evidence-traceable evaluation
of input-derived cycle-subspace information — as an exact classical constraint
and as a soft penalty — on one deterministic synthetic
lifting family. We make no claim of priority over any cited system, and §3 is
positioning rather than systematic review.

## 2. Background and definitions

Each concept is stated in plain language before notation.

### 2.1 Graphs, cell complexes, and cellular sheaves

A **graph** records which pairs of objects are related. A **cell complex** adds
higher-dimensional pieces: on top of vertices (degree 0) and edges (degree 1) it
attaches faces (degree 2) glued along closed edge loops. A **cellular sheaf**
attaches a small vector space to each cell and a linear transport map to each
incidence, so data can move along the complex.

Formally a two-dimensional complex has boundary operators `d1` (edges to
vertices) and `d2` (faces to edges) satisfying `d1 d2 = 0`.

### 2.2 Typed maps and chain-map compatibility

A **typed map** carries degree-`n` data to degree-`n` data. It is a **chain map**
when it commutes with the boundaries — informally, when it does not tear the
complex. For a source with boundary `dC` and target with boundary `dD`,

    dD F1 − F0 dC = 0.

The left-hand side is the **chain-map defect**. Our historically named
`ExactChainMapLayer` parameterizes `(F0, F1)` inside the nullspace of that
expression, so the chain-map equation holds exactly rather than being penalized.
This use of “exactly” means equality to numerical tolerance; it does **not** mean
that a sequence is exact in the algebraic sense `im d_(n+1) = ker d_n`. That
utility layer is not used in either campaign reported here: the annulus model
mixes a fixed basis of exact chain maps, while the continuous lifting uses a
soft penalty.

### 2.3 The implied complex

This paper's central move. When a model learns a lift `W` from edge signals to
cycle-basis coordinates, `Wᵀ` is a candidate face boundary. The learned lifting
therefore **implies a complex** with boundaries `(B1, Wᵀ)`, and that complex is legitimate
only if `B1 Wᵀ = 0`. Every structural term below is written as a condition on the
implied complex rather than bolted on:

| campaign key | form | meaning |
|---|---|---|
| `exact` | `mean((B1 Wᵀ)²)` | boundary-of-boundary compatibility: the implied operators satisfy `d∘d = 0` |
| `cone` | `exp(−2·σ_min(W))` | singular-value cone surrogate that rewards increasing the smallest singular value |
| `rtd` | MSE between normalized pairwise distances | RTD-inspired distance-preservation surrogate |

All three short keys are retained because they were frozen in the campaign
protocol and machine-readable result. They must not be read as identities:
`exact` is **not exactness of a sequence** — `W = 0` already gives zero loss
without implying `im Wᵀ = ker B1`; `cone` is **not mapping-cone homology**;
and `rtd` is **not the published RTD or SRTD statistic** described in §2.5.
Exact mapping-cone homology and the corrected RTD/SRTD implementation appear
only as separate diagnostics elsewhere in this work.

### 2.4 Kernels, cokernels, and mapping cones

The **kernel** of the induced map on homology is what a conversion destroys; the
**cokernel** is what it cannot reach. A **mapping cone** is acyclic exactly when
the map is a quasi-isomorphism, and its homology is the sum of the two:

    dim H_n(cone F) = dim coker H_n(F) + dim ker H_(n−1)(F).

We verify this identity numerically on 24 maps: the twelve dihedral candidates
of §4.1 and twelve cycle-killing regression fixtures. It matters here because it
shows the cone **bundles** two directional facts into one number and loses their
separation — the kernel and cokernel are strictly more informative than the
acyclicity bit.

### 2.5 RTD and SRTD

**Representation Topology Divergence** compares two point clouds describing the
same items by tracking when topological features appear and disappear as a
distance threshold grows. **SRTD** is a symmetric variant. Our corrected
reference normalizes each dissimilarity matrix by its full-matrix 0.9 quantile,
reports persistence separately by degree, uses degree 1 for the published scalar,
and caps exact enumeration at 64 entities.

## 3. Related work

§1.2 and §2.5 cite the primary sources. In summary: RTD [1], RTD-AE [2], and
SRTD [3] define the divergence reference, with quotient homology [4] an adjacent
algebraic treatment; DCM [5] and DiffLift [6] are the closest precedents for
learning higher-order topological structure from graphs; Neural Sheaf
Diffusion [7] and Knowledge Sheaves [8] for learned sheaf transports; DeepMoE
[11], GMoE [12], GraphMETRO [13], and MvCGE [14] for per-example expert routing.

Categorical deep learning [9] argues architectures are usefully described as
algebras over a theory, and functor learning [10] gives an instance where the
learned object must preserve structure rather than merely fit data. Our annulus
model forms a softmax convex mixture of twelve fixed chain-map templates, so its
chain law is inherited from that basis; the continuous lifting experiment
instead optimizes an unconstrained matrix with a finite compatibility penalty.
§10 states plainly that neither validates a categorical claim.

Equality-constrained least squares is classical: Björck [15] surveys constrained
least-squares methods, and Eldén [16] develops a direct method for the
equality-constrained problem. Lim [17] places graph incidence operators, cycle
spaces, and Hodge Laplacians in a unified linear-algebraic treatment; Hoppe and
Schaub [18] connect edge flows and higher-order cell-complex structure. These
works explain why constraining or shrinking a linear estimator toward a correct
cycle subspace can reduce variance. Our contribution is not that mechanism; it
is the controlled evaluation and audit trail for deriving the soft penalty from
the observed input in this lifting generator.

## 4. Two settings

### 4.1 Selection over a fixed family

<figure>
  <img src="figures/architecture.svg" alt="Architecture and data flow of the identifiable typed-map experiment: a planted group element, a selector over all source node and edge features, a fixed twelve-map basis, its softmax convex mixture, and four scored outputs." width="680">
  <figcaption><strong>Figure 1. The selection setting.</strong> A group element is planted in a six-sector cellular annulus (12 vertices, 18 edges, 6 faces, Betti (1, 1, 0)). The flattened MLP consumes all source node and edge features; identifying markers carry the target identity among nuisance channels. No per-example target signals or labels enter selector inference, but the fixed annulus boundaries and twelve action templates are built into the hypothesis class, and target signals supervise the declared training objectives. During training the model forms a softmax convex mixture of the twelve fixed signed chain maps; hard evaluation uses the argmax vertex. Every mixture satisfies the chain law because every template does, so the residual is a numerical check rather than a learned objective.</figcaption>
</figure>

Twelve dihedral maps act on the annulus. Each is built as a **signed permutation
in every degree**: `F0` permutes vertices, `F1` permutes edges with an
orientation sign, `F2` is fixed by matching the mapped boundary to a unique
oriented face. We verify numerically that all twelve satisfy `Fᵀ F = I` at
degrees 0, 1, and 2.

The executed `IdentifiableTypedMapModel` encodes the complete source node and
edge feature tensors with a flattened MLP, predicts twelve logits, and uses
their softmax weights to linearly mix the fixed degree-wise templates. It does
not instantiate `ExactChainMapLayer` or a generic nullspace parameterization.
The fixed source/target boundaries and all twelve action templates are registered
inside the model; only per-example target signals and labels are absent from the
selector's forward inputs.
An argmax over the same logits supplies the hard decoded transformation used by
the discrete accuracy and exact-cone evaluations.

That property determines the exact hard-map certificate in this setting; it does
not determine the behavior of soft mixtures during training (§5).

### 4.2 Learned continuous edge-to-cycle lifting

We use a generator in which the lifting is learnable. Its design is forced by a
conflict: the routing benchmark used elsewhere in this project deliberately hides
cell structure from the graph observation so a router cannot shortcut, which
makes conversion impossible by construction. Routing needs targets *not*
inferable from the graph; conversion needs them *inferable*. One generator cannot
serve both, so this is a separate one.

- The **2-cells are a cycle basis of the graph**, so the cell complex is
  determined by the graph rather than drawn beside it. Graph size and density
  vary, so the cycle rank varies, and the lifting defect varies with it.
- **Face activity thresholds the circulation** of the edge cochain around each
  cycle — exactly `B2ᵀ x1`. Integrating an edge feature around a cycle is the
  operation the lifting must perform.
- **Sheaf transport** combines an endpoint frame difference with a per-edge
  twist. The frame difference telescopes to the identity around any closed cycle,
  so without the twist every holonomy would be trivially zero; the twist rides a
  separate edge channel so holonomy and face activity are not the same quantity
  twice.

**The learning task.** For each topology, fit a separate
`W: R^E → R^F` from edge signals to cycle-basis coordinates. Its transpose
`Wᵀ: R^F → R^E` is a candidate `d2`; `W` is degree-changing and is not a typed
chain map between complexes. `B2` is not supplied explicitly to the optimizer,
but paired responses generated as `Y = X B2 + epsilon` provide ordinary
supervised signal about it. There is no shared model that generalizes `W` to a
new topology. `B1` is observable because it is the graph. Ground truth is
`W = B2ᵀ`, a median of 242 free parameters across the 29
eligible topologies (range 30–770). The generator uses NetworkX
`cycle_basis`, whose basis coordinates are noncanonical. The
boundary-compatibility objective identifies the cycle **subspace**, not an
arbitrary choice of basis coordinates; paired targets supply the coordinate
convention for each fitted topology. The generator enforces `B1 B2 = 0`; the
degree-1 kernel of the canonical inclusion equals the graph's cycle rank.

This non-explicit provision is not an information-theoretic barrier. With the
deterministic generator algorithm known, the NetworkX basis `B2` is algorithmically
recoverable from the observed graph, and `B1` alone determines the target cycle
subspace `ker B1`. The penalty is therefore strong input-derived structural side
information, not direct use of `B2` or response labels. The unpenalized baseline
ignores `B1`. The historical campaign omits stronger graph-aware analytic
comparators: one could reconstruct the generator's cycle basis directly,
parameterize the rows of
`W` in `ker B1`, or project an estimator into that subspace with a Hodge/nullspace
operator. The sealed v2 replication of §6.4 adds exactly these comparators —
least squares restricted to `ker B1`, a dimension-matched seeded random
subspace, and training-only cross-validated ridge — while the generator's own
cycle basis appears only as a withheld-knowledge oracle ceiling (§7.2). The
historical contrast is against a graph-blind full-matrix baseline,
not against an optimal cycle-basis or Hodge oracle.

For each eligible seed, the executed estimation problem is

    X_train[i,:] ~ N(0, I_E)
    Y_train = X_train B2 + epsilon,   epsilon[i,:] ~ N(0, 0.02^2 I_F)
    X_test[i,:] ~ N(0, I_E),         Y_test = X_test B2
    W_hat_lambda = argmin_W mean((X_train W^T - Y_train)^2) + lambda R(W).

The optimizer is Adam for 2,500 steps at learning rate 0.05, initialized at
`W = 0` in float64. The held-out target is noiseless, so its MSE evaluates map
recovery. For the distance surrogate, `R(W)` forms the two `16 x 16` matrices
`cdist(X_train, X_train)` and
`cdist(X_train W^T, X_train W^T)`, including their zero diagonals; each matrix is
divided by its own full-matrix mean plus `1e-12`, and the elementwise squared
difference is averaged.

The sample geometry makes the regularization problem deliberately data-scarce.
With `N_train = 16`, the unconstrained row-wise linear system has `E > 16` in 21
of 29 eligible seeds. For a connected graph, each row in the compatibility-zero
set lies in the `F`-dimensional cycle space; `F <= 16` in 24 seeds. Sixteen seeds
therefore move from an underdetermined ambient system (`E > 16`) to a
potentially identifiable hard cycle-subspace system (`F <= 16`). Median
dimensions are `E = 23`, `F = 11`; five seeds still have `F > 16`. The executed finite
penalty does **not** perform that hard dimension reduction, but it shrinks toward
the lower-dimensional subspace. A large gain is therefore plausible as
scarce-probe system identification, without implying a general advantage in
well-sampled regimes.

## 5. Retrospective no-fit diagnostics for structural objectives

During the post-campaign audit we used two inexpensive checks that require no
training. They were not used prospectively to select the reported objectives or
predict the reported results. We propose them as pre-fit screens for future
structural objectives:

1. **Sample variability:** does the proposed quantity vary over a supplied
   candidate sample? No detected variation means that sample offers no evidence
   of discrimination; it does not prove class-wide constancy unless the sample
   exhausts the class.
2. **Truth-relative ordering:** is the known truth's score no higher than every
   supplied alternative? If not, the objective favors at least one supplied
   alternative over the truth. This does not locate a local or class-wide
   optimum.

These are diagnostic heuristics, not necessary or sufficient conditions for
better held-out prediction. In finite samples, even a target-misaligned
regularizer can improve generalization through a bias–variance tradeoff; a
target-aligned, nonconstant objective can still fail through optimization,
scaling, or model misspecification. The claims below therefore rest on the
algebraic proof for the annulus and the frozen empirical contrasts, not on the
screen alone.

### 5.1 The hard-map cone certificate is constant on annulus vertices

A mapping cone is acyclic exactly when its chain map is a quasi-isomorphism. Every
hard decoded candidate in §4.1 is a signed permutation, hence invertible, hence a
quasi-isomorphism. **Exact cone acyclicity therefore takes the same value on all
twelve decoded vertices.**

The generalization is broader than this annulus: **any hypothesis class of
invertible chain maps between fixed complexes has a constant cone-acyclicity
signal.** Acyclicity can certify that a candidate is a quasi-isomorphism; it
cannot select among candidates that already have that property.

This statement does **not** extend to the complete training trajectory. Training
uses `cone_soft_betti` on convex mixtures of the twelve maps; those mixtures
need not be invertible and the differentiable proxy can vary and provide
gradients. The RTD-style annulus loss is also not constant: it compares
cross-example distances of predicted and target typed representations, the
selectors vary by example, and soft mixtures are non-orthogonal. It consumes
target batch structure. Accordingly, the empirical chance accuracies in §7.5
cannot be deduced from a constancy theorem for either soft objective.

### 5.2 The continuous surrogates are target-misaligned

In §4.2 the truth `W = B2ᵀ` has full row rank. Nevertheless,
`exp(−2·σ_min(W))` decreases monotonically as the smallest singular value
grows, so its optimum is not the finite ground-truth scale: it continually
rewards singular-value inflation. The term therefore imposes a directional bias.
The frozen experiment establishes that this bias harmed held-out error at the
tested weight; the analytic observation alone would not have proved harm.

The RTD-inspired normalized pairwise-distance term has a different mismatch. It
asks the source edge vectors `X` and lifted vectors `X Wᵀ` to preserve normalized
pairwise geometry. The truth `W = B2ᵀ` is intentionally lossy: because `F < E`,
it discards cut-space components and cannot preserve the full source geometry.
The surrogate therefore opposes a defining property of the target lifting. Its
no-detected-improvement result concerns this target-misaligned surrogate only and
says nothing negative about published RTD or SRTD.

### 5.3 Operational screen

`screen_structural_term` evaluates a candidate term on a known truth and a
supplied candidate sample, returning the operational labels
`truth-no-higher-than-supplied-candidates-and-varies`,
`truth-higher-than-a-supplied-candidate`, or
`constant-over-supplied-candidates`. The labels are sample-relative prompts for
inspection, not theorems about the full hypothesis class or generalization. The
exact annulus class-wide constancy result is instead established analytically;
all effect-size and held-out-performance claims come from the reported campaigns.

## 6. Experimental design

### 6.1 Prospectively locked same-family replication

The historically named conversion campaign for the edge-to-cycle lifting was frozen in a protocol document committed **before
execution**: SHA-256
`503cc282f40d118ba1739c2afe1bfc77eaf2b1733baaddb91c0c3363e75ae2b8`, committed at
`d5d18af`, campaign run at `11644c6` from a clean worktree. No endpoint,
weight, decision rule, or sample size changed after the freeze. One objective
implementation did: the protocol writes the squared Frobenius norm
`‖B1 Wᵀ‖²`, while the runner executed the elementwise mean
`mean((B1 Wᵀ)²)`. Because the number of entries varies by topology, this is not
one global rescaling. The paper reports the executed objective and treats the
result as a prospectively specified analysis with a disclosed implementation
deviation, not a pristine protocol replication or independent confirmation.

The design was locked prospectively only relative to this run. Protocol 27 says
that the hypotheses, effect directions, endpoint, training-set size, and tested
weights were chosen after exploratory results on the same generator family.
Document 26 does not retain the exploratory seed identities, so possible overlap
with the frozen seed block 20261001–20261030 cannot be audited. Prospective
locking prevents changes after the freeze; it does not remove this
outcome-informed design history. Independent confirmation requires an untouched,
disjoint generator family or seed block. §6.4 reports that next step: a sealed
replication on an untouched, disjoint seed block of the same generator family.
Its design remains outcome-informed, so it is still not independent-lab or
independent-generator confirmation; what the untouched block removes is any
selection on the new outcomes.

| item | value |
|---|---|
| declared topologies | 30 (seeds 20261001–20261030) |
| eligible | **29**; seed 20261025 had fewer than three faces and was skipped, **not replaced** |
| training pairs | 16 |
| held-out pairs | 3072 |
| training-label noise | Gaussian, sigma 0.02 |
| held-out target | noiseless `B2ᵀ x` |
| primary statistical unit | eligible generator seed: joint topology, data, and noise realization (`n = 29`) |
| optimiser | Adam, lr 0.05, 2500 steps, `W` initialised to zeros, float64 |
| weights | `exact` 3.0, `cone` 0.01, `rtd` 0.1 |

Weights were selected after exploratory work and then frozen before this run.
The publication bundle does not retain a machine-verifiable record of every
exploratory candidate or the exploratory seeds, so it does not claim that these
are globally or even exhaustively “best” weights or that exploratory and frozen
seed sets were disjoint. Weights are not comparable across terms because the
terms have different scales. The keys `exact`, `cone`, and `rtd` are frozen
historical labels defined and qualified in §2.3.

### 6.2 Endpoints and decisions

Primary endpoint is the paired quantity, one value per eligible generator seed:

    d = log10( held-out MSE with term / held-out MSE without )

Negative means the term improves the lifting. Each seed jointly determines the
topology, training and evaluation inputs, and training noise; the penalized and
unpenalized fits share that realization. Pairing therefore controls the shared
realization in each contrast but does not remove topology variance. The estimand
is the mean paired log ratio over eligible generator seeds, and the Student-t
interval treats the 29 seed-level replicates as exchangeable. Training responses
contain Gaussian noise with sigma 0.02, whereas held-out targets are noiseless,
so held-out MSE measures recovery of the planted linear lifting rather than
prediction of fresh response noise.

**Multiplicity is adjusted.** The three primary contrasts form one family; the
governing interval is Bonferroni-corrected to two-sided 98.333%, giving a
family-wise error rate of 0.05. Unadjusted 95% intervals are reported alongside
and the adjusted interval governs. An exact two-sided sign test is reported as
distribution-free sensitivity and governs nothing.

H4 and H5 were prespecified but are **secondary analyses outside this
multiplicity-controlled family**. H4 has an unadjusted Student-t 95% interval;
H5 has no valid inferential interval or decision. H4 measures the mean
within-seed Pearson correlation, over nine fixed weights for
the historically named `exact` objective, between
`log10(max(||B1 Wᵀ||_F, 1e-30))` and
`log10(max(held-out MSE, 1e-300))`. The reported defect is a
Frobenius norm; it is distinct from the elementwise mean-square compatibility
term used during training. All nine fits within a seed reuse
the same generated data and initialization and differ in the common driver,
penalty weight `lambda`. They are points on one deterministic regularization
path, not nine independent observations.

The frozen H5 endpoint contains a design error discovered during audit. It
divides the selected route's error by the per-row minimum of those same two
route errors. Consequently every ratio is at least one and every `log10` ratio
is nonnegative, while the decision rule required an interval strictly below
zero. Support was impossible by construction. We preserve the result for
transparency but draw **no inferential routing conclusion** from H5.

The campaign ran once over all declared topologies. No interim analysis, no early
stop, no extension.

### 6.3 Hardware, software, provenance

Campaign: CPU, float64, PyTorch 2.13.0+cu130, Python 3.12.3. The annulus campaign
and all timing benchmarks ran on one NVIDIA GB10, Linux 6.17.0-1026-nvidia
aarch64, CUDA 13.0, deterministic algorithms enabled with
`CUBLAS_WORKSPACE_CONFIG=:4096:8`.

| item | value |
|---|---|
| annulus campaign commit | `8021292e97abfec91768f1b5437c883a42c29c60` |
| annulus launch fingerprint | `44408d7adf8467e594879b46e25a1cb7fd89a7e7a5d5f3446548bcbf3ed1096e` |
| annulus strict summary SHA-256 | `0cd0defb0b0d41b5f7563c364cfdda62cb72c5e5a845bdd5d1ab76a2e1cb953c` |
| routing campaign commit | `e69b07707950b6abe332366c51fe8c94254899f3` |
| scheduler steps completed | 56 of 56 |

The sealed scheduler receipt lists all 296 produced files with byte counts and
SHA-256 digests; all 296 verify against the on-disk artifacts.

### 6.4 Sealed untouched-seed replication (v2)

The follow-up demanded above — an untouched, disjoint seed block of the same
generator family — is complete. The design was frozen in protocol
[`docs/31`](31-independent-lifting-replication-protocol.md) with a dedicated
runner (`scripts/run_lifting_replication_v2.py`) and tests, committed as design
commit A; a machine-readable design seal
([`docs/32`](32-independent-lifting-replication-seal.json)) was committed
immediately after as commit B and pushed to the private remote (timestamp
2026-08-24 22:31:13 -0400) before any declared seed was instantiated. At
preflight the runner re-verified the seal schema, every embedded hash against
both its frozen constants and the actual file bytes, its own runtime SHA-256,
the seal's presence at HEAD, and a clean worktree. No design file changed
between commit A and execution; the recorded execution revision equals commit
B. The canonical command was

```bash
env CUDA_VISIBLE_DEVICES=-1 .venv/bin/python \
  scripts/run_lifting_replication_v2.py \
  --output results/campaigns/lifting-replication-v2.json
```

| item | value |
|---|---|
| protocol | `docs/31`, SHA-256 `6288eade4755aa188299760303b389ce13acd42659b8b9bc340cb9d4024afec0` |
| design commit (A) | `044322c7dc6a6255eec941dbcb76c45288a9666c` |
| seal / execution commit (B) | `9baae6b8322120724e7f5aff3c47fd7ef343086c` |
| result | `results/campaigns/lifting-replication-v2.json`, committed at `64a1c3bf6b5f824fb5990392f5efb08bf36559b9` |
| declared seeds | 36 (`20270101..20270136`) |
| eligible seeds | **33**, meeting the frozen minimum of 30; `20270103`, `20270116`, `20270120` generated two faces each, were retained, and were **never replaced** |
| generation failures | 0 |
| training pairs | 16 |
| held-out pairs | 3072 |
| training-label noise | Gaussian, sigma 0.02 |
| held-out target | noiseless `B2ᵀ x` |
| environment | CPU, float64, single torch thread, CUDA hidden; Python 3.12.3, PyTorch 2.13.0 (build `2.13.0+cu130`), NetworkX 3.6.1, NumPy 2.5.2; Linux aarch64; recorded environment matched the frozen requirement exactly |

Nine fitted arms enter the seven-claim family: two graph-blind references
(finite-step Adam; minimum-norm least squares of the same unpenalized
objective), the soft boundary penalty at frozen weight 3.0 in finite-step Adam
and in closed form, exact least squares restricted to `ker B1`, least squares
restricted to a dimension-matched seeded random subspace, training-only
four-fold-selected ridge, and the historical singular-value (weight 0.01) and
RTD-inspired distance surrogates. A tenth arm sets the map to the withheld
generator cycle basis; it is an attainability ceiling and enters no contrast.

The endpoint of every claim is the paired
`log10(MSE_arm / MSE_reference)` within a seed; the estimand is its mean over
eligible seeds, the conditional synthetic-generator quantity
`E_seed[log10(MSE_arm/MSE_reference) | connected, F>=3]`. Decisions use
one-sided Student-t bounds at Bonferroni `alpha = 0.05/7` with `n = 33`, df 32,
critical value `2.5912722991315227`. Sign tests are direction-neutral
sensitivity analyses and never govern support. One eligible generator seed is
the sole unit of inference: it jointly determines topology and, through a
frozen SHA-256 sub-seed schedule, the training predictors, label noise, test
predictors, and random specificity subspace. A prespecified off-path secondary
analysis (C1) sits outside the Bonferroni family and receives no support
decision.

This is an **untouched-seed, outcome-informed, same-generator-family
replication**. The seed block was never instantiated before the seal, but the
hypotheses, arms, directions, and weights were chosen after the v1 outcomes and
a hostile post-campaign diagnostic on the old seeds were known. It is not an
independent-lab or independent-generator replication and not a pristine
preregistration. V1 and v2 estimates are never pooled, meta-analyzed, or
jointly interval-estimated. After execution, every primary summary was
independently recomputed from the retained raw rows (maximum absolute
discrepancy at most `1e-13`, summation-order roundoff), and an independent
re-derivation reproduced every estimate, bound, and support flag to
`4.45e-16`. The retention and enforcement gaps found by that validation are
disclosed in §9 and in the full result record,
[`docs/33`](33-lifting-replication-v2-results.md).

## 7. Results

### 7.1 Boundary compatibility improves an edge-to-cycle lifting

| objective | weight | 95% interval | Bonferroni 98.33% | sign test | decision |
|---|---:|---|---|---:|---|
| boundary compatibility (`exact`) | 3.0 | [−2.628, −1.632] | **[−2.749, −1.511]** | <1e-6 | **improves** |
| singular-value cone surrogate (`cone`) | 0.01 | [+0.125, +0.254] | **[+0.109, +0.270]** | <1e-6 | **harms** |
| RTD-inspired distance surrogate (`rtd`) | 0.1 | [+0.002, +0.034] | [−0.002, +0.038] | 0.458 | **no detected improvement** |

<figure>
  <img src="figures/fig-campaign.svg" alt="Forest plot of three structural objectives. The boundary-compatibility interval lies below zero, the singular-value surrogate interval lies above zero, and the RTD-inspired distance-surrogate interval crosses zero." width="680">
  <figcaption><strong>Figure 2. Three structural objectives under one protocol.</strong> Paired <code>log10</code>(held-out error with term / without), one value per eligible generator seed, 29 joint topology/data/noise realizations. The thick bar is the Bonferroni-adjusted 98.33% interval governing the primary decision; the thin bar is the unadjusted 95% interval. Boundary compatibility improves held-out map recovery; the singular-value surrogate harms it; the RTD-inspired distance surrogate shows no detected improvement. The short labels in the graphic reproduce frozen campaign keys and do not identify <code>exact</code> with sequence exactness, <code>cone</code> with mapping-cone homology, or <code>rtd</code> with published RTD. Generated from <code>results/campaigns/conversion-campaign-v1-corrected.json</code>.</figcaption>
</figure>

Median `log10` ratio for boundary compatibility is −2.858, roughly a
**700-fold** reduction in held-out error; the mean is −2.130. The adjusted
interval lies far from zero.

The executed term is `mean((W B1ᵀ)²)`, which is zero at the truth because
`B2ᵀ B1ᵀ = (B1 B2)ᵀ = 0`. It does not directly use `B2` or response labels, but
it uses `B1`, which determines the target cycle subspace; the unpenalized
baseline ignores this strong structural side information. With the generator
algorithm known, `B2` is algorithmically recoverable from the graph, and no
analytic cycle-basis, nullspace, or Hodge-projection oracle was included.
Training still optimizes all `F×E` entries of `W`; weight 3.0 penalizes but does
not prohibit off-cycle components. §7.2 reports the sealed untouched-seed
replication that adds the omitted classical comparators.

### 7.2 Sealed untouched-seed replication: the exact cycle constraint outperforms soft shrinkage

All seven prespecified claims of §6.4 were decided once, on the sealed block,
under the frozen decision rules. The endpoint is the paired `log10` ratio of
held-out MSE (numerator arm / reference arm); negative estimates mean the
numerator arm improves recovery. [`docs/33`](33-lifting-replication-v2-results.md)
is the full record; Figure 3 summarizes it.

Exact values, standard errors, and bounds at full precision are recorded in
[`docs/33`](33-lifting-replication-v2-results.md); the table rounds to four
decimals.

| id | endpoint (numerator / reference) | estimate | SE | one-sided bound | support |
|---|---|---:|---:|---:|---|
| h1-soft-vs-ambient-adam | `soft_boundary_lambda3 / ambient_adam` | −1.8726 | 0.2399 | upper −1.2509 | **supported** |
| h2-hard-cycle-vs-ambient-ls | `hard_cycle_ls / ambient_min_norm_ls` | −1.8769 | 0.2120 | upper −1.3275 | **supported** |
| h3-hard-cycle-vs-soft-closed-form | `hard_cycle_ls / soft_boundary_closed_form_lambda3` | −0.1274 | 0.0305 | upper −0.0484 | **supported** |
| h4-hard-cycle-vs-hard-random | `hard_cycle_ls / hard_random_subspace_ls` | −3.2319 | 0.2547 | upper −2.5720 | **supported** |
| h5-ridge-vs-ambient-ls | `inner_cv_ridge / ambient_min_norm_ls` | +0.0152 | 0.0297 | upper +0.0922 | **not supported** |
| h6-singular-surrogate-harm | `singular_value_surrogate / ambient_adam` | +0.1906 | 0.0404 | lower +0.0858 | **supported** |
| h7-rtd-bounded-benefit-futility | `rtd_inspired_distance_surrogate / ambient_adam` | +0.0188 | 0.0087 | lower −0.0038 vs margin −0.0458 | **supported** |

<figure>
  <img src="figures/fig-replication.svg" alt="Forest plot of the seven frozen claims of the untouched-seed replication. Six claims are supported: the soft penalty and the exact cycle constraint each improve on their graph-blind references, the exact constraint improves on the closed-form soft solution and on a dimension-matched random subspace, the singular-value surrogate harms, and the RTD-inspired surrogate is bounded below its futility margin. The ridge claim is not supported. A muted square marks the descriptive optimizer diagnostic." width="680">
  <figcaption><strong>Figure 3. The seven-claim family of the sealed untouched-seed replication.</strong> Point: mean paired <code>log10</code>(held-out MSE, numerator arm / reference arm) over 33 eligible generator seeds; whisker: the governing one-sided Bonferroni bound in the claim's direction (familywise <code>alpha = 0.05/7</code>, critical value 2.5912722991315227). H5 (ridge) is not supported. The dashed marker on H7 is the prespecified futility margin <code>log10(0.90)</code>; H7 is a bounded-benefit/futility statement only and cannot establish equality or the absence of every benefit. The muted square below the divider is the descriptive optimizer diagnostic (ambient Adam / ambient minimum-norm LS), which carries no support decision. Same-generator-family replication on synthetic seeds, not real-data validation. Generated from <code>results/campaigns/lifting-replication-v2.json</code>.</figcaption>
</figure>

**H1 replicates the soft-penalty effect on untouched seeds.** The soft
boundary-compatibility penalty at frozen weight 3.0 improves over the
graph-blind ambient Adam reference on the disjoint block: geometric mean MSE
ratio `0.01340985679332218`, descriptive two-sided 95% interval
`[−2.36129485375057, −1.3838568663407198]`, sign test 33 negative of 33
(p = `2.3283064365386963e-10`).

**H2 improves on the solver-matched graph-blind baseline.** Exact least squares
restricted to `ker B1` improves over the ambient minimum-norm least-squares
solution of the same unpenalized objective: geometric mean ratio
`0.013277071629811591`, descriptive interval
`[−2.3087619478822723, −1.4450334555558149]`, sign test 33 of 33.

**H3 is the central comparison.** Hard cycle-subspace least squares improves
over the closed-form solution of the same frozen soft objective, with optimizer
confounding removed by construction: geometric mean ratio
`0.745769272372031` — real but modest — descriptive interval
`[−0.18951209171471245, −0.06527893761520454]`, sign test 28 negative, 3
positive, 2 ties discarded (p = `4.649162292480469e-06`). The mechanism is
dimensional: on the 10 eligible seeds with cycle-feature dimension at least the
training budget (`F >= N_train = 16`) the two solutions are effectively tied
(|log10 ratio| at most `7.93e-13`), and the advantage is carried by the 23
lower-dimensional seeds (mean log10 ratio `−0.18278486886714318`); a
tolerance-aware sign analysis treating |log10 ratio| `<= 1e-10` as a tie remains
strongly positive (22 negative, 1 positive, 10 ties; p =
`5.7220458984375e-06`). Under the frozen
design the exact classical constraint is preferable to soft shrinkage of the
same `B1`-derived information, and the payoff concentrates exactly where the
cycle subspace is smaller than the training budget. The soft penalty is a differentiable
approximation to that constraint, not a new superior method, and constrained
least squares is classical (§3), not a novel algorithm.

**H4 establishes specificity against the prespecified control.** The true cycle
subspace improves over the dimension-matched seeded random subspace: geometric
mean ratio `0.0005862327707089732`, descriptive interval
`[−3.7506683788961146, −2.7131914369786294]`, sign test 33 of 33. The exact
constraint thus outperformed the prespecified dimension-matched random-subspace
control; a single such control cannot exclude every alternative subspace family,
so we do not claim a more general specificity.

**H5 is not supported, exactly as frozen.** Training-only four-fold-selected
ridge did not improve over ambient minimum-norm least squares: the governing
one-sided upper bound `+0.09221008861621441` is not below zero (geometric mean
ratio `1.0355395685149351`; sign test 19 positive, 14 negative,
p = `0.48685024166479707`). The claim is reported as frozen and is not
reinterpreted; this null does not show that ridge cannot help under any design,
only that no benefit was detected here.

**H6 replicates the singular-value surrogate's harm.** The one-sided lower
bound `+0.0858061772078059` exceeds zero (geometric mean ratio
`1.5508000886966786`; sign test 29 positive, 4 negative,
p = `1.0928604751825333e-05`). The claim is scoped to the exact implemented
formula, not mapping-cone homology.

**H7 is supported as a bounded-benefit/futility statement.** The one-sided
lower bound `−0.003807143571896345` exceeds the prespecified margin
`log10(0.90) = −0.045757490560675115`, ruling out an RTD-inspired benefit of
10% or more in geometric-mean held-out MSE versus ambient Adam. This is **not**
a noninferiority or equivalence conclusion — the direction and estimand
semantics are reversed relative to those procedures — and it cannot establish
equality or the absence of every benefit. The claim is scoped to this
target-misaligned surrogate, not published RTD/SRTD.

**Descriptive optimizer diagnostics** (prespecified, outside the family, no
support decision). Ambient Adam is measurably worse than the minimum-norm
least-squares solution of the same unpenalized objective: mean log10 ratio
`+0.22888064836548128` (median `+0.2811655771640295`), exact sign test 25
positive, 8 negative, p = `0.004551384132355452`. Finite-step Adam also does
not fully reach the closed-form optimum of the penalized objective: soft Adam
versus its closed form averages `+0.10580697537392178`, with per-seed solution
gap `||A_adam − A_closed_form||_F` mean `0.8556475807071382` (maximum
`4.005285286298402`). Part of the H1 margin is therefore optimizer slack rather
than penalty geometry; H3 removes this confound. This solver gap is why the
soft penalty's reference is Adam while the classical estimators are referenced
to the same-solver least-squares solution.

**C1 (prespecified secondary, off-path, no decision).** Per eligible topology,
twelve independent training-input/label-noise replicates of the minimum-norm
estimator share one independently generated 3,072-row noiseless test set. The
cycle-projector defect covaries positively with held-out error across these
replicates — mean Fisher-z `+0.7712258159811779`, unadjusted 95% interval
`[+0.43398720808657104, +1.1084644238757848]`, back-transformed mean r
`+0.6476416755579272` — while the dimension-matched random-subspace defect does
not (mean z `−0.3168499119888693`, interval
`[−0.5622502040999926, −0.07144961987774598]`). The relevant descriptive
contrast is the paired delta-z: `+1.0880757279700473`, interval
`[+0.8873984853168706, +1.288752970623224]`, lying above zero. Two positive
correlations alone would not establish specificity; the paired contrast is what
separates the projectors. Unlike the historical path correlation of §7.3, these
replicates share no common penalty-weight driver. This remains association, not
causation: the analysis does not establish that either defect is independently
calibrated, causal, or useful for routing, and it says nothing about real data.

**The generator-cycle-basis oracle is a withheld-knowledge ceiling.** Setting
the map to the withheld generator basis yields held-out MSE, and error relative
to the mean squared test target, of exactly `0.0` on all 33 eligible seeds. It
is a deterministic attainability ceiling and numerical-integrity check — it
shows how much generator knowledge the learning problem withholds — and it is
isolated from every fitted arm and inferential table. It is not evidence that
any learned method discovered the generator basis.

### 7.3 Historical secondary analysis: covariation along the compatibility-penalty path

Within each topology, nine prespecified weights on the compatibility penalty
produce learned maps of varying quality. The endpoint is the correlation between
a map's boundary-compatibility defect and its held-out error along that path.

| quantity | value |
|---|---|
| mean within-seed correlation | **+0.961** |
| 95% interval | **[+0.935, +0.987]** |
| eligible seeds with positive correlation | **29 / 29** |

This prespecified analysis is outside the three-contrast multiplicity family, so
its 95% interval is unadjusted. The same data and initialization are reused for
the nine fits, and `lambda` is a common driver of both compatibility defect and
held-out error. The nine path points are therefore not independent observations.
The result establishes covariation along this regularization path; it does not
show that defect carries independent predictive information, calibrates error
off path or on unseen topologies, isolates an intervention effect, or is causal.

### 7.4 The singular-value surrogate harms; the distance surrogate does not improve (historical campaign)

Both belong to the multiplicity-controlled primary family. The singular-value
cone surrogate harms held-out error at the tested weight: its adjusted interval
lies entirely above zero. For the RTD-inspired normalized pairwise-distance
surrogate, the adjusted interval contains zero and the sign test is 0.458. This
supports **no detected improvement**, not equivalence, inertness, or a general
claim about published RTD/SRTD; no equivalence margin was prespecified. The
sealed replication strengthens both: H6 (§7.2) replicates the harm, and H7
upgrades the distance surrogate's null to a bounded-benefit/futility result at
a prespecified margin.

§5.2 identifies the singular-value surrogate's scale-inflating bias; the frozen
contrast establishes that it harmed here. §5.1 proves a different fact in the
selection setting: exact cone acyclicity is constant across that finite class.

### 7.5 The selection setting, and why its nulls are weak

On the annulus, all six objectives containing task or reconstruction supervision
decode the planted transformation perfectly: transformation and cell-face
accuracy are 1.000 on all five seeds. Their mean map MSE spans
`2.618e-17` to `2.504e-8`. The engineering recovery gate applies only to
`task_reconstruction` and `combined` — five seeds each — and passes **10 of 10**
applicable runs.

<figure>
  <img src="figures/fig-recovery.svg" alt="Two-panel bar chart. Left: transformation accuracy by objective, six task- or reconstruction-supervised objectives at 1.000 and cone-only and RTD-only at the 0.0833 chance line. Right: map mean-squared error on a log scale; supervised means span 2.618e-17 to 2.504e-8, while cone-only and RTD-only are 0.109 and 0.191." width="680">
  <figcaption><strong>Figure 4. Recovery by objective in the selection setting.</strong> Mean over five seeds. All six objectives carrying task or reconstruction supervision have perfect decoded accuracy, with mean map MSE from <code>2.618e-17</code> to <code>2.504e-8</code>. The cone-only and RTD-only controls sit at chance with mean map MSE 0.109 and 0.191, about 7–16 orders of magnitude above the supervised range depending on the comparison. The RTD-only loss consumes target batch geometry and is not an unsupervised control. Generated from <code>results/summaries/identifiable-campaign-summary.json</code>.</figcaption>
</figure>

All 21 declared continuous contrast intervals contain zero, so adding the
annulus cone proxy or RTD-style training surrogate changed nothing. **That null
is weak evidence**: an analytic marker decoder also reaches 1.000, so the ceiling
was attainable without learning and no candidate could improve on a control
already at the maximum.

The two single-loss controls are descriptive negative results. `cone_only` reaches
transformation accuracy 0.0815 and `rtd_only` 0.0833 against a 0.0833 chance
baseline — **while producing acyclic cones in 6,000 of 6,000 evaluated examples
each**. This shows that the exact hard-map certificate is satisfied by all
decoded predictions while identification remains at chance. It does not show
that the differentiable cone proxy was constant during training, and it supplies
no analogous constancy proof for the RTD-style loss.

### 7.6 Specificity: the cycle subspace against the prespecified controls

Earlier exploratory notes report ridge, random-subspace, and nonlinear-link
checks, but no generating script or machine-readable result for those analyses
is preserved in the publication evidence. Their quantitative values are
therefore excluded from this paper. The frozen, machine-recorded
matched-control campaign this question required is the sealed v2 replication of
§7.2, and it resolves the question in both directions. H4 establishes that the
true cycle subspace outperformed the prespecified dimension-matched
random-subspace control; one such control cannot exclude every alternative
subspace family, so we do not claim a more general specificity. H5 — frozen,
and not supported — finds no detected benefit from training-only
four-fold-selected ridge over ambient minimum-norm least squares under this
design. H3 locates the remaining margin: the exact cycle constraint improves on
even the closed-form soft penalty, by a modest geometric-mean ratio of
`0.745769272372031`, concentrated in the 23 seeds whose cycle-feature dimension
is below the training budget. The historical campaign's improvement over the
unpenalized fit was therefore not merely generic shrinkage, and its soft penalty
is not the best use of the same `B1`-derived information.

### 7.7 Routing: a flawed accuracy endpoint and a separate compute result

The prespecified routing H5 accuracy endpoint is **non-informative because its decision
rule is impossible to satisfy**. It divides the chosen route's error by the
per-row minimum of the cell and graph errors, so every ratio is at least one and
every log ratio is nonnegative. An interval strictly below zero cannot occur.

The original summary had a second flaw: its 28 rows are two tested weights for
each of only 14 topology clusters. Treating them as 28 independent observations
and using a Student-t interval with 27 degrees of freedom is pseudoreplication.
The formerly reported interval and 25-of-28 selection count are withdrawn from
inferential use. The corrected record reports the historical row-naive interval
`[−0.111, +0.382]` and a topology-clustered descriptive interval
`[−0.123, +0.393]`; neither is inferential, and H5 receives no decision. A
redesigned, cluster-aware endpoint is required.

A separate routing result, from a different experiment, is retained as a
descriptive compute comparison rather than an accuracy claim.

<figure>
  <img src="figures/fig-compute.svg" alt="Horizontal bar chart of median inference latency for the routed path, three single fixed routes, and the dense three-expert path, with whiskers marking mean p95." width="680">
  <figcaption><strong>Figure 5. Trained routing inference latency on GB10.</strong> Median over 100 timed iterations averaged across five seeds; whiskers mark mean p95. Batch 64, bfloat16. All paths were timed in the plotted order inside one process, so residual thermal or allocator drift is confounded with path order. Generated from <code>results/summaries/compute-campaign.json</code>.</figcaption>
</figure>

Across five seeds, the mean dense/routed speed ratio is **1.532** (Student-t 95%
CI **[1.489, 1.575]**), and the mean routed/fastest-fixed latency ratio is
**2.269** (95% CI **[2.215, 2.322]**). Routed peak allocated memory is below
dense in every seed. The defensible statement is that routing saves compute
against dense evaluation, not that it saves compute overall and not that a
measured defect selects the view.

An earlier five-seed campaign descriptively found a hard-minus-best-fixed
accuracy margin of +0.1098, 95% interval [+0.0953, +0.1243]. That result used privileged
latent-regime distillation and had target structured views available at
inference, and a disclosed pre-freeze procedural deviation makes it
protocol-aligned rather than pristine preregistration. It is a different
experiment from the defect-based routing tested here and the two must not be
conflated.

### 7.8 Corruption diagnostics

<figure>
  <img src="figures/fig-contrasts.svg" alt="Forest plot of twelve corruption contrasts with 95% intervals; every interval crosses the zero line." width="680">
  <figcaption><strong>Figure 6. Every corruption contrast interval contains zero.</strong> Nine Gate-3 base contrasts use a paired complete-block bootstrap conditional on a fixed checkpoint pair; three gauge contrasts use a Student-t interval with df = 7 across eight training seeds. No multiplicity adjustment. Generated from <code>results/gate3/paired_comparison_final.json</code> and <code>results/summaries/gauge-corruption-campaign.json</code>.</figcaption>
</figure>

These compare clean and corrupted **fixed-expert embeddings**. They never invoke a
translator or learned map and therefore cannot support or refute any conversion
claim. All twelve intervals contain zero.

## 8. Discussion

**The precise objective matters.** In a sealed untouched-seed, outcome-informed,
same-generator-family replication, exact least squares restricted to the
graph's cycle kernel improves held-out recovery of an edge-to-cycle lifting
over both the graph-blind minimum-norm baseline and the closed-form soft
boundary penalty at its frozen weight; the soft penalty's improvement over a
graph-blind Adam reference replicates from the historical campaign; a
singular-value cone surrogate's harm replicates; and an RTD-inspired normalized
pairwise-distance surrogate is bounded below a 10% geometric-mean benefit.
Reporting only that
"homological structure helps" or "does not help" would erase the distinctions
among an exact classical constraint, a boundary-of-boundary penalty, and two
motivated but nonidentical surrogates.

**Retrospective no-fit inspection is useful but not dispositive.** §5's checks expose the
hard-map cone constancy result and the singular-value surrogate's scale-inflating
bias. They guide experimental design; they neither predict an effect size nor
replace held-out evaluation.

**Acyclicity is not correctness.** An acyclic mapping cone certifies that a chain
map is a quasi-isomorphism. In the annulus class every candidate is already an
invertible chain map, so that certificate is constant; a model trained on the
cone proxy alone can satisfy the certificate while failing to identify the
planted map.

**The cone bundles what the kernel and cokernel separate.** `dim H_n(cone) =
dim coker H_n + dim ker H_(n−1)` — the cone sums a directional pair and loses the
distinction between *destroyed* and *unreachable*. Where a measurement is wanted,
the unbundled pair is strictly more informative.

**The cycle-subspace information, not the penalty form, carries the gain.** The
compatibility penalty's zero set has every row of `W` in the cycle space, which has
dimension `F` for these connected topologies. The executed finite-weight
regularizer merely shrinks off-cycle components; it does not hard-constrain the
optimizer, reduce the parameter count from `F×E`, or prove exactness. The
replication's matched controls isolate what matters: kernel-restricted least
squares improves over the closed-form solution of the same soft objective
(modest geometric-mean ratio `0.745769272372031`), a dimension-matched random
subspace is worse by orders of magnitude, and training-only cross-validated
ridge shows no detected benefit over the minimum-norm baseline. The exact
constraint is classical constrained least squares, not a new algorithm; the
soft penalty is a differentiable approximation to it, not a superior method.
The withheld generator-basis oracle still attains exactly zero error on every
eligible seed, so no fitted arm reaches generator knowledge.

**The soft penalty's finite-step form is optimizer-confounded, and the design
measures how much.** Ambient Adam is measurably worse than the minimum-norm
least-squares solution of the same unpenalized objective (descriptive mean
log10 ratio `+0.22888064836548128`; sign test 25 positive, 8 negative,
p = `0.004551384132355452`), and finite-step Adam does not reach the
closed-form penalized optimum (per-seed solution gap mean
`0.8556475807071382`). Part of the soft penalty's margin over Adam is therefore
optimizer slack rather than penalty geometry; the closed-form contrast of H3
removes that confound, and the exact constraint still wins it.

**The effect occurs in a favorable scarce-probe geometry.** With only 16 probes,
21 of 29 ambient systems have `E > 16`; the hard cycle-subspace dimension is at
most 16 in 24 seeds, and 16 seeds cross from `E > 16` to `F <= 16`. The finite
penalty does not literally reduce the fitted `F x E` parameter count, but its
shrinkage targets the subspace that a hard graph-aware estimator would use. The
sealed replication shares this frozen geometry (`N_train = 16`, training noise
sigma 0.02, noiseless held-out targets) on 33 further eligible seeds of the
same family, and its hard-constraint arm performs exactly that dimension
reduction. The
large observed effect is therefore consistent with classical variance reduction
in underdetermined system identification.

## 9. Limitations

- **One synthetic linear lifting family.** 29 eligible generator seeds in the
  historical campaign and 33 in the sealed v2 replication, one training-set
  size (`N_train = 16`), one noise level (`sigma = 0.02`), and one soft-penalty
  weight (`lambda = 3.0`). Nothing
  transfers automatically to real data, unseen generator families, unseen
  topologies, or nonlinear transformations.
- **Seed-level inference.** Each primary replicate jointly instantiates topology,
  data, and training noise. Intervals assume the eligible seed-level paired
  effects (29 historical, 33 in v2) are exchangeable; pairing controls their
  shared realizations but does not remove topology variance. V2 inference
  quantifies Monte Carlo variation over the frozen seed mechanism, not
  uncertainty for real data.
- **No unseen-topology learning.** A separate `W` is fitted and evaluated
  within each topology. Neither campaign learns a shared converter that
  generalizes to a new graph.
- **Noncanonical target coordinates.** NetworkX chooses one cycle basis among
  many. The penalty identifies the cycle subspace; paired labels identify that
  run's arbitrary basis coordinates.
- **Strong structural side information and a withheld-knowledge oracle.** Although the
  penalty does not directly use `B2` or response labels, `B1` determines the
  target cycle subspace, and the known deterministic generator makes `B2`
  algorithmically recoverable from the graph. The historical baseline ignores
  `B1`. The v2 replication adds the kernel-restricted least-squares estimator,
  a dimension-matched random-subspace control, and training-only ridge, and the
  exact constraint wins; the exact generator-basis reconstruction appears only
  as a withheld-knowledge oracle whose error is exactly zero on all 33 eligible
  seeds and which no fitted arm attains.
- **Favorable scarce-probe geometry.** `N_train = 16`; 21/29 seeds have
  `E > 16`, 24/29 have `F <= 16`, and 16/29 cross from the former ambient
  underdetermination to the latter hard-subspace dimension. Medians are
  `E = 23`, `F = 11`, while five seeds still have `F > 16`. The executed finite
  penalty is shrinkage, not a hard dimension reduction, and the effect may
  diminish with more probes. The v2 block shares the frozen geometry; its
  ineligible-seed dimensions are disclosed in docs/33.
- **Protocol implementation deviation.** The runner used an elementwise mean
  where frozen protocol 27 wrote a Frobenius sum. The result is not a pristine
  protocol replication.
- **V2 protocol-implementation gaps.** The executed v2 runner did not retain or
  assert the per-seed closed-form stationarity residual that protocol 31 §7.3
  lists; the residual was test-verified below `1e-10` before the seal, the
  affected endpoint depends only on retained per-arm held-out MSEs, and
  rerunning a sealed, consumed seed block is forbidden, so the gap is disclosed
  in docs/33 rather than repaired. Post-campaign audit found three further
  enforcement gaps that likewise did not fire: the gelsd rank/positivity
  rejection, the raw boundary defect in the C1 positivity guard, and full
  seal-content verification. Post-hoc checks confirmed all omitted conditions
  hold on the retained rows (minimum gelsd rank 3, minimum singular value
  `0.0386`, minimum C1 boundary defect `0.0372`); the runner is hardened and
  the automated validator now re-enforces each testable condition.
- **Optimizer confounding in the soft penalty's finite-step form.** Finite-step
  Adam does not reach the closed-form penalized optimum (per-seed solution gap
  mean `0.8556475807071382`, maximum `4.005285286298402`), so part of the soft
  penalty's margin over ambient Adam is optimizer slack; the closed-form H3
  contrast is the confound-free comparison.
- **Regularization specificity is established only within this family.** The
  sealed v2 campaign is the matched-control campaign earlier drafts required:
  the true cycle subspace beats a dimension-matched seeded random subspace
  (H4), and training-only four-fold-selected ridge shows no detected benefit
  over ambient minimum-norm least squares (H5, frozen and not supported). H5's
  null does not prove ridge cannot help anywhere — only that no benefit was
  detected under this frozen design.
- **Outcome-informed same-family design.** The hypotheses, directions, endpoint,
  training size, and weights were selected after exploration on the same
  generator family. Exploratory seed identities were not retained, so overlap
  with the frozen 20261001–20261030 block is unverifiable. The v2 replication
  supplies the untouched disjoint seed block the historical campaign lacked —
  sealed and pushed before execution, never previewed — but it remains
  outcome-informed and same-family: not an independent-lab or
  independent-generator replication, and never pooled with v1. An untouched
  disjoint generator family and any real-data validation remain open.
- **The selection setting saturates.** Six of eight objectives reach exactly
  1.000, so its structural nulls cannot detect improvement. Its RTD-only loss
  consumes target batch geometry and is not an unsupervised control.
- **Five seeds in the annulus campaign**, where the exact two-sided sign test
  floor is p = 0.0625 and can never be decisive.
- **Defect-based routing accuracy is unresolved.** The frozen H5 decision was
  impossible to satisfy because its per-row oracle denominator forces every
  endpoint observation to be nonnegative. Its 28 rows also comprise only 14
  topology clusters at two weights, so the former df=27 interval and 25/28 count
  cannot be treated as independent-trial inference. H5 receives no decision.
- **Privileged supervision** in the historical routing campaign, and
  **target-view translators** in the historical conversion modules, which consumed
  target structure and are not conversions.
- **Timing caveats.** All paths timed in fixed order inside one process; raw
  per-iteration timings not retained; identifiable runner reports p90 and routing
  runner p95, never pooled.
- **Secondary analyses are unadjusted.** The historical path C1 and the v2
  off-path C1, the separate routing campaign, and the corruption diagnostics
  lie outside the multiplicity-controlled families; the routing H5 has no valid
  inferential decision, and the v2 C1 receives no support decision.
- **The historical C1 is a regularization-path association.** Within each eligible seed,
  its nine fits share data and initialization and vary the common driver
  `lambda`. They are not independent observations, so that C1 does not
  establish independent predictive information or off-path calibration. The v2
  C1 replaces the path design with twelve independent training/noise replicates
  per topology sharing one noiseless test set; its paired delta-z contrast
  against a dimension-matched random-subspace defect is descriptive and remains
  association, not causation.
- **No equivalence test.** The historical RTD-inspired surrogate interval
  includes zero, and no equivalence margin was specified for it. The v2 H7 is a
  bounded-benefit/futility result at one prespecified margin (`log10(0.90)`):
  it rules out a benefit of 10% or more but is not a noninferiority or
  equivalence conclusion and cannot establish exact inertness.
- **The distance surrogate is target-misaligned.** It asks a rank-`F` lifting to
  preserve full source geometry even though the truth intentionally discards
  cut-space components. Its no-detected-improvement result does not test or
  criticize published RTD/SRTD.
- **Artifact boundaries.** The 8.8 GB `artifacts/` tree is untracked; tracked
  evidence is the curated `results/` bundle.

## 10. Claim boundary

This work provides evidence, on one synthetic family, from a sealed
untouched-seed, outcome-informed, same-generator-family replication (33
eligible of 36 declared seeds), that graph-derived cycle-subspace information
improves scarce-probe identification of an edge-to-cycle lifting. Six of seven
prespecified claims are supported at one-sided Bonferroni `alpha = 0.05/7`:
exact least squares restricted to `ker B1` improves over the graph-blind
ambient minimum-norm solution (H2) and over the closed-form soft boundary
penalty at its frozen weight (H3, by a modest geometric-mean ratio of
`0.745769272372031`); the soft penalty improves over the graph-blind ambient
Adam reference (H1), replicating the historical campaign's direction on an
untouched block; the exact constraint outperformed the prespecified
dimension-matched random-subspace control (H4); the singular-value cone surrogate's
harm replicates (H6); and any benefit of the RTD-inspired surrogate is bounded
below 10% of geometric-mean held-out MSE (H7, a bounded-benefit/futility result
at the prespecified margin `log10(0.90)`). The frozen ridge claim (H5) is not
supported. Every estimand is the mean paired `log10` MSE ratio conditional on
connected, `F>=3` seeds of this generator family; intervals quantify Monte
Carlo variation over the frozen seed mechanism, not uncertainty for real data.
The historical campaign — prospectively locked, with a disclosed sum-versus-mean
implementation deviation — is retained as the prior account and is never pooled
with v2. A prespecified, unadjusted off-path secondary analysis finds that the
cycle-projector defect covaries with held-out error across independent
training/noise replicates where a dimension-matched random-subspace defect does
not (paired Fisher-z contrast `+1.0880757279700473`, interval
`[+0.8873984853168706, +1.288752970623224]`); it is descriptive and carries no
support decision.

It does **not** establish:

- any general conclusion about mapping-cone homology or published RTD/SRTD as
  training objectives;
- that a measured defect can select a representation; the frozen routing test
  of this claim had an impossible decision rule;
- lifting quality on real or out-of-distribution data, transfer to unseen
  topologies or generator families, neural nonlinear translators, or sheaves,
  or any broader conversion claim;
- that any fitted arm attains the withheld generator-basis oracle, whose error
  is exactly zero on every eligible seed; constrained least squares on
  `ker B1` is classical and is not claimed as a new algorithm, and the soft
  penalty is not claimed as a superior method;
- an independent-lab or independent-generator replication: the v2 seed block
  was untouched, disjoint, sealed, and pushed before execution, but the design
  was outcome-informed and the generator family is shared with the historical
  campaign;
- that generic shrinkage cannot help this task under any design; H5's frozen
  null covers training-only four-fold ridge under this frozen design only;
- equality, inertness, or the absence of every benefit for the RTD-inspired
  surrogate; H7 rules out only a benefit of 10% or more at its margin;
- independent predictive information, calibration, causation, or routing
  utility from the C1 associations (historical on-path; v2 off-path);
- general equivalence between graphs, cellular complexes, and sheaves;
- a general method for learning quasi-isomorphisms — the annulus candidates are
  preconstructed invertible chain maps, and the verified chain-map identity uses
  a fixed 1e-5 tolerance;
- any Langlands, eigensheaf, Fourier–Mukai, or category-theoretic result.
  Imposing a chain-map constraint by construction is not a categorical claim in
  the sense of [9] or [10]: we fix one finite hypothesis class by hand in §4.1,
  §4.2 optimizes a full matrix under a finite soft penalty, and the v2
  exact-constraint arm is classical equality-constrained least squares [15,
  16] — none of these is a categorical construction.

## 11. Code and data availability

**Availability status.** The project repository is private and is licensed
proprietarily (Copyright © 2026 Sean Mahdavian, all rights reserved). This is
**not an open-source or open-data release.**

**Access for peer review.** The source, configurations, frozen protocols, and
compact evidence bundle are made available to editors and reviewers for the
duration of review through a read-only snapshot supplied to the handling editor.
The snapshot is built by `scripts/build_review_snapshot.py`, which archives the
repository at one commit together with the tracked `results/` bundle and its
manifest and refuses to run against an uncommitted worktree. Reviewers can
rehash the compact evidence, trace every reported value to it, rerun the
historical CPU edge-to-cycle lifting campaign (the sealed v2 replication record
is verify-only by construction, §12), run the test suite, and regenerate the
figures and paper.
The snapshot intentionally excludes the 8.8 GB raw `artifacts/` tree. A full
rerun of the GB10 annulus campaign or trained-checkpoint benchmarks therefore
requires separately supplied raw artifacts/checkpoints and compatible GPU
hardware. Access is granted for review only and confers no license to
redistribute or reuse the material; the terms in `LICENSE` continue to apply.

Readers outside the review process should treat this section as a description of
what exists and how it is organised, not as a download.

All code, configurations, and compact evidence are in that repository. The
tracked bundle under `results/` contains 51 files (4,034,915 bytes) with a
`MANIFEST.json` recording
path, byte count, SHA-256, generating commit, and generating command for each.

| location | contents |
|---|---|
| `results/campaigns/` | the frozen edge-to-cycle lifting campaign records, including the sealed untouched-seed v2 replication result |
| `results/summaries/` | strict compact summaries for the annulus, gauge, compute, and routing campaigns |
| `results/gate3/`, `results/gate3g/` | gate decisions and per-batch corruption-report derivatives |
| `results/benchmarks/` | ten identifiable and five routing trained benchmark records |

The v2 replication's sealed lineage is tracked alongside the bundle: the frozen
protocol (`docs/31-independent-lifting-replication-protocol.md`), the
machine-readable design seal
(`docs/32-independent-lifting-replication-seal.json`), the dedicated runner
(`scripts/run_lifting_replication_v2.py`), and the result
(`results/campaigns/lifting-replication-v2.json`). Whenever the bundle is
rebuilt, the exporter revalidates that record: the seal schema and its full
claim content and stop rules, every embedded hash, the pinned design and
execution commits (including commit ancestry where the git object store is
present), the eligibility accounting, and — recomputed independently from the
retained raw rows — every published estimate, standard error, bound, support
decision, sign test, C1 correlation and Fisher-z summary, descriptive
diagnostic, and audit-block mean, together with the protocol's raw-row validity
conditions. It fails closed on any mismatch; `--verify-only` re-hashes every
bundle file, the v2 record included, against the manifest.

Corruption reports are exported as per-batch derivatives: the `per_example` array
is dropped and the `per_batch` array — the unit of analysis — retained. This is
lossless for every published statistic, verified by recomputing the adjusted
partial Spearman from retained rows and matching to 1e-15.

Synthetic generators are deterministic given the committed configs and seeds; no
external dataset is required.

## 12. Reproducibility

The reviewer snapshot supports the following checks without a GPU:

```bash
uv sync --frozen --extra dev --python 3.12.3
env CUDA_VISIBLE_DEVICES=-1 .venv/bin/python \
  scripts/run_conversion_campaign.py \
  --output /tmp/conversion-campaign.json
.venv/bin/python scripts/export_publication_evidence.py --verify-only
.venv/bin/python scripts/render_figures.py
.venv/bin/python scripts/render_paper.py
```

The campaign runner fails before fitting unless the lockfile, Python, NetworkX,
NumPy, base Torch version, generator, and frozen protocol match the recorded
environment. Hiding CUDA keeps this float64 CPU run independent of accelerator
occupancy.

The sealed v2 replication is verified rather than rerun. Its canonical
execution command was

```bash
env CUDA_VISIBLE_DEVICES=-1 .venv/bin/python \
  scripts/run_lifting_replication_v2.py \
  --output results/campaigns/lifting-replication-v2.json
```

The runner binds `--output` to the sealed path and fails before fitting if that
output already exists, so the consumed seed block cannot be re-executed in
place; any substantive change would create a new campaign with a new untouched
seed block under protocol 31's freeze rules. The v2 record is instead checked through the
exporter: rebuilding the bundle revalidates its sealed lineage (§11), and
`--verify-only` re-hashes it against the manifest.

With separately supplied raw campaign artifacts and checkpoints, the full
private workspace additionally supports:

```bash
python scripts/summarize_gauge_corruption_campaign.py \
  --output results/summaries/gauge-corruption-campaign.json
python scripts/summarize_compute_campaign.py \
  --output results/summaries/compute-campaign.json
```

Each summarizer revalidates provenance before aggregating and fails closed on any
hash, seed, pairing, schema, or receipt mismatch. The exporter works from an
explicit allowlist and refuses checkpoints, prediction dumps, histories, logs, and
caches. Its manifest carries no timestamp, so re-exporting unchanged evidence
reproduces the bundle byte for byte.

Thus the compact snapshot supports verification of every reported value and a
fresh run of the historical CPU conversion campaign; the sealed v2 replication
record supports hash and sealed-lineage verification, not re-execution. It does
not, by itself, support a fresh GB10 campaign
or checkpoint benchmark; those require the excluded raw artifacts and hardware
described in §11.

The paper PDF is rendered with `python scripts/render_paper.py`; figures are
regenerated from tracked evidence with `python scripts/render_figures.py`.

## 13. Declarations

**Author contributions.** Sean Mahdavian designed and implemented the system, ran
the campaigns, and wrote the manuscript.

**Affiliation.** Phillips Exeter Academy.

**ORCID.** [0009-0000-8432-7825](https://orcid.org/0009-0000-8432-7825).

**Funding.** This work received no external funding.

**Competing interests.** The author declares no competing interests.

**License.** The software, source code, documentation, and evidence described
here are made available under a proprietary license, Copyright © 2026 Sean
Mahdavian, all rights reserved. The terms are in the repository's `LICENSE`.
**This is not an open-source or open-data release**, and §11 should be read
accordingly.

**Ethics.** Every experiment reported here uses synthetic data generated by
committed deterministic code. No human subjects, personal data, or personally
identifiable information are involved. Compute is small: the annulus campaign is
25.8 minutes of GPU time and the edge-to-cycle lifting campaign runs on CPU.

A molecular transfer experiment on the public OGBG-MOLHIV benchmark was run
earlier in this project and is **deliberately not reported here**, because it
answers a different question from the one this paper asks. Its result — the
ring-lift route losing to the graph route, with later variants contaminated by
repeated inspection of the official test split — is recorded in the project's
claims ledger as C6.

**Reporting.** Null and negative results are reported in the body rather than an
appendix, and every quantitative experimental result is traceable to a tracked
machine-readable file under `results/`.

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
   Framework for Representation Analysis](https://arxiv.org/abs/2606.06342).
   TMLR, 2026. arXiv:2606.06342.
4. Beshkov.
   [A Quotient Homology Theory of Representation in Neural
   Networks](https://openreview.net/forum?id=RluspxztzS).
   TMLR, 2026.
5. Battiloro, Spinelli, Telyatnikov, Bronstein, Scardapane, and Di Lorenzo.
   [From Latent Graph to Latent Topology Inference: Differentiable Cell Complex
   Module](https://arxiv.org/abs/2305.16174).
   ICLR, 2024.
6. Franco, Duarte, Nikitin, Ponti, Mesquita, and Souza.
   [Differentiable Lifting for Topological Neural
   Networks](https://neurips.cc/virtual/2025/123646).
   NeurIPS 2025 Workshop on Non-Euclidean Foundation Models and Geometric
   Learning. Workshop poster, not a main-conference paper.
7. Bodnar, Di Giovanni, Chamberlain, Liò, and Bronstein.
   [Neural Sheaf Diffusion: A Topological Perspective on Heterophily and
   Oversmoothing in GNNs](https://arxiv.org/abs/2202.04579).
   NeurIPS, 2022.
8. Gebhart, Hansen, and Schrater.
   [Knowledge Sheaves: A Sheaf-Theoretic Framework for Knowledge Graph
   Embedding](https://proceedings.mlr.press/v206/gebhart23a.html).
   AISTATS, 2023.
9. Gavranović, Lessard, Dudzik, von Glehn, Madeira Araújo, and Veličković.
   [Position: Categorical Deep Learning is an Algebraic Theory of All
   Architectures](https://proceedings.mlr.press/v235/gavranovic24a.html).
   ICML, 2024.
10. Gavranović.
    [Learning Functors using Gradient Descent](https://arxiv.org/abs/2009.06837).
    EPTCS 323, 2020.
11. Wang, Yu, Dunlap, Ma, Wang, Mirhoseini, Darrell, and Gonzalez.
    [Deep Mixture of Experts via Shallow
    Embedding](https://proceedings.mlr.press/v115/wang20d.html).
    UAI, 2020.
12. H. Wang, Jiang, You, Han, Liu, Srinivasa, Kompella, and Z. Wang.
    [Graph Mixture of Experts: Learning on Large-Scale Graphs with Explicit
    Diversity
    Modeling](https://papers.nips.cc/paper_files/paper/2023/hash/9f4064d145bad5e361206c3303bda7b8-Abstract-Conference.html).
    NeurIPS, 2023.
13. S. Wu, Cao, Ribeiro, Zou, and Leskovec.
    [GraphMETRO: Mitigating Complex Graph Distribution Shifts via Mixture of
    Aligned
    Experts](https://papers.nips.cc/paper_files/paper/2024/hash/11c892a9fcc430cc0f4c7d457e5d60ea-Abstract-Conference.html).
    NeurIPS, 2024.
14. Z. Wu, Cai, Zhang, Lu, Chen, Zhuang, and Wang.
   [Where Graph Meets Heterogeneity: Multi-View Collaborative Graph
   Experts](https://neurips.cc/virtual/2025/poster/116976).
   NeurIPS, 2025.
15. Björck.
   [Numerical Methods for Least Squares Problems, Chapter 5: Constrained Least
   Squares Problems](https://epubs.siam.org/doi/10.1137/1.9781611971484.ch5).
   SIAM, 1996.
16. Eldén.
   [Perturbation Theory for the Least Squares Problem with Linear Equality
   Constraints](https://epubs.siam.org/doi/10.1137/0717028).
   SIAM Journal on Numerical Analysis 17(3):338–350, 1980.
17. Lim.
   [Hodge Laplacians on
   Graphs](https://epubs.siam.org/doi/10.1137/18M1223101).
   SIAM Review 62(3):685–715, 2020.
18. Hoppe and Schaub.
   [Representing Edge Flows on Graphs via Sparse Cell
   Complexes](https://proceedings.mlr.press/v231/hoppe24a.html).
   Proceedings of the Second Learning on Graphs Conference, PMLR 231:1:1–1:22,
   2024.
