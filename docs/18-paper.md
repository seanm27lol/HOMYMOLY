# Typed Representation Routing and Homological Diagnostics: An Audit-Corrected Experimental Report

**Sean Mahdavian — HOMYMOLY project**

**Audit revision: 2026-08-22**

## Abstract

HOMYMOLY studies two related questions: whether a model can route individual
examples among graph, cell-complex, and cellular-sheaf experts, and whether
homological diagnostics can measure or reduce the damage caused by moving
between those representations. An audit materially narrows the answers.
The first untouched five-seed routing campaign had a mean advantage of
+0.036 accuracy over the best fixed expert, with a 95% Student-t interval
of [−0.018, +0.090]; the interval crosses zero. A later +0.108 result used
a stabilization selected after inspecting weak seeds and is development
evidence, not confirmation. A new five-seed campaign under the committed v2
freeze produced a mean advantage of +0.1098, with a 95% Student-t interval of
[+0.0953, +0.1243]. This supports a narrow synthetic routing result, but not a
graph-only or conversion claim: training retained privileged latent-regime
distillation, routing used summaries of all available structured views, and
the historical translators read target cell activity or sheaf transports.
The campaign was protocol-aligned rather than pristine preregistration because
validation metrics from an aborted pre-freeze seed-20260906 attempt had been
observed before the committed rerun.

The repository now contains a target-held-out translation setting and a
separate layer whose parameters satisfy a two-term chain-map equation by
construction. In a one-seed controlled permutation experiment, the latter
recovered the planted map to numerical precision and produced an acyclic
mapping cone. This is an implementation sanity check, not evidence that a cone
objective improves a downstream model. Historical ablations support only a
scoped null for reconstruction, local-consistency, and H0-distance surrogates.
Corrected fixed-expert corruption diagnostics found no supported paired
ablation contrast after severity and repeated-block adjustment; they still do
not evaluate a typed conversion. On OGBG-MOLHIV, the initial ring-cell model underperformed
the graph model; later redesign scores are exploratory because the official
test set was reused. The present evidence establishes working mathematical
and systems components, several negative observations, and a set of open
experiments—a scoped routing gain under a privileged synthetic protocol, not a
demonstrated downstream benefit from homological loss.

## 1. Questions and current evidential status

The motivating idea is broader than topology-aware feature learning. Different
mathematical representations expose different operations: graphs emphasize
pairwise relations, cell complexes encode higher-order incidence, and sheaves
attach local vector spaces and transport maps. A model might choose among
these views per example, while exactness defects and mapping-cone homology
describe what a declared conversion fails to preserve.

That motivation is not itself a result. The project currently separates five
claims:

| question | current status |
|---|---|
| Does hard per-example routing beat every fixed expert? | Supported in the protocol-aligned v2 synthetic campaign: mean +0.1098, 95% Student-t interval [+0.0953, +0.1243], with the procedural and supervision boundaries stated below. |
| Do the historical translators learn graph-to-cell or graph-to-sheaf conversion? | No. They are target-view encoders because they consume target structure. |
| Can a learnable finite map satisfy a chain-map equation and yield an evaluable cone? | Yes, in a small controlled permutation experiment; one seed, no downstream task. |
| Do structural losses improve task behavior? | Historical local surrogates and corrected paired fixed-expert diagnostics show no supported benefit; a direct identifiable learned-map ablation is reported separately below. |
| Do ring 2-cells improve MOLHIV prediction? | No in the initial v1 comparison; later redesign observations are post-test exploratory results. |

No claim of firstness or broad novelty is made. The closest-work comparison
below is a positioning exercise, not a systematic-review certificate.

## 2. Related work

### 2.1 Comparing and regularizing paired representations

[Representation Topology Divergence
(RTD)](https://proceedings.mlr.press/v162/barannikov22a.html) compares paired
point clouds, including clouds in different ambient dimensions, through
cross-barcodes. [RTD-AE](https://openreview.net/forum?id=lIu-ixf-Tzf)
differentiates the RTD construction for topology-preserving autoencoder
training. [Symmetric Representation Topology Divergence
(SRTD)](https://openreview.net/forum?id=pGgJ9qB2Io) replaces an ad hoc
directional combination with a symmetric union/intersection construction and
reports degree-specific barcode totals. These papers are the direct basis for
HOMYMOLY’s representation-divergence reference. The project’s earlier scalar
did not follow their degree and normalization conventions and its historical
outputs have been withdrawn.

### 2.2 Learning higher-order structure

[Differentiable Cell Complex Module
(DCM)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6b97236d90d945be7c58268207a14f4f-Abstract-Conference.html)
learns probabilities for higher-order cells jointly with a downstream task.
[Differentiable Lifting
(DiffLift)](https://openreview.net/forum?id=eC89CbINIw) learns graph liftings
to hypergraphs and cellular, simplicial, or combinatorial complexes.
These are close precedents for learning a target topological domain from graph
data. [Neural Sheaf Diffusion](https://arxiv.org/abs/2202.04579) learns sheaf
restriction maps for graph representation learning. HOMYMOLY differs in its
specific experimental combination of fixed representation families,
conversion diagnostics, and routing, but the current experiments do not
establish that this combination is new or superior.

### 2.3 Dynamic routing and graph mixtures of experts

[DeepMoE](https://proceedings.mlr.press/v115/wang20d.html) provides an early
per-example dynamic-routing comparison with computation-aware motivation.
[GMoE](https://papers.nips.cc/paper_files/paper/2023/hash/9f4064d145bad5e361206c3303bda7b8-Abstract-Conference.html)
routes graph nodes among message-passing experts with different aggregation
behavior. [GraphMETRO](https://papers.nips.cc/paper_files/paper/2024/hash/11c892a9fcc430cc0f4c7d457e5d60ea-Abstract-Conference.html)
uses aligned experts and a gate to model combinations of graph distribution
shifts. [MvCGE](https://openreview.net/forum?id=dsp8dUlZFq) dynamically
activates collaborative graph experts across multiple graph views. These
systems make dynamic specialization on graph-structured data an established
research direction. HOMYMOLY’s scoped question is whether the expert choice
can correspond to graph, cell, or sheaf processing under a controlled
benchmark and an honest compute comparison.

## 3. System and method

### 3.1 Representations and experts

All three experts return a 64-dimensional embedding and binary logits:

- The graph expert performs edge-conditioned message passing and includes an
  endpoint-pair pathway needed by the synthetic graph label.
- The cell expert augments a graph backbone with active faces represented by
  padded oriented boundary-edge lists. This representation supports rings of
  arbitrary encoded length; it does not replace a long ring with a triangle.
- The connection-sheaf expert uses rank-2 planar rotations as edge transports
  and includes an explicit face-holonomy pathway.

The experts are approximately, not exactly, parameter matched. Historical
Gate-2 configurations contain about 0.85–0.93 million parameters per expert;
the molecular models use 0.918 million graph parameters and 0.956–0.973
million cell parameters, depending on the variant.

### 3.2 Synthetic benchmark

ConfirmatoryStructuredSignal constructs counterfactual groups containing one
example for each regime-label pair on a shared oriented complex. Splits are
group-disjoint. The labels are deliberately relational:

- graph: a sign relation across two unmarked anchor edges;
- cell: activity of a probe face with the edge cochain held fixed;
- sheaf: a cycle-holonomy defect.

Regime reliability amplitudes overlap, so the regime is not perfectly
identifiable from cheap, label-independent statistics. This reduces a trivial
router shortcut but does not prove a universal identifiability ceiling.

### 3.3 Routing supervision: historical and canonical

At inference, the hard router selects one expert from observation summaries
that omit the label and regime tensors but include graph features,
candidate/active-face statistics, and sheaf-transport statistics. Historical
routing training also was not fully non-privileged: validation
regime labels indexed a regime-by-expert accuracy table, and the table
provided the router’s utility targets. The regime was not an inference input,
but this is privileged regime distillation and is part of the claim boundary.

The canonical configuration now uses a per-example supervised utility target
instead of the latent-regime table. This removes regime-label privilege but is
a different experiment; its outcome must not be inferred from historical
runs.

### 3.4 Typed translators and their boundaries

The historical compatibility mode exposes target face activity to the
graph-to-cell module and target edge transports to the graph-to-sheaf module.
Those modules can be useful target-view encoders, but they do not demonstrate
conversion from graph observations.

The canonical target-held-out mode makes the following changes:

- graph-to-cell predicts soft activity over a supplied candidate face
  incidence and is supervised against the target cellular boundary operator;
- graph-to-sheaf predicts SO(2) edge transports from graph edge features and
  endpoint latents and is supervised against target transports.

“Graph-only” in repository configuration names therefore means that target
activity and transport values are held out. Candidate face incidence is still
provided. The cell loss is typed structure reconstruction, not a learned
chain-map residual, and neither neural translator currently constructs a
mapping cone of its learned map.

More fundamentally, these held-out targets are not identifiable from graph
inputs in the current synthetic generator: the cell bit selects probe versus
decoy activity without changing the graph cochain, while the sheaf transport
defect is generated independently of graph-observed node fields. The current
mode therefore implements target-reconstruction objectives but cannot validate
competent graph-to-cell/sheaf conversion on this benchmark. The transport MSE
also fixes a gauge rather than defining a gauge-invariant loss.

### 3.5 Homological evaluation

The corrected non-differentiable RTD/SRTD reference:

- accepts paired dissimilarity matrices with a one-to-one entity
  correspondence;
- normalizes each matrix by its full-matrix 0.9 quantile by default;
- reports persistence separately by homological degree;
- uses degree 1 for the published scalar RTD convention;
- constructs an additional simplex degree internally for deaths while
  excluding truncation-frontier generators from reported scores;
- limits exact enumeration to at most 64 entities.

The training-time H0 distance surrogate is not an exact cross-barcode and may
disagree with exact RTD even in directional ordering. It is labeled as a
surrogate throughout the audited code.

### 3.6 Constraint-respecting finite chain maps

For a two-term source complex with boundary dC and target complex with
boundary dD, the separate ExactChainMapLayer parameterizes degree maps F0 and
F1 in the nullspace of

    dD F1 − F0 dC = 0.

Thus every parameter value satisfies the chain-map equation up to floating-
point roundoff. A zero map is always legal, so paired-signal or task
supervision remains necessary. The implementation also constructs the
mapping-cone differentials, a differentiable soft-nullity proxy for cone
acyclicity, an exact cone-homology evaluation, and forward/reverse
cycle-consistency losses.

## 4. Results

### 4.1 Historical routing evidence

The first five-seed campaign was untouched with respect to its frozen
configuration:

| seed | hard routed | best fixed | margin | route accuracy | graph/cell/sheaf utilization |
|---|---:|---:|---:|---:|---|
| 20260901 | 0.767 | 0.668 | +0.099 | 0.546 | .33/.30/.36 |
| 20260902 | 0.693 | 0.704 | −0.011 | 0.378 | .08/.57/.35 |
| 20260903 | 0.720 | 0.692 | +0.028 | 0.434 | .10/.50/.40 |
| 20260904 | 0.749 | 0.691 | +0.058 | 0.504 | .19/.50/.31 |
| 20260905 | 0.687 | 0.680 | +0.007 | 0.388 | .11/.57/.32 |

The mean hard-minus-best-fixed margin was **+0.036**, with a two-sided 95%
Student-t interval of **[−0.018, +0.090]** across five seeds. One seed lost to
the best fixed route, and weak seeds substantially under-used the graph
expert. Under the prespecified interval rule, this result is inconclusive.

A higher router learning rate and longer warmup were selected after examining
the weak seeds and then rerun on the same five seeds. The resulting mean
margin was +0.108, with every reused seed positive. That is useful
stabilization evidence, but its nominal interval has no confirmatory coverage
because model selection and evaluation reused the same seeds.

The committed v2 campaign retained the stabilized hyperparameters and the
historical structured-view/regime-distilled setting, changed to five new
experiment/data seeds, and froze the primary endpoint before its valid runs:

| seed | hard routed | best fixed | margin | dense | route accuracy | route MI (nats) |
|---|---:|---:|---:|---:|---:|---:|
| 20260906 | 0.8028 | 0.6743 | +0.1285 | 0.7723 | 0.6100 | 0.1622 |
| 20260907 | 0.7745 | 0.6754 | +0.0991 | 0.7571 | 0.5468 | 0.0972 |
| 20260908 | 0.7680 | 0.6667 | +0.1013 | 0.7582 | 0.5381 | 0.0934 |
| 20260909 | 0.7789 | 0.6667 | +0.1122 | 0.7342 | 0.5752 | 0.1264 |
| 20260910 | 0.7745 | 0.6667 | +0.1078 | 0.7505 | 0.5501 | 0.1006 |

The mean hard-minus-best-fixed margin was **+0.1098** (sample SD 0.0117),
with a two-sided 95% Student-t interval of **[+0.0953, +0.1243]**. All five
margins were positive, and the frozen interval decision rule labels the result
supported. A distribution-free sign-test sensitivity analysis is less
decisive (two-sided p = 0.0625), as expected with only five seeds. Mean
hard-minus-dense accuracy was +0.0253; dense logits were an unweighted mean of
the trained expert logits rather than an independently optimized ensemble.

All 4,590 prediction records were independently rechecked. The five valid
runs share Git revision `e69b077`, executable fingerprint `473fb0f6…`, frozen
config hashes, Torch 2.13.0+cu130, CUDA 13.0, and the NVIDIA GB10. The campaign
nevertheless has a procedural deviation: an aborted precommit seed-20260906
attempt recorded validation metrics under different executable code before
the committed seed was rerun; it never produced the primary test endpoint.
The core endpoint, rule, and hashes were committed before the valid runs, but a
provenance-safeguard paragraph was added during seed 5. We therefore call this
result protocol-aligned under the committed freeze, not pristine
preregistration. Its scope remains a synthetic router trained by privileged
regime distillation and evaluated with target structured views available.

### 4.2 Controlled exact-chain-map experiment

A source cycle complex with 8 vertices and 8 edges was transformed by hidden
vertex and edge permutations. Bidirectional constraint-respecting maps were
trained on 4,096 paired random signals and evaluated on 2,048 held-out paired
signals for one seed and 1,200 optimization steps.

| endpoint | value |
|---|---:|
| mean held-out bidirectional signal MSE | 1.54 × 10⁻¹⁴ |
| forward planted-map MSE | 2.29 × 10⁻¹⁵ |
| maximum forward chain residual | 1.19 × 10⁻⁷ |
| maximum reverse chain residual | 2.38 × 10⁻⁷ |
| forward/reverse cycle-consistency MSE | 1.95 × 10⁻¹⁵ |
| exact forward-cone Betti numbers | (0, 0, 0) |

The planted map is an isomorphism, so an acyclic mapping cone is the expected
answer. This experiment verifies that the parameterization, training path,
and exact cone oracle agree on a controlled case. It does not compare against
an unconstrained map, does not isolate the contribution of the soft cone term,
and does not demonstrate usefulness for graph/cell/sheaf prediction.

> **PENDING RESULT — CANONICAL GRAPH-ONLY/TARGET-HELD-OUT GATE-2 RUN**
> The canonical configuration holds out face activity and sheaf transports,
> adds typed structure-reconstruction terms, and replaces latent-regime
> routing targets with per-example supervised utility. Candidate face
> incidence remains supplied, but the held-out targets are unidentifiable from
> current graph inputs. Report its failed/null outcomes as an integration
> diagnostic, not a conversion test. It changes multiple scientific factors
> and must not be treated as a direct replication of historical Gate 2.

### 4.3 Historical structural-surrogate ablations

Four historical variants shared one seed and schedule and changed only the
target-view encoder losses:

| target-view objective | hard routed accuracy |
|---|---:|
| task only | 0.746 |
| task + reconstruction | 0.733 |
| task + reconstruction + local consistency | 0.751 |
| full historical objective, including H0 surrogate | 0.746 |

There is no monotonic task improvement. On the gauge-tier revision, the local
consistency term reliably held its own surrogate near 0.18 rather than the
1.4–2.0 drift seen without it, while recorded task accuracy remained similar.
Across eight paired gauge runs, this manipulation remained reliable and did
not produce a recorded task benefit. These are scoped negative observations
about the implemented reconstruction, local-consistency, and H0-distance
surrogates. The modules were target-view encoders, and no run trained a
learned-map mapping-cone objective. Causal claims about a direct cone loss
remain untested.

The historical corruption report cannot be used for inference. It reused five
sample blocks across five severity levels, treated repeated rows as
independent, omitted severity and block adjustment, used process-salted random
draws, and scored an undocumented sum across homological degrees under the
wrong normalization. Its earlier correlation and partial-correlation claims
are withdrawn.

The corrected fixed-expert diagnostic used deterministic paired draws, degree-1
SRTD with full-matrix 0.9-quantile normalization, 306 unique examples and 13
complete batch blocks per corruption kind, and five repeated severity levels
(65 batch observations). The statistic is the Pearson correlation of rank
residuals after controlling for ranked embedding displacement, ranked
severity, and block fixed effects. Uncertainty uses a complete-block bootstrap
and inference uses within-block residual permutation.

Eleven of twelve per-model bootstrap intervals included zero. The exception
was edge-cochain noise for the full historical objective: adjusted correlation
0.214, 95% block-bootstrap interval [0.002, 0.488], but its unadjusted
permutation p-value was 0.115. This isolated, internally mixed diagnostic is
not evidence after considering the twelve unadjusted model/kind reads.

The ablation question is better represented by paired contrasts against
task-only, with identical block/severity corruption draws. This paired analysis
was designed during the audit and is not a preregistered endpoint:

| candidate − task-only | corruption | adjusted-correlation difference | 95% paired block-bootstrap interval | exact paired p |
|---|---|---:|---:|---:|
| + reconstruction | edge cochain | +0.018 | [−0.301, +0.355] | 0.903 |
| + reconstruction | node anchor | +0.020 | [−0.234, +0.278] | 0.896 |
| + reconstruction | transport rotation | −0.011 | [−0.169, +0.120] | 0.894 |
| + local consistency | edge cochain | −0.022 | [−0.318, +0.313] | 0.878 |
| + local consistency | node anchor | +0.030 | [−0.122, +0.154] | 0.671 |
| + local consistency | transport rotation | −0.021 | [−0.181, +0.088] | 0.854 |
| full historical objective | edge cochain | +0.057 | [−0.185, +0.332] | 0.701 |
| full historical objective | node anchor | −0.011 | [−0.219, +0.233] | 0.936 |
| full historical objective | transport rotation | −0.072 | [−0.210, +0.021] | 0.324 |

Every paired interval includes zero, and every exact whole-block model-label
randomization p-value is at least 0.324. No multiplicity adjustment was made.
These statistics are conditional on each fixed checkpoint pair and sampled
blocks; they do not estimate training-seed variation. Most importantly, the
program compares clean and corrupted fixed-expert embeddings. It never invokes
a translator or learned map, so it cannot support or refute a typed-conversion
claim.

### 4.4 GB10 forward-pass benchmark

The audit added a synchronous CUDA forward-pass benchmark on an NVIDIA GB10,
batch size 64, bfloat16, with the batch already resident on the device,
20 warmups, and 100 measured iterations:

| execution path | median latency | p95 latency | mean throughput |
|---|---:|---:|---:|
| fixed graph | 16.91 ms | 25.02 ms | 3,413 examples/s |
| fixed cell | 21.17 ms | 27.91 ms | 2,802 examples/s |
| fixed sheaf | 21.98 ms | 30.29 ms | 2,677 examples/s |
| hard routed | 38.34 ms | 45.63 ms | 1,627 examples/s |
| dense three-expert ensemble | 67.81 ms | 71.15 ms | 969 examples/s |

The median dense-to-routed ratio is 1.77. This replaces the historical
declared-cost proxy with an execution measurement, but it is not yet a
matched-accuracy result: the benchmark artifact records checkpoint = null,
so it timed an untrained canonical model and did not record a trained route
distribution. It excludes data loading and preprocessing and does not show a
memory advantage. It establishes the latency of these code paths on this
batch shape, not the compute/accuracy tradeoff of a confirmed trained model.

### 4.5 Molecular transfer

The initial MOLHIV experiment used the official 32,901/4,113/4,113 scaffold
split, early stopping on validation AUROC, and the official evaluator. The
three values behind each mean are initialization seeds on the same fixed
split; their sample SD is not uncertainty over molecule sampling.

| model | status | valid mean | official-test AUROC, mean ± seed SD |
|---|---|---:|---:|
| graph | initial comparison | 0.794 | 0.771 ± 0.014 |
| cell v1, AtomRing faces | initial comparison | 0.782 | 0.723 ± 0.017 |
| cell v2, boundary-edge max + ring size | post-test development | 0.771 | 0.757 ± 0.002 |
| cell v3, plus bond-type counts | post-test development | 0.761 | 0.729 ± 0.025 |

In the initial v1 read, graph exceeded cell by 0.0481 AUROC and won all three
paired seeds. The smaller validation gap is consistent with greater
scaffold-test degradation for v1 cell, but training scores were not recorded,
so it does not identify optimization versus generalization.

V2 was designed after inspecting v1 on the official test, and v3 was designed
after inspecting v2; v2 was then restored after inspecting v3. Their test
scores are therefore exploratory development observations. V2 adds an
elementwise maximum over learned boundary-edge embeddings plus normalized
ring size; it does not identify a physically “strongest bond.” V3 adds raw
counts for five OGB bond-type categories, not normalized aromatic-bond
shares.

RDKit produced at least one AtomRing for 31,316/4,111/4,111 molecules
(95.18%/99.951%/99.951%), yielding 95,393/15,894/14,974 faces across
train/validation/test. Seven of 41,127 SMILES failed RDKit parsing
(3/2/2 by split, 0.017%) and were retained with zero faces. A graph-cycle-rank
audit finds 31,319/4,113/4,113 cyclic graphs, so validation and test contain
no acyclic examples. The two zero-face test examples are parser failures,
both negative labels, not evidence for ring-free transfer. Acyclic molecular
transfer is unevaluable on this test set.

## 5. Discussion

Three separations are central.

First, representation routing and representation conversion are different
claims. A router can choose among fixed experts without learning a meaningful
map between their domains. The v2 campaign supports the former only in its
privileged synthetic setting; it provides no evidence for the latter.

Second, satisfying an algebraic constraint and improving a task are different
claims. The exact-chain-map layer successfully enforces its declared equation
and recovers a planted isomorphism. Only matched downstream ablations can show
whether the constraint or cone objective supplies useful inductive bias.

Third, a responsive diagnostic and incremental predictive value are different
claims. Corrected RTD/SRTD measures degree-specific representation differences,
but the corrected fixed-checkpoint study found no supported paired ablation
contrast beyond ordinary geometric displacement. Its isolated unadjusted
within-model interval does not establish a robust effect.

The most informative next experiments are consequently narrow: replicate the
routing result under non-regime-distilled supervision and graph-identifiable
inputs; evaluate target-held-out conversion on an identifiable benchmark; and
compare cone-only, RTD-only, combined, and task/reconstruction objectives over
multiple training seeds and less symmetric map families.

## 6. Limitations

- The synthetic evidence comes from one benchmark family with deliberately
  engineered regimes and reliability cues.
- The supported v2 routing result uses privileged latent-regime supervision
  during training and structured target-view summaries at inference. Its five
  seeds are too few to check Student-t assumptions, and the sign-test
  sensitivity is p = 0.0625. The canonical experiment removes regime
  privilege but remains unvalidated.
- The target-held-out cell translator still receives candidate incidence; it
  does not discover arbitrary cells from a graph. Current cell/sheaf targets
  are also unidentifiable from graph inputs, so this setting cannot establish
  translation competence without a redesigned benchmark.
- The exact-chain-map result is one seed on an 8-vertex planted isomorphism.
  It has no unconstrained or loss-component ablation.
- The historical target-view ablations do not test graph-only conversion or a
  direct mapping-cone loss.
- Corrected corruption inference is conditional on fixed checkpoints and 13
  held-out blocks per kind, with no across-seed or multiplicity-adjusted
  inference. It evaluates experts, not conversions. All historical correlation
  interpretations remain withdrawn.
- The compute result times an untrained model’s forward paths on one GB10,
  one batch size, and one precision. It is not an end-to-end or
  matched-accuracy benchmark.
- The molecular test set was adaptively reused after v1, and it contains no
  acyclic graphs. Later molecular variants need a fresh locked split or an
  external benchmark.
- Exact cross-barcode enumeration is exponential and capped at 64 entities.
- GPU scatter-add operations are not guaranteed bit-deterministic even when
  deterministic configuration flags are enabled.
- No categorical equivalence, derived-category, Fourier–Mukai, eigensheaf,
  Langlands, or co-exactness construction is implemented. Those ideas remain
  motivation, not results.

## 7. Reproducibility and artifact boundary

The local worktree records:

- group-disjoint synthetic generators and anti-shortcut tests;
- phased training with atomic checkpoints and configuration/code
  fingerprints;
- frozen hashes and the completed routing-confirmatory-v2 protocol in
  [docs/19](19-routing-confirmatory-v2-protocol.md);
- the corrected degree-specific exact RTD/SRTD implementation and fixtures;
- the exact-chain-map implementation and
  [controlled summary](../artifacts/chain-map-exact/summary.json);
- the [GB10 compute summary](../artifacts/benchmarks/compute-remediation-e69b077.json);
- molecular result JSON files and the audited Gate-5 record.

At audit time, the large artifacts directory is ignored by Git. Local paths
alone are not a portable evidence package. Reproducibility claims should be
conditioned on exporting and committing a compact checksummed bundle that
contains configurations, summaries, environment metadata, and hashes of any
excluded large files. The PDF corresponding to the previous manuscript is
stale until this Markdown report is finalized and regenerated.

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
   [From Latent Graph to Latent Topology Inference: Differentiable Cell
   Complex
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

---

**Claim boundary.** This report contains one controlled exact-chain-map
demonstration, one protocol-aligned positive routing campaign under privileged
synthetic supervision, scoped negative observations for historical surrogates
and corrected fixed-expert diagnostics, and an initial negative molecular
comparison followed by post-test development. It does not establish
graph-only conversion, a task benefit from mapping-cone or RTD loss, general
molecular superiority, or a categorical/Langlands construction.
