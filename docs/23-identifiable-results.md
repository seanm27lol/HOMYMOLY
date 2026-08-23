# Identifiable typed-map campaign: methods and results

Status: **complete and frozen.** This document is the standalone record of the
40-run identifiable typed-map campaign and the trained benchmarks that followed
it. It is written to be readable without the source code.

The historical protocol is [`docs/21`](21-identifiable-typed-map-protocol.md) and
is unchanged by this record. The journal manuscript is
[`docs/18-paper.md`](18-paper.md).

- Campaign ID: `identifiable-gb10-factorial-v1`
- Generating commit: `8021292e97abfec91768f1b5437c883a42c29c60`
- Launch fingerprint: `44408d7adf8467e594879b46e25a1cb7fd89a7e7a5d5f3446548bcbf3ed1096e`
- Scheduler steps: 56 of 56 completed, receipt sealed
- Date completed: 2026-08-23

## 1. What the experiment does

A **cellular annulus** is a ring-shaped complex — a disc with a hole. Ours is
divided into six sectors and has 12 vertices, 18 edges, and 6 faces, with Betti
numbers (1, 1, 0): one piece, one independent cycle, no enclosed volume.

A **typed map** carries data of each degree to data of the same degree —
vertices to vertices, edges to edges, faces to faces. It is a **chain map** when
it commutes with the boundary operators, meaning it does not tear the complex.
The **exactness defect** is how badly that commuting fails; we call its largest
observed magnitude the **chain-map residual**.

The experiment plants one of twelve dihedral group elements as the true map,
shows the model explicit identifying markers, and asks it to recover which
element was planted. The model is a flattened multilayer perceptron that
predicts weights over twelve hard-coded group-action templates and materializes
their mixture. Every produced map lies in the nullspace of the chain-map
equation by construction, so exactness is architectural rather than learned.

A **mapping cone** packages what a map fails to preserve. An **acyclic** cone
certifies that the map is invertible. As §4 shows, that certificate is much
weaker than it sounds.

## 2. Frozen design

| item | value |
|---|---|
| source config | `configs/identifiable-maps/gb10-full.yaml` |
| config SHA-256 | `22abb205e8a89586b38799d7f7b8d53f0c24cef45f872453533ddf34e20fad73` |
| ablations | 8 (see below) |
| seeds | 20260821, 20260822, 20260823, 20260824, 20260825 |
| runs | 40 (8 × 5); 0 missing, 0 replaced, 0 excluded |
| samples per run | 4,800 train / 1,200 validation / 1,200 test |
| map tolerance | 1e-5 (fixed) |
| RTD training entities | 48 |
| chance, transformation accuracy | 1/12 = 0.0833 |
| chance, cell-face accuracy | 1/6 = 0.1667 |
| analytic marker decoder | 1.000 (closed form, no learning) |

### Objectives

| ablation | task | reconstruction | cone | rtd | role |
|---|:--:|:--:|:--:|:--:|---|
| `task_only` | yes | – | – | – | identification supervision alone |
| `reconstruction_only` | – | yes | – | – | paired-signal supervision alone |
| `task_reconstruction` | yes | yes | – | – | control for the declared contrasts |
| `task_reconstruction_cone` | yes | yes | yes | – | does a cone term add anything? |
| `task_reconstruction_rtd` | yes | yes | – | yes | does an RTD term add anything? |
| `combined` | yes | yes | yes | yes | do both add anything? |
| `cone_only` | – | – | yes | – | identifiability control |
| `rtd_only` | – | – | – | yes | identifiability control |

Three contrasts were declared in advance, each against `task_reconstruction`:
`combined`, `task_reconstruction_cone`, and `task_reconstruction_rtd`. The two
`*_only` cells are controls, not contrasts.

### Engineering recovery gate

An absolute threshold gate, applicable to `task_reconstruction` and `combined`
only. A run passes when **all** of:

| check | threshold |
|---|---|
| cell-face accuracy | ≥ 0.95 |
| transformation accuracy | ≥ 0.95 |
| map MSE | ≤ 1e-3 |
| chain-map residual | ≤ 1e-5 |
| hard-cone acyclic fraction | = 1.0 |

This is an implementation gate, not a comparison between objectives.

## 3. Result: exact recovery

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

All standard deviations across the five seeds are zero for the saturated
accuracies.

**The engineering recovery gate passed in 10 of 10 applicable runs.** Zero
failures. In every applicable run both accuracies were exactly 1.0, map errors
were at numerical precision, chain-map residuals met the 1e-5 tolerance (largest
observed 1.42e-14), and hard cones were acyclic.

Because the analytic marker decoder also reaches 1.000, the ceiling was
attainable without learning. A learned 1.000 is recovery of a known-attainable
answer, not evidence of a powerful model.

## 4. Result: the structural nulls

### 4.1 Adding structure changes nothing

All **21** declared continuous contrast endpoints — three contrasts × seven
registered endpoints — have Student-t intervals (df = 4) containing zero.
Accuracy endpoints are exactly tied at 1.000, so their differences are
identically zero and their sign tests are fully tied.

This is a null under a hard ceiling. The benchmark cannot distinguish "these
terms are useless" from "this task is too easy to reveal their value." Say the
weaker thing.

### 4.2 Structure alone cannot identify the map, and provably cannot

| control | transformation accuracy | chance | cell-face accuracy | chance | hard-cone Betti |
|---|---:|---:|---:|---:|---|
| `cone_only` | 0.0815 | 0.0833 | 0.1697 | 0.1667 | `[0,0,0,0]` in 6,000 / 6,000 |
| `rtd_only` | 0.0833 | 0.0833 | 0.1703 | 0.1667 | `[0,0,0,0]` in 6,000 / 6,000 |

Both controls sit at chance on both accuracy endpoints, with map MSEs fifteen
orders of magnitude worse than the supervised objectives — **and every decoded
cone is acyclic in all 6,000 evaluated examples for both controls.**

This is the sharpest finding in the campaign. A model trained only to make its
mapping cone acyclic succeeds completely at making its mapping cone acyclic and
learns nothing about which map was planted.

**It could not have gone otherwise.** Both structural signals are constant on
the hypothesis space, so their information content about the planted element is
exactly zero — this is a property of the construction, not an optimization
failure. Each of the twelve basis maps is built as a signed permutation in every
degree: `F0` permutes vertices, `F1` permutes edges with an orientation sign,
and `F2` is fixed by matching the mapped cellular boundary to a unique oriented
face (`build_annulus_map_system` in
`src/homymoly/experiments/identifiable_maps.py`). Numerically, all twelve
satisfy `Fᵀ F = I` at degrees 0, 1, and 2, and each has exactly one
unit-magnitude entry per row and column.

- **Cone signal.** A mapping cone is acyclic exactly when its chain map is a
  quasi-isomorphism. A signed permutation is invertible, so all twelve
  candidates are isomorphisms of chain complexes and all twelve have acyclic
  cones. `[0,0,0,0]` in 6,000 of 6,000 examples is that constant made visible.
- **RTD signal.** A signed permutation is orthogonal, hence an isometry, so the
  mapped point cloud carries the same pairwise dissimilarity matrix as the
  source under every candidate (maximum observed distance change 1.5e-07, at
  float precision). The paired matrices RTD consumes are identical across the
  hypothesis space, so the divergence is constant too.

Both `*_only` objectives are therefore fully satisfiable by any of the twelve
candidates, and chance identification is the only attainable outcome.

**The scope of the warning is the point.** Any hypothesis class whose candidate
maps are all invertible has a constant cone-acyclicity signal — and that is
precisely the class a practitioner has in mind when a cone objective looks
attractive. The circularity objection (the class was chosen to be isomorphisms)
restates the finding rather than rebutting it: wanting the learned map to be
structure-preserving is the same property that makes acyclicity useless for
choosing among structure-preserving candidates.

What makes it a trap rather than a triviality is the view from inside the
training loop. The structural objective is driven to full satisfaction, the
diagnostic returns a clean certificate on every single example, and
identification never rises above chance. Nothing in the training signal reveals
the problem. **Acyclicity certifies invertibility within the template family; it
does not certify correctness.** A cone term is a legitimate constraint and a
useless discriminator.

## 5. Trained compute benchmarks

Ten identifiable-map checkpoint benchmarks and five routing benchmarks ran as
scheduler steps 42–56. Both measure a warmed, synchronized CUDA forward pass on
the same GB10. **They report different tail statistics and must never be
pooled**: the identifiable runner records p10/p90, the routing runner records
p95.

Identifiable — batch 192, 20 warmup and 100 timed iterations, model forward only:

| ablation | median latency (mean ± SD, 5 seeds) | p90 (mean) | peak allocated bytes |
|---|---:|---:|---:|
| `combined` | 0.2753 ± 0.0037 ms | 0.2899 ms | 35,069,440 |
| `task_reconstruction` | 0.2762 ± 0.0022 ms | 0.2979 ms | 35,069,440 |

The paired difference is −0.00089 ms, 95% interval [−0.0075, +0.0057] ms. The two
ablations execute the **same inference graph**, so this is a runner-noise check,
not an architectural comparison. Peak allocated memory is byte-identical across
all ten runs.

Routing — batch 64, bfloat16, 100 timed iterations:

| quantity | mean ± SD over 5 seeds |
|---|---|
| dense-to-routed median-latency ratio | 1.532 ± 0.035 |
| routed-to-fastest-fixed median-latency ratio | 2.269 ± 0.043 |

Routed inference is ~1.53× faster than dense three-expert evaluation and ~2.27×
slower than the fastest single fixed route (`fixed_graph` in all five seeds).
Routed peak allocated memory was below dense in every seed (119,415,296 vs
169,401,344 bytes).

> **Correction.** An earlier handoff recorded the routed-to-fastest-fixed ratio
> as `1.863 ± 0.071`. That figure is not reproducible from any artifact in this
> repository under any ratio definition we could construct, and it appears in no
> machine-readable result. The value above is recomputed from the five sealed
> trained benchmarks and is *less* favorable to routing. Two earlier
> `compute-remediation*.json` files record `checkpoint: null` — they timed an
> untrained model and are excluded from every reported compute result.

## 6. Evidence map

Every number above is traceable to a tracked file under `results/`.

| claim | tracked evidence |
|---|---|
| frozen design, 40 runs, per-ablation endpoints, 21 contrast intervals, recovery gate | `results/summaries/identifiable-campaign-summary.json` |
| trained benchmark validation and aggregates, p90/p95 separation | `results/summaries/compute-campaign.json` |
| per-benchmark raw records | `results/benchmarks/identifiable/` (10), `results/benchmarks/routing/` (5) |
| gauge eight-seed corruption contrasts | `results/summaries/gauge-corruption-campaign.json` |
| Gate-3 base nine paired contrasts | `results/gate3/paired_comparison_final.json` |
| corruption reports (per-batch derivatives) | `results/gate3/*/`, `results/gate3g/*/` |
| routing endpoint table | `results/summaries/routing-confirmatory-v2-summary.json` |
| file-level provenance for all of the above | `results/MANIFEST.json` |
| the §4.2 structural argument, machine-checked | `tests/test_identifiable_maps.py::test_cone_and_rtd_signals_are_constant_across_the_whole_hypothesis_space` and `::test_dihedral_basis_is_signed_orthogonal_and_closed_in_all_degrees` |

### Exact provenance

| item | value |
|---|---|
| identifiable campaign commit | `8021292e97abfec91768f1b5437c883a42c29c60` |
| routing campaign commit | `e69b07707950b6abe332366c51fe8c94254899f3` |
| frozen full-config SHA-256 | `22abb205e8a89586b38799d7f7b8d53f0c24cef45f872453533ddf34e20fad73` |
| strict campaign summary SHA-256 | `0cd0defb0b0d41b5f7563c364cfdda62cb72c5e5a845bdd5d1ab76a2e1cb953c` |
| identifiable code fingerprint | `5908adf7d445524c52797d779478945a184b4e1f10056c1d21bcde044bedb360` |
| launch fingerprint | `44408d7adf8467e594879b46e25a1cb7fd89a7e7a5d5f3446548bcbf3ed1096e` |
| environment | NVIDIA GB10, Linux 6.17.0-1026-nvidia aarch64, glibc 2.39, Python 3.12.3, PyTorch 2.13.0+cu130, CUDA 13.0, NumPy 2.5.2, PyYAML 6.0.3 |

All 296 files listed in the sealed scheduler receipt verify against the on-disk
artifacts by byte count and SHA-256. All 48 gauge report, checkpoint, and config
hashes verify. No provenance mismatch was found, so no stop condition fired.

The 8.8 GB `artifacts/` tree is untracked and is not durable evidence by itself;
it is pinned by hash from the tracked bundle but not distributed.

## 7. What this does not establish

- **Not** a general graph neural network. The model is a flattened MLP over
  explicit markers with a hard-coded twelve-element basis.
- **Not** a universal representation translator, and not conversion quality on
  real or out-of-distribution data.
- **Not** general equivalence between graphs, cellular complexes, and sheaves.
- **Not** a learned quasi-isomorphism or exact sequence. The verified identity is
  the chain-map law up to a fixed 1e-5 tolerance on one synthetic template
  family.
- **Not** any Langlands, eigensheaf, Fourier–Mukai, or category-theoretic machine
  learning result. Those remain motivation.
- **Not** a benefit from cone or RTD losses. Both are null here, and neither can
  identify the map alone.
- **Not** a matched-compute Pareto claim. The benchmarks are descriptive timing
  from one runner on one machine, with paths timed in a fixed order and raw
  per-iteration timings not retained.

## 8. Reproducing this record

```bash
python scripts/summarize_gauge_corruption_campaign.py \
  --output results/summaries/gauge-corruption-campaign.json
python scripts/summarize_compute_campaign.py \
  --output results/summaries/compute-campaign.json
python scripts/export_publication_evidence.py --output-root results
python scripts/export_publication_evidence.py --verify-only
```

Each summarizer revalidates provenance before aggregating and fails closed on
any hash, seed, pairing, schema, or receipt mismatch. Re-running over unchanged
evidence reproduces the bundle byte for byte.

Re-executing the 40-run campaign itself is **not** part of release validation and
should not be done absent a real provenance mismatch.
