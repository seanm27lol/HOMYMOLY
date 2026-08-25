# HOMYMOLY

**Boundary compatibility as a prior on learned edge-to-cycle liftings.**

## Thesis

Revised 2026-08-24 to state what survived testing rather than what was originally
hoped for. The previous thesis — that cone- or RTD-inspired defects would improve
conversion and select views during routing — was tested across two campaign
families and **did not survive as stated**. See
[`docs/28`](docs/28-conversion-campaign-results.md) for the prospectively locked
same-generator-family replication
and [`docs/00`](docs/00-original-idea.md) for how this relates to the original
idea.

> **When an edge-to-cycle-coordinate lifting is learned from scarce paired data,
> a boundary-of-boundary penalty derived from the input is a strong prior on
> that lifting. Along the prespecified penalty path, the
> resulting compatibility defect covaries with held-out damage within the
> studied synthetic family.**

The first claim comes from a prospectively specified primary analysis in a
same-family replication, with a disclosed sum-versus-mean implementation
deviation. It is not independent confirmation: the hypotheses, directions,
endpoint, training size, and weights were informed by exploration on the same
generator family, and the exploratory seeds were not retained, so overlap with
the frozen seed block cannot be audited. The second is a prespecified secondary
regularization-path association with an unadjusted interval:

| claim | evidence |
|---|---|
| boundary compatibility improves an edge-to-cycle lifting | Bonferroni-adjusted interval **[−2.749, −1.511]** on `log10` held-out map-recovery error |
| compatibility defect covaries with damage along the penalty path (**C1**) | within-seed Pearson correlation **+0.961**, unadjusted CI [+0.935, +0.987], **29/29** eligible seeds |

For C1, the nine fits within each topology reuse the same data and initialization
and differ in the common driver `lambda`; they are not independent observations.
The result does not establish independent predictive information or off-path
calibration.

The continuous task fits `W: R^E -> R^F` from edge signals to coordinates in a
chosen cycle basis; `W^T` is a candidate `d2`. This is a degree-changing lifting,
not a degree-preserving typed chain map or a conversion between complexes. The
annulus experiment, separately, does select genuine typed chain maps.

The executed penalty is `mean((W·B1ᵀ)²)`. Its zero set annihilates
coboundaries, so the implied operators satisfy `d∘d = 0`; at finite weight it
encourages rather than enforces that set, and all `F×E` entries remain
trainable. The
frozen key `exact` is historical shorthand: this is **not exactness of a
sequence**, because `W = 0` also has zero defect without
`im Wᵀ = ker B1`. The penalty does not directly use `B2` or response labels, but
`B1` determines the target cycle subspace and is strong input-derived structural
side information; the unpenalized baseline ignores it. With the known
deterministic generator, `B2` is algorithmically recoverable from the graph. No
analytic cycle-basis, Hodge-projection, or nullspace oracle was included. Frozen
protocol 27 wrote the Frobenius sum rather than the executed elementwise mean;
the difference is disclosed and the protocol file remains immutable.

The large effect occurs in favorable scarce-probe geometry. With 16 training
probes, 21/29 eligible seeds have `E > 16`, 24/29 have cycle dimension
`F <= 16`, and 16/29 move from the former underdetermined ambient system to the
latter potentially identifiable hard-subspace system (median `E = 23`,
`F = 11`; five have `F > 16`). The finite penalty remains shrinkage of a full
`F x E` matrix, not a hard parameter reduction.

## What did not survive

- A **singular-value cone surrogate**,
  `exp(−2·σ_min(W))`, harms held-out lifting-recovery error at the tested weight;
  this is not a test of mapping-cone homology.
- An **RTD-inspired normalized pairwise-distance surrogate** shows no detected
  improvement. It is target-misaligned here: it asks the rank-`F` lifting to
  preserve full source geometry even though the planted truth discards cut-space
  components. This is not an equivalence result and says nothing negative about
  published RTD/SRTD.
- Exact mapping-cone homology and the corrected RTD/SRTD implementation remain
  useful as **diagnostics**. Their mathematical names are not transferred to the
  conversion surrogates.
- The defect-routing H5 result is **non-informative**: its frozen per-row
  oracle denominator forces every endpoint observation to be nonnegative, while
  support required an interval below zero. Its 28 rows are also two weights for
  each of 14 topology clusters, so the former df=27 interval and 25/28 count are
  pseudoreplicated and withdrawn from inferential use. The defensible routing
  result is a **compute** comparison: over five seeds, mean dense/routed is 1.532
  (t95 CI [1.489, 1.575]) and mean routed/fastest-fixed is 2.269
  ([2.215, 2.322]).

## Retrospective no-fit diagnostics proposed for future screens

The post-campaign audit identified two truth-aware checks that require a known
answer and supplied representative candidates but no model fitting. They were
**not** prospectively applied to choose the reported objectives, so they do not
provide prospective validation of the observed outcomes. We propose applying
them to synthetic or oracle-known tasks before future protocols are frozen:

1. the known truth scores higher than a supplied candidate, showing that the
   term favors at least one supplied alternative;
2. no variation is detected over the supplied candidate sample, so that sample
   provides no evidence that the term can discriminate among those candidates.

`homymoly.topology.screen_structural_term` implements these retrospective
diagnostics as candidate future screens. Their verdicts are relative to the
supplied sample, not proofs about the entire hypothesis class. They are neither
necessary nor sufficient conditions for improved generalization: bias can help
finite-sample prediction, and a varying term that ranks the supplied truth first
can still fail. Use the screen to inspect a proposal, then test it with held-out
data.

This repository contains the research specification, the executable Stage 1
foundation, three structured experts, graph-to-cell/sheaf translators, a
cost-aware router, degree-specific RTD/SRTD references, exact finite chain-map
layers, a conversion generator, a corruption suite, and resumable GB10
experiments.

## Current outcome

The 40-run identifiable typed-map campaign is **complete and frozen**. Full
record: [`docs/23`](docs/23-identifiable-results.md). Manuscript:
[`docs/18-paper.md`](docs/18-paper.md).

**The implementation decodes the planted map perfectly.** On a synthetic
six-sector cellular annulus (12 vertices, 18 edges, 6 faces, Betti (1, 1, 0))
with a finite dihedral family of twelve exact three-term maps, every objective
containing task or reconstruction supervision reached transformation accuracy
1.000 and cell-face accuracy 1.000 on all five seeds. Across these six
objectives, mean map MSE ranges from `2.618e-17` to `2.504e-8`; the cone-only
and RTD-only controls have means 0.109 and 0.191, about 7–16 orders higher
depending on the comparison. The prespecified engineering recovery gate applies
only to `task_reconstruction` and `combined` (five seeds each) and passed **10 of
10** applicable runs. An analytic marker decoder also reaches 1.000, so this is
recovery of a known-attainable ceiling, not evidence of a powerful model.

**The structural results are negative, and they are the interesting part.**

- Adding the annulus cone proxy, the RTD-style training surrogate, or both
  changed nothing: all 21 declared continuous contrast intervals contain zero,
  against a saturated accuracy ceiling.
- Under the two single-loss controls, the model sits at chance —
  transformation accuracy 0.0815 (cone-only) and 0.0833 (RTD-only) against a
  0.0833 baseline — **while producing acyclic cones in 6,000 of 6,000 evaluated
  examples.** Every hard decoded candidate is a signed permutation, so exact cone
  acyclicity cannot distinguish the twelve decoded vertices. This does not make
  the differentiable `cone_soft_betti` objective constant over soft mixtures.
  The RTD-only loss also varies and consumes target batch geometry, so it is not
  an unsupervised control and has no analogous constancy proof. The chance
  accuracies are empirical negative results, not consequences of a theorem about
  the soft training losses.

**Routing** (frozen five-seed v2 campaign): hard-minus-best-fixed margin
**+0.1098** (SD 0.0117; Student-t 95% CI [0.0953, 0.1243]), meeting the frozen
decision rule. Training used privileged latent-regime distillation and inference
used structured target views, so this is not a graph-only or conversion result;
an aborted pre-commit seed-20260906 attempt makes it protocol-aligned rather
than pristine preregistration. See
[`docs/19`](docs/19-routing-confirmatory-v2-protocol.md).

**Corruption diagnostics** are fixed-expert embedding diagnostics only and test
no translator, learned map, or conversion. All nine Gate-3 base paired intervals
contain zero, and all three eight-seed gauge intervals contain zero (exact sign
tests p ≥ 0.727). No multiplicity adjustment is applied anywhere.

**Trained compute** (GB10, five seeds): mean dense/routed speed ratio is 1.532
(Student-t 95% CI [1.489, 1.575]); mean routed/fastest-fixed latency ratio is
2.269 ([2.215, 2.322]). The identifiable and routing runners report p90 and p95
respectively and are never pooled.

## What this does not show

- Not a general graph neural network — the model is a flattened MLP over
  explicit markers selecting from a hard-coded twelve-element basis.
- Not a universal representation translator, and no lifting quality on real
  or out-of-distribution data.
- Not general equivalence between graphs, cellular complexes, and sheaves.
- Not a learned quasi-isomorphism or exact sequence; the verified identity is
  the chain-map law up to a fixed 1e-5 tolerance on one synthetic template.
- Not any Langlands, eigensheaf, Fourier–Mukai, or category-theoretic machine
  learning result. Those remain motivation, not results.
- Not a benefit from mapping-cone homology or published RTD/SRTD losses: the
  historically named conversion campaign tested two explicitly narrower surrogates. Also not a
  matched-compute Pareto claim.

## Five-minute smoke path

No GPU required. Installs the package, runs the suite, checks the exact oracles,
and verifies the tracked evidence bundle against its manifest:

```bash
uv sync --frozen --extra dev --python 3.12.3
.venv/bin/python -m pytest -q
.venv/bin/homymoly validate-foundation --config configs/stage1.yaml
.venv/bin/python scripts/export_publication_evidence.py --verify-only
```

The last command re-hashes all 48 tracked evidence files and reports
`{"verified": true, ...}` when the bundle matches its manifest.

## Where the evidence lives

Tracked, checksummed, and readable without a GPU — 48 files, 2.85 MB:

| path | contents |
|---|---|
| `results/MANIFEST.json` | path, byte count, SHA-256, generating commit, and command for every file |
| `results/summaries/` | strict compact summaries: identifiable, gauge, compute, routing |
| `results/gate3/`, `results/gate3g/` | gate decisions and per-batch corruption-report derivatives |
| `results/benchmarks/` | ten identifiable and five routing trained benchmark records |

Corruption reports are exported as per-batch derivatives: the `per_example`
array is dropped and the `per_batch` array — the unit of analysis — is kept. This
is lossless for every published statistic, verified by recomputing the adjusted
partial Spearman from the retained rows and matching to 1e-15. Each derivative
records the SHA-256 of the untruncated source.

The 8.8 GB `artifacts/` tree (checkpoints, per-example dumps, histories,
scheduler logs) is intentionally untracked and is not durable evidence by
itself.

The read-only reviewer snapshot contains the source and this compact evidence
bundle. It can rehash and trace every reported value, rerun the CPU conversion
campaign, run the tests, and regenerate the paper. It cannot independently rerun
the GB10 annulus campaign or trained-checkpoint benchmarks without separately
supplied raw artifacts/checkpoints and compatible GPU hardware.

## Should HOMYMOLY use RTD?

**Yes as a diagnostic and reference; not as an established training loss.**
Representation Topology Divergence (RTD) remains an important foundation for
the project, but it is a component and baseline rather than the novelty claim.

HOMYMOLY will use RTD in four roles:

1. as an independently implemented, hand-validated diagnostic for paired
   metric representations;
2. as a reference against which narrower distance-preservation surrogates are
   identified honestly;
3. as a possible future auxiliary loss only when the published construction is
   actually implemented and passes held-out evaluation;
4. as a source of established constructions, tests, and differentiation strategies.

RTD compares the Vietoris–Rips filtrations induced by two paired point-cloud
representations. HOMYMOLY's broader research program considers explicit typed
transformations between graph, cell, and sheaf complexes, structural diagnostics,
task utility, and compute-aware routing. The present paper establishes only the
narrow boundary-compatibility result stated above. Directional RTD remains useful
for asymmetric diagnosis; SRTD is a natural symmetric comparison.

## Repository map

- [Original idea and reconstruction](docs/00-original-idea.md)
- [Research brief](docs/01-research-brief.md)
- [Literature review](docs/02-literature-review.md)
- [Mathematical contract](docs/03-mathematical-contract.md)
- [Proposed method](docs/04-method.md)
- [RTD integration](docs/05-rtd-integration.md)
- [Experimental protocol](docs/06-experimental-protocol.md)
- [Derived and Langlands guardrails](docs/07-derived-langlands-scope.md)
- [Claims ledger](docs/08-claims-ledger.md)
- [Stage 1 runtime build](docs/09-stage1-build.md)
- [GB10 experimental plan](docs/10-gb10-experimental-plan.md)
- [Stage 1 validation record](docs/11-stage1-validation.md)
- [Gate 2 training and automatic GB10 launch](docs/12-gate2-training.md)
- [Gate 2 run handoff](docs/13-gate2-run-handoff.md)
- [Gate 2 review](docs/14-gate2-review.md)
- [Gate 3 record](docs/15-gate3-record.md)
- [Gate 5 record: molecular transfer](docs/16-gate5-record.md)
- [Gate 2 confirmatory campaign](docs/17-gate2-confirmatory.md)
- [Paper: Boundary Compatibility, Not Acyclicity](docs/18-paper.md)
- [Routing confirmatory v2 protocol](docs/19-routing-confirmatory-v2-protocol.md)
- [Audit remediation and continuation record](docs/20-audit-remediation.md)
- [Identifiable typed-map protocol](docs/21-identifiable-typed-map-protocol.md)
- [Release handoff](docs/22-overnight-handoff.md)
- [**Identifiable typed-map results record**](docs/23-identifiable-results.md)
- [Continuous-map probe](docs/24-continuous-map-probe.md)
- [Conversion generator specification](docs/25-conversion-generator-spec.md)
- [Exploratory boundary-compatibility study](docs/26-exactness-as-a-prior.md)
- [Frozen edge-to-cycle lifting protocol (historical filename)](docs/27-conversion-campaign-protocol.md)
- [Corrected edge-to-cycle lifting results](docs/28-conversion-campaign-results.md)
- [Post-campaign audit corrections](docs/29-audit-corrections.md)
- [Bibliography](references.bib)

## Stage 1 foundation

For GB10, use the pinned NGC base described in the [Stage 1 runtime build](docs/09-stage1-build.md); a fresh PyPI environment may resolve a different CUDA stack. In an existing compatible PyTorch environment, install the development package and run the exact-oracle gate:

```bash
python -m pip install -e . --no-deps
python -m pytest
homymoly validate-foundation --config configs/stage1.yaml
```

The gate verifies deterministic balanced data, canonical oriented structures, the boundary law, graph-to-cell chain maps, mapping-cone chain laws, connection-sheaf operators, and hand-checkable Betti and holonomy sentinels. These are implementation invariants, not evidence that a structural loss improves learning.

## Current architecture and experiments

The system keeps two claims separate:

- The routing experiment asks whether a cheap router over the available
  structured observations can select one expert per example. Its summaries
  include graph features, candidate/active-face statistics, and sheaf
  transport statistics; no label or regime tensor is an inference input.
  The routing-v2 campaign used privileged latent-regime supervision during
  training to distill a regime-by-expert utility table. The canonical config
  exposes a per-example utility mode that removes that training privilege,
  but the reported v2 result does not use that mode.
- Historical graph-to-cell/sheaf translators hold target cell activity and sheaf
  transports out of translator inputs and predict them from the graph view, while
  supplying candidate face incidence. In the current synthetic generator,
  those held-out targets are not identifiable from the graph inputs, so this
  remains an implemented reconstruction objective, not conversion evidence.
- The historically named continuous conversion campaign is separate. For each
  topology it fits a full `F×E` matrix `W` from 16 paired observations and
  observes `B1`. `B2` is not supplied explicitly to the optimizer, but paired
  responses `Y = X B2 + epsilon` provide supervised signal about it. Evaluation
  uses held-out pairs from that same topology. A finite boundary-compatibility penalty encourages rows
  toward the cycle subspace; it does not hard-constrain `W`, identify sequence
  exactness, or learn a shared converter for unseen topologies. Here `W` maps
  edge signals to cycle-basis coordinates and `W^T` is a candidate `d2`; it is
  not a typed chain map. Training labels have Gaussian noise sigma 0.02, while
  held-out targets are noiseless, so the endpoint measures map recovery.
- The annulus experiment uses a distinct finite twelve-map architecture whose
  chain-map equations hold by construction and whose hard mapping cones are
  evaluated exactly.

On the GB10, trained-checkpoint benchmarks over five seeds give mean
dense/routed 1.532 (Student-t 95% CI [1.489, 1.575]) and mean
routed/fastest-fixed 2.269 ([2.215, 2.322]), at batch 64 in
bfloat16. Routed peak allocated memory is below dense in every seed. These are
descriptive medians from one runner on one machine with paths timed in a fixed
order; they do not establish an accuracy/compute Pareto win. Earlier
`compute-remediation*.json` benchmarks recorded `checkpoint: null` — they timed
an untrained model and are excluded from all reported results.

## Contribution boundary

HOMYMOLY does **not** claim to introduce:

- topological losses for machine learning;
- mapping-cone comparison of neural representations;
- learned graph liftings;
- neural cellular sheaves;
- categorical descriptions of neural architectures; or
- mixture-of-experts routing.
- equality-constrained least squares, cycle-space projection, or Hodge theory.

The supported contribution is narrower: an auditable empirical evaluation of an
input-derived boundary-compatibility penalty on one deterministic synthetic
lifting family, plus carefully bounded negative and compute results. Classical
constrained least squares and Hodge theory explain the subspace mechanism; the
paper does not claim a new theorem.

## Status

Stage 1, the fixed experts, and the identifiable typed-map campaign are
complete. No further large training run is planned; the remaining open work is
scientific, not computational.

- The identifiable campaign gives perfect decoded transformation/cell accuracy
  for all six supervised objectives. Its 10/10 recovery gate applies only to
  `task_reconstruction` and `combined`, and an analytic decoder reaches the same
  saturated ceiling. The structural contrasts are therefore weak nulls, and the
  informative next step is a harder benchmark where the correct map is *not*
  analytically attainable — not more seeds on this one.
- The continuous lifting campaign supports a boundary-compatibility penalty
  over the unpenalized fit, subject to the disclosed sum-versus-mean protocol
  deviation. It is a prospectively locked same-family replication, not
  independent confirmation: design choices were outcome-informed, and
  exploratory seed overlap is unverifiable. Its compatibility/error result is a
  prespecified, unadjusted regularization-path association; `lambda` is a common
  driver, so independent prediction and off-path calibration remain open. The
  baseline ignores `B1`, and no analytic cycle-basis/Hodge/nullspace oracle was
  tested.
- The frozen defect-routing H5 endpoint is non-informative because success was
  mathematically impossible under its per-row oracle denominator. Its 28 rows
  also pseudoreplicate 14 topology clusters at two weights, so the former df=27
  interval and 25/28 count are withdrawn. It requires a redesigned,
  cluster-aware follow-up rather than reinterpretation.
- The five-seed routing-v2 result supports only the scoped historical
  regime-distilled, structured-view routing endpoint; n=5 leaves distributional
  assumptions uncheckable and the exact two-sided sign-test sensitivity floor is
  p=0.0625.
- The published scalar RTD convention is degree 1 with full-matrix 0.9-quantile
  normalization; multi-degree results are returned explicitly rather than
  summed. All pre-audit "exact SRTD" corruption scalars remain withdrawn.
- Molecular results are exploratory because the official test split was
  consulted across architecture iterations, and that split contains no acyclic
  graphs.
- Literature and novelty conclusions are research judgments, not a patent search
  or a guarantee of priority.
