# HOMYMOLY

**Homological routing between structured representations.**

HOMYMOLY is a research project investigating whether a machine-learning system can choose among vector, graph, cell-complex, and cellular-sheaf representations while explicitly measuring the structural damage caused by each conversion.

The working thesis is:

> A task- and compute-aware mixture of structured experts can learn typed representation changes, constrain those changes to behave like chain maps, and use mapping-cone defects to quantify topology lost or introduced during routing.

This repository contains the research specification, the executable Stage 1
foundation, three structured experts, graph-to-cell/sheaf translators, a
cost-aware router, degree-specific RTD/SRTD references, exact finite chain-map
layers, a corruption suite, and resumable GB10 experiments.

## Current outcome

The 40-run identifiable typed-map campaign is **complete and frozen**. Full
record: [`docs/23`](docs/23-identifiable-results.md). Manuscript:
[`docs/18-paper.md`](docs/18-paper.md).

**The implementation recovers the planted map exactly.** On a synthetic
six-sector cellular annulus (12 vertices, 18 edges, 6 faces, Betti (1, 1, 0))
with a finite dihedral family of twelve exact three-term maps, every objective
containing task or reconstruction supervision reached transformation accuracy
1.000 and cell-face accuracy 1.000 on all five seeds, with map MSE at the 1e-16
level. A prespecified engineering recovery gate passed **10 of 10** applicable
runs. An analytic marker decoder also reaches 1.000, so this is recovery of a
known-attainable ceiling, not evidence of a powerful model.

**The structural results are negative, and they are the interesting part.**

- Adding a mapping-cone term, an RTD term, or both changed nothing: all 21
  declared continuous contrast intervals contain zero, against a saturated
  accuracy ceiling.
- Trained on structural losses *alone*, the model sits at chance —
  transformation accuracy 0.0815 (cone-only) and 0.0833 (RTD-only) against a
  0.0833 baseline — **while producing acyclic cones in 6,000 of 6,000 evaluated
  examples.** This is not an optimization failure. Every candidate map is a
  signed permutation, hence an invertible isometry, so cone acyclicity and RTD
  are both *constant* across the hypothesis space and carry exactly zero
  information about which map was planted. Cone acyclicity certifies that the
  decoded map is invertible; it does not certify that it is the correct map —
  and the same degeneracy holds for **any** hypothesis class of invertible maps,
  which is the setting where a cone objective looks most attractive.

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

**Trained compute** (GB10): routed inference is 1.532 ± 0.035× faster than dense
three-expert evaluation and 2.269 ± 0.043× slower than the fastest fixed route.
The identifiable and routing runners report p90 and p95 respectively and are
never pooled.

## What this does not show

- Not a general graph neural network — the model is a flattened MLP over
  explicit markers selecting from a hard-coded twelve-element basis.
- Not a universal representation translator, and no conversion quality on real
  or out-of-distribution data.
- Not general equivalence between graphs, cellular complexes, and sheaves.
- Not a learned quasi-isomorphism or exact sequence; the verified identity is
  the chain-map law up to a fixed 1e-5 tolerance on one synthetic template.
- Not any Langlands, eigensheaf, Fourier–Mukai, or category-theoretic machine
  learning result. Those remain motivation, not results.
- Not a benefit from cone or RTD losses, and not a matched-compute Pareto claim.

## Five-minute smoke path

No GPU required. Installs the package, runs the suite, checks the exact oracles,
and verifies the tracked evidence bundle against its manifest:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
homymoly validate-foundation --config configs/stage1.yaml
python scripts/export_publication_evidence.py --verify-only
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

## Should HOMYMOLY use RTD?

**Yes.** Representation Topology Divergence (RTD) is one of the strongest foundations for the project, but it is a component and baseline rather than the novelty claim.

HOMYMOLY will use RTD in four roles:

1. as an independently implemented, hand-validated diagnostic for paired
   metric representations;
2. as an auxiliary topology-preservation loss when two views have one-to-one sample correspondence;
3. as a baseline against which a map-aware mapping-cone loss is tested;
4. as a source of established constructions, tests, and differentiation strategies.

RTD compares the Vietoris–Rips filtrations induced by two paired point-cloud representations. HOMYMOLY's proposed extension operates on explicit typed transformations between graph, cell, and sheaf complexes and combines structural defects with task utility and compute-aware routing. Directional RTD remains useful for asymmetric diagnosis; SRTD is a natural symmetric comparison.

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
- [Paper: Typed Representation Routing with Homological Structure](docs/18-paper.md)
- [Routing confirmatory v2 protocol](docs/19-routing-confirmatory-v2-protocol.md)
- [Audit remediation and continuation record](docs/20-audit-remediation.md)
- [Identifiable typed-map protocol](docs/21-identifiable-typed-map-protocol.md)
- [Release handoff](docs/22-overnight-handoff.md)
- [**Identifiable typed-map results record**](docs/23-identifiable-results.md)
- [Bibliography](references.bib)

## Stage 1 foundation

For GB10, use the pinned NGC base described in the [Stage 1 runtime build](docs/09-stage1-build.md); a fresh PyPI environment may resolve a different CUDA stack. In an existing compatible PyTorch environment, install the development package and run the exact-oracle gate:

```bash
python -m pip install -e '.[dev]'
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
- The conversion experiment holds target cell activity and sheaf transports
  out of translator inputs and predicts them from the graph view, while
  supplying candidate face incidence. In the current synthetic generator,
  those held-out targets are not identifiable from the graph inputs, so this
  is an implemented reconstruction objective awaiting an identifiable
  benchmark—not conversion evidence. A separate exact-chain-map layer
  parameterizes degree maps in the nullspace of the chain-map equations and
  evaluates their mapping cones exactly.

On the GB10, trained-checkpoint benchmarks over five seeds measure routed
inference at 1.532 ± 0.035× the throughput of the dense three-expert path and
2.269 ± 0.043× slower than the fastest single fixed route, at batch 64 in
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

The candidate contribution is their disciplined intersection: dynamic, cost-aware routing among genuinely different structured representations, using typed transformations, categorical coherence, and map-aware homological defects.

## Status

Stage 1, the fixed experts, and the identifiable typed-map campaign are
complete. No further large training run is planned; the remaining open work is
scientific, not computational.

- The identifiable campaign recovers its planted map exactly and passes its
  recovery gate 10/10, but under a saturated ceiling that an analytic decoder
  also reaches. The structural contrasts are therefore weak nulls, and the
  informative next step is a harder benchmark where the correct map is *not*
  analytically attainable — not more seeds on this one.
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
