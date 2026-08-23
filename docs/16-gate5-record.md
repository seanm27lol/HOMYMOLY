# Gate-5 record: molecular transfer (2026-08-03)

> **Validity correction (2026-08-13).** The v1 graph/cell comparison was the
> initial official-test read. V2 and v3 were designed after inspecting earlier
> official-test results, and v2 was restored after inspecting v3. Their test
> AUROCs are post-test development scores, not unbiased final-test estimates.
> This record also corrects ring prevalence, parser-failure handling, and the
> precise v2/v3 feature definitions below.

Gate 5 moves to a molecular benchmark after the synthetic gates, testing
the graph route against chemically valid ring/2-cell lifts on OGBG-MOLHIV
with official splits and evaluator. Claims entry: C6 in
[the ledger](08-claims-ledger.md).

## Verdict

**The initial v1 comparison does not support C6:** the ring-lift route loses
to the graph route by 0.0481 AUROC on the official test split, with graph
winning all three paired initialization seeds. Later redesigns are exploratory
because the test set had already been inspected.

| route | params | valid AUROC (mean) | test AUROC (3 seeds) | test mean ± std |
|---|---|---|---|---|
| graph | 917,954 | 0.794 | 0.777 / 0.780 / 0.755 | **0.771 ± 0.014** |
| cell (ring 2-cells) | 955,842 | 0.782 | 0.713 / 0.742 / 0.713 | **0.723 ± 0.017** |

## Setup (per the plan's constraints)

- `MolecularHIVDataset` (`src/homymoly/data/molecular.py`): OGBG-MOLHIV
  with the official scaffold split (32901/4113/4113) and OGB evaluator.
  One-hot atom/bond features (174/13 dims, OGB conventions), canonical
  undirected edges, rdkit `AtomRings` as 2-cells **only** in the padded
  boundary-edge representation — no long ring was encoded as a
  nonexistent triangle (the plan's prerequisite, migrated in the Gate-5
  build). Ring extraction verified against independent graph-cycle
  presence on all splits. Seven SMILES (0.017%) fail RDKit parsing; they are
  retained with zero faces rather than excluded.
- Trainer (`scripts/train_molhiv.py`): BCE, AdamW 3e-4, cosine, batch 64,
  early stopping on validation AUROC (patience 8), official evaluator for
  test. Matched configs (hidden 128, 3 layers, embedding 64); parameter
  counts within 4%. Results: `artifacts/gate5/molhiv_results.json`.
- The sheaf route was not run: per the plan it enters only with an
  explicit molecular interpretation for its local frames, which does not
  exist yet.

## Recorded properties of the official split

The official split contains 32,901/4,113/4,113 molecules. RDKit produced at
least one `AtomRing` for 31,316/4,111/4,111 molecules
(95.18%/99.951%/99.951%), yielding 95,393/15,894/14,974 ring faces. A separate
graph-cycle-rank audit finds 31,319/4,113/4,113 cyclic graphs: validation and
test contain no acyclic examples. Their two zero-face cases per split are
RDKit parse failures, not ring-free molecules; both test failures are negative
labels, so subgroup AUROC is undefined. Acyclic transfer is therefore not
evaluable on this test set. Ring-presence agreement is 41,120/41,127; exact
`AtomRings` counts and graph cyclomatic rank are different ring bases and were
not expected to match count-for-count.

## Interpretation (measured, not speculative)

- All three paired seeds favor graph (graph worst seed 0.755 still exceeds
  cell best seed 0.742). The validation gap is 0.0119 versus 0.0481 on test;
  this is consistent with greater scaffold-test degradation for cell but does
  not identify optimization versus generalization because training scores
  were not recorded.
- This is coherent with the Gate-3 record: higher-order structural
  machinery has not shown benefit at these scales in any of our three
  data designs (independent, gauge, molecular). The face gating pathway
  that works on synthetic (cell expert 1.0 on its regime) does not
  transfer as-is to molecular rings; a molecularly-informed cell
  architecture (e.g., ring-size-aware aggregation, bond-type-aware
  boundary features) is the redesign direction if this thread is pursued.
- The graph route's 0.771 ± 0.014 is a usable MOLHIV baseline for the
  repository's architecture family.

## Consequences

- The initial synthetic Gate-2 confirmatory interval crossed zero, and its
  later stabilized interval was post-selection. A fresh frozen routing
  campaign is reported separately; this molecular experiment does not add
  evidence for routed multi-representation benefit.
- Historical Gate-3 partial correlations are invalidated by the 2026-08-13
  audit and provide no molecular design signal.

## The molecularly-informed redesign (2026-08-04, v2 evaluation)

The redesign directions above were implemented as an optional
`molecular_mode` on the cell expert: the face encoder additionally receives
an elementwise masked max over learned boundary-edge embeddings and normalized
ring size. It was evaluated after the v1 official-test result on the same
split (3 seeds,
`artifacts/gate5/molhiv_results_v2.json`):

| route | test AUROC mean ± std | valid mean |
|---|---|---|
| graph | 0.771 ± 0.014 | 0.794 |
| cell (v1) | 0.723 ± 0.017 | 0.782 |
| **cell_molecular (v2)** | **0.757 ± 0.002** | 0.771 |

- Descriptively, v2 is +0.0343 above v1 and has the smallest observed
  initialization-seed SD (0.0015). Its validation mean is lower than v1
  (0.7713 vs 0.7817), and its official-test score is post-test development
  evidence.
- It still trails the graph route by 0.014 on average, so C6's strict
  form (beat the graph route) remains unsupported — but the direction
  generates a hypothesis that ring-aware pooling, not the mere presence of
  2-cells, may matter. It does not validate that direction on held-out data.
  Next iteration if pursued: bond-type and
  stereo-conditioned face aggregation (the one-hot bond channels are
  currently pooled indiscriminately) and a second face message round
  rather than deeper readout.

## v3 exploratory iteration: raw bond-type counts

Iteration two added raw per-ring counts for the five OGB bond-type one-hot
categories to the v2 face encoder. It was evaluated after inspecting v2 on
the same official test split
(`artifacts/gate5/molhiv_results_v3.json`):

| variant | test AUROC mean ± std | valid mean |
|---|---|---|
| v2 (elementwise edge-embedding max + ring size) | **0.757 ± 0.002** | 0.771 |
| v3 (+ raw bond-type counts) | 0.729 ± 0.025 | 0.761 |

v3 is descriptively worse on mean, seed SD, and validation. Because this was
adaptive reuse of the test set, it is not a clean held-out ablation. A
scaffold-specific bond-distribution effect is one hypothesis, not a measured
cause. The shipped molecular default reverts to v2, but any next molecular
iteration must select architectures on a fresh locked split or external
benchmark before reporting a final test score.
