# Gate-5 record: molecular transfer (2026-08-03)

Gate 5 moves to a molecular benchmark after the synthetic gates, testing
the graph route against chemically valid ring/2-cell lifts on OGBG-MOLHIV
with official splits and evaluator. Claims entry: C6 in
[the ledger](08-claims-ledger.md).

## Verdict

**C6 not supported at this architecture and recipe: the ring-lift route
loses to the plain graph route by ~5 AUROC points on the official test
split, consistently across three seeds.**

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
  detection on all splits (only ~0.2% rdkit-unparseable metal complexes
  differ).
- Trainer (`scripts/train_molhiv.py`): BCE, AdamW 3e-4, cosine, batch 64,
  early stopping on validation AUROC (patience 8), official evaluator for
  test. Matched configs (hidden 128, 3 layers, embedding 64); parameter
  counts within 4%. Results: `artifacts/gate5/molhiv_results.json`.
- The sheaf route was not run: per the plan it enters only with an
  explicit molecular interpretation for its local frames, which does not
  exist yet.

## Recorded properties of the official split

The scaffold split is strongly skewed by ring content: train is 79%
ring-bearing while valid/test are ~100% ring-bearing (2/4113 ring-free
molecules on test). Ring-free transfer is therefore **not evaluable** on
the official test split; the test AUROC above is effectively the
ring-bearing subset for both routes — the regime where the cell route was
built to win, and it still loses.

## Interpretation (measured, not speculative)

- The gap is not noise: all three seeds agree (graph worst seed 0.755
  still beats cell best seed 0.742). Valid AUROC is within 0.012, so the
  cell route's deficit is generalization to unseen scaffolds, not
  optimization.
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

- The routing contribution currently stands on the synthetic Gate-2
  result (C3, supported); molecular transfer of the routed system is not
  warranted until a fixed route wins on molecules — routing between a
  losing cell route and the graph route adds nothing.
- Remaining open thread from Gate 3: the gauge-tier observation that
  topological defects carry more independent damage signal when
  translations are consistency-constrained (partial ρ 0.31 vs 0.14) —
  a multi-seed pilot is the cheap next measurement if pursued.

## The molecularly-informed redesign (2026-08-04, v2 evaluation)

The redesign directions above were implemented as an optional
`molecular_mode` on the cell expert: the face encoder additionally
receives a per-face masked max over boundary edge features (the
strongest bond in the ring, which the oriented boundary sum can cancel)
and the ring size as a normalized scalar. Evaluated on the identical
protocol (3 seeds, official split/evaluator,
`artifacts/gate5/molhiv_results_v2.json`):

| route | test AUROC mean ± std | valid mean |
|---|---|---|
| graph | 0.771 ± 0.014 | 0.794 |
| cell (v1) | 0.723 ± 0.017 | 0.782 |
| **cell_molecular (v2)** | **0.757 ± 0.002** | 0.771 |

- The redesign recovers **+0.034** of the v1 cell route's −0.048
  deficit (~70%), and it is the most consistent route of the three
  (test std 0.002 vs 0.014–0.017).
- It still trails the graph route by 0.014 on average, so C6's strict
  form (beat the graph route) remains unsupported — but the direction
  is validated: ring-aware features, not the mere presence of 2-cells,
  carry the molecular signal. Next iteration if pursued: bond-type and
  stereo-conditioned face aggregation (the one-hot bond channels are
  currently pooled indiscriminately) and a second face message round
  rather than deeper readout.
