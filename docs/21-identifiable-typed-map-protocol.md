# Identifiable graph-only typed-map pilot

Status: pre-campaign implementation protocol. This experiment is separate from
historical Gate 2 and Gate 3 runs and does not repair them by reinterpretation.
The confirmatory campaign must remain disabled until its shared manifest and
strict summarizer are regenerated against the final source and config hashes.

## Question and claim boundary

This controlled pilot asks whether a graph observation that explicitly
identifies one member of a finite transformation family can drive the
associated oriented cellular and discrete-connection targets while every
learned degree map satisfies declared chain-map equations.

The implementation is a flattened multilayer perceptron. It consumes flattened
node and edge features, predicts weights over twelve hard-coded dihedral
group-action templates, and materializes a linear mixture of those templates.
It is not a general graph neural network, a learned translator between arbitrary
data structures, or evidence of categorical equivalence. The edgewise
\(SO(2)\) rotations are a finite discrete connection target; calling this a general
sheaf translator would overstate the implementation. Nothing here implements
the Langlands program or a Fourier--Mukai transform.

## Cellular annulus and action basis

The source and target share one six-sector cellular annulus:

- 12 vertices: six inner and six outer;
- 18 one-cells: six inner-cycle, six outer-cycle, and six radial edges;
- six quadrilateral two-cells, one per sector;
- real Betti numbers \((1,1,0)\).

In a sector-oriented abstract basis, write inner and outer ring edges as
\(a_i,b_i\), radial edges as \(r_i\), and sector faces as \(q_i\). The cellular
boundary is

\[
\partial q_i=r_i+b_i-r_{i+1}-a_i,
\]

with indices modulo six. Stored edges use the repository canonical
lower-vertex orientation, so the corresponding row signs are canonicalized.
The constructed matrices satisfy \(B_1B_2=0\) exactly in the integer-valued
float64 basis.

The twelve actions are six rotations and six reflections preserving inner and
outer rings. \(P_0\) permutes vertices. \(P_1\) is the induced signed edge
permutation and records orientation reversal. \(P_2\) is not guessed from
vertex parity: for every source face the implementation matches
\(P_1B_2[:,i]\) to exactly one signed target column
\(\pm B_2[:,j]\). Every basis triple is a signed orthogonal permutation, the
basis closes under all 144 pairwise products in degrees zero, one, and two,
and it obeys

\[
B_1P_1=P_0B_1,\qquad B_2P_2=P_1B_2.
\]

## Identifiable graph-only observations

Ordered one-hot node markers encode the images of inner vertices zero and one.
All twelve ordered pairs are distinct, so an analytic lookup decoder is perfect
on clean generated markers. Its held-out accuracy is reported beside the model.
The uniform transformation chance baseline is \(1/12\); the uniform active-face
chance baseline is \(1/6\).

The model input is exclusively:

- node channels: source degree-zero signal, two ordered markers, nuisance noise;
- edge channels: source degree-one signal, source connection angle, nuisance
  noise.

The face coefficient is computed from the graph edge cochain as
\(x_2=B_2^\top x_1\). Target degree signals, target cells, connection transports,
and the class index are supervision only and never enter the forward method.
Held-out targets are

\[
y_i=P_ix_i\quad(i=0,1,2),\qquad \theta^{\prime}=P_1\theta,
\]

with edge transports \(R(\theta^{\prime}_e)\in SO(2)\). Cell activity is
transported by \(\lvert P_2\rvert\). Split seeds and every sample are deterministic,
and full split sizes are multiples of twelve.

This identifiability setup is intentionally easy. The one-hot markers make the
analytic solution exact and create a likely learned-accuracy ceiling. A
pre-freeze CPU development pilot with 1200 training and 240 held-out examples
reached 1.0 held-out transformation accuracy. That diagnostic informed the
full configuration before freeze; it is not a pristine campaign result and
not evidence of general representation conversion.

## Architecturally exact learned maps and cone oracle

For predicted weights \(w_g\), the relaxed maps are

\[
F_i=\sum_g w_gP_i^{(g)}.
\]

Linearity makes both chain equations architectural invariants. Evaluation
fails immediately when the maximum entry of either relaxed or argmax-decoded
residual exceeds \(10^{-5}\).

The mapping-cone signature uses a fixed-tolerance float64 numerical-rank oracle,
not symbolic exact arithmetic. The rank absolute tolerance is \(10^{-7}\), the
chain-map construction tolerance is \(10^{-5}\), and both are recorded. A pure
basis automorphism has cone Betti signature \((0,0,0,0)\). The uniform mixture of
all twelve annulus maps remains an exact chain map but has cone signature
\((0,1,1,0)\), showing that this cone check is no longer vacuous. At temperature
0.05 the regression fixture gives a cone proxy above 0.05 for the uniform map
and more than 0.05 above a pure map. A declared nonuniform wrong mixture has a
finite nonzero cone gradient.

The uniform mixture is symmetry-stationary: its gradient can cancel even though
its cone proxy is nonzero. More generally, cone acyclicity tests whether the
induced homology action is invertible, not which sample transformation is
correct. Thus `cone_only` cannot identify the target map and is retained as a
negative control.

## Objectives, bounded RTD, and calibration

The runner exposes eight explicit modes:

- `task_only`;
- `reconstruction_only`;
- `task_reconstruction`;
- `task_reconstruction_cone`;
- `task_reconstruction_rtd`;
- `cone_only`;
- `rtd_only`;
- `combined`.

The full combined weights are task 1.0, reconstruction 1.0, cell 0.25, discrete
connection 0.25, cone 0.1, and RTD 0.25. The 0.1 cone weight was frozen from
pre-campaign scale diagnostics, not campaign outcomes. At deterministic full
batch initialization on CPU, the supervised weighted gradient norm was 0.4734,
the unweighted cone norm was 0.1193, and the weighted cone norm was 0.01193.
The GB10 diagnostic reproduced 0.4714, 0.1197, and 0.01197. The cone term is
therefore a modest roughly 2.5 percent gradient-scale perturbation at
initialization. These are single-initialization calibration measurements and
do not establish downstream benefit.

Training RTD is the qualified differentiable H0 RTD-style surrogate, not the
published exact cross-barcode algorithm. Its cubic entity scope is strict:
`loss.rtd_training_entities` is 48 in the full config and 24 in smoke, must lie
in `[2,batch_size]`, and selects the deterministic leading prefix of each
already deterministically shuffled batch. A final short batch uses its entire
available prefix. The pairwise-distance implementation forces an exact zero
diagonal. On the GB10, batch 192 with the 48-entity prefix took a median 14.81
ms for RTD-only model forward plus backward across three measured steps and
peaked at 75,073,024 allocated CUDA bytes. This timing excludes optimizer work
and the cone objective; it bounds the requested RTD step specifically.

Held-out exact RTD/SRTD evaluation is separate. Typed representations are cast
to CPU float64 before distances are formed, use a fixed deterministic held-out
prefix, full-matrix 0.9-quantile normalization, and report degrees separately.
No values from different homological degrees are added.

## Pre-specified engineering gate and descriptive comparisons

The primary outcome is an implementation recovery/exactness gate, evaluated
for every `task_reconstruction` and `combined` seed:

- transformation accuracy at least 0.95;
- cell accuracy at least 0.95;
- map MSE at most 0.001;
- relaxed and hard chain residuals at most 0.00001;
- 100 percent of hard decoded cones have signature \((0,0,0,0)\).

These absolute thresholds are engineering gates informed by pre-freeze CPU
development, not hypothesis-test p-values. The summary records the individual
checks. `passed` is null for ablations where the gate is not applicable.

Because clean markers and the pre-freeze pilot imply a likely accuracy ceiling,
`combined` minus baseline is not a superiority primary endpoint. All
`+cone`/`+RTD` accuracy and MSE contrasts are descriptive, unadjusted secondary
analyses. Continuous map MSE and typed degree-zero, degree-one, and degree-two
MSE are prominent secondary endpoints because they can reveal changes hidden
by saturated class accuracy. Five paired seeds, 20260821 through 20260825, and
all eight modes are retained to estimate null and secondary behavior honestly.
No hyperparameter may be tuned to held-out campaign outcomes.

## Execution and artifacts

CPU smoke:

```bash
.venv/bin/python scripts/train_identifiable_maps.py \
  --config configs/identifiable-maps/smoke.yaml
```

One full GB10 run:

```bash
.venv/bin/python scripts/train_identifiable_maps.py \
  --config configs/identifiable-maps/gb10-full.yaml \
  --seed 20260821 \
  --ablation combined \
  --output artifacts/identifiable-maps/campaign/seed-20260821/combined
```

Repeat all eight modes at seeds 20260821 through 20260825 with distinct output
directories. Each training run atomically emits `effective_config.yaml`,
`provenance.json`, `checkpoint.pt`, `history.json`,
`test_predictions.jsonl`, `summary.json`, and `manifest.json`. Provenance
records seed, ablation, command, configuration and source hashes, whole-code
fingerprint, Git state, platform, deterministic mode, and cuBLAS workspace.

The standalone bounded-step diagnostic is:

```bash
.venv/bin/python scripts/benchmark_identifiable_training_step.py \
  --config configs/identifiable-maps/gb10-full.yaml \
  --output artifacts/identifiable-maps/diagnostics/annulus-gb10-training-step.json \
  --device cuda \
  --warmup 1 \
  --iterations 3
```

Checkpoint-specific inference is:

```bash
.venv/bin/python scripts/benchmark_identifiable_maps.py \
  --config artifacts/identifiable-maps/campaign/seed-20260821/combined/effective_config.yaml \
  --checkpoint artifacts/identifiable-maps/campaign/seed-20260821/combined/checkpoint.pt \
  --output artifacts/identifiable-maps/benchmarks/gb10-s1-combined.json \
  --warmup 20 \
  --iterations 100 \
  --batch-size 192 \
  --device cuda
```

The inference benchmark rejects a checkpoint unless its schema, entire embedded
effective configuration, seed, and ablation match the supplied effective
configuration. It measures trained model forward latency only and excludes data
loading, losses, RTD, and the float64 cone numerical-rank oracle.

## Frozen validation and descriptive analysis

The analysis was frozen before campaign execution. At freeze time the campaign
directory contained no run outputs and the campaign manifest had
`execution_enabled: false`. The full configuration SHA-256 is
`22abb205e8a89586b38799d7f7b8d53f0c24cef45f872453533ddf34e20fad73`.

After all forty training runs finish, validate and summarize them with:

```bash
.venv/bin/python scripts/summarize_identifiable_campaign.py \
  --campaign-root artifacts/identifiable-maps/campaign \
  --source-config configs/identifiable-maps/gb10-full.yaml \
  --output artifacts/identifiable-maps/campaign-summary.json
```

The summarizer fails closed on an incomplete or substituted 5-by-8 grid, a
nonempty recorded Git status, differing commit or executable fingerprints,
config drift, a non-GB10/CUDA run, manifest hash or size mismatch, checkpoint
identity mismatch, changed denominators or tolerance, duplicate or reordered
sample IDs, or paired targets that differ across ablations. It recomputes
prediction accuracy, analytic marker accuracy, cell accuracy, map MSE, residual
maxima, cone histograms, and every recorded engineering-gate check. Checkpoints
are loaded with `weights_only=True` and must embed the exact effective config,
ablation, best epoch, schema, and a nonempty model state.

The output records SHA-256 hashes for this protocol and the summarizer itself.
It reports the ten applicable absolute recovery-gate decisions without dropping
failed runs. All `+cone`, `+RTD`, and `+cone+RTD` paired differences, Student-t
intervals, and exact sign tests are descriptive and unadjusted. With five pairs,
the minimum attainable two-sided sign-test p-value without ties is 0.0625. No
multiplicity correction is applied and no structural-loss benefit is inferred.

## Interpretation

This is a pilot of a finite, explicitly identifiable action with exact
architectural chain laws. Passing the engineering gate demonstrates that the
implementation can recover its controlled targets. It does not demonstrate
that structural losses improve a ceiling-limited task, that arbitrary graph,
cell, or connection representations are interchangeable, or that a generally
learned conversion architecture has been achieved.

---

## Appendix: frozen result

Added 2026-08-23. **The protocol above is historical and unchanged.** This
appendix records only the outcome of executing it. The full readable record is
[`docs/23-identifiable-results.md`](23-identifiable-results.md).

The campaign completed all 40 prespecified runs — 8 ablations × 5 seeds — with
zero missing, replaced, or excluded runs, under launch fingerprint
`44408d7adf8467e594879b46e25a1cb7fd89a7e7a5d5f3446548bcbf3ed1096e` at commit
`8021292e97abfec91768f1b5437c883a42c29c60`. The strict summary is
`artifacts/identifiable-maps/campaign-summary.json`, SHA-256
`0cd0defb0b0d41b5f7563c364cfdda62cb72c5e5a845bdd5d1ab76a2e1cb953c`, exported to
`results/summaries/identifiable-campaign-summary.json`.

**The engineering recovery gate passed in 10 of 10 applicable runs.** In every
applicable run transformation accuracy and cell-face accuracy were exactly 1.0,
map errors were at numerical precision, chain-map residuals met the fixed 1e-5
tolerance (largest observed 1.42e-14), and hard cones were acyclic.

Every objective containing task or reconstruction supervision reached 1.000 on
both accuracy endpoints across all five seeds. The two identifiability controls
did not: `cone_only` reached transformation accuracy 0.0815 and `rtd_only`
0.0833, against a chance baseline of 0.0833 — **while producing acyclic hard
cones in 6,000 of 6,000 evaluated examples each.** As the protocol anticipated,
passing a structural check is not evidence of recovery: acyclicity certifies
that the decoded map is invertible within the template family, not that it is
the planted map.

All 21 declared continuous contrast endpoints have Student-t intervals (df = 4)
containing zero. No structural-loss benefit is established. The interpretation
in the protocol stands: this is a pilot of a finite, explicitly identifiable
action with exact architectural chain laws, and the accuracy ceiling — also
reached by a closed-form analytic marker decoder — means the nulls are weak
evidence of absence rather than strong evidence against the structural terms.
