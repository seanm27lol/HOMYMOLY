# Specification: a generator where conversion is actually learnable

Status: **specification**. The blocking analysis and the identifiability gate in
§4 have been run and are reported with their numbers. The generator itself is not
built.

This document specifies the one piece of data infrastructure that every real test
of the project's original idea is blocked behind.

## 1. The idea this is meant to test

> Different data structures — graphs, cell complexes, sheaves — are different
> views of the same thing. Moving between views loses something. Homological
> algebra has exact tools for measuring what a map destroys: kernels, cokernels,
> mapping cones. Build a system that holds several views, moves between them, and
> uses those defects as real measurements — both to decide which view to use and
> to train the maps.

Nothing in this repository has tested that. Every homological experiment so far
ran on a synthetic annulus mapping to itself, or on abstract cycle complexes.
**None ran on a conversion between the representations the idea is about.**

## 2. Why it was never tested

The existing generator, `ConfirmatoryStructuredSignal`, makes conversion
impossible *by design*. Its own docstring says so:

> "cell labels change whether a fixed, energized probe face is active **while the
> edge cochain remains fixed**"

The code energizes the probe face's boundary in the edge features regardless of
the cell label, then flips which face is active. The cell bit is deliberately
invisible in the graph observation. Sheaf frames and the holonomy defect are
likewise drawn independently of the graph-observed node fields.

This is a *correct* design for the experiment it was built for. Routing needs
each label to require its own view, or the router shortcuts. But it means:

| goal | requires |
|---|---|
| routing (anti-shortcut) | targets **not** inferable from the graph |
| conversion | targets **are** inferable from the graph |

These are contradictory. One generator cannot serve both, which is why the
conversion arm of the project quietly never happened.

**Therefore: this is a new, separate generator.** `ConfirmatoryStructuredSignal`
is unchanged — it underpins the sealed routing evidence and must stay
reproducible.

## 3. Design

### 3.1 The requirement the previous four attempts all failed

Every homological result in this project so far came out **constant** across the
thing being varied, and a constant carries no information:

| setting | what was constant |
|---|---|
| annulus, fixed basis of 12 | cone acyclicity — all candidates invertible |
| annulus, 24 candidates | data could not distinguish twins at low cycle weight |
| C8 → C6 learned map | structure preservation was generic (200/200 random maps) |
| any matched-Betti pair | quasi-isomorphism is generic |

The fix is **variable topology across examples**. If different examples have
different cycle rank, the conversion destroys different amounts, and the defect
becomes a per-example quantity with real spread. §4 verifies this.

### 3.2 Graph layer

A connected random graph per example, with size and density drawn so that the
**cycle rank varies widely** across the dataset. Node features carry a 2-vector
frame; edge features carry a scalar cochain plus noise.

### 3.3 Cell layer — determined by the graph

The 2-cells are a **cycle basis of the graph**. This is canonical: the graph
determines its own cycle space, so the cell complex is a function of the graph
rather than an independent draw.

Face activity is a function of graph observables — for example, a face is active
when the circulation of the edge cochain around its cycle exceeds a threshold.
This is a genuine lifting: computing it requires integrating an edge feature
around a cycle, which is exactly the operation a graph→cell converter must learn.

### 3.4 Sheaf layer — determined by the graph

Edge transports are the rotations aligning the node frames at their endpoints:
the transport on edge `(u, v)` is the rotation carrying `u`'s frame vector to
`v`'s. Cycle holonomy follows from the node features. Nothing is drawn
independently.

### 3.5 The property that makes this a fair test

Because every target is a deterministic function of the graph observation, a
**perfect converter exists**. So a learned converter can be measured against an
attainable ceiling, and a null result means "the homological terms did not help
close a gap that was closable" rather than "the task was impossible" — which is
the criticism that invalidates every previous null in this project.

## 4. Identifiability gate — run, and it passes

The design's central claim is that variable graph topology produces varying
defects. Tested directly on the canonical graph→cell inclusion, over 81 connected
random graphs on 8–14 vertices:

- cycle ranks observed: **0 through 21**
- **21 distinct defect profiles**
- **21 distinct cone Betti signatures**

The degree-1 kernel equals the cycle rank exactly. The inclusion sends every
independent graph cycle to the boundary of a face, so `ker H1` counts precisely
the cycles the conversion destroys.

That is the project's original question — *what does this conversion destroy?* —
answered by a number that varies per example. It is the first setting in this
repository where a homological defect is not constant.

The cokernel is zero throughout, because the inclusion is surjective on homology.
All the variation lives in the kernel. Any claim about unreachable structure
needs a conversion that is *not* an inclusion, and this spec does not provide
one.

## 5. What this unblocks

In dependency order:

1. **Learn a real graph→cell chain map.** `ExactChainMapLayer` already accepts
   different source and target complexes, so the machinery exists.
2. **Measure its kernel and cokernel** with `topology/defects.py`. What did the
   learned conversion destroy, and what could it not reach?
3. **Claim C1** — do those defects predict downstream damage? Currently marked
   untested in the claims ledger, and untestable without this generator.
4. **Claim C2** — does a cone or RTD term improve a learned converter? Every
   previous attempt failed because the defect was constant or generic. Here it is
   neither.
5. **Route on defects instead of on learned utility.** The half of the original
   idea that has never been touched: pick the view using the measured cost of
   converting into it, rather than a distilled utility table.

## 6. Stop conditions

Abandon or redesign if any of these turns out to hold:

- The defect profile stops varying once face activity and transports are made
  functions of the graph — that is the failure mode of all four previous
  attempts, and it must be re-checked on the built generator, not assumed from
  §4.
- A trivial baseline recovers the cell or sheaf structure exactly. If a linear
  probe on edge features reproduces face activity, the conversion is a lookup and
  no learned map is being tested.
- The perfect converter is not representable inside the exact chain-map
  nullspace. Then a null result would again mean "impossible," not "unhelpful."

Each of these is cheap to check and should be checked before any campaign is run.

## 7. Scope

This specifies a generator and the experiments it unblocks. It claims no result.
The identifiability gate in §4 is a property of the canonical inclusion on random
graphs; it is evidence that the design direction is sound, not that any learned
converter works.
