# Original idea and reconstruction

## Raw transcription

> Idea Homologies are based Off of 10 complexes which are measuring the exactness and Codes sadness of between transformations between different data structures so logical factor spaces you know, sheafs, groups, anything else you can think of to do this by taking the quotient between the colonel and the image of the all the possible outcomes of its example and matrixes so what if you built a architecture for machine, learning models to like deeper than just the topology of it because I’ve seen there are top logical deep learning applications from it, but think about the representations of data structures that you can switch between and you can see how constant changing them with their exactness and their co-exactness I feel like there’s definitely something you can do with that and then you also like a proud too you know the work of the langlands program And everything else that can really come out of this when it comes to this pure math, but I think about it I think about like changing between different types of data structures when you’re going through, at times sheafs can be more efficient, then spaces, all of the process data in different efficient ways and there’s I think I should come back to it. Core idea would be measuring like the loss and the reverse the loss. The co-exactness and exactness between these different structures. There’s something about like just needed for categories and derived categories have the Fourier mukai transform, which allows them to have Eigensheafs which could be helpful. (Some of the idea was lost to the transcriber)

## Likely transcription corrections

- “10 complexes” → **chain complexes**.
- “colonel” → **kernel**.
- “codes sadness” → probably **co-exactness**.
- “top logical” → **topological**.
- “Fourier mukai” → **Fourier–Mukai**.
- “Eigensheafs” → **eigensheaves**.

## Reconstructed research question

Can a learning system maintain several mathematical views of the same underlying data—such as vectors, graphs, higher-order complexes, and cellular sheaves—and dynamically select or transform between those views while measuring:

1. which structural features are destroyed by a forward conversion;
2. which features in the target are not reachable from the source;
3. what fails to return under a reverse conversion;
4. whether different conversion paths agree; and
5. whether the benefit of a richer representation justifies its computational cost?

The proposed mathematical language is a typed category of representations whose usable objects are linearized into chain or cochain complexes. Representation changes become chain maps or functors. Kernels, cokernels, mapping cones, cycle/unit–counit defects, and persistent topology provide different—not interchangeable—measurements of conversion behavior.

---

## What the idea turned into (2026-08-24)

The reconstructed question above asked five things. After two campaign families,
here is what each became. This section exists so the original document says out
loud what survived, rather than leaving the reader to infer it.

| the original asked | outcome |
|---|---|
| which structural features a forward conversion destroys | **answered.** The kernel of the induced homology map counts them, and the measurement is now implemented in `topology/defects.py` |
| which target features are unreachable | implemented as the cokernel; **untested** as a predictor |
| what fails to return under a reverse conversion | machinery exists; **no ablation** |
| whether conversion paths agree | **never tested** |
| whether a richer representation justifies its cost | **no on accuracy, yes on compute.** Routed inference is 1.532× faster than dense evaluation but 2.269× slower than the fastest single fixed route |

The transcript's own phrase — *"taking the quotient between the [kernel] and the
image"* — is homology, and it turned out to be closer to the mark than the
constructions built on top of it.

**What survived is one clause, sharpened.** Exactness — the requirement that a
conversion commute with the boundaries, so that `d∘d = 0` on the complex it
implies — is a strong prior on a learned conversion when paired data is scarce,
and its violation is a calibrated measure of the damage done. Confirmed under a
preregistered protocol: adjusted interval [−2.802, −1.458] on `log10` held-out
error, and a within-topology defect/damage correlation of +0.854 across 29 of 29
topologies.

**What did not survive is the part the project spent its campaigns on.** The
mapping cone fails as a training term — confirmatorily it *harms* — and RTD is
inert. Selecting a view by measured conversion cost is not supported. Nothing in
the derived-category, Fourier–Mukai, eigensheaf, or Langlands direction was ever
implemented, and nothing here bears on it.

There is an irony worth recording. Exactness was present from the first
executable version, enforced architecturally by `ExactChainMapLayer`, and was
never itself the object under test. The mechanism that works was in the
architecture the whole time while every campaign was aimed at the two mechanisms
that do not.

Full records: [`docs/26`](26-exactness-as-a-prior.md) for the exploratory work,
[`docs/27`](27-conversion-campaign-protocol.md) for the frozen protocol, and
[`docs/28`](28-conversion-campaign-results.md) for the confirmatory result.
