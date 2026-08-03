# Derived-category, Fourier–Mukai, and Langlands scope

## Derived language that is immediately justified

HOMYMOLY may legitimately use:

- chain and cochain complexes;
- chain maps and chain homotopies;
- mapping cones;
- quasi-isomorphisms;
- adjunction units and counits;
- derived functors when a concrete exactness failure is specified;
- dg or stable categories when functorial cones are required.

Derived functors produce objects or groups, not automatic scalar losses. Every optimization term needs a declared scalarization.

## Fourier–Mukai guardrail

A Fourier–Mukai transform is an integral-kernel functor of the form

\[
\Phi_{\mathcal P}(E)
=Rp_{Y*}\left(Lp_X^*E\otimes^L\mathcal P\right)
\]

under appropriate geometric or dg hypotheses. It is not a generic name for an arbitrary learned conversion among vectors, graphs, groups, or sheaves.

The computationally plausible future analogue is a finite dg-bimodule layer. For finite-dimensional algebras or dg categories \(A\) and \(B\), a learned kernel \(P\) could induce

\[
F_P(M)=M\otimes_A^L P.
\]

A reverse kernel \(Q\) could be evaluated using the cones of unit/counit-like maps

\[
A\to P\otimes_B^LQ,
\qquad
B\to Q\otimes_A^LP.
\]

This deserves the label “Fourier–Mukai-inspired” only when the bimodule/kernel construction is actually implemented.

Primary mathematical references include [Mukai's derived duality](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/BDBEBAC584BE15236C2D62C383A34245/S002776300001922Xa.pdf/duality-between-dx-and-with-its-application-to-picard-sheaves.pdf), [Orlov's representability theorem](https://arxiv.org/abs/alg-geom/9606006), and [Toën's derived Morita theory](https://arxiv.org/abs/math/0408337).

## Geometric Langlands guardrail

A Hecke eigensheaf is an eigenobject for a compatible family of Hecke functors associated with representations of a Langlands dual group. Fourier–Mukai theory captures important abelian or torus phenomena but does not generally create Hecke eigensheaves.

Geometric Langlands should enter HOMYMOLY only if a future construction genuinely contains Hecke actions, dual reductive groups, or the appropriate moduli categories. At present it is motivation for functor actions, eigenobjects, duality, and categorical transforms—not an architectural ingredient or novelty claim.

Relevant primary sources include [Laumon's generalized Fourier transform](https://arxiv.org/abs/alg-geom/9603004) and Beilinson–Drinfeld's [Quantization of Hitchin's Integrable System and Hecke Eigensheaves](https://math.uchicago.edu/~drinfeld/langlands/QuantizationHitchin.pdf).

## Approved terminology

Preferred description:

> Functorial multi-representation learning with homological round-trip defects.

Avoid “co-exact learning,” “Fourier–Mukai layer,” or “Langlands architecture” unless the corresponding mathematical construction is present and tested.
