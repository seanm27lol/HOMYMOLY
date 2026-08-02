# Mathematical contract

This document defines the terms that HOLYMOLY may use. Implementations and claims must conform to these definitions.

## Chain complexes and exactness

A chain complex over a field \(k\) is a sequence

\[
\cdots \to C_{n+1}\xrightarrow{d_{n+1}}C_n\xrightarrow{d_n}C_{n-1}\to\cdots
\]

with \(d_nd_{n+1}=0\). Its degree-\(n\) homology is

\[
H_n(C)=\ker d_n/\operatorname{im}d_{n+1}.
\]

The sequence is exact at \(C_n\) exactly when \(H_n(C)=0\).

Two distinct failures must not be conflated:

- **chain-law defect:** \(d_nd_{n+1}\ne0\), so the proposed operators do not define a complex;
- **exactness defect:** \(\operatorname{im}d_{n+1}\subsetneq\ker d_n\) in a valid complex.

## Coexactness and reverse loss

“Coexactness” is not a generic reverse reconstruction loss. In a finite Hilbert complex, the Hodge decomposition is

\[
C_n=\operatorname{im}d_{n+1}\oplus\ker\Delta_n\oplus\operatorname{im}d_n^*,
\]

whose components are often called exact, harmonic, and coexact. A learned reverse transformation should instead be described using cycle consistency, a unit/counit, or an explicit cokernel/cohomology defect.

## Typed linearizations

Arbitrary spaces, groups, graphs, and sheaves do not belong to one common abelian category. Kernels, images, and quotients therefore cannot be applied to them indiscriminately. HOLYMOLY assigns each supported type a declared linearization:

| Type | Initial complex |
|---|---|
| Vector point cloud | filtered Vietoris–Rips chains when a metric comparison is required |
| Graph | graph cellular chains/cochains |
| Simplicial or cell complex | simplicial/cellular chains |
| Cellular sheaf | cellular sheaf cochain complex |
| Group, future | bar-resolution chains |
| Small category/logical structure, future | nerve followed by simplicial chains |

Every learned conversion must declare its source, target, degree maps, coefficient field, bases or gauge conventions, and structure-preservation requirement.

## Chain maps

For complexes \(C\) and \(D\), a conversion \(F:C\to D\) is a chain map when

\[
d^D_nF_n=F_{n-1}d^C_n
\]

at every degree. An approximate implementation may use

\[
\mathcal L_{\mathrm{chain}}(F)
=\sum_n\left\|d^D_nF_n-F_{n-1}d^C_n\right\|_F^2.
\]

Induced maps \(H_n(F)\) are interpreted only after the chain-map contract is met or the approximation error is reported.

For a linear induced map:

- \(\ker H_n(F)\) contains source homology classes destroyed by the conversion;
- \(\operatorname{coker}H_n(F)\) contains target classes not reached from the source.

These are structural statements, not complete measures of semantic information.

## Mapping cones

For a chain map \(F:C\to D\), define

\[
\operatorname{Cone}(F)_n=D_n\oplus C_{n-1}
\]

with the standard cone differential. The cone is acyclic exactly when \(F\) is a quasi-isomorphism. Its homology combines kernel and cokernel information through the associated long exact sequence.

Mapping-cone homology is therefore the preferred map-aware defect. It differs from RTD: RTD constructs an auxiliary cone-like filtration from two paired distance matrices, whereas HOLYMOLY's direct cone is constructed from the declared transformation \(F\).

## Reverse maps and path coherence

For \(F:C\to D\) and \(G:D\to C\), round-trip preservation asks for

\[
GF\simeq I_C,\qquad FG\simeq I_D,
\]

potentially up to chain homotopy. It does not ask for \(GF=0\).

When \(F\dashv G\), the relevant transformations are the unit and counit

\[
\eta:I_C\Rightarrow GF,\qquad \epsilon:FG\Rightarrow I_D.
\]

In a dg or stable setting, the homology of \(\operatorname{Cone}(\eta)\) and \(\operatorname{Cone}(\epsilon)\) can quantify the two round-trip defects. For several paths between the same typed endpoints, a naturality/path loss measures disagreement between the resulting composites.

## Scalarization

Homology and derived functors produce objects or groups, not canonical scalar losses. Candidate scalarizations include:

- persistent-bar lengths;
- Betti-number discrepancies for evaluation;
- soft ranks or singular-value thresholds;
- low-eigenvalue statistics of Hodge Laplacians;
- heat traces or regularized log-determinants;
- normalized cone energy.

All scalarizations must state normalization, coefficient field, degree range, tolerance, and sampling procedure.
