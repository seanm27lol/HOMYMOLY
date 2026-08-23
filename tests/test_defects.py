from __future__ import annotations

import torch

from homymoly.experiments.identifiable_maps import DegreeMaps, build_annulus_map_system
from homymoly.topology import (
    ChainComplex,
    ChainMap,
    cone_betti_from_defects,
    cone_betti_numbers,
    exactness_defects,
    harmonic_basis,
    induced_homology_map,
    is_quasi_isomorphism,
)

DTYPE = torch.float64


def _annulus() -> tuple[ChainComplex, object, torch.Tensor]:
    system = build_annulus_map_system(6)
    boundary_1 = system.boundary_1.to(DTYPE)
    boundary_2 = system.boundary_2.to(DTYPE)
    complex_ = ChainComplex(
        (system.num_vertices, system.num_edges, system.num_faces),
        (boundary_1, boundary_2),
    )
    laplacian = boundary_1.mT @ boundary_1 + boundary_2 @ boundary_2.mT
    _, vectors = torch.linalg.eigh(laplacian)
    harmonic = vectors[:, 0:1] / vectors[:, 0:1].norm()
    return complex_, system, harmonic


def _identity(complex_: ChainComplex) -> DegreeMaps:
    return DegreeMaps(
        *(
            torch.eye(complex_.space_dim(degree), dtype=DTYPE)
            for degree in range(3)
        )
    )


def test_identity_destroys_and_misses_nothing() -> None:
    complex_, _, _ = _annulus()
    chain_map = ChainMap(complex_, complex_, _identity(complex_))

    defects = exactness_defects(chain_map)

    assert all(defect.chain_kernel == 0 for defect in defects)
    assert all(defect.chain_cokernel == 0 for defect in defects)
    assert all(defect.is_iso_on_homology for defect in defects)
    assert is_quasi_isomorphism(chain_map)


def test_harmonic_basis_spans_the_homology_of_each_degree() -> None:
    complex_, _, _ = _annulus()

    for degree in range(3):
        basis = harmonic_basis(complex_, degree)
        assert basis.shape[1] == complex_.betti(degree)
        if basis.shape[1]:
            # Orthonormal columns, and annihilated by the Hodge Laplacian.
            identity = torch.eye(basis.shape[1], dtype=DTYPE)
            torch.testing.assert_close(basis.mT @ basis, identity, atol=1e-10, rtol=0)
            residual = complex_.hodge_laplacian(degree).to(DTYPE) @ basis
            assert float(residual.abs().max()) < 1e-8


def test_a_cycle_killing_map_reports_one_destroyed_and_one_unreachable_class() -> None:
    complex_, system, harmonic = _annulus()
    projector = torch.eye(system.num_edges, dtype=DTYPE) - harmonic @ harmonic.mT
    rotation = DegreeMaps(*(degree[0].to(DTYPE) for degree in system.basis))
    killed = DegreeMaps(
        rotation.degree_zero, rotation.degree_one @ projector, rotation.degree_two
    )

    defects = exactness_defects(ChainMap(complex_, complex_, killed, atol=1e-8))
    degree_one = defects[1]

    assert degree_one.homology_kernel == 1, "the cycle class is destroyed"
    assert degree_one.homology_cokernel == 1, "the target cycle is unreachable"
    assert degree_one.chain_kernel == 1
    assert not degree_one.is_iso_on_homology
    # Degrees 0 and 2 are untouched.
    assert defects[0].is_iso_on_homology
    assert defects[2].is_iso_on_homology


def test_induced_map_ranks_zero_when_homology_is_killed() -> None:
    """Regression: a purely relative rank threshold rescales to its own noise.

    The induced map of a cycle-killing chain map is uniformly at float64 noise
    level. Ranking it against its own largest singular value would report full
    rank and hide the destroyed class entirely.
    """

    complex_, system, harmonic = _annulus()
    projector = torch.eye(system.num_edges, dtype=DTYPE) - harmonic @ harmonic.mT
    rotation = DegreeMaps(*(degree[0].to(DTYPE) for degree in system.basis))
    killed = DegreeMaps(
        rotation.degree_zero, rotation.degree_one @ projector, rotation.degree_two
    )
    chain_map = ChainMap(complex_, complex_, killed, atol=1e-8)

    induced = induced_homology_map(chain_map, 1)

    assert induced.shape == (1, 1)
    assert float(induced.abs().max()) < 1e-10
    assert exactness_defects(chain_map)[1].homology_kernel == 1


def test_the_cone_bundles_kernel_and_cokernel_for_every_candidate() -> None:
    """dim H_n(cone) == dim coker H_n(F) + dim ker H_(n-1)(F), on all 24 maps."""

    complex_, system, harmonic = _annulus()
    projector = torch.eye(system.num_edges, dtype=DTYPE) - harmonic @ harmonic.mT

    checked = 0
    for index in range(system.num_transformations):
        rotation = DegreeMaps(*(degree[index].to(DTYPE) for degree in system.basis))
        killed = DegreeMaps(
            rotation.degree_zero, rotation.degree_one @ projector, rotation.degree_two
        )
        for candidate in (rotation, killed):
            chain_map = ChainMap(complex_, complex_, candidate, atol=1e-8)
            predicted = cone_betti_from_defects(exactness_defects(chain_map))
            assert predicted == cone_betti_numbers(chain_map, map_atol=1e-8)
            checked += 1
    assert checked == 2 * system.num_transformations


def test_quasi_isomorphism_separates_the_two_candidate_classes() -> None:
    complex_, system, harmonic = _annulus()
    projector = torch.eye(system.num_edges, dtype=DTYPE) - harmonic @ harmonic.mT

    for index in range(system.num_transformations):
        rotation = DegreeMaps(*(degree[index].to(DTYPE) for degree in system.basis))
        killed = DegreeMaps(
            rotation.degree_zero, rotation.degree_one @ projector, rotation.degree_two
        )
        assert is_quasi_isomorphism(ChainMap(complex_, complex_, rotation, atol=1e-8))
        assert not is_quasi_isomorphism(
            ChainMap(complex_, complex_, killed, atol=1e-8)
        )
