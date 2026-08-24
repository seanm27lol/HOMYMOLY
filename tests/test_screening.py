from __future__ import annotations

import pytest
import torch

from homymoly.data.conversion import ConversionDataset
from homymoly.experiments.identifiable_maps import DegreeMaps, build_annulus_map_system
from homymoly.topology import (
    CONSTANT,
    NOT_SATISFIED,
    SATISFIED,
    ChainComplex,
    ChainMap,
    cone_betti_numbers,
    exactness_term,
    screen_structural_term,
)

DTYPE = torch.float64


def _conversion_case() -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Truth W = B2^T, plus wrong candidates from the same shape."""

    sample = ConversionDataset(1, seed=20261001, dtype=DTYPE)[0]
    truth = sample.boundary_2.mT.clone()
    generator = torch.Generator().manual_seed(7)
    candidates = [
        truth + 0.5 * torch.randn(truth.shape, generator=generator, dtype=DTYPE)
        for _ in range(16)
    ]
    return sample.boundary_1, truth, candidates


def test_the_gate_passes_the_term_that_was_confirmed() -> None:
    """Exactness is satisfied by the truth and separates candidates."""

    boundary_1, truth, candidates = _conversion_case()

    result = screen_structural_term(exactness_term(boundary_1), truth, candidates)

    assert result.verdict == SATISFIED
    assert result.usable
    assert result.satisfied_by_truth
    assert result.varies_over_class
    assert "separates candidates" in result.explain()


def test_the_gate_rejects_a_term_the_truth_violates() -> None:
    """A term the answer does not satisfy can only pull away from it.

    This is the mapping cone's failure mode in the conversion campaign: rewarding
    large singular values biases W away from the true, rank-deficient answer.
    """

    _boundary_1, truth, candidates = _conversion_case()

    def collapse_penalty(candidate: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.linalg.svdvals(candidate).min() * 2.0)

    result = screen_structural_term(collapse_penalty, truth, candidates)

    assert result.verdict == NOT_SATISFIED
    assert not result.usable
    assert "does not satisfy" in result.explain()


def test_the_gate_rejects_a_term_that_is_constant_over_the_class() -> None:
    """Cone acyclicity on the identifiable annulus: every candidate is invertible.

    This is the section 6.3 failure. All twelve dihedral maps are isomorphisms, so
    every cone is acyclic and the term carries no information at any weight.
    """

    system = build_annulus_map_system(6)
    complex_ = ChainComplex(
        (system.num_vertices, system.num_edges, system.num_faces),
        (system.boundary_1, system.boundary_2),
    )

    def cone_defect(index: int) -> float:
        maps = DegreeMaps(*(degree[index] for degree in system.basis))
        return float(sum(cone_betti_numbers(ChainMap(complex_, complex_, maps))))

    result = screen_structural_term(
        cone_defect, 0, list(range(1, system.num_transformations))
    )

    assert result.verdict == CONSTANT
    assert not result.usable
    assert result.satisfied_by_truth, "every candidate is acyclic, truth included"
    assert "constant" in result.explain()


def test_screening_needs_candidates() -> None:
    _boundary_1, truth, _ = _conversion_case()
    with pytest.raises(ValueError, match="at least one candidate"):
        screen_structural_term(exactness_term(_boundary_1), truth, [])


def test_relative_spread_is_scale_free() -> None:
    """Rescaling a term must not change whether it is judged to vary."""

    _boundary_1, truth, candidates = _conversion_case()
    base = exactness_term(_boundary_1)
    scaled = screen_structural_term(
        lambda candidate: base(candidate) * 1e6, truth, candidates
    )
    plain = screen_structural_term(base, truth, candidates)

    assert scaled.verdict == plain.verdict
    assert scaled.relative_spread == pytest.approx(plain.relative_spread, rel=1e-9)
