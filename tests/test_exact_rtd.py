"""Acceptance tests for the exact cone cross-barcode module.

The properties checked here are the RTD-module acceptance list from the
RTD-integration design document: identity and normalized-rescaling zeros,
permutation invariance, directional asymmetry with a symmetric half-sum,
collapse behavior, localized-difference detection, and the per-interval
stability bound.  The exact module is the evaluation reference; the
differentiable H0 surrogate is intentionally not asserted to match it.
"""

from __future__ import annotations

from math import inf

import numpy as np
import pytest
import torch

from homymoly.metrics.distances import normalize_dissimilarity
from homymoly.metrics.exact_rtd import (
    cone_cross_barcode,
    exact_rtd,
    exact_rtd_by_degree,
    exact_rtd_directional,
    exact_rtd_directional_by_degree,
    exact_srtd,
    exact_srtd_by_degree,
    total_persistence,
)


def _euclidean(points: np.ndarray) -> torch.Tensor:
    distances = np.sqrt(((points[:, None] - points[None]) ** 2).sum(-1))
    return torch.tensor(distances, dtype=torch.float64)


@pytest.fixture(scope="module")
def paired_clouds() -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(7)
    source_points = rng.normal(size=(10, 4))
    target_points = source_points + rng.normal(scale=0.4, size=(10, 4))
    return _euclidean(source_points), _euclidean(target_points)


def test_identical_representations_give_zero_everywhere() -> None:
    rng = np.random.default_rng(11)
    distances = _euclidean(rng.normal(size=(9, 3)))
    assert exact_srtd(distances, distances) == 0.0
    assert exact_rtd_directional(distances, distances) == 0.0
    assert exact_rtd(distances, distances) == 0.0


def test_isometry_and_rescaling_give_zero_after_normalization() -> None:
    rng = np.random.default_rng(13)
    points = rng.normal(size=(9, 3))
    shifted = 3.7 * points + 5.0
    assert exact_srtd(_euclidean(points), _euclidean(shifted)) < 1e-9


def test_joint_permutation_preserves_scores(paired_clouds) -> None:  # type: ignore[no-untyped-def]
    source, target = paired_clouds
    permutation = np.random.default_rng(17).permutation(source.shape[0])
    perm_source = source[permutation][:, permutation]
    perm_target = target[permutation][:, permutation]
    assert exact_srtd(perm_source, perm_target) == pytest.approx(
        exact_srtd(source, target), abs=1e-12
    )
    assert exact_rtd(perm_source, perm_target) == pytest.approx(
        exact_rtd(source, target), abs=1e-12
    )


def test_directional_scores_and_half_sum_symmetry(paired_clouds) -> None:  # type: ignore[no-untyped-def]
    source, target = paired_clouds
    forward = exact_rtd_directional(source, target)
    reverse = exact_rtd_directional(target, source)
    assert forward > 0 and reverse > 0
    # Swap consistency: forward(R, S) equals reverse(S, R) by construction.
    assert exact_rtd_directional(target, source, max_dim=2) == pytest.approx(
        reverse, abs=1e-12
    )
    half_sum_ab = 0.5 * (forward + reverse)
    half_sum_ba = 0.5 * (
        exact_rtd_directional(target, source) + exact_rtd_directional(source, target)
    )
    assert half_sum_ab == pytest.approx(half_sum_ba, abs=1e-12)
    assert exact_rtd(source, target) == pytest.approx(half_sum_ab, abs=1e-12)


def test_collapse_produces_structured_nonzero_barcode() -> None:
    rng = np.random.default_rng(19)
    distances = _euclidean(rng.normal(size=(8, 3)))
    collapsed = torch.zeros_like(distances)
    barcode = cone_cross_barcode(distances, collapsed, mode="srtd")
    finite = [interval for interval in barcode if interval.death != inf]
    assert finite, "collapse must produce finite discrepancy intervals"
    assert all(interval.length > 0 for interval in finite)
    assert exact_srtd(distances, collapsed) > 0


def test_localized_difference_is_detected() -> None:
    rng = np.random.default_rng(23)
    base = rng.normal(size=(10, 3))
    moved = base.copy()
    moved[0] += 2.0  # a single localized displacement
    score = exact_srtd(_euclidean(base), _euclidean(moved))
    assert score > 0


def test_interval_lengths_respect_stability_bound(paired_clouds) -> None:  # type: ignore[no-untyped-def]
    source, target = paired_clouds
    normalized_source = normalize_dissimilarity(source, mode="quantile")
    normalized_target = normalize_dissimilarity(target, mode="quantile")
    bound = float((normalized_source - normalized_target).abs().max())
    for mode in ("forward", "reverse", "srtd"):
        for interval in cone_cross_barcode(source, target, mode=mode):
            if interval.death == inf:
                continue
            assert interval.length <= bound + 1e-9


def test_srtd_degree_one_equals_sum_of_directionals(paired_clouds) -> None:  # type: ignore[no-untyped-def]
    source, target = paired_clouds
    symmetric = total_persistence(
        cone_cross_barcode(source, target, mode="srtd"), degree=1
    )
    forward = total_persistence(
        cone_cross_barcode(source, target, mode="forward"), degree=1
    )
    reverse = total_persistence(
        cone_cross_barcode(source, target, mode="reverse"), degree=1
    )
    assert symmetric == pytest.approx(forward + reverse, abs=1e-12)


def test_published_three_point_directional_fixture() -> None:
    """Hand reduction of the degree-one cross-barcode on three vertices."""

    source = torch.tensor(
        ((0.0, 1.0, 5.0), (1.0, 0.0, 4.0), (5.0, 4.0, 0.0)),
        dtype=torch.float64,
    )
    target = torch.tensor(
        ((0.0, 2.0, 6.0), (2.0, 0.0, 3.0), (6.0, 3.0, 0.0)),
        dtype=torch.float64,
    )
    forward = cone_cross_barcode(
        source, target, mode="forward", max_dim=1, normalization="none"
    )
    reverse = cone_cross_barcode(
        source, target, mode="reverse", max_dim=1, normalization="none"
    )
    symmetric = cone_cross_barcode(
        source, target, mode="srtd", max_dim=1, normalization="none"
    )

    assert [(bar.degree, bar.birth, bar.death) for bar in forward] == [(1, 3.0, 4.0)]
    assert [(bar.degree, bar.birth, bar.death) for bar in reverse] == [(1, 1.0, 2.0)]
    assert sorted((bar.degree, bar.birth, bar.death) for bar in symmetric) == [
        (1, 1.0, 2.0),
        (1, 3.0, 4.0),
    ]
    assert exact_rtd_directional(
        source, target, max_dim=1, normalization="none"
    ) == pytest.approx(1.0)
    assert exact_rtd(source, target, max_dim=1, normalization="none") == pytest.approx(
        1.0
    )
    assert exact_srtd(source, target, max_dim=1, normalization="none") == pytest.approx(
        2.0
    )


def test_collapse_fixture_matches_shifted_ordinary_h0_barcode() -> None:
    """RTD collapse theorem: degree-one cross-bars are finite H0 bars."""

    source = torch.tensor(
        ((0.0, 1.0, 3.0), (1.0, 0.0, 2.0), (3.0, 2.0, 0.0)),
        dtype=torch.float64,
    )
    collapsed = torch.zeros_like(source)
    barcode = cone_cross_barcode(
        source, collapsed, mode="forward", max_dim=1, normalization="none"
    )
    assert sorted((bar.birth, bar.death) for bar in barcode) == [
        (0.0, 1.0),
        (0.0, 2.0),
    ]
    assert exact_rtd_directional(
        source, collapsed, max_dim=1, normalization="none"
    ) == pytest.approx(3.0)


def test_quantile_normalization_matches_official_full_matrix_convention() -> None:
    matrix = torch.zeros((5, 5), dtype=torch.float64)
    upper = torch.triu_indices(5, 5, offset=1)
    values = torch.arange(1, 11, dtype=torch.float64).square()
    matrix[upper[0], upper[1]] = values
    matrix = matrix + matrix.mT

    normalized = normalize_dissimilarity(matrix, mode="quantile", quantile=0.9)
    # np.quantile/torch.quantile on all 25 entries gives 81.0. Taking only
    # unique off-diagonal distances would give 82.9 and is not the official
    # RTD implementation's convention.
    torch.testing.assert_close(normalized, matrix / 81.0)
    assert float(torch.quantile(values, 0.9)) == pytest.approx(82.9)


def test_degree_specific_scores_exclude_truncation_frontier() -> None:
    rng = np.random.default_rng(321)
    source = rng.random((7, 7))
    source = (source + source.T) * 0.5
    np.fill_diagonal(source, 0.0)
    target = rng.random((7, 7))
    target = (target + target.T) * 0.5
    np.fill_diagonal(target, 0.0)
    source_tensor = torch.tensor(source, dtype=torch.float64)
    target_tensor = torch.tensor(target, dtype=torch.float64)

    shallow = exact_srtd_by_degree(
        source_tensor, target_tensor, max_dim=1, normalization="none"
    )
    deep = exact_srtd_by_degree(
        source_tensor, target_tensor, max_dim=3, normalization="none"
    )
    assert shallow == pytest.approx(deep[:2], abs=1e-12)
    assert deep[2] > 0.0
    assert exact_srtd(
        source_tensor, target_tensor, max_dim=3, normalization="none"
    ) == pytest.approx(deep[1], abs=1e-12)
    assert exact_srtd(
        source_tensor,
        target_tensor,
        degree=2,
        max_dim=3,
        normalization="none",
    ) == pytest.approx(deep[2], abs=1e-12)
    assert all(
        bar.degree <= 1
        for bar in cone_cross_barcode(
            source_tensor,
            target_tensor,
            max_dim=1,
            normalization="none",
        )
    )


def test_per_degree_rtd_api_and_degree_validation(paired_clouds) -> None:  # type: ignore[no-untyped-def]
    source, target = paired_clouds
    forward = exact_rtd_directional_by_degree(source, target, max_dim=2)
    reverse = exact_rtd_directional_by_degree(target, source, max_dim=2)
    symmetric = exact_rtd_by_degree(source, target, max_dim=2)
    assert symmetric == pytest.approx(
        tuple(
            0.5 * (left + right) for left, right in zip(forward, reverse, strict=True)
        )
    )
    assert exact_rtd(source, target) == pytest.approx(symmetric[1])
    with pytest.raises(ValueError, match="must not exceed"):
        exact_srtd(source, target, degree=2, max_dim=1)
    with pytest.raises(ValueError, match="nonnegative integer"):
        cone_cross_barcode(source, target, max_dim=-1)
    with pytest.raises(ValueError, match="quantile"):
        normalize_dissimilarity(source, mode="quantile", quantile=0.0)


def test_size_guard_and_shape_validation() -> None:
    big = torch.ones((65, 65), dtype=torch.float64) - torch.eye(65, dtype=torch.float64)
    with pytest.raises(ValueError, match="bounded"):
        cone_cross_barcode(big, big)
    mismatched_a = torch.zeros((4, 4), dtype=torch.float64)
    mismatched_b = torch.zeros((5, 5), dtype=torch.float64)
    with pytest.raises(ValueError, match="same number of entities"):
        cone_cross_barcode(mismatched_a, mismatched_b)
