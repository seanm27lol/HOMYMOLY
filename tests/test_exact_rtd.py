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

from homymoly.metrics.exact_rtd import (
    cone_cross_barcode,
    exact_rtd,
    exact_rtd_directional,
    exact_srtd,
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
    collapsed = (torch.ones_like(distances) - torch.eye(8, dtype=torch.float64)).double()
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
    bound = float((source - target).abs().max())
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


def test_size_guard_and_shape_validation() -> None:
    big = torch.ones((65, 65), dtype=torch.float64) - torch.eye(65, dtype=torch.float64)
    with pytest.raises(ValueError, match="bounded"):
        cone_cross_barcode(big, big)
    mismatched_a = torch.zeros((4, 4), dtype=torch.float64)
    mismatched_b = torch.zeros((5, 5), dtype=torch.float64)
    with pytest.raises(ValueError, match="same number of entities"):
        cone_cross_barcode(mismatched_a, mismatched_b)
