from __future__ import annotations

import numpy as np
import pytest
import torch

from homymoly.metrics import (
    aggregate_route_diagnostics,
    chain_complex_residual,
    chain_map_residual,
    directional_h0_rtd_surrogate,
    h0_persistence_death_discrepancy,
    paired_h0_rtd_surrogates,
    pairwise_euclidean_distances,
    single_linkage_connectivity,
    symmetric_h0_srtd_surrogate,
    validate_dissimilarity_matrix,
    zero_dimensional_persistence_deaths,
)


def _distance_matrix(values: list[list[float]]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def test_identical_views_and_uniform_rescaling_are_zero() -> None:
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.5, 1.0], [0.0, 2.0]],
        dtype=torch.float64,
    )
    distances = pairwise_euclidean_distances(points)

    diagnostics = paired_h0_rtd_surrogates(distances, distances)
    assert diagnostics.forward.item() == pytest.approx(0.0, abs=1e-12)
    assert diagnostics.reverse.item() == pytest.approx(0.0, abs=1e-12)
    assert diagnostics.symmetric.item() == pytest.approx(0.0, abs=1e-12)
    assert symmetric_h0_srtd_surrogate(
        distances, 7.0 * distances
    ).item() == pytest.approx(0.0, abs=1e-12)
    assert h0_persistence_death_discrepancy(distances, distances) == 0.0


def test_directionality_and_symmetric_half_sum() -> None:
    source = _distance_matrix([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    collapsed = torch.zeros_like(source)

    forward = directional_h0_rtd_surrogate(source, collapsed)
    reverse = directional_h0_rtd_surrogate(collapsed, source)
    symmetric = symmetric_h0_srtd_surrogate(source, collapsed)

    assert forward.item() == pytest.approx(0.0)
    assert reverse.item() > 0.0
    assert symmetric.item() == pytest.approx(0.5 * (forward + reverse).item())
    assert symmetric.item() == pytest.approx(
        symmetric_h0_srtd_surrogate(collapsed, source).item()
    )


def test_joint_permutation_preserves_paired_scores() -> None:
    generator = torch.Generator().manual_seed(17)
    first_points = torch.randn((7, 3), generator=generator, dtype=torch.float64)
    second_points = torch.randn((7, 5), generator=generator, dtype=torch.float64)
    first = pairwise_euclidean_distances(first_points)
    second = pairwise_euclidean_distances(second_points)
    permutation = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    permuted_first = first[permutation][:, permutation]
    permuted_second = second[permutation][:, permutation]

    before = paired_h0_rtd_surrogates(first, second)
    after = paired_h0_rtd_surrogates(permuted_first, permuted_second)
    torch.testing.assert_close(before.forward, after.forward)
    torch.testing.assert_close(before.reverse, after.reverse)
    torch.testing.assert_close(before.symmetric, after.symmetric)


def test_exact_h0_deaths_and_localized_difference_reference() -> None:
    first = _distance_matrix([[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]])
    second = _distance_matrix([[0.0, 3.0, 1.0], [3.0, 0.0, 2.0], [1.0, 2.0, 0.0]])
    deaths = zero_dimensional_persistence_deaths(first, normalization="none")

    np.testing.assert_allclose(deaths, np.asarray([1.0, 2.0]))
    permutation = torch.tensor([2, 0, 1])
    permuted_first = first[permutation][:, permutation]
    np.testing.assert_allclose(
        zero_dimensional_persistence_deaths(
            permuted_first,
            normalization="none",
        ),
        deaths,
    )
    assert (
        h0_persistence_death_discrepancy(
            first,
            second,
            normalization="none",
        )
        == 0.0
    )
    # Ordinary H0 death multisets agree, but the paired merge localization does not.
    assert symmetric_h0_srtd_surrogate(first, second, normalization="none") > 0


def test_single_linkage_connectivity_matches_minimax_paths() -> None:
    distances = _distance_matrix([[0.0, 1.0, 9.0], [1.0, 0.0, 2.0], [9.0, 2.0, 0.0]])
    expected = _distance_matrix([[0.0, 1.0, 2.0], [1.0, 0.0, 2.0], [2.0, 2.0, 0.0]])
    torch.testing.assert_close(single_linkage_connectivity(distances), expected)


def test_paired_surrogate_has_finite_nonzero_gradients() -> None:
    first_points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.2], [3.0, 0.0], [4.0, 1.5]],
        dtype=torch.float64,
        requires_grad=True,
    )
    second_points = torch.tensor(
        [[0.0], [0.5], [2.0], [5.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    loss = symmetric_h0_srtd_surrogate(
        pairwise_euclidean_distances(first_points),
        pairwise_euclidean_distances(second_points),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert first_points.grad is not None and torch.isfinite(first_points.grad).all()
    assert second_points.grad is not None and torch.isfinite(second_points.grad).all()
    assert first_points.grad.abs().sum() > 0
    assert second_points.grad.abs().sum() > 0


def test_float32_pairwise_distances_have_exact_zero_diagonal_at_training_scale() -> (
    None
):
    generator = torch.Generator().manual_seed(20260821)
    points = torch.randn((192, 256), generator=generator, dtype=torch.float32)
    distances = pairwise_euclidean_distances(points)

    assert torch.count_nonzero(torch.diagonal(distances)) == 0
    canonical = validate_dissimilarity_matrix(distances)
    torch.testing.assert_close(canonical, distances)


@pytest.mark.parametrize("size", [0, 1, 4])
def test_degenerate_collapsed_spaces_are_finite(size: int) -> None:
    collapsed = torch.zeros((size, size), dtype=torch.float64, requires_grad=True)
    score = symmetric_h0_srtd_surrogate(collapsed, collapsed)
    deaths = zero_dimensional_persistence_deaths(collapsed)

    assert torch.isfinite(score)
    assert score.item() == 0.0
    assert deaths.shape == (max(0, size - 1),)
    assert np.isfinite(deaths).all()
    score.backward()
    assert collapsed.grad is not None and torch.isfinite(collapsed.grad).all()


def test_chain_residuals_are_zero_for_identity_and_differentiable_when_perturbed() -> (
    None
):
    boundary = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    identity_vertices = torch.eye(2, dtype=torch.float64)
    identity_edges = torch.ones((1, 1), dtype=torch.float64)

    assert (
        chain_map_residual(
            boundary,
            boundary,
            identity_edges,
            identity_vertices,
        ).item()
        == 0.0
    )
    assert (
        chain_complex_residual(
            (boundary,),
            (boundary,),
            (identity_vertices, identity_edges),
        ).item()
        == 0.0
    )

    learned_vertices = torch.tensor(
        [[1.0, 0.2], [0.0, 0.8]],
        dtype=torch.float64,
        requires_grad=True,
    )
    loss = chain_map_residual(
        boundary,
        boundary,
        identity_edges,
        learned_vertices,
    )
    loss.backward()
    assert loss.item() > 0
    assert learned_vertices.grad is not None
    assert torch.isfinite(learned_vertices.grad).all()
    assert learned_vertices.grad.abs().sum() > 0


def test_route_diagnostic_aggregation_preserves_gradients() -> None:
    task = torch.tensor([1.0, 2.0], requires_grad=True)
    chain = torch.tensor([0.5, 0.25], requires_grad=True)
    total = aggregate_route_diagnostics(
        {"task": task, "chain": chain, "compute": 0.1},
        {"task": 1.0, "chain": 2.0, "compute": 3.0},
        reduction="mean",
    )
    total.backward()

    assert total.item() == pytest.approx(((1.0 + 1.0 + 0.3) + (2.0 + 0.5 + 0.3)) / 2)
    torch.testing.assert_close(task.grad, torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(chain.grad, torch.tensor([1.0, 1.0]))


def test_invalid_dissimilarities_are_rejected() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        validate_dissimilarity_matrix(torch.tensor([[0.0, 1.0], [2.0, 0.0]]))
    with pytest.raises(ValueError, match="nonnegative"):
        validate_dissimilarity_matrix(torch.tensor([[0.0, -1.0], [-1.0, 0.0]]))
