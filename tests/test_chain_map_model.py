from __future__ import annotations

import torch

from homymoly.models.chain_map import (
    ExactChainMapLayer,
    cone_soft_betti,
    cycle_consistency_loss,
    mapping_cone_boundaries,
)
from homymoly.topology import build_oriented_incidence


def _cycle_boundary(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    incidence = build_oriented_incidence(
        4,
        ((0, 1), (1, 2), (2, 3), (0, 3)),
        dtype=dtype,
    )
    return incidence.boundary_1


def test_exact_chain_map_layer_enforces_the_law_architecturally() -> None:
    source = _cycle_boundary()
    vertex_permutation = torch.eye(4, dtype=torch.float64)[torch.tensor([2, 0, 3, 1])]
    edge_permutation = torch.eye(4, dtype=torch.float64)[torch.tensor([1, 3, 0, 2])]
    target = vertex_permutation @ source @ edge_permutation.mT
    layer = ExactChainMapLayer(source, target, dtype=torch.float64)
    with torch.no_grad():
        layer.coefficients.normal_()
    assert float(layer.residual().detach().abs().max()) < 1e-12
    mapped_zero, mapped_one = layer(torch.randn(5, 4), torch.randn(5, 4))
    assert mapped_zero.shape == (5, 4)
    assert mapped_one.shape == (5, 4)


def test_mapping_cone_differentials_square_to_chain_residual() -> None:
    source = _cycle_boundary()
    layer = ExactChainMapLayer(source, source, dtype=torch.float64)
    with torch.no_grad():
        layer.coefficients.normal_()
    d1, d2 = mapping_cone_boundaries(source, source, layer.matrices())
    torch.testing.assert_close(d1 @ d2, layer.residual(), atol=1e-12, rtol=0)


def test_paired_signal_training_recovers_an_invertible_chain_map() -> None:
    torch.manual_seed(7)
    source = _cycle_boundary(torch.float32)
    p0 = torch.eye(4)[torch.tensor([2, 0, 3, 1])]
    p1 = torch.eye(4)[torch.tensor([1, 3, 0, 2])]
    target = p0 @ source @ p1.mT
    forward = ExactChainMapLayer(source, target)
    reverse = ExactChainMapLayer(target, source)
    optimizer = torch.optim.Adam(
        (*forward.parameters(), *reverse.parameters()), lr=0.05
    )
    x0 = torch.randn(128, 4)
    x1 = torch.randn(128, 4)
    y0 = x0 @ p0.mT
    y1 = x1 @ p1.mT
    for _ in range(350):
        predicted_y0, predicted_y1 = forward(x0, x1)
        predicted_x0, predicted_x1 = reverse(y0, y1)
        loss = (
            (predicted_y0 - y0).square().mean()
            + (predicted_y1 - y1).square().mean()
            + (predicted_x0 - x0).square().mean()
            + (predicted_x1 - x1).square().mean()
            + 0.1 * cycle_consistency_loss(forward.matrices(), reverse.matrices())
            + 1e-4 * cone_soft_betti(source, target, forward.matrices())
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    learned = forward.matrices()
    torch.testing.assert_close(learned.degree_zero, p0, atol=2e-3, rtol=0)
    torch.testing.assert_close(learned.degree_one, p1, atol=2e-3, rtol=0)
    assert float(forward.residual().detach().abs().max()) < 2e-6
    assert float(cycle_consistency_loss(learned, reverse.matrices()).detach()) < 1e-5
