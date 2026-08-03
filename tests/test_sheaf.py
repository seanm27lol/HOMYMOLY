from __future__ import annotations

from math import cos, sin

import torch

from homymoly.topology import (
    connection_coboundary,
    connection_residual,
    cycle_holonomy,
)


def _rotation(angle: float) -> torch.Tensor:
    return torch.tensor(
        ((cos(angle), -sin(angle)), (sin(angle), cos(angle))),
        dtype=torch.float64,
    )


def _pure_gauge_triangle() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    angles = (0.2, -0.4, 0.7)
    edge_index = torch.tensor(((0, 0, 1), (1, 2, 2)), dtype=torch.long)
    transports = torch.stack(
        [
            _rotation(angles[tail] - angles[head])
            for tail, head in edge_index.t().tolist()
        ]
    )
    global_vector = torch.tensor((0.3, -0.8), dtype=torch.float64)
    node_values = torch.stack(
        [_rotation(-angle) @ global_vector for angle in angles]
    )
    return edge_index, transports, node_values


def test_connection_coboundary_matches_edge_residual_convention() -> None:
    edge_index, transports, node_values = _pure_gauge_triangle()
    coboundary = connection_coboundary(
        edge_index,
        transports,
        num_vertices=3,
    )
    residual = connection_residual(node_values, edge_index, transports)

    torch.testing.assert_close(
        coboundary @ node_values.flatten(),
        residual.flatten(),
        atol=1e-12,
        rtol=0,
    )
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-12, rtol=0)


def test_pure_gauge_holonomy_is_identity_and_defect_is_detected() -> None:
    edge_index, transports, node_values = _pure_gauge_triangle()
    identity = torch.eye(2, dtype=torch.float64)
    torch.testing.assert_close(
        cycle_holonomy((0, 1, 2), edge_index, transports),
        identity,
        atol=1e-12,
        rtol=0,
    )

    defective = transports.clone()
    defective[0] = _rotation(0.5) @ defective[0]
    assert torch.linalg.matrix_norm(
        cycle_holonomy((0, 1, 2), edge_index, defective) - identity
    ) > 0.1
    assert torch.linalg.vector_norm(
        connection_residual(node_values, edge_index, defective)
    ) > 0.1


def test_connection_rejects_noncanonical_edges() -> None:
    edge_index, transports, _ = _pure_gauge_triangle()
    reversed_edge = edge_index.clone()
    reversed_edge[:, 0] = reversed_edge[:, 0].flip(0)

    try:
        connection_coboundary(reversed_edge, transports, num_vertices=3)
    except ValueError as error:
        assert "tail < head" in str(error)
    else:
        raise AssertionError("noncanonical connection edge was accepted")
