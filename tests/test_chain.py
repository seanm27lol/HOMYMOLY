from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from homymoly.topology import (
    ChainComplex,
    ChainMap,
    build_oriented_incidence,
    graph_to_cell_inclusion,
    nullity,
    numerical_rank,
)


def triangle_complexes() -> tuple[ChainComplex, ChainComplex]:
    incidence = build_oriented_incidence(
        3,
        [(0, 1), (1, 2), (0, 2)],
        [(0, 1, 2)],
    )
    graph = ChainComplex((3, 3), (incidence.boundary_1,))
    cell = ChainComplex(
        (3, 3, 1), (incidence.boundary_1, incidence.boundary_2)
    )
    return graph, cell


class ChainComplexTests(unittest.TestCase):
    def test_float64_rank_and_nullity_respect_tolerance(self) -> None:
        matrix = torch.diag(torch.tensor([1.0, 1e-12], dtype=torch.float32))
        self.assertEqual(numerical_rank(matrix, rtol=1e-10), 1)
        self.assertEqual(numerical_rank(matrix, rtol=1e-14), 2)
        self.assertEqual(nullity(matrix, rtol=1e-10), 1)

    def test_graph_and_filled_triangle_betti_numbers(self) -> None:
        graph, cell = triangle_complexes()
        self.assertEqual(graph.betti_numbers(), (1, 1))
        self.assertEqual(cell.betti_numbers(), (1, 0, 0))
        self.assertEqual(graph.max_chain_residual(), 0.0)
        self.assertEqual(cell.max_chain_residual(), 0.0)

        for degree, expected_betti in enumerate(cell.betti_numbers()):
            laplacian = cell.hodge_laplacian(degree)
            self.assertEqual(nullity(laplacian), expected_betti)

    def test_invalid_chain_complex_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "chain law failed"):
            ChainComplex(
                (1, 1, 1),
                (
                    torch.ones((1, 1), dtype=torch.float64),
                    torch.ones((1, 1), dtype=torch.float64),
                ),
            )

    def test_identity_and_graph_to_cell_maps_are_exact(self) -> None:
        graph, cell = triangle_complexes()
        identity = ChainMap.identity(graph)
        inclusion = graph_to_cell_inclusion(graph, cell)
        self.assertEqual(identity.max_residual(), 0.0)
        self.assertEqual(inclusion.max_residual(), 0.0)
        torch.testing.assert_close(inclusion.map(0), torch.eye(3, dtype=torch.float64))
        torch.testing.assert_close(inclusion.map(1), torch.eye(3, dtype=torch.float64))
        self.assertEqual(tuple(inclusion.map(2).shape), (1, 0))

    def test_non_chain_map_is_rejected(self) -> None:
        graph, _ = triangle_complexes()
        with self.assertRaisesRegex(ValueError, "chain-map law failed"):
            ChainMap(
                graph,
                graph,
                [
                    torch.zeros((3, 3), dtype=torch.float64),
                    torch.eye(3, dtype=torch.float64),
                ],
            )

    def test_betti_numbers_are_basis_invariant(self) -> None:
        graph, _ = triangle_complexes()
        vertex_permutation = torch.eye(3, dtype=torch.float64)[[2, 0, 1]]
        edge_orientation = torch.diag(
            torch.tensor([-1.0, 1.0, -1.0], dtype=torch.float64)
        )
        transformed_b1 = vertex_permutation @ graph.boundary(1) @ edge_orientation
        transformed = ChainComplex((3, 3), (transformed_b1,))
        self.assertEqual(transformed.betti_numbers(), graph.betti_numbers())


if __name__ == "__main__":
    unittest.main()
