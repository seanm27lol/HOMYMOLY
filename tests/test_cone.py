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
    cone_betti_numbers,
    graph_to_cell_inclusion,
    hodge_projector,
    mapping_cone,
)


def triangle_complexes() -> tuple[ChainComplex, ChainComplex, torch.Tensor, torch.Tensor]:
    incidence = build_oriented_incidence(
        3,
        [(0, 1), (1, 2), (0, 2)],
        [(0, 1, 2)],
    )
    graph = ChainComplex((3, 3), (incidence.boundary_1,))
    cell = ChainComplex(
        (3, 3, 1), (incidence.boundary_1, incidence.boundary_2)
    )
    return graph, cell, incidence.boundary_1, incidence.boundary_2


class MappingConeTests(unittest.TestCase):
    def test_identity_cone_is_acyclic_and_has_square_zero(self) -> None:
        graph, _, _, _ = triangle_complexes()
        cone = mapping_cone(ChainMap.identity(graph))
        self.assertEqual(cone.betti_numbers(), (0, 0, 0))
        self.assertEqual(cone.max_chain_residual(), 0.0)

    def test_zero_map_cone_is_direct_sum_with_shift(self) -> None:
        graph, _, _, _ = triangle_complexes()
        zero = ChainMap.zero(graph, graph)
        self.assertEqual(cone_betti_numbers(zero), (1, 2, 1))

    def test_filling_one_cycle_creates_degree_two_cone_class(self) -> None:
        graph, cell, _, _ = triangle_complexes()
        inclusion = graph_to_cell_inclusion(graph, cell)
        self.assertEqual(graph.betti_numbers(), (1, 1))
        self.assertEqual(cell.betti_numbers(), (1, 0, 0))
        self.assertEqual(cone_betti_numbers(inclusion), (0, 0, 1))

    def test_hodge_projector_gives_valid_reverse_chain_map(self) -> None:
        graph, cell, boundary_1, boundary_2 = triangle_complexes()
        projector = hodge_projector(boundary_2)
        torch.testing.assert_close(projector @ projector, projector, atol=1e-10, rtol=0)
        torch.testing.assert_close(
            projector @ boundary_2,
            torch.zeros_like(boundary_2),
            atol=1e-10,
            rtol=0,
        )
        torch.testing.assert_close(
            boundary_1 @ projector, boundary_1, atol=1e-10, rtol=0
        )

        projection = ChainMap(
            cell,
            graph,
            [
                torch.eye(3, dtype=torch.float64),
                projector,
                torch.zeros((0, 1), dtype=torch.float64),
            ],
        )
        self.assertLess(projection.max_residual(), 1e-10)
        self.assertEqual(cone_betti_numbers(projection), (0, 1, 0, 0))

    def test_cone_rejects_an_approximate_non_chain_map(self) -> None:
        graph, _, _, _ = triangle_complexes()
        invalid = ChainMap(
            graph,
            graph,
            [
                torch.zeros((3, 3), dtype=torch.float64),
                torch.eye(3, dtype=torch.float64),
            ],
            validate=False,
        )
        with self.assertRaisesRegex(ValueError, "chain-map law failed"):
            mapping_cone(invalid)


if __name__ == "__main__":
    unittest.main()
