from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from homymoly.topology.incidence import (
    build_boundary_2,
    build_oriented_incidence,
    canonical_cycle,
    canonical_edge,
    validate_boundary_squared_zero,
)


class IncidenceTests(unittest.TestCase):
    def test_edges_and_cycles_have_deterministic_orientation(self) -> None:
        self.assertEqual(canonical_edge(7, 2), (2, 7))
        expected = (0, 1, 3, 2)
        for cycle in (
            (0, 1, 3, 2),
            (1, 3, 2, 0),
            (0, 2, 3, 1),
            (3, 1, 0, 2, 3),
        ):
            self.assertEqual(canonical_cycle(cycle), expected)

    def test_triangle_incidence_is_deterministic_and_exact(self) -> None:
        first = build_oriented_incidence(
            3,
            edges=[(2, 1), (0, 2), (1, 0)],
            faces=[(2, 0, 1)],
        )
        second = build_oriented_incidence(
            3,
            edges=[(0, 1), (1, 2), (2, 0)],
            faces=[(1, 2, 0, 1)],
        )

        self.assertEqual(first.edges, ((0, 1), (0, 2), (1, 2)))
        self.assertEqual(first.faces, ((0, 1, 2),))
        torch.testing.assert_close(first.boundary_1, second.boundary_1)
        torch.testing.assert_close(first.boundary_2, second.boundary_2)
        torch.testing.assert_close(
            first.boundary_2[:, 0],
            torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64),
        )
        self.assertEqual(
            validate_boundary_squared_zero(first.boundary_1, first.boundary_2),
            0.0,
        )

    def test_missing_face_edge_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not present"):
            build_boundary_2([(0, 1), (1, 2)], [(0, 1, 2)])

    def test_duplicate_cells_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate edges"):
            build_oriented_incidence(2, [(0, 1), (1, 0)])
        with self.assertRaisesRegex(ValueError, "duplicate faces"):
            build_oriented_incidence(
                3,
                [(0, 1), (1, 2), (0, 2)],
                [(0, 1, 2), (2, 1, 0)],
            )

    def test_invalid_boundary_product_is_rejected(self) -> None:
        incidence = build_oriented_incidence(
            3,
            [(0, 1), (1, 2), (0, 2)],
            [(0, 1, 2)],
        )
        corrupted = incidence.boundary_2.clone()
        corrupted[0, 0] = 0
        self.assertGreater(
            validate_boundary_squared_zero(
                incidence.boundary_1, corrupted, raise_on_error=False
            ),
            0.0,
        )
        with self.assertRaisesRegex(ValueError, "boundary law failed"):
            validate_boundary_squared_zero(incidence.boundary_1, corrupted)


if __name__ == "__main__":
    unittest.main()
