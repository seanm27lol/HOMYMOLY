"""Deterministic mixed-regime structured signal generator.

Each block of six samples contains every ``(regime, binary label)`` pair over
the same graph, candidate faces, and size.  This counterfactual grouping keeps
padding, graph density, and vertex count from becoming route shortcuts.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import ceil, cos, pi, sin
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .types import (
    SignalRegime,
    StructuredObservations,
    StructuredSample,
    count_values,
)

_REGIME_LABEL_PAIRS = tuple(
    (regime, label) for regime in SignalRegime for label in (0, 1)
)


def _local_canonical_structure(
    edges: Iterable[tuple[int, int]],
    faces: Iterable[tuple[int, int, int]],
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int, int], ...]]:
    """Fallback used only when the topology package has not been installed."""

    canonical_edges = tuple(sorted({tuple(sorted(edge)) for edge in edges}))
    edge_set = set(canonical_edges)
    canonical_faces = tuple(sorted({tuple(sorted(face)) for face in faces}))
    for a, b, c in canonical_faces:
        if not {(a, b), (a, c), (b, c)}.issubset(edge_set):
            raise ValueError(f"candidate face {(a, b, c)} is missing a boundary edge")
    return canonical_edges, canonical_faces


def _canonical_structure(
    num_vertices: int,
    edges: Iterable[tuple[int, int]],
    faces: Iterable[tuple[int, int, int]],
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int, int], ...]]:
    """Use the public topology interface, with an import-isolated fallback."""

    edge_tuple = tuple(edges)
    face_tuple = tuple(faces)
    try:
        from homymoly.topology.incidence import build_oriented_incidence
    except ModuleNotFoundError as exc:
        if exc.name not in {"homymoly.topology", "homymoly.topology.incidence"}:
            raise
        return _local_canonical_structure(edge_tuple, face_tuple)

    incidence = build_oriented_incidence(
        num_vertices,
        edge_tuple,
        face_tuple,
        dtype=torch.float64,
    )
    return tuple(incidence.edges), tuple(incidence.faces)


def _rotation(angle: float, *, dtype: torch.dtype) -> Tensor:
    return torch.tensor(
        [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]],
        dtype=dtype,
    )


def _rng(seed: int, *coordinates: int) -> np.random.Generator:
    entropy = [int(seed) & 0xFFFFFFFF]
    entropy.extend(int(value) & 0xFFFFFFFF for value in coordinates)
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _find_triangles(graph: nx.Graph) -> list[tuple[int, int, int]]:
    triangles: list[tuple[int, int, int]] = []
    for clique in nx.enumerate_all_cliques(graph):
        if len(clique) < 3:
            continue
        if len(clique) > 3:
            break
        triangles.append(tuple(sorted(int(vertex) for vertex in clique)))
    return sorted(set(triangles))


def _base_graph(
    num_vertices: int,
    rng: np.random.Generator,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int, int], ...]]:
    graph = nx.Graph()
    graph.add_nodes_from(range(num_vertices))
    for vertex in range(num_vertices):
        graph.add_edge(vertex, (vertex + 1) % num_vertices)
        graph.add_edge(vertex, (vertex + 2) % num_vertices)

    target_edges = 2 * num_vertices + num_vertices // 4
    while graph.number_of_edges() < target_edges:
        u, v = (int(item) for item in rng.choice(num_vertices, size=2, replace=False))
        if u != v:
            graph.add_edge(u, v)

    edges = [tuple(sorted((int(u), int(v)))) for u, v in graph.edges()]
    faces = _find_triangles(graph)
    return _canonical_structure(num_vertices, edges, faces)


def _edge_boundary(face: tuple[int, int, int]) -> tuple[tuple[tuple[int, int], float], ...]:
    """Boundary coefficients for the ascending orientation of a triangle."""

    a, b, c = face
    return (((b, c), 1.0), ((a, c), -1.0), ((a, b), 1.0))


class MixedStructuredSignal(Dataset[StructuredSample]):
    """Balanced bring-up signals with shared raw observations.

    This generator is the mechanically audited, easy Stage-1 benchmark. Its
    reliability amplitudes intentionally make the active regime identifiable;
    it is not the confirmatory benchmark for structural-routing claims.

    Parameters
    ----------
    num_samples:
        Number of samples.  Joint regime/label counts differ by at most one and
        are exactly equal when this is divisible by six.
    seed:
        Dataset seed.  A sample is a pure function of ``(seed, index)`` and can
        therefore be regenerated consistently across worker processes.
    min_vertices, max_vertices:
        Inclusive range, constrained to the benchmark contract of 24--96.
    num_vertices:
        Optional fixed count or inclusive ``(minimum, maximum)`` pair.  It
        overrides ``min_vertices`` and ``max_vertices``.
    """

    def __init__(
        self,
        num_samples: int = 300,
        *,
        seed: int = 0,
        min_vertices: int = 24,
        max_vertices: int = 96,
        num_vertices: int | tuple[int, int] | None = None,
        node_feature_dim: int = 4,
        edge_feature_dim: int = 2,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not isinstance(num_samples, int) or num_samples < len(_REGIME_LABEL_PAIRS):
            raise ValueError("num_samples must be an integer of at least 6")
        if num_vertices is not None:
            if isinstance(num_vertices, int):
                min_vertices = max_vertices = num_vertices
            elif isinstance(num_vertices, tuple) and len(num_vertices) == 2:
                min_vertices, max_vertices = num_vertices
            else:
                raise TypeError("num_vertices must be an int or a (minimum, maximum) tuple")
        if not (24 <= min_vertices <= max_vertices <= 96):
            raise ValueError("vertex bounds must satisfy 24 <= minimum <= maximum <= 96")
        if node_feature_dim < 4:
            raise ValueError("node_feature_dim must be at least 4")
        if edge_feature_dim < 2:
            raise ValueError("edge_feature_dim must be at least 2")
        if not isinstance(dtype, torch.dtype) or not torch.empty(
            (), dtype=dtype
        ).is_floating_point():
            raise TypeError("dtype must be a floating torch dtype")

        self.num_samples = num_samples
        self.seed = int(seed)
        self.min_vertices = int(min_vertices)
        self.max_vertices = int(max_vertices)
        self.node_feature_dim = int(node_feature_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        self.dtype = dtype
        self._schedule = self._make_schedule()

    def _make_schedule(self) -> tuple[tuple[SignalRegime, int], ...]:
        schedule: list[tuple[SignalRegime, int]] = []
        block_count = ceil(self.num_samples / len(_REGIME_LABEL_PAIRS))
        for block in range(block_count):
            schedule_rng = _rng(self.seed, 0x53434845, block)
            regimes = tuple(
                list(SignalRegime)[int(index)]
                for index in schedule_rng.permutation(len(SignalRegime))
            )
            first_label = int(schedule_rng.integers(0, 2))
            schedule.extend(
                (
                    (regimes[0], first_label),
                    (regimes[1], 1 - first_label),
                    (regimes[2], first_label),
                    (regimes[0], 1 - first_label),
                    (regimes[1], first_label),
                    (regimes[2], 1 - first_label),
                )
            )
        return tuple(schedule[: self.num_samples])

    def __len__(self) -> int:
        return self.num_samples

    @property
    def regimes(self) -> tuple[SignalRegime, ...]:
        return tuple(regime for regime, _ in self._schedule)

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(label for _, label in self._schedule)

    @property
    def group_ids(self) -> tuple[int, ...]:
        block_size = len(_REGIME_LABEL_PAIRS)
        return tuple(index // block_size for index in range(len(self)))

    @property
    def regime_counts(self) -> dict[SignalRegime, int]:
        return count_values(self.regimes)

    @property
    def label_counts(self) -> dict[int, int]:
        return count_values(self.labels)

    @property
    def joint_counts(self) -> dict[tuple[SignalRegime, int], int]:
        return count_values(self._schedule)

    def indices_for(
        self,
        *,
        regime: SignalRegime | str | None = None,
        label: int | None = None,
    ) -> tuple[int, ...]:
        selected_regime = SignalRegime.coerce(regime) if regime is not None else None
        if label is not None and label not in (0, 1):
            raise ValueError("label must be 0, 1, or None")
        return tuple(
            index
            for index, (item_regime, item_label) in enumerate(self._schedule)
            if (selected_regime is None or item_regime is selected_regime)
            and (label is None or item_label == label)
        )

    def distribution(self, indices: Sequence[int] | None = None) -> dict[str, Any]:
        """Return inspectable counts for a full dataset or proposed split."""

        selected = tuple(range(len(self))) if indices is None else tuple(int(i) for i in indices)
        if any(index < 0 or index >= len(self) for index in selected):
            raise IndexError("distribution indices must lie inside the dataset")
        pairs = tuple(self._schedule[index] for index in selected)
        return {
            "num_samples": len(selected),
            "regime_counts": count_values(regime for regime, _ in pairs),
            "label_counts": count_values(label for _, label in pairs),
            "joint_counts": count_values(pairs),
        }

    def split_indices(
        self,
        *,
        train_fraction: float = 0.7,
        validation_fraction: float = 0.15,
        seed: int | None = None,
    ) -> dict[str, tuple[int, ...]]:
        """Create deterministic group-disjoint train/validation/test indices."""

        if train_fraction <= 0 or validation_fraction < 0:
            raise ValueError("split fractions must be non-negative and train must be positive")
        if train_fraction + validation_fraction >= 1:
            raise ValueError("train_fraction + validation_fraction must be less than 1")

        groups = np.arange(max(self.group_ids) + 1, dtype=np.int64)
        if len(groups) < 2:
            raise ValueError("at least 12 samples are required for a group-disjoint split")
        _rng(self.seed if seed is None else seed, 0x53504C49).shuffle(groups)
        num_groups = len(groups)
        num_train = round(train_fraction * num_groups)
        num_validation = round(validation_fraction * num_groups)
        num_train = min(max(num_train, 1), num_groups - 1)
        num_validation = min(num_validation, num_groups - num_train - 1)
        if validation_fraction > 0 and num_groups >= 3:
            num_validation = max(num_validation, 1)
            if num_train + num_validation >= num_groups:
                num_train = num_groups - num_validation - 1

        assignments = {
            "train": frozenset(int(group) for group in groups[:num_train]),
            "validation": frozenset(
                int(group) for group in groups[num_train : num_train + num_validation]
            ),
            "test": frozenset(int(group) for group in groups[num_train + num_validation :]),
        }
        return {
            name: tuple(
                index for index, group in enumerate(self.group_ids) if group in selected_groups
            )
            for name, selected_groups in assignments.items()
        }

    def __getitem__(self, index: int) -> StructuredSample:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("MixedStructuredSignal indices must be integers")
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        regime, label = self._schedule[index]
        group_id = self.group_ids[index]
        group_rng = _rng(self.seed, 0x47524F55, group_id)
        sample_rng = _rng(self.seed, 0x53414D50, index)

        num_vertices = int(
            group_rng.integers(self.min_vertices, self.max_vertices + 1)
        )
        edges, faces = _base_graph(num_vertices, group_rng)
        if len(faces) < 2:
            raise RuntimeError("the mixed structured base graph must contain at least two faces")

        edge_to_index = {edge: position for position, edge in enumerate(edges)}
        anchor_edge_positions = group_rng.choice(len(edges), size=2, replace=False)
        graph_anchor = edges[int(anchor_edge_positions[0])]
        sheaf_anchor_position = int(anchor_edge_positions[1])

        face_anchor_positions = group_rng.choice(len(faces), size=2, replace=False)
        face_anchor_positions = tuple(int(item) for item in face_anchor_positions)

        nuisance_bits = {
            SignalRegime.GRAPH: int(sample_rng.integers(0, 2)),
            SignalRegime.CELL: int(sample_rng.integers(0, 2)),
            SignalRegime.SHEAF: int(sample_rng.integers(0, 2)),
        }
        nuisance_bits[regime] = label
        graph_bit = nuisance_bits[SignalRegime.GRAPH]
        cell_bit = nuisance_bits[SignalRegime.CELL]
        sheaf_bit = nuisance_bits[SignalRegime.SHEAF]

        target_strength = 2.5
        nuisance_strength = 0.65
        strengths = {
            item: target_strength if item is regime else nuisance_strength for item in SignalRegime
        }

        node_array = sample_rng.normal(
            loc=0.0,
            scale=0.20,
            size=(num_vertices, self.node_feature_dim),
        )
        u, v = graph_anchor
        orientation = 1.0 if int(sample_rng.integers(0, 2)) else -1.0
        relation = 1.0 if graph_bit else -1.0
        node_array[u, 0] += orientation * strengths[SignalRegime.GRAPH]
        node_array[v, 0] += orientation * relation * strengths[SignalRegime.GRAPH]

        frame_angles = group_rng.uniform(-pi, pi, size=num_vertices)
        global_vector = sample_rng.normal(size=2)
        global_vector /= np.linalg.norm(global_vector) + 1e-12
        for vertex, angle in enumerate(frame_angles):
            local_vector = np.asarray(
                [
                    cos(angle) * global_vector[0] + sin(angle) * global_vector[1],
                    -sin(angle) * global_vector[0] + cos(angle) * global_vector[1],
                ]
            )
            node_array[vertex, -2:] += strengths[SignalRegime.SHEAF] * local_vector

        edge_array = sample_rng.normal(
            loc=0.0,
            scale=0.15,
            size=(len(edges), self.edge_feature_dim),
        )
        for edge_position, (tail, head) in enumerate(edges):
            edge_array[edge_position, 0] += node_array[head, 0] - node_array[tail, 0]

        active_count = max(1, min(8, len(faces) // 4))
        excluded = set(face_anchor_positions)
        remaining = [position for position in range(len(faces)) if position not in excluded]
        background_positions = group_rng.choice(
            remaining,
            size=active_count - 1,
            replace=False,
        )
        probe_face_position = face_anchor_positions[0]
        selected_anchor_position = (
            probe_face_position if cell_bit else face_anchor_positions[1]
        )
        active_positions = {int(item) for item in background_positions}
        active_positions.add(selected_anchor_position)
        face_active = torch.zeros(len(faces), dtype=torch.bool)
        face_active[sorted(active_positions)] = True

        # Keep the observed cochain independent of the cell label. The label is
        # whether this fixed energized probe cycle is filled, which requires
        # relating edge circulation to the active 2-cell mask.
        circulation_face = faces[probe_face_position]
        circulation_orientation = 1.0 if int(sample_rng.integers(0, 2)) else -1.0
        for edge, coefficient in _edge_boundary(circulation_face):
            edge_array[edge_to_index[edge], 1] += (
                circulation_orientation * coefficient * strengths[SignalRegime.CELL]
            )

        transports = []
        defect_angle = (pi / 2 if regime is SignalRegime.SHEAF else pi / 8) * sheaf_bit
        defect = _rotation(defect_angle, dtype=self.dtype)
        for edge_position, (tail, head) in enumerate(edges):
            base = _rotation(
                float(frame_angles[tail] - frame_angles[head]),
                dtype=self.dtype,
            )
            transports.append(defect @ base if edge_position == sheaf_anchor_position else base)

        node_features = torch.as_tensor(node_array, dtype=self.dtype)
        edge_features = torch.as_tensor(edge_array, dtype=self.dtype)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        face_index = torch.tensor(faces, dtype=torch.long).t().contiguous()
        transport = torch.stack(transports, dim=0)

        return StructuredSample(
            observations=StructuredObservations(
                node_features=node_features,
                edge_features=edge_features,
            ),
            edge_index=edge_index,
            face_index=face_index,
            face_active=face_active,
            transport=transport,
            label=torch.tensor(label, dtype=torch.long),
            regime=regime,
            sample_id=f"mixed-{self.seed}-{index:06d}",
            metadata={
                "generator": "MixedStructuredSignal",
                "generator_version": 1,
                "benchmark_tier": "bringup",
                "group_id": group_id,
                "sample_index": index,
                "num_vertices": num_vertices,
                "num_edges": len(edges),
                "num_faces": len(faces),
            },
        )


__all__ = ["MixedStructuredSignal"]
