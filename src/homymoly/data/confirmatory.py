"""Confirmatory mixed-regime data with explicit anti-shortcut controls.

The Stage-1 :class:`~homymoly.data.mixed_structured.MixedStructuredSignal`
generator is intentionally easy to inspect.  This module provides the harder
Gate-2 counterpart.  Every counterfactual group contains all six
``(regime, binary label)`` pairs over one canonical complex.  Within each
regime, the two labels also share nuisance noise and route-reliability draws.

The three targets are deliberately relational:

* graph labels change which signs meet across two anchor edges while preserving
  the node-feature multiset;
* cell labels change whether a fixed, energized probe face is active while the
  edge cochain remains fixed;
* sheaf labels change cycle holonomy while individual transport marginals and
  isotropic node-field marginals remain matched.

Route reliability is observable but label-independent.  Active-route and
nuisance strengths are drawn from overlapping intervals, preventing a single
amplitude comparison from being an exact hidden-regime code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import pi
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .mixed_structured import _base_graph, _edge_boundary, _rng, _rotation
from .types import (
    SignalRegime,
    StructuredObservations,
    StructuredSample,
    count_values,
)

_REGIMES = tuple(SignalRegime)
_REGIME_LABEL_PAIRS = tuple((regime, label) for regime in _REGIMES for label in (0, 1))


@dataclass(frozen=True, slots=True)
class ConfirmatoryConfig:
    """Configuration for :class:`ConfirmatoryStructuredSignal`.

    The strength intervals intentionally overlap.  They describe a weak,
    label-independent reliability cue, not the class target itself.
    """

    num_samples: int = 600
    seed: int = 0
    min_vertices: int = 24
    max_vertices: int = 96
    node_feature_dim: int = 4
    edge_feature_dim: int = 2
    node_noise: float = 0.12
    edge_noise: float = 0.10
    reliable_strength_min: float = 0.90
    reliable_strength_max: float = 1.30
    nuisance_strength_min: float = 0.80
    nuisance_strength_max: float = 1.20
    holonomy_angle: float = pi / 2
    stalk_mode: str = "independent"
    gauge_noise_std: float = 0.3
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if not isinstance(self.num_samples, int) or self.num_samples < 18:
            raise ValueError("num_samples must be an integer of at least 18")
        if self.num_samples % len(_REGIME_LABEL_PAIRS) != 0:
            raise ValueError("num_samples must be divisible by six for complete groups")
        if not (24 <= self.min_vertices <= self.max_vertices <= 96):
            raise ValueError(
                "vertex bounds must satisfy 24 <= minimum <= maximum <= 96"
            )
        if self.node_feature_dim < 4:
            raise ValueError("node_feature_dim must be at least 4")
        if self.edge_feature_dim < 2:
            raise ValueError("edge_feature_dim must be at least 2")
        if self.node_noise < 0 or self.edge_noise < 0:
            raise ValueError("feature-noise scales must be non-negative")
        reliable = (self.reliable_strength_min, self.reliable_strength_max)
        nuisance = (self.nuisance_strength_min, self.nuisance_strength_max)
        if reliable[0] <= 0 or nuisance[0] <= 0:
            raise ValueError("reliability strengths must be positive")
        if reliable[0] >= reliable[1] or nuisance[0] >= nuisance[1]:
            raise ValueError("each reliability interval must have positive width")
        overlap = min(reliable[1], nuisance[1]) - max(reliable[0], nuisance[0])
        if overlap <= 0:
            raise ValueError("reliable and nuisance strength intervals must overlap")
        if not (0 < self.holonomy_angle < 2 * pi):
            raise ValueError("holonomy_angle must lie strictly between zero and 2*pi")
        if self.stalk_mode not in ("independent", "gauge"):
            raise ValueError("stalk_mode must be 'independent' or 'gauge'")
        if self.gauge_noise_std < 0:
            raise ValueError("gauge_noise_std must be nonnegative")
        if (
            not isinstance(self.dtype, torch.dtype)
            or not torch.empty((), dtype=self.dtype).is_floating_point()
        ):
            raise TypeError("dtype must be a floating torch dtype")


def _two_vertex_disjoint_edges(
    edges: tuple[tuple[int, int], ...],
    rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[int, int]]:
    order = [int(item) for item in rng.permutation(len(edges))]
    for left_position, left_index in enumerate(order):
        left = edges[left_index]
        left_vertices = set(left)
        for right_index in order[left_position + 1 :]:
            right = edges[right_index]
            if left_vertices.isdisjoint(right):
                return left, right
    raise RuntimeError("the confirmatory graph requires two vertex-disjoint edges")


def _face_edges(face: tuple[int, int, int]) -> frozenset[tuple[int, int]]:
    a, b, c = face
    return frozenset(((a, b), (a, c), (b, c)))


def _two_edge_disjoint_face_positions(
    faces: tuple[tuple[int, int, int], ...],
    rng: np.random.Generator,
) -> tuple[int, int]:
    order = [int(item) for item in rng.permutation(len(faces))]
    for left_offset, left_position in enumerate(order):
        left_edges = _face_edges(faces[left_position])
        for right_position in order[left_offset + 1 :]:
            if left_edges.isdisjoint(_face_edges(faces[right_position])):
                return left_position, right_position
    # The relation remains valid if the two candidates share an edge.  This
    # fallback is retained for future sparse graph generators.
    return order[0], order[1]


def _edge_position_for_face(
    face: tuple[int, int, int],
    edge_to_position: dict[tuple[int, int], int],
    rng: np.random.Generator,
) -> int:
    boundary = tuple(edge for edge, _ in _edge_boundary(face))
    return edge_to_position[boundary[int(rng.integers(0, len(boundary)))]]


class ConfirmatoryStructuredSignal(Dataset[StructuredSample]):
    """Hard graph/cell/sheaf counterfactual benchmark for Gate 2.

    The constructor mirrors ``MixedStructuredSignal`` for trainer integration.
    A :class:`ConfirmatoryConfig` can instead be supplied positionally or via
    ``config=``; when present it is authoritative.
    """

    def __init__(
        self,
        num_samples: int | ConfirmatoryConfig = 600,
        *,
        config: ConfirmatoryConfig | None = None,
        seed: int = 0,
        min_vertices: int = 24,
        max_vertices: int = 96,
        num_vertices: int | tuple[int, int] | None = None,
        node_feature_dim: int = 4,
        edge_feature_dim: int = 2,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if isinstance(num_samples, ConfirmatoryConfig):
            if config is not None:
                raise ValueError("provide a ConfirmatoryConfig only once")
            config = num_samples
        if config is None:
            if num_vertices is not None:
                if isinstance(num_vertices, int):
                    min_vertices = max_vertices = num_vertices
                elif isinstance(num_vertices, tuple) and len(num_vertices) == 2:
                    min_vertices, max_vertices = num_vertices
                else:
                    raise TypeError(
                        "num_vertices must be an int or a (minimum, maximum) tuple"
                    )
            config = ConfirmatoryConfig(
                num_samples=int(num_samples),
                seed=seed,
                min_vertices=min_vertices,
                max_vertices=max_vertices,
                node_feature_dim=node_feature_dim,
                edge_feature_dim=edge_feature_dim,
                dtype=dtype,
            )
        elif num_vertices is not None:
            raise ValueError("num_vertices cannot override an explicit config")

        self.config = config
        self.num_samples = config.num_samples
        self.seed = config.seed
        self.min_vertices = config.min_vertices
        self.max_vertices = config.max_vertices
        self.node_feature_dim = config.node_feature_dim
        self.edge_feature_dim = config.edge_feature_dim
        self.dtype = config.dtype
        self._schedule = self._make_schedule()

    def _make_schedule(self) -> tuple[tuple[SignalRegime, int], ...]:
        schedule: list[tuple[SignalRegime, int]] = []
        group_count = self.num_samples // len(_REGIME_LABEL_PAIRS)
        for group_id in range(group_count):
            schedule_rng = _rng(self.seed, 0x434F4E46, group_id)
            order = schedule_rng.permutation(len(_REGIME_LABEL_PAIRS))
            schedule.extend(_REGIME_LABEL_PAIRS[int(position)] for position in order)
        return tuple(schedule)

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
        return tuple(index // len(_REGIME_LABEL_PAIRS) for index in range(len(self)))

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
        selected = (
            tuple(range(len(self)))
            if indices is None
            else tuple(int(i) for i in indices)
        )
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
        """Return deterministic, complete-group train/validation/test indices."""

        if train_fraction <= 0 or validation_fraction < 0:
            raise ValueError(
                "split fractions must be non-negative and train must be positive"
            )
        if train_fraction + validation_fraction >= 1:
            raise ValueError("train_fraction + validation_fraction must be less than 1")

        groups = np.arange(self.num_samples // len(_REGIME_LABEL_PAIRS), dtype=np.int64)
        if len(groups) < 3:
            raise ValueError("at least three complete groups are required")
        _rng(self.seed if seed is None else seed, 0x53504C49).shuffle(groups)
        num_groups = len(groups)
        num_train = min(max(round(train_fraction * num_groups), 1), num_groups - 2)
        num_validation = round(validation_fraction * num_groups)
        if validation_fraction > 0:
            num_validation = max(num_validation, 1)
        num_validation = min(num_validation, num_groups - num_train - 1)

        assignments = {
            "train": frozenset(int(group) for group in groups[:num_train]),
            "validation": frozenset(
                int(group) for group in groups[num_train : num_train + num_validation]
            ),
            "test": frozenset(
                int(group) for group in groups[num_train + num_validation :]
            ),
        }
        return {
            name: tuple(
                index
                for index, group in enumerate(self.group_ids)
                if group in selected_groups
            )
            for name, selected_groups in assignments.items()
        }

    def __getitem__(self, index: int) -> StructuredSample:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("ConfirmatoryStructuredSignal indices must be integers")
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        regime, label = self._schedule[index]
        regime_position = _REGIMES.index(regime)
        group_id = index // len(_REGIME_LABEL_PAIRS)
        structure_rng = _rng(self.seed, 0x53545255, group_id)
        pair_rng = _rng(self.seed, 0x50414952, group_id, regime_position)

        num_vertices = int(
            structure_rng.integers(self.min_vertices, self.max_vertices + 1)
        )
        edges, faces = _base_graph(num_vertices, structure_rng)
        if len(faces) < 3:
            raise RuntimeError("the confirmatory complex requires at least three faces")

        edge_to_position = {edge: position for position, edge in enumerate(edges)}
        graph_edges = _two_vertex_disjoint_edges(edges, structure_rng)
        cell_decoy_position, cell_probe_position = _two_edge_disjoint_face_positions(
            faces, structure_rng
        )
        remaining_holonomy_faces = [
            position
            for position in range(len(faces))
            if position not in (cell_decoy_position, cell_probe_position)
        ]
        holonomy_face_position = int(structure_rng.choice(remaining_holonomy_faces))
        holonomy_edge_position = _edge_position_for_face(
            faces[holonomy_face_position], edge_to_position, structure_rng
        )

        active_count = max(1, min(8, len(faces) // 4))
        excluded_faces = {cell_decoy_position, cell_probe_position}
        background_candidates = [
            position for position in range(len(faces)) if position not in excluded_faces
        ]
        background_positions = structure_rng.choice(
            background_candidates, size=active_count - 1, replace=False
        )

        nuisance_bits = [
            int(item) for item in pair_rng.integers(0, 2, size=len(_REGIMES))
        ]
        nuisance_bits[regime_position] = label
        graph_bit, cell_bit, sheaf_bit = nuisance_bits

        qualities = pair_rng.uniform(
            self.config.nuisance_strength_min,
            self.config.nuisance_strength_max,
            size=len(_REGIMES),
        )
        qualities[regime_position] = pair_rng.uniform(
            self.config.reliable_strength_min,
            self.config.reliable_strength_max,
        )
        graph_quality, cell_quality, sheaf_quality = (float(item) for item in qualities)

        node_array = pair_rng.normal(
            loc=0.0,
            scale=self.config.node_noise,
            size=(num_vertices, self.node_feature_dim),
        )
        (u, v), (p, q) = graph_edges
        graph_vertices = (u, v, p, q)
        node_array[list(graph_vertices), 0] = 0.0
        graph_sign = 1.0 if int(pair_rng.integers(0, 2)) else -1.0
        if graph_bit:
            signs = (1.0, 1.0, -1.0, -1.0)
        else:
            signs = (1.0, -1.0, -1.0, 1.0)
        for vertex, sign in zip(graph_vertices, signs, strict=True):
            node_array[vertex, 0] = graph_sign * graph_quality * sign

        # Independent mode (default) preserves the shipped rng stream and
        # produces a field that is deliberately decoupled from the frames.
        # Gauge mode negates the connection frame angles for the stalks —
        # T_e = R(phi_tail - phi_head) transports R(-phi_tail)u to
        # R(-phi_head)u exactly — and adds per-vertex noise, so clean
        # samples are approximate global sections (the doc-03 pure-gauge
        # sentinel) and consistency objectives have a zero noise floor; its
        # stream differs from independent mode.
        field_angles = pair_rng.uniform(-pi, pi, size=num_vertices)
        if self.config.stalk_mode == "gauge":
            frame_angles = field_angles
            field_angles = pair_rng.normal(
                0.0, self.config.gauge_noise_std, size=num_vertices
            ) - frame_angles
        node_array[:, -2] = sheaf_quality * np.cos(field_angles)
        node_array[:, -1] = sheaf_quality * np.sin(field_angles)

        edge_array = pair_rng.normal(
            loc=0.0,
            scale=self.config.edge_noise,
            size=(len(edges), self.edge_feature_dim),
        )
        cell_sign = 1.0 if int(pair_rng.integers(0, 2)) else -1.0
        for edge, coefficient in _edge_boundary(faces[cell_probe_position]):
            edge_array[edge_to_position[edge], 1] += (
                cell_sign * cell_quality * coefficient
            )

        selected_cell_position = (
            cell_probe_position if cell_bit else cell_decoy_position
        )
        active_positions = {int(item) for item in background_positions}
        active_positions.add(selected_cell_position)
        face_active = torch.zeros(len(faces), dtype=torch.bool)
        face_active[sorted(active_positions)] = True

        if self.config.stalk_mode == "gauge":
            pass  # frame_angles already shares the stalk base draw above
        else:
            frame_angles = pair_rng.uniform(-pi, pi, size=num_vertices)
        defect_sign = 1.0 if int(pair_rng.integers(0, 2)) else -1.0
        defect = _rotation(
            defect_sign * self.config.holonomy_angle,
            dtype=self.dtype,
        )
        transports: list[Tensor] = []
        for edge_position, (tail, head) in enumerate(edges):
            base = _rotation(
                float(frame_angles[tail] - frame_angles[head]),
                dtype=self.dtype,
            )
            if edge_position == holonomy_edge_position and sheaf_bit:
                base = defect @ base
            transports.append(base)

        return StructuredSample(
            observations=StructuredObservations(
                node_features=torch.as_tensor(node_array, dtype=self.dtype),
                edge_features=torch.as_tensor(edge_array, dtype=self.dtype),
            ),
            edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
            face_index=torch.tensor(faces, dtype=torch.long).t().contiguous(),
            face_active=face_active,
            transport=torch.stack(transports, dim=0),
            label=torch.tensor(label, dtype=torch.long),
            regime=regime,
            sample_id=f"confirmatory-{self.seed}-{index:06d}",
            metadata={
                "generator": "ConfirmatoryStructuredSignal",
                "generator_version": 1,
                "group_id": group_id,
                "sample_index": index,
                "num_vertices": num_vertices,
                "num_edges": len(edges),
                "num_faces": len(faces),
            },
        )


__all__ = ["ConfirmatoryConfig", "ConfirmatoryStructuredSignal"]
