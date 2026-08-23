"""A generator in which graph-to-cell and graph-to-sheaf conversion is learnable.

``ConfirmatoryStructuredSignal`` deliberately hides cell and sheaf structure from
the graph observation: it energises a probe face's boundary regardless of the
cell label and draws sheaf frames independently of the node fields. That is the
right design for routing, where each label must require its own view or the
router shortcuts. It makes conversion impossible by construction, because no
function of the graph determines the targets.

This generator inverts that choice, and is therefore a separate object rather
than a mode of the existing one. Every target here is a deterministic function of
the graph observation, so a perfect converter exists and a learned one can be
measured against an attainable ceiling.

Three facts make the construction work.

* The two-cells are a **cycle basis of the graph**, so the cell complex is
  determined by the graph rather than drawn beside it. Because graph size and
  density vary, the cycle rank varies, and the homological defect of a conversion
  varies with it. Every earlier experiment in this repository failed because that
  defect came out constant.
* Face activity thresholds the **circulation of the edge cochain** around each
  cycle, which is exactly ``B2^T x1``. Integrating an edge feature around a cycle
  is the operation a graph-to-cell converter has to perform.
* Sheaf transports combine a frame difference with a per-edge twist. The frame
  difference telescopes to the identity around any closed cycle, so on its own it
  would make every holonomy trivial; the twist channel is what gives cycle
  holonomy content. It is carried on a different edge-feature channel from the
  cochain, so holonomy and face activity are not the same quantity in disguise.

See ``docs/25-conversion-generator-spec.md``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .types import StructuredObservations

NODE_FEATURE_DIM = 4
EDGE_FEATURE_DIM = 3
COCHAIN_CHANNEL = 0
TWIST_CHANNEL = 1


@dataclass(frozen=True, slots=True)
class ConversionConfig:
    """Frozen sampling design for the conversion generator."""

    min_vertices: int = 8
    max_vertices: int = 14
    min_density: float = 0.22
    max_density: float = 0.42
    twist_scale: float = 0.6
    node_noise: float = 0.05
    edge_noise: float = 0.05
    # Face activity thresholds |circulation| at this quantile of the observed
    # circulations, so activity is neither almost-always nor almost-never true.
    activity_quantile: float = 0.5
    max_attempts: int = 64

    def __post_init__(self) -> None:
        if self.min_vertices < 4:
            raise ValueError("min_vertices must be at least four")
        if self.max_vertices < self.min_vertices:
            raise ValueError("max_vertices must be at least min_vertices")
        if not 0.0 < self.min_density <= self.max_density < 1.0:
            raise ValueError("densities must satisfy 0 < min <= max < 1")
        if self.twist_scale <= 0:
            raise ValueError("twist_scale must be positive")
        if self.node_noise < 0 or self.edge_noise < 0:
            raise ValueError("noise scales must be nonnegative")
        if not 0.0 < self.activity_quantile < 1.0:
            raise ValueError("activity_quantile must lie strictly in (0, 1)")


@dataclass(frozen=True, slots=True)
class ConversionSample:
    """One graph observation and the cell and sheaf structure it determines."""

    sample_id: str
    observations: StructuredObservations
    edge_index: Tensor
    boundary_1: Tensor
    boundary_2: Tensor
    face_cycles: tuple[tuple[int, ...], ...]
    face_circulation: Tensor
    face_active: Tensor
    edge_transport_angle: Tensor
    cycle_holonomy_angle: Tensor
    cycle_rank: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_vertices(self) -> int:
        return int(self.boundary_1.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.boundary_1.shape[1])

    @property
    def num_faces(self) -> int:
        return int(self.boundary_2.shape[1])


def _sample_seed(seed: int, index: int) -> int:
    digest = hashlib.sha256(f"homymoly-conversion:{seed}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _cycle_basis_boundary(
    graph: nx.Graph,
    order: list[int],
    edge_position: dict[tuple[int, int], int],
    *,
    dtype: torch.dtype,
) -> tuple[Tensor, tuple[tuple[int, ...], ...]]:
    """Build ``B2`` from a cycle basis, oriented so ``B1 @ B2 == 0``.

    Consecutive vertices in a ``networkx`` cycle basis entry are adjacent, so each
    step of the walk is an edge of the graph. Traversing ``a -> b`` agrees with the
    canonical orientation ``(min, max)`` when ``a < b`` and opposes it otherwise;
    summing those signed edges telescopes the vertex boundary to zero.
    """

    relabel = {node: position for position, node in enumerate(order)}
    cycles = [tuple(relabel[node] for node in cycle) for cycle in nx.cycle_basis(graph)]
    boundary = torch.zeros((len(edge_position), len(cycles)), dtype=dtype)
    for face, cycle in enumerate(cycles):
        for tail, head in zip(cycle, cycle[1:] + cycle[:1], strict=True):
            key = (min(tail, head), max(tail, head))
            if key not in edge_position:
                raise RuntimeError(
                    "a cycle-basis step is not an edge of the graph; the cycle "
                    "basis and edge set disagree"
                )
            boundary[edge_position[key], face] = 1.0 if tail < head else -1.0
    return boundary, tuple(cycles)


class ConversionDataset(Dataset[ConversionSample]):
    """Deterministic graphs whose cell and sheaf structure the graph determines."""

    def __init__(
        self,
        num_samples: int,
        *,
        seed: int,
        config: ConversionConfig | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if isinstance(num_samples, bool) or not isinstance(num_samples, int):
            raise TypeError("num_samples must be an integer")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.num_samples = num_samples
        self.seed = int(seed)
        self.config = config or ConversionConfig()
        self.dtype = dtype

    def __len__(self) -> int:
        return self.num_samples

    def _draw_graph(self, rng: np.random.Generator) -> nx.Graph:
        config = self.config
        for _ in range(config.max_attempts):
            vertices = int(rng.integers(config.min_vertices, config.max_vertices + 1))
            density = float(rng.uniform(config.min_density, config.max_density))
            graph = nx.gnp_random_graph(
                vertices, density, seed=int(rng.integers(1 << 31))
            )
            if nx.is_connected(graph) and graph.number_of_edges() > 0:
                return graph
        raise RuntimeError(
            "could not draw a connected graph within the attempt budget; "
            "widen the density range"
        )

    def __getitem__(self, index: int) -> ConversionSample:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("dataset indices must be integers")
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        config = self.config
        rng = np.random.default_rng(_sample_seed(self.seed, index))
        graph = self._draw_graph(rng)
        order = sorted(graph.nodes())
        relabel = {node: position for position, node in enumerate(order)}
        edges = sorted(
            (min(relabel[u], relabel[v]), max(relabel[u], relabel[v]))
            for u, v in graph.edges()
        )
        edge_position = {edge: position for position, edge in enumerate(edges)}

        num_vertices, num_edges = len(order), len(edges)
        boundary_1 = torch.zeros((num_vertices, num_edges), dtype=self.dtype)
        for position, (tail, head) in enumerate(edges):
            boundary_1[tail, position] = -1.0
            boundary_1[head, position] = 1.0
        boundary_2, cycles = _cycle_basis_boundary(
            graph, order, edge_position, dtype=self.dtype
        )

        # --- graph observation -------------------------------------------------
        frame_angle = torch.as_tensor(
            rng.uniform(-np.pi, np.pi, size=num_vertices), dtype=self.dtype
        )
        node_features = torch.stack(
            (
                torch.cos(frame_angle),
                torch.sin(frame_angle),
                torch.as_tensor(
                    rng.normal(0.0, config.node_noise, size=num_vertices),
                    dtype=self.dtype,
                ),
                torch.as_tensor(
                    rng.normal(0.0, config.node_noise, size=num_vertices),
                    dtype=self.dtype,
                ),
            ),
            dim=-1,
        )
        cochain = torch.as_tensor(
            rng.normal(0.0, 1.0, size=num_edges), dtype=self.dtype
        )
        twist = torch.as_tensor(
            rng.uniform(-config.twist_scale, config.twist_scale, size=num_edges),
            dtype=self.dtype,
        )
        edge_features = torch.stack(
            (
                cochain,
                twist,
                torch.as_tensor(
                    rng.normal(0.0, config.edge_noise, size=num_edges),
                    dtype=self.dtype,
                ),
            ),
            dim=-1,
        )

        # --- cell structure, determined by the observation ---------------------
        # Circulation of the edge cochain around each cycle. This is exactly the
        # degree-two coefficient vector B2^T x1.
        circulation = boundary_2.mT @ cochain
        if circulation.numel():
            threshold = torch.quantile(
                circulation.abs().to(torch.float32), config.activity_quantile
            ).to(self.dtype)
            face_active = circulation.abs() > threshold
        else:
            face_active = torch.zeros(0, dtype=torch.bool)

        # --- sheaf structure, determined by the observation --------------------
        # Transport combines the endpoint frame difference with the edge twist.
        # The frame difference telescopes around any closed cycle, so holonomy is
        # carried entirely by the twist channel: B2^T twist.
        tails = torch.tensor([tail for tail, _ in edges], dtype=torch.long)
        heads = torch.tensor([head for _, head in edges], dtype=torch.long)
        transport_angle = frame_angle[heads] - frame_angle[tails] + twist
        holonomy_angle = boundary_2.mT @ twist

        return ConversionSample(
            sample_id=f"conversion-{self.seed}-{index:07d}",
            observations=StructuredObservations(
                node_features=node_features, edge_features=edge_features
            ),
            edge_index=torch.tensor([tails.tolist(), heads.tolist()], dtype=torch.long),
            boundary_1=boundary_1,
            boundary_2=boundary_2,
            face_cycles=cycles,
            face_circulation=circulation,
            face_active=face_active,
            edge_transport_angle=transport_angle,
            cycle_holonomy_angle=holonomy_angle,
            cycle_rank=len(cycles),
            metadata={
                "num_vertices": num_vertices,
                "num_edges": num_edges,
                "activity_threshold": float(threshold) if circulation.numel() else 0.0,
            },
        )
