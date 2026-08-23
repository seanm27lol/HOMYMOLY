"""Identifiable graph-only typed maps on a finite cellular annulus.

This experiment is intentionally narrower than a general graph-to-cell or
graph-to-sheaf translator.  It supplies a controlled setting in which an
ordered pair of node markers in the graph observation uniquely identifies one
element of a dihedral action.  That element determines held-out oriented cell
coefficients and rank-two sheaf transports.

Every decoded map is a linear combination of signed cellular automorphisms.
Consequently the two declared chain-map equations

``B1 @ F1 = F0 @ B1`` and ``B2 @ F2 = F1 @ B2``

hold architecturally (up to floating-point roundoff).  This construction does
not assert categorical equivalence between arbitrary representation types.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import gcd
from typing import Literal, NamedTuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from homymoly.metrics import (
    pairwise_euclidean_distances,
    symmetric_h0_srtd_surrogate,
)
from homymoly.topology import build_oriented_incidence

Ablation = Literal[
    "combined",
    "task_only",
    "reconstruction_only",
    "task_reconstruction",
    "task_reconstruction_cone",
    "task_reconstruction_rtd",
    "cone_only",
    "rtd_only",
]

ABLATIONS: tuple[Ablation, ...] = (
    "combined",
    "task_only",
    "reconstruction_only",
    "task_reconstruction",
    "task_reconstruction_cone",
    "task_reconstruction_rtd",
    "cone_only",
    "rtd_only",
)


class DegreeMaps(NamedTuple):
    """Batched degree maps ``F0``, ``F1``, and ``F2``."""

    degree_zero: Tensor
    degree_one: Tensor
    degree_two: Tensor


@dataclass(frozen=True, slots=True)
class AnnulusMapSystem:
    """A cellular annulus and all signed maps in its dihedral action."""

    sectors: int
    edges: tuple[tuple[int, int], ...]
    faces: tuple[tuple[int, ...], ...]
    boundary_1: Tensor
    boundary_2: Tensor
    basis: DegreeMaps
    marker_pairs: tuple[tuple[int, int], ...]
    transformations: tuple[tuple[int, int], ...]

    @property
    def num_vertices(self) -> int:
        return 2 * self.sectors

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    @property
    def num_transformations(self) -> int:
        return len(self.transformations)


def build_annulus_map_system(
    sectors: int = 6,
    *,
    dtype: torch.dtype = torch.float64,
    tolerance: float = 1e-12,
) -> AnnulusMapSystem:
    """Build a cellular annulus and its orientation-aware dihedral maps.

    Vertices ``0..n-1`` form the inner ring and vertices ``n..2n-1`` form
    the outer ring. The one-cells are both ring cycles plus the ``n`` radial
    edges. Each sector is closed by one quadrilateral two-cell. Degree-two
    signs are derived by matching the mapped cellular boundary.
    """

    if isinstance(sectors, bool) or not isinstance(sectors, int):
        raise TypeError("sectors must be an integer")
    if sectors < 4:
        raise ValueError("sectors must be at least four")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("annulus maps require a floating dtype")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    inner = tuple(range(sectors))
    outer = tuple(range(sectors, 2 * sectors))
    edges = tuple(
        [(inner[index], inner[(index + 1) % sectors]) for index in range(sectors)]
        + [(outer[index], outer[(index + 1) % sectors]) for index in range(sectors)]
        + [(inner[index], outer[index]) for index in range(sectors)]
    )
    faces = tuple(
        (
            inner[index],
            inner[(index + 1) % sectors],
            outer[(index + 1) % sectors],
            outer[index],
        )
        for index in range(sectors)
    )
    incidence = build_oriented_incidence(
        2 * sectors,
        edges,
        faces,
        dtype=torch.float64,
    )
    edge_position = {edge: index for index, edge in enumerate(incidence.edges)}

    maps_zero: list[Tensor] = []
    maps_one: list[Tensor] = []
    maps_two: list[Tensor] = []
    marker_pairs: list[tuple[int, int]] = []
    transformations: list[tuple[int, int]] = []
    for orientation in (1, -1):
        for shift in range(sectors):
            vertex_map: dict[int, int] = {}
            for source_position in range(sectors):
                target_position = (shift + orientation * source_position) % sectors
                vertex_map[source_position] = target_position
                vertex_map[sectors + source_position] = sectors + target_position

            degree_zero = torch.zeros((2 * sectors, 2 * sectors), dtype=torch.float64)
            for source, target in vertex_map.items():
                degree_zero[target, source] = 1.0

            degree_one = torch.zeros(
                (len(incidence.edges), len(incidence.edges)), dtype=torch.float64
            )
            for source_index, (tail, head) in enumerate(incidence.edges):
                mapped_tail = vertex_map[tail]
                mapped_head = vertex_map[head]
                target_edge = tuple(sorted((mapped_tail, mapped_head)))
                sign = 1.0 if mapped_tail < mapped_head else -1.0
                degree_one[edge_position[target_edge], source_index] = sign

            degree_two = torch.zeros(
                (len(incidence.faces), len(incidence.faces)), dtype=torch.float64
            )
            for source_index in range(len(incidence.faces)):
                mapped_boundary = degree_one @ incidence.boundary_2[:, source_index]
                matches = [
                    (target_index, sign)
                    for target_index in range(len(incidence.faces))
                    for sign in (1.0, -1.0)
                    if torch.equal(
                        mapped_boundary,
                        sign * incidence.boundary_2[:, target_index],
                    )
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        "mapped face boundary does not identify one oriented cell: "
                        f"source={source_index}, matches={matches}"
                    )
                target_index, sign = matches[0]
                degree_two[target_index, source_index] = sign

            residual_one = (
                incidence.boundary_1 @ degree_one - degree_zero @ incidence.boundary_1
            )
            residual_two = (
                incidence.boundary_2 @ degree_two - degree_one @ incidence.boundary_2
            )
            residual = max(
                float(residual_one.abs().max()),
                float(residual_two.abs().max()),
            )
            if residual > tolerance:
                raise RuntimeError(
                    "a generated dihedral basis map violates the chain-map law: "
                    f"{residual:.3e} > {tolerance:.3e}"
                )
            maps_zero.append(degree_zero)
            maps_one.append(degree_one)
            maps_two.append(degree_two)
            marker_pairs.append((vertex_map[0], vertex_map[1]))
            transformations.append((orientation, shift))

    if len(set(marker_pairs)) != 2 * sectors:
        raise RuntimeError("ordered graph markers do not identify the dihedral action")
    return AnnulusMapSystem(
        sectors=sectors,
        edges=tuple(incidence.edges),
        faces=tuple(incidence.faces),
        boundary_1=incidence.boundary_1.to(dtype=dtype),
        boundary_2=incidence.boundary_2.to(dtype=dtype),
        basis=DegreeMaps(
            torch.stack(maps_zero).to(dtype=dtype),
            torch.stack(maps_one).to(dtype=dtype),
            torch.stack(maps_two).to(dtype=dtype),
        ),
        marker_pairs=tuple(marker_pairs),
        transformations=tuple(transformations),
    )


def decode_ordered_markers(node_features: Tensor, system: AnnulusMapSystem) -> Tensor:
    """Decode the transformation index from graph-only marker channels.

    Channel one marks the image of the first canonical rim vertex and channel
    two marks the image of its oriented successor.  The ordered pair is a
    bijective code for the finite transformation family.
    """

    values = torch.as_tensor(node_features)
    if values.ndim not in (2, 3) or values.shape[-1] < 3:
        raise ValueError("node_features must have shape [V,D] or [B,V,D], D >= 3")
    unbatched = values.ndim == 2
    if unbatched:
        values = values.unsqueeze(0)
    if values.shape[1] != system.num_vertices:
        raise ValueError("node_features have the wrong annulus vertex count")
    anchor = values[..., 1].argmax(dim=1)
    successor = values[..., 2].argmax(dim=1)
    lookup = {pair: index for index, pair in enumerate(system.marker_pairs)}
    decoded: list[int] = []
    for left, right in zip(anchor.tolist(), successor.tolist(), strict=True):
        pair = (int(left), int(right))
        if pair not in lookup:
            raise ValueError(f"marker pair {pair} is not a declared transformation")
        decoded.append(lookup[pair])
    result = torch.tensor(decoded, dtype=torch.long, device=values.device)
    return result[0] if unbatched else result


def _coordinate_seed(seed: int, index: int) -> int:
    digest = hashlib.sha256(f"homymoly-identifiable:{seed}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


class IdentifiableTypedMapDataset(Dataset[dict[str, Tensor | str]]):
    """Deterministic paired graph observations and held-out typed targets.

    Only ``node_features`` and ``edge_features`` are model inputs.  Source
    degree-two coefficients are derived from the observed edge cochain as
    ``B2.T @ x1``.  Cell and sheaf targets are never inserted into either
    observation tensor.
    """

    def __init__(
        self,
        num_samples: int,
        *,
        seed: int,
        sectors: int = 6,
        noise_std: float = 0.05,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if isinstance(num_samples, bool) or not isinstance(num_samples, int):
            raise TypeError("num_samples must be an integer")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if noise_std < 0:
            raise ValueError("noise_std must be nonnegative")
        self.num_samples = num_samples
        self.seed = int(seed)
        self.noise_std = float(noise_std)
        self.dtype = dtype
        self.system = build_annulus_map_system(sectors, dtype=dtype)
        self._class_stride = self._coprime_stride(self.system.num_transformations)
        self._class_offset = _coordinate_seed(self.seed, 0xD1ED) % (
            self.system.num_transformations
        )

    def _coprime_stride(self, classes: int) -> int:
        candidate = 1 + _coordinate_seed(self.seed, 0x57A1DE) % (classes - 1)
        while gcd(candidate, classes) != 1:
            candidate = 1 + candidate % (classes - 1)
        return int(candidate)

    def __len__(self) -> int:
        return self.num_samples

    def transformation_index(self, index: int) -> int:
        return int(
            (self._class_offset + self._class_stride * index)
            % self.system.num_transformations
        )

    @staticmethod
    def graph_observation(sample: dict[str, Tensor | str]) -> dict[str, Tensor]:
        """Return the complete and exclusive model input view."""

        return {
            "node_features": torch.as_tensor(sample["node_features"]),
            "edge_features": torch.as_tensor(sample["edge_features"]),
        }

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("dataset indices must be integers")
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        generator = torch.Generator().manual_seed(_coordinate_seed(self.seed, index))
        transform = self.transformation_index(index)
        maps = DegreeMaps(*(degree[transform] for degree in self.system.basis))

        source_zero = torch.randn(
            self.system.num_vertices, generator=generator, dtype=self.dtype
        )
        source_one = torch.randn(
            self.system.num_edges, generator=generator, dtype=self.dtype
        )
        source_two = self.system.boundary_2.mT @ source_one
        source_sheaf_angle = 0.75 * torch.tanh(
            torch.randn(self.system.num_edges, generator=generator, dtype=self.dtype)
        )

        node_noise = self.noise_std * torch.randn(
            self.system.num_vertices, generator=generator, dtype=self.dtype
        )
        edge_noise = self.noise_std * torch.randn(
            self.system.num_edges, generator=generator, dtype=self.dtype
        )
        anchor, successor = self.system.marker_pairs[transform]
        anchor_marker = torch.zeros(self.system.num_vertices, dtype=self.dtype)
        successor_marker = torch.zeros(self.system.num_vertices, dtype=self.dtype)
        anchor_marker[anchor] = 1.0
        successor_marker[successor] = 1.0
        node_features = torch.stack(
            (source_zero, anchor_marker, successor_marker, node_noise), dim=-1
        )
        edge_features = torch.stack(
            (source_one, source_sheaf_angle, edge_noise), dim=-1
        )

        source_active = F.one_hot(
            source_two.abs().argmax(), num_classes=self.system.num_faces
        ).to(dtype=self.dtype)
        target_active = maps.degree_two.abs() @ source_active
        return {
            "sample_id": f"identifiable-{self.seed}-{index:07d}",
            "node_features": node_features,
            "edge_features": edge_features,
            "source_degree_zero": source_zero,
            "source_degree_one": source_one,
            "source_degree_two": source_two,
            "source_sheaf_angle": source_sheaf_angle,
            "source_cell_active": source_active,
            "target_degree_zero": maps.degree_zero @ source_zero,
            "target_degree_one": maps.degree_one @ source_one,
            "target_degree_two": maps.degree_two @ source_two,
            "target_sheaf_angle": maps.degree_one @ source_sheaf_angle,
            "target_cell_active": target_active,
            "transformation": torch.tensor(transform, dtype=torch.long),
        }


class IdentifiableMapOutput(NamedTuple):
    logits: Tensor
    weights: Tensor
    maps: DegreeMaps
    source_degree_two: Tensor
    source_cell_active: Tensor
    target_degree_zero: Tensor
    target_degree_one: Tensor
    target_degree_two: Tensor
    target_sheaf_angle: Tensor
    target_cell_active: Tensor


class IdentifiableTypedMapModel(nn.Module):
    """Graph-only finite-action decoder with architecturally exact maps."""

    def __init__(
        self,
        system: AnnulusMapSystem,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        map_temperature: float = 1.0,
        basis_tolerance: float = 1e-10,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if map_temperature <= 0:
            raise ValueError("map_temperature must be positive")
        self.sectors = system.sectors
        self.num_vertices = system.num_vertices
        self.num_edges = system.num_edges
        self.num_faces = system.num_faces
        self.num_transformations = system.num_transformations
        self.map_temperature = float(map_temperature)
        self.register_buffer("boundary_1", system.boundary_1.to(torch.float32))
        self.register_buffer("boundary_2", system.boundary_2.to(torch.float32))
        self.register_buffer("basis_zero", system.basis.degree_zero.to(torch.float32))
        self.register_buffer("basis_one", system.basis.degree_one.to(torch.float32))
        self.register_buffer("basis_two", system.basis.degree_two.to(torch.float32))
        input_dim = self.num_vertices * 4 + self.num_edges * 3
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_transformations),
        )
        basis_residual = self.basis_residual_max()
        if basis_residual > basis_tolerance:
            raise RuntimeError(
                "registered basis violates the chain-map equations: "
                f"{basis_residual:.3e} > {basis_tolerance:.3e}"
            )

    def basis_residual_max(self) -> float:
        first = torch.einsum("ve,gej->gvj", self.boundary_1, self.basis_one)
        first = first - torch.einsum("gvi,ie->gve", self.basis_zero, self.boundary_1)
        second = torch.einsum("ef,gfj->gej", self.boundary_2, self.basis_two)
        second = second - torch.einsum("gei,if->gef", self.basis_one, self.boundary_2)
        return max(float(first.abs().max()), float(second.abs().max()))

    def forward(
        self, node_features: Tensor, edge_features: Tensor
    ) -> IdentifiableMapOutput:
        if node_features.ndim != 3 or tuple(node_features.shape[1:]) != (
            self.num_vertices,
            4,
        ):
            raise ValueError(f"node_features must have shape [B,{self.num_vertices},4]")
        if edge_features.ndim != 3 or tuple(edge_features.shape[1:]) != (
            self.num_edges,
            3,
        ):
            raise ValueError(f"edge_features must have shape [B,{self.num_edges},3]")
        if node_features.shape[0] != edge_features.shape[0]:
            raise ValueError("node and edge feature batches must agree")
        encoded = torch.cat(
            (node_features.flatten(1), edge_features.flatten(1)), dim=1
        ).to(dtype=self.boundary_1.dtype)
        logits = self.encoder(encoded)
        weights = torch.softmax(logits / self.map_temperature, dim=-1)
        maps = DegreeMaps(
            torch.einsum("bg,gij->bij", weights, self.basis_zero),
            torch.einsum("bg,gij->bij", weights, self.basis_one),
            torch.einsum("bg,gij->bij", weights, self.basis_two),
        )

        source_zero = node_features[..., 0].to(maps.degree_zero.dtype)
        source_one = edge_features[..., 0].to(maps.degree_one.dtype)
        source_sheaf = edge_features[..., 1].to(maps.degree_one.dtype)
        source_two = torch.einsum("ef,be->bf", self.boundary_2, source_one)
        source_active = F.one_hot(
            source_two.abs().argmax(dim=1), num_classes=self.num_faces
        ).to(dtype=maps.degree_two.dtype)
        unsigned_two = torch.einsum("bg,gij->bij", weights, self.basis_two.abs())
        return IdentifiableMapOutput(
            logits=logits,
            weights=weights,
            maps=maps,
            source_degree_two=source_two,
            source_cell_active=source_active,
            target_degree_zero=torch.einsum(
                "bij,bj->bi", maps.degree_zero, source_zero
            ),
            target_degree_one=torch.einsum("bij,bj->bi", maps.degree_one, source_one),
            target_degree_two=torch.einsum("bij,bj->bi", maps.degree_two, source_two),
            target_sheaf_angle=torch.einsum(
                "bij,bj->bi", maps.degree_one, source_sheaf
            ),
            target_cell_active=torch.einsum("bij,bj->bi", unsigned_two, source_active),
        )

    def residuals(self, maps: DegreeMaps) -> tuple[Tensor, Tensor]:
        """Return both batched chain-map equation residuals."""

        first = torch.einsum("ve,bej->bvj", self.boundary_1, maps.degree_one)
        first = first - torch.einsum("bvi,ie->bve", maps.degree_zero, self.boundary_1)
        second = torch.einsum("ef,bfj->bej", self.boundary_2, maps.degree_two)
        second = second - torch.einsum("bei,if->bef", maps.degree_one, self.boundary_2)
        return first, second

    def hard_maps(self, logits: Tensor) -> DegreeMaps:
        indices = logits.argmax(dim=-1)
        return DegreeMaps(
            self.basis_zero.index_select(0, indices),
            self.basis_one.index_select(0, indices),
            self.basis_two.index_select(0, indices),
        )


def mapping_cone_boundaries(
    boundary_1: Tensor,
    boundary_2: Tensor,
    maps: DegreeMaps,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build the batched cone differentials for a three-term self-map."""

    batch = maps.degree_zero.shape[0]
    b1 = boundary_1.to(maps.degree_zero).expand(batch, -1, -1)
    b2 = boundary_2.to(maps.degree_zero).expand(batch, -1, -1)
    zeros_02 = maps.degree_zero.new_zeros(
        (batch, boundary_1.shape[0], boundary_2.shape[1])
    )
    degree_one = torch.cat((b1, maps.degree_zero), dim=2)
    degree_two = torch.cat(
        (
            torch.cat((b2, maps.degree_one), dim=2),
            torch.cat((zeros_02, -b1), dim=2),
        ),
        dim=1,
    )
    degree_three = torch.cat((maps.degree_two, -b2), dim=1)
    return degree_one, degree_two, degree_three


def cone_soft_betti_loss(
    boundary_1: Tensor,
    boundary_2: Tensor,
    maps: DegreeMaps,
    *,
    temperature: float = 0.05,
) -> Tensor:
    """Differentiable spectral acyclicity proxy for the learned mapping cone."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    d1, d2, d3 = mapping_cone_boundaries(boundary_1, boundary_2, maps)
    laplacians = (
        d1 @ d1.mT,
        d1.mT @ d1 + d2 @ d2.mT,
        d2.mT @ d2 + d3 @ d3.mT,
        d3.mT @ d3,
    )
    per_degree = [
        torch.exp(
            -torch.linalg.eigvalsh(laplacian.float()).clamp_min(0.0)
            / float(temperature)
        ).mean(dim=1)
        for laplacian in laplacians
    ]
    return torch.stack(per_degree, dim=1).sum(dim=1).mean()


@dataclass(frozen=True, slots=True)
class LossWeights:
    task: float
    reconstruction: float
    cell: float
    sheaf: float
    cone: float
    rtd: float

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.as_dict().values()):
            raise ValueError("loss weights must be nonnegative")
        if not any(value > 0 for value in self.as_dict().values()):
            raise ValueError("at least one loss weight must be positive")

    def as_dict(self) -> dict[str, float]:
        return {
            "task": float(self.task),
            "reconstruction": float(self.reconstruction),
            "cell": float(self.cell),
            "sheaf": float(self.sheaf),
            "cone": float(self.cone),
            "rtd": float(self.rtd),
        }


def loss_weights_for_ablation(
    ablation: Ablation,
    *,
    combined: LossWeights | None = None,
) -> LossWeights:
    """Resolve named controls without silently retaining another objective."""

    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation {ablation!r}; expected one of {ABLATIONS}")
    base = combined or LossWeights(1.0, 1.0, 0.25, 0.25, 0.1, 0.25)
    if ablation == "combined":
        return base
    table: dict[str, LossWeights] = {
        "task_only": LossWeights(base.task, 0.0, 0.0, 0.0, 0.0, 0.0),
        "reconstruction_only": LossWeights(
            0.0, base.reconstruction, base.cell, base.sheaf, 0.0, 0.0
        ),
        "task_reconstruction": LossWeights(
            base.task, base.reconstruction, base.cell, base.sheaf, 0.0, 0.0
        ),
        "task_reconstruction_cone": LossWeights(
            base.task,
            base.reconstruction,
            base.cell,
            base.sheaf,
            base.cone,
            0.0,
        ),
        "task_reconstruction_rtd": LossWeights(
            base.task,
            base.reconstruction,
            base.cell,
            base.sheaf,
            0.0,
            base.rtd,
        ),
        "cone_only": LossWeights(0.0, 0.0, 0.0, 0.0, base.cone, 0.0),
        "rtd_only": LossWeights(0.0, 0.0, 0.0, 0.0, 0.0, base.rtd),
    }
    return table[ablation]


def typed_representation(
    degree_zero: Tensor,
    degree_one: Tensor,
    degree_two: Tensor,
    sheaf_angle: Tensor,
) -> Tensor:
    """Concatenate typed coordinates for paired-distance objectives."""

    return torch.cat(
        (
            degree_zero,
            degree_one,
            degree_two,
            torch.sin(sheaf_angle),
            torch.cos(sheaf_angle),
        ),
        dim=1,
    )


def compute_identifiable_losses(
    model: IdentifiableTypedMapModel,
    output: IdentifiableMapOutput,
    batch: dict[str, Tensor | list[str]],
    weights: LossWeights,
    *,
    cone_temperature: float = 0.05,
    rtd_entities: int | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute all controls and the explicitly weighted training objective."""

    transformation = torch.as_tensor(
        batch["transformation"], device=output.logits.device
    )
    target_zero = torch.as_tensor(
        batch["target_degree_zero"], device=output.logits.device
    )
    target_one = torch.as_tensor(
        batch["target_degree_one"], device=output.logits.device
    )
    target_two = torch.as_tensor(
        batch["target_degree_two"], device=output.logits.device
    )
    target_sheaf = torch.as_tensor(
        batch["target_sheaf_angle"], device=output.logits.device
    )
    target_cell = torch.as_tensor(
        batch["target_cell_active"], device=output.logits.device
    )

    task = F.cross_entropy(output.logits, transformation.long())
    reconstruction = (
        F.mse_loss(output.target_degree_zero, target_zero)
        + F.mse_loss(output.target_degree_one, target_one)
        + F.mse_loss(output.target_degree_two, target_two)
    ) / 3.0
    cell = F.binary_cross_entropy(
        output.target_cell_active.clamp(1e-6, 1.0 - 1e-6), target_cell
    )
    sheaf = (1.0 - torch.cos(output.target_sheaf_angle - target_sheaf)).mean()

    zero = output.logits.sum() * 0.0
    cone = (
        cone_soft_betti_loss(
            model.boundary_1,
            model.boundary_2,
            output.maps,
            temperature=cone_temperature,
        )
        if weights.cone > 0
        else zero
    )
    if rtd_entities is not None and (
        isinstance(rtd_entities, bool)
        or not isinstance(rtd_entities, int)
        or rtd_entities < 2
    ):
        raise ValueError("rtd_entities must be None or an integer >= 2")
    selected_entities = (
        output.logits.shape[0]
        if rtd_entities is None
        else min(rtd_entities, output.logits.shape[0])
    )
    if weights.rtd > 0 and selected_entities > 1:
        predicted = typed_representation(
            output.target_degree_zero[:selected_entities],
            output.target_degree_one[:selected_entities],
            output.target_degree_two[:selected_entities],
            output.target_sheaf_angle[:selected_entities],
        )
        target = typed_representation(
            target_zero[:selected_entities],
            target_one[:selected_entities],
            target_two[:selected_entities],
            target_sheaf[:selected_entities],
        )
        rtd = symmetric_h0_srtd_surrogate(
            pairwise_euclidean_distances(predicted),
            pairwise_euclidean_distances(target),
            normalization="quantile",
        )
    else:
        rtd = zero
    terms = {
        "task": task,
        "reconstruction": reconstruction,
        "cell": cell,
        "sheaf": sheaf,
        "cone": cone,
        "rtd": rtd,
    }
    objective = sum(weights.as_dict()[name] * value for name, value in terms.items())
    return objective, terms


__all__ = [
    "ABLATIONS",
    "Ablation",
    "AnnulusMapSystem",
    "DegreeMaps",
    "IdentifiableMapOutput",
    "IdentifiableTypedMapDataset",
    "IdentifiableTypedMapModel",
    "LossWeights",
    "build_annulus_map_system",
    "compute_identifiable_losses",
    "cone_soft_betti_loss",
    "decode_ordered_markers",
    "loss_weights_for_ablation",
    "mapping_cone_boundaries",
    "typed_representation",
]
