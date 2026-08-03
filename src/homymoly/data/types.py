"""Typed containers for structured synthetic observations.

The central invariant in this module is that a sample owns exactly one
``StructuredObservations`` object.  Graph, cell, and sheaf routes may interpret
the accompanying structure differently, but they do not receive separately
generated features.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import Tensor


class SignalRegime(str, Enum):
    """Latent mechanism that determines a synthetic sample's label."""

    GRAPH = "graph"
    CELL = "cell"
    SHEAF = "sheaf"

    @classmethod
    def coerce(cls, value: SignalRegime | str) -> SignalRegime:
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            valid = ", ".join(item.value for item in cls)
            raise ValueError(f"unknown signal regime {value!r}; expected one of {valid}") from exc


def _require_tensor(name: str, value: Tensor, *, ndim: int | None = None) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got shape {tuple(value.shape)}")


class _FrozenMetadata(Mapping[str, Any]):
    """Small immutable mapping that remains picklable by DataLoader workers."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = dict(values)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"_FrozenMetadata({self._values!r})"

    def __reduce__(self):  # type: ignore[no-untyped-def]
        return (_FrozenMetadata, (self._values,))


def _freeze_metadata(value: Any, *, path: str = "metadata") -> Any:
    """Recursively freeze metadata and reject feature-bearing array objects."""

    if isinstance(value, (Tensor, np.ndarray)):
        raise TypeError(f"{path} must not contain tensor or ndarray values")
    if isinstance(value, Mapping):
        return _FrozenMetadata(
            {
                str(key): _freeze_metadata(item, path=f"{path}.{key}")
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_metadata(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_metadata(item, path=path) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True, slots=True)
class StructuredObservations:
    """The raw observations shared, by identity, by every expert route."""

    node_features: Tensor
    edge_features: Tensor

    def __post_init__(self) -> None:
        _require_tensor("node_features", self.node_features)
        _require_tensor("edge_features", self.edge_features)
        if self.node_features.ndim not in (2, 3):
            raise ValueError("node_features must have shape [V, D] or [B, V, D]")
        if self.edge_features.ndim != self.node_features.ndim:
            raise ValueError("node_features and edge_features must have the same rank")
        if not self.node_features.is_floating_point():
            raise TypeError("node_features must use a floating dtype")
        if not self.edge_features.is_floating_point():
            raise TypeError("edge_features must use a floating dtype")
        if self.node_features.device != self.edge_features.device:
            raise ValueError("node_features and edge_features must share a device")
        if not torch.isfinite(self.node_features).all() or not torch.isfinite(
            self.edge_features
        ).all():
            raise ValueError("observation features must contain only finite values")

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> StructuredObservations:
        return StructuredObservations(
            node_features=self.node_features.to(device=device, non_blocking=non_blocking),
            edge_features=self.edge_features.to(device=device, non_blocking=non_blocking),
        )


def _columns_as_tuples(index: Tensor) -> list[tuple[int, ...]]:
    return [tuple(int(item) for item in column) for column in index.t().tolist()]


def _validate_canonical_edges(edge_index: Tensor, num_vertices: int) -> None:
    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must have dtype torch.long")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    if edge_index.numel() == 0:
        return
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= num_vertices:
        raise ValueError("edge_index contains a vertex outside the sample")
    edges = _columns_as_tuples(edge_index)
    if any(u >= v for u, v in edges):
        raise ValueError("canonical undirected edges must satisfy u < v")
    if edges != sorted(set(edges)):
        raise ValueError("edge_index columns must be unique and lexicographically sorted")


def _validate_canonical_faces(face_index: Tensor, edge_index: Tensor) -> None:
    if face_index.dtype != torch.long:
        raise TypeError("face_index must have dtype torch.long")
    if face_index.ndim != 2 or face_index.shape[0] != 3:
        raise ValueError(f"face_index must have shape [3, F], got {tuple(face_index.shape)}")
    faces = _columns_as_tuples(face_index)
    if any(not (a < b < c) for a, b, c in faces):
        raise ValueError("canonical triangular faces must satisfy a < b < c")
    if faces != sorted(set(faces)):
        raise ValueError("face_index columns must be unique and lexicographically sorted")
    edges = set(_columns_as_tuples(edge_index))
    for a, b, c in faces:
        if not {(a, b), (a, c), (b, c)}.issubset(edges):
            raise ValueError(f"face {(a, b, c)} references an edge absent from edge_index")


_RESERVED_METADATA_KEYS = frozenset(
    {
        "label",
        "regime",
        "node_features",
        "edge_features",
        "observations",
        "face_active",
    }
)


@dataclass(frozen=True, slots=True)
class StructuredSample:
    """One graph with candidate faces, transports, and isolated supervision.

    ``transport[e]`` maps the rank-2 stalk vector at the canonical edge tail
    into the head frame. The corresponding connection residual is
    ``x_head - transport[e] @ x_tail``.
    """

    observations: StructuredObservations
    edge_index: Tensor
    face_index: Tensor
    face_active: Tensor
    transport: Tensor
    label: Tensor
    regime: SignalRegime
    sample_id: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.observations, StructuredObservations):
            raise TypeError("observations must be StructuredObservations")
        if self.observations.node_features.ndim != 2:
            raise ValueError("sample observations must be unbatched rank-2 tensors")

        num_vertices = int(self.observations.node_features.shape[0])
        num_edges = int(self.observations.edge_features.shape[0])
        if num_vertices <= 0:
            raise ValueError("a structured sample must contain at least one vertex")

        _validate_canonical_edges(self.edge_index, num_vertices)
        _validate_canonical_faces(self.face_index, self.edge_index)
        if self.edge_index.shape[1] != num_edges:
            raise ValueError("edge_features and edge_index disagree on the number of edges")

        _require_tensor("face_active", self.face_active, ndim=1)
        if self.face_active.dtype != torch.bool:
            raise TypeError("face_active must have dtype torch.bool")
        if self.face_active.shape[0] != self.face_index.shape[1]:
            raise ValueError("face_active and face_index disagree on the number of faces")

        _require_tensor("transport", self.transport, ndim=3)
        if tuple(self.transport.shape) != (num_edges, 2, 2):
            raise ValueError(
                f"transport must have shape [E, 2, 2], got {tuple(self.transport.shape)}"
            )
        if not self.transport.is_floating_point():
            raise TypeError("transport must use a floating dtype")
        if not torch.isfinite(self.transport).all():
            raise ValueError("transport must contain only finite values")

        _require_tensor("label", self.label)
        sample_device = self.node_features.device
        structural_tensors = (
            self.edge_index,
            self.face_index,
            self.face_active,
            self.transport,
            self.label,
        )
        if any(tensor.device != sample_device for tensor in structural_tensors):
            raise ValueError("all sample tensors must share the observation device")
        if self.label.ndim != 0 or self.label.dtype != torch.long:
            raise TypeError("label must be a scalar torch.long tensor")
        if int(self.label) not in (0, 1):
            raise ValueError("label must be binary")

        object.__setattr__(self, "regime", SignalRegime.coerce(self.regime))
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string")

        metadata_keys = {str(key) for key in self.metadata}
        overlap = metadata_keys & _RESERVED_METADATA_KEYS
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"metadata contains reserved supervision/observation keys: {names}")
        object.__setattr__(self, "metadata", _freeze_metadata(dict(self.metadata)))

    @property
    def node_features(self) -> Tensor:
        return self.observations.node_features

    @property
    def edge_features(self) -> Tensor:
        return self.observations.edge_features

    @property
    def num_vertices(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_faces(self) -> int:
        return int(self.face_index.shape[1])

    def observations_for(self, route: SignalRegime | str) -> StructuredObservations:
        """Return the same raw object for every legal route."""

        SignalRegime.coerce(route)
        return self.observations

    def model_inputs(self, route: SignalRegime | str) -> dict[str, Tensor]:
        """Return a typed route view without targets or privileged structure."""

        selected = SignalRegime.coerce(route)
        inputs = {
            "node_features": self.node_features,
            "edge_features": self.edge_features,
            "edge_index": self.edge_index,
        }
        if selected is SignalRegime.CELL:
            inputs.update(
                {
                    "face_index": self.face_index,
                    "face_active": self.face_active,
                }
            )
        elif selected is SignalRegime.SHEAF:
            inputs.update(
                {
                    "transport": self.transport,
                    "face_index": self.face_index,
                }
            )
        return inputs

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> StructuredSample:
        return StructuredSample(
            observations=self.observations.to(device, non_blocking=non_blocking),
            edge_index=self.edge_index.to(device=device, non_blocking=non_blocking),
            face_index=self.face_index.to(device=device, non_blocking=non_blocking),
            face_active=self.face_active.to(device=device, non_blocking=non_blocking),
            transport=self.transport.to(device=device, non_blocking=non_blocking),
            label=self.label.to(device=device, non_blocking=non_blocking),
            regime=self.regime,
            sample_id=self.sample_id,
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class StructuredBatch:
    """Padded structured samples with validity and activity masks separated."""

    observations: StructuredObservations
    node_mask: Tensor
    edge_index: Tensor
    edge_mask: Tensor
    face_index: Tensor
    face_mask: Tensor
    face_active: Tensor
    transport: Tensor
    labels: Tensor
    regimes: tuple[SignalRegime, ...]
    sample_ids: tuple[str, ...]
    metadata: tuple[Mapping[str, Any], ...]
    num_vertices: Tensor
    num_edges: Tensor
    num_faces: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.observations, StructuredObservations):
            raise TypeError("observations must be StructuredObservations")
        batch_size = int(self.observations.node_features.shape[0])
        if self.observations.node_features.ndim != 3:
            raise ValueError("batched node_features must have shape [B, V, D]")
        if self.observations.edge_features.ndim != 3:
            raise ValueError("batched edge_features must have shape [B, E, D]")
        expected_lengths = (len(self.regimes), len(self.sample_ids), len(self.metadata))
        if any(length != batch_size for length in expected_lengths):
            raise ValueError("batch metadata lengths must equal batch size")
        if tuple(self.node_mask.shape) != tuple(self.observations.node_features.shape[:2]):
            raise ValueError("node_mask shape must match padded nodes")
        if tuple(self.edge_mask.shape) != tuple(self.observations.edge_features.shape[:2]):
            raise ValueError("edge_mask shape must match padded edges")
        if tuple(self.edge_index.shape) != (batch_size, 2, self.edge_mask.shape[1]):
            raise ValueError("edge_index must have shape [B, 2, E]")
        if tuple(self.face_index.shape) != (batch_size, 3, self.face_mask.shape[1]):
            raise ValueError("face_index must have shape [B, 3, F]")
        if self.face_active.shape != self.face_mask.shape:
            raise ValueError("face_active and face_mask must have identical shapes")
        if tuple(self.transport.shape) != (batch_size, self.edge_mask.shape[1], 2, 2):
            raise ValueError("transport must have shape [B, E, 2, 2]")
        if tuple(self.labels.shape) != (batch_size,) or self.labels.dtype != torch.long:
            raise TypeError("labels must have shape [B] and dtype torch.long")
        for name, mask in (
            ("node_mask", self.node_mask),
            ("edge_mask", self.edge_mask),
            ("face_mask", self.face_mask),
            ("face_active", self.face_active),
        ):
            if mask.dtype != torch.bool:
                raise TypeError(f"{name} must have dtype torch.bool")
        if torch.any(self.face_active & ~self.face_mask):
            raise ValueError("padded faces cannot be active")

        batch_tensors = (
            self.node_mask,
            self.edge_index,
            self.edge_mask,
            self.face_index,
            self.face_mask,
            self.face_active,
            self.transport,
            self.labels,
            self.num_vertices,
            self.num_edges,
            self.num_faces,
        )
        device = self.node_features.device
        if any(tensor.device != device for tensor in batch_tensors):
            raise ValueError("all batch tensors must share the observation device")

        for name, counts, mask in (
            ("num_vertices", self.num_vertices, self.node_mask),
            ("num_edges", self.num_edges, self.edge_mask),
            ("num_faces", self.num_faces, self.face_mask),
        ):
            if tuple(counts.shape) != (batch_size,) or counts.dtype != torch.long:
                raise TypeError(f"{name} must have shape [B] and dtype torch.long")
            if not torch.equal(counts, mask.sum(dim=1, dtype=torch.long)):
                raise ValueError(f"{name} must equal the corresponding mask count")

        expected_node_mask = (
            torch.arange(self.node_mask.shape[1], device=device).unsqueeze(0)
            < self.num_vertices.unsqueeze(1)
        )
        expected_edge_mask = (
            torch.arange(self.edge_mask.shape[1], device=device).unsqueeze(0)
            < self.num_edges.unsqueeze(1)
        )
        expected_face_mask = (
            torch.arange(self.face_mask.shape[1], device=device).unsqueeze(0)
            < self.num_faces.unsqueeze(1)
        )
        if not torch.equal(self.node_mask, expected_node_mask):
            raise ValueError("node_mask must be a contiguous valid prefix")
        if not torch.equal(self.edge_mask, expected_edge_mask):
            raise ValueError("edge_mask must be a contiguous valid prefix")
        if not torch.equal(self.face_mask, expected_face_mask):
            raise ValueError("face_mask must be a contiguous valid prefix")

        for index in range(batch_size):
            num_vertices = int(self.num_vertices[index])
            num_edges = int(self.num_edges[index])
            num_faces = int(self.num_faces[index])
            valid_edges = self.edge_index[index, :, :num_edges]
            valid_faces = self.face_index[index, :, :num_faces]
            if valid_edges.numel() and (
                torch.any(valid_edges < 0) or torch.any(valid_edges >= num_vertices)
            ):
                raise ValueError("valid edge indices must reference real vertices")
            if valid_faces.numel() and (
                torch.any(valid_faces < 0) or torch.any(valid_faces >= num_vertices)
            ):
                raise ValueError("valid face indices must reference real vertices")
            if torch.any(self.edge_index[index, :, num_edges:] != -1):
                raise ValueError("padded edge indices must use the -1 sentinel")
            if torch.any(self.face_index[index, :, num_faces:] != -1):
                raise ValueError("padded face indices must use the -1 sentinel")
            if torch.count_nonzero(self.node_features[index, num_vertices:]):
                raise ValueError("padded node features must be zero")
            if torch.count_nonzero(self.edge_features[index, num_edges:]):
                raise ValueError("padded edge features must be zero")
            if torch.count_nonzero(self.transport[index, num_edges:]):
                raise ValueError("padded transports must be zero")

        object.__setattr__(self, "regimes", tuple(SignalRegime.coerce(r) for r in self.regimes))
        object.__setattr__(self, "sample_ids", tuple(self.sample_ids))
        object.__setattr__(self, "metadata", tuple(self.metadata))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    @property
    def node_features(self) -> Tensor:
        return self.observations.node_features

    @property
    def edge_features(self) -> Tensor:
        return self.observations.edge_features

    def observations_for(self, route: SignalRegime | str) -> StructuredObservations:
        SignalRegime.coerce(route)
        return self.observations

    def model_inputs(self, route: SignalRegime | str) -> dict[str, Tensor]:
        """Return a typed padded route view without supervision metadata."""

        selected = SignalRegime.coerce(route)
        inputs = {
            "node_features": self.node_features,
            "edge_features": self.edge_features,
            "node_mask": self.node_mask,
            "edge_index": self.edge_index,
            "edge_mask": self.edge_mask,
        }
        if selected is SignalRegime.CELL:
            inputs.update(
                {
                    "face_index": self.face_index,
                    "face_mask": self.face_mask,
                    "face_active": self.face_active,
                }
            )
        elif selected is SignalRegime.SHEAF:
            inputs.update(
                {
                    "transport": self.transport,
                    "face_index": self.face_index,
                    "face_mask": self.face_mask,
                }
            )
        return inputs

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> StructuredBatch:
        return StructuredBatch(
            observations=self.observations.to(device, non_blocking=non_blocking),
            node_mask=self.node_mask.to(device=device, non_blocking=non_blocking),
            edge_index=self.edge_index.to(device=device, non_blocking=non_blocking),
            edge_mask=self.edge_mask.to(device=device, non_blocking=non_blocking),
            face_index=self.face_index.to(device=device, non_blocking=non_blocking),
            face_mask=self.face_mask.to(device=device, non_blocking=non_blocking),
            face_active=self.face_active.to(device=device, non_blocking=non_blocking),
            transport=self.transport.to(device=device, non_blocking=non_blocking),
            labels=self.labels.to(device=device, non_blocking=non_blocking),
            regimes=self.regimes,
            sample_ids=self.sample_ids,
            metadata=self.metadata,
            num_vertices=self.num_vertices.to(device=device, non_blocking=non_blocking),
            num_edges=self.num_edges.to(device=device, non_blocking=non_blocking),
            num_faces=self.num_faces.to(device=device, non_blocking=non_blocking),
        )


# A readable alias for callers that prefer the dataset-oriented name.
StructuredData = StructuredSample


def count_values(values: Iterable[Any]) -> dict[Any, int]:
    """Count values without depending on collection-specific utilities."""

    result: dict[Any, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


__all__ = [
    "SignalRegime",
    "StructuredBatch",
    "StructuredData",
    "StructuredObservations",
    "StructuredSample",
]
