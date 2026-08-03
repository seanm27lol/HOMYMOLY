"""Graded structural corruptions for the Gate-3 corruption suite.

The plan's Gate-3 criterion requires held-out corruptions so that conversion
damage varies continuously and structural diagnostics can be tested for
predictive value beyond reconstruction error.  Each transform here preserves
the dataset's structural contract (validated :class:`StructuredSample`
invariants) while degrading exactly one signal channel:

* ``transport_rotation``: compose a random rotation ``R(theta_e)`` with
  ``theta_e ~ N(0, sigma)`` onto every transport.  Holonomy defects grow
  smoothly with sigma while per-edge marginals stay rotation-shaped;
* ``edge_cochain_noise``: add ``N(0, sigma)`` to the edge-feature channel
  that carries the cell circulation, diluting the probe-boundary signal;
* ``node_anchor_noise``: add ``N(0, sigma)`` to the node-feature channel
  that carries the graph anchor signs.

Severity is per sample so that damage and diagnostics correlate across a
batch rather than only across levels.  All transforms are deterministic by
``(seed, sample_id, kind)`` — regenerating the evaluation reproduces
corruptions bit-for-bit.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Literal

import torch
from torch import Tensor

from .types import StructuredSample

CorruptionKind = Literal["transport_rotation", "edge_cochain_noise", "node_anchor_noise"]

_KINDS: tuple[CorruptionKind, ...] = (
    "transport_rotation",
    "edge_cochain_noise",
    "node_anchor_noise",
)


def corruption_kinds() -> tuple[CorruptionKind, ...]:
    return _KINDS


def _sample_rng(seed: int, sample_id: str, kind: str) -> torch.Generator:
    key = f"{seed}:{sample_id}:{kind}".encode()
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return torch.Generator().manual_seed(value)


def _rotation_batch(angles: Tensor, *, dtype: torch.dtype) -> Tensor:
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    upper = torch.stack((cos, -sin), dim=-1)
    lower = torch.stack((sin, cos), dim=-1)
    return torch.stack((upper, lower), dim=-2).to(dtype)


def apply_corruption(
    sample: StructuredSample,
    *,
    kind: CorruptionKind,
    sigma: float,
    seed: int,
) -> StructuredSample:
    """Return a corrupted copy of ``sample``; the input is left untouched."""

    if kind not in _KINDS:
        raise ValueError(f"unknown corruption kind: {kind}")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    if sigma == 0:
        return sample
    rng = _sample_rng(seed, sample.sample_id, kind)

    if kind == "transport_rotation":
        angles = sigma * torch.randn(sample.transport.shape[0], generator=rng)
        noise = _rotation_batch(angles, dtype=sample.transport.dtype)
        transport = noise @ sample.transport
        # Corruptions must keep the transport a valid rotation: the product
        # of two rotations is a rotation, so no reprojection is needed.
        return replace(sample, transport=transport)

    if kind == "edge_cochain_noise":
        noise = sigma * torch.randn(
            sample.observations.edge_features.shape[0], generator=rng
        ).to(sample.observations.edge_features.dtype)
        edge_features = sample.observations.edge_features.clone()
        edge_features[:, 1] = edge_features[:, 1] + noise
        observations = replace(
            sample.observations, edge_features=edge_features
        )
        return replace(sample, observations=observations)

    noise = sigma * torch.randn(
        sample.observations.node_features.shape[0], generator=rng
    ).to(sample.observations.node_features.dtype)
    node_features = sample.observations.node_features.clone()
    node_features[:, 0] = node_features[:, 0] + noise
    observations = replace(sample.observations, node_features=node_features)
    return replace(sample, observations=observations)


__all__ = ["CorruptionKind", "apply_corruption", "corruption_kinds"]
