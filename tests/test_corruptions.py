"""Tests for the Gate-3 corruption transforms."""

from __future__ import annotations

import torch

from homymoly.data import ConfirmatoryStructuredSignal
from homymoly.data.corruptions import apply_corruption, corruption_kinds


def _sample():
    dataset = ConfirmatoryStructuredSignal(24, seed=5, num_vertices=24)
    return dataset[0]


def test_zero_sigma_returns_the_same_sample() -> None:
    sample = _sample()
    for kind in corruption_kinds():
        assert apply_corruption(sample, kind=kind, sigma=0.0, seed=1) is sample


def test_transport_rotation_preserves_orthogonality_and_structure() -> None:
    sample = _sample()
    corrupted = apply_corruption(sample, kind="transport_rotation", sigma=0.4, seed=1)
    identity = torch.eye(2, dtype=corrupted.transport.dtype)
    deviation = (
        corrupted.transport.double().mT @ corrupted.transport.double() - identity
    ).abs().max()
    # The product of rotations is a rotation; only fp32 rounding is expected.
    assert deviation < 1e-5
    assert torch.equal(corrupted.edge_index, sample.edge_index)
    assert torch.equal(corrupted.face_index, sample.face_index)
    assert torch.equal(corrupted.label, sample.label)
    assert not torch.equal(corrupted.transport, sample.transport)


def test_edge_and_node_corruptions_touch_only_the_declared_channel() -> None:
    sample = _sample()
    edge_corrupted = apply_corruption(sample, kind="edge_cochain_noise", sigma=0.5, seed=1)
    edge_delta = edge_corrupted.observations.edge_features - sample.observations.edge_features
    assert torch.all(edge_delta[:, 0] == 0)
    assert torch.any(edge_delta[:, 1] != 0)

    node_corrupted = apply_corruption(sample, kind="node_anchor_noise", sigma=0.5, seed=1)
    node_delta = node_corrupted.observations.node_features - sample.observations.node_features
    assert torch.all(node_delta[:, 1:] == 0)
    assert torch.any(node_delta[:, 0] != 0)


def test_corruptions_are_deterministic_by_seed_and_sample() -> None:
    sample = _sample()
    first = apply_corruption(sample, kind="transport_rotation", sigma=0.3, seed=11)
    second = apply_corruption(sample, kind="transport_rotation", sigma=0.3, seed=11)
    third = apply_corruption(sample, kind="transport_rotation", sigma=0.3, seed=12)
    assert torch.equal(first.transport, second.transport)
    assert not torch.equal(first.transport, third.transport)


def test_unknown_kind_and_negative_sigma_rejected() -> None:
    sample = _sample()
    try:
        apply_corruption(sample, kind="bogus", sigma=0.1, seed=1)  # type: ignore[arg-type]
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("unknown kind must be rejected")
    try:
        apply_corruption(sample, kind="transport_rotation", sigma=-0.1, seed=1)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("negative sigma must be rejected")
