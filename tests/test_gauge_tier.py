"""Tests for the gauge stalk mode of the confirmatory generator.

Gauge mode makes clean samples approximate global sections (the doc-03
pure-gauge sentinel) so consistency objectives have a zero noise floor; it
is the substrate for the Gate-3 revision experiments.  The tier trades
shortcut-hardness for structural interpretability: at low gauge noise the
defect edge's residual is visible per edge, which is intended — the
mechanism experiments need structural defects to be real and measurable.
"""

from __future__ import annotations

import torch

from homymoly.data import ConfirmatoryStructuredSignal, SignalRegime
from homymoly.data.confirmatory import ConfirmatoryConfig


def _gauge_dataset(noise: float = 0.0, samples: int = 60) -> ConfirmatoryStructuredSignal:
    config = ConfirmatoryConfig(
        num_samples=samples,
        seed=7,
        min_vertices=24,
        max_vertices=24,
        stalk_mode="gauge",
        gauge_noise_std=noise,
    )
    return ConfirmatoryStructuredSignal(config)


def _edge_residuals(sample) -> torch.Tensor:  # type: ignore[no-untyped-def]
    vectors = sample.observations.node_features[:, -2:].double()
    tails = vectors[sample.edge_index[0]]
    heads = vectors[sample.edge_index[1]]
    transported = torch.einsum("eij,ej->ei", sample.transport.double(), tails)
    return (heads - transported).norm(dim=1)


def test_zero_noise_clean_samples_are_exact_sections() -> None:
    dataset = _gauge_dataset(noise=0.0)
    for index in dataset.indices_for(regime=SignalRegime.SHEAF, label=0):
        assert _edge_residuals(dataset[index]).max() < 1e-5


def test_gauge_mode_preserves_the_holonomy_label_signal() -> None:
    dataset = _gauge_dataset(noise=0.0)
    defects: dict[int, list[float]] = {0: [], 1: []}
    for index in dataset.indices_for(regime=SignalRegime.SHEAF):
        sample = dataset[index]
        edges = [tuple(edge) for edge in sample.edge_index.t().tolist()]
        position = {edge: k for k, edge in enumerate(edges)}
        transports = sample.transport.double()
        worst = 0.0
        for face in sample.face_index.t().tolist():
            a, b, c = sorted(face)
            holonomy = (
                transports[position[(a, b)]]
                @ transports[position[(b, c)]]
                @ transports[position[(a, c)]].T
            )
            worst = max(worst, (holonomy - torch.eye(2, dtype=torch.float64)).norm().item())
        defects[int(sample.label)].append(worst)
    assert max(defects[0]) < 1e-5
    assert min(defects[1]) > 1.9


def test_gauge_noise_raises_the_residual_floor_deterministically() -> None:
    noisy = _gauge_dataset(noise=0.4)
    sample = noisy[noisy.indices_for(regime=SignalRegime.SHEAF, label=0)[0]]
    residuals = _edge_residuals(sample)
    assert residuals.max() > 0.05  # noise floor is present...
    again = _gauge_dataset(noise=0.4)[noisy.indices_for(regime=SignalRegime.SHEAF, label=0)[0]]
    assert torch.equal(residuals, _edge_residuals(again))  # ...and deterministic


def test_independent_mode_remains_the_default() -> None:
    config = ConfirmatoryConfig(num_samples=60, seed=11, min_vertices=24, max_vertices=24)
    assert config.stalk_mode == "independent"
    dataset = ConfirmatoryStructuredSignal(config)
    sample = dataset[dataset.indices_for(regime=SignalRegime.SHEAF, label=0)[0]]
    # Independent mode has no section property: residuals stay O(quality).
    assert _edge_residuals(sample).max() > 0.5


def test_gauge_mode_validation() -> None:
    for kwargs in ({"stalk_mode": "bogus"}, {"gauge_noise_std": -0.1}):
        try:
            ConfirmatoryConfig(num_samples=60, seed=1, min_vertices=24, max_vertices=24, **kwargs)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"config must reject {kwargs}")
