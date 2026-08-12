from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from homymoly.data.confirmatory import ConfirmatoryStructuredSignal
from homymoly.training.baselines import (
    PermutationInvariantBaseline,
    ShortcutBaselineConfig,
    run_shortcut_baselines,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _small_config() -> ShortcutBaselineConfig:
    return ShortcutBaselineConfig(
        num_samples=120,
        data_seed=123,
        split_seed=456,
        training_seed=789,
        min_vertices=24,
        max_vertices=32,
        epochs=8,
        batch_size=32,
        hidden_dim=12,
        learning_rate=5e-3,
        patience=4,
        num_threads=1,
    )


def test_shortcut_campaign_is_deterministic_group_disjoint_and_complete() -> None:
    first = run_shortcut_baselines(_small_config())
    second = run_shortcut_baselines(_small_config())
    assert first == second
    assert first["split"] == {
        "sizes": {"train": 84, "validation": 18, "test": 18},
        "group_counts": {"train": 14, "validation": 3, "test": 3},
        "group_disjoint": True,
    }
    assert set(first["baselines"]) == {
        "constant_majority",
        "scalar_amplitude",
        "pooled_mlp",
        "permutation_invariant_deepsets",
    }
    assert first["baselines"]["constant_majority"]["test_accuracy"] == 0.5
    assert first["relational_oracles"]["uses_hidden_regime"] is True
    assert first["relational_oracles"]["routed_test_accuracy"] > 0.90


def test_permutation_invariant_baseline_ignores_set_order() -> None:
    dataset = ConfirmatoryStructuredSignal(18, seed=91, num_vertices=24)
    sample = dataset[0]
    model = PermutationInvariantBaseline(
        node_dim=sample.node_features.shape[1],
        edge_dim=sample.edge_features.shape[1],
        structure_dim=4,
        hidden_dim=8,
    )
    model.eval()
    node = sample.node_features.unsqueeze(0)
    edge = sample.edge_features.unsqueeze(0)
    transport = sample.transport.unsqueeze(0)
    node_mask = torch.ones((1, sample.num_vertices), dtype=torch.bool)
    edge_mask = torch.ones((1, sample.num_edges), dtype=torch.bool)
    structure = torch.tensor(
        [[sample.num_vertices, sample.num_edges, sample.num_faces, 1.0]],
        dtype=torch.float32,
    ).log1p()
    node_order = torch.randperm(sample.num_vertices, generator=torch.Generator().manual_seed(1))
    edge_order = torch.randperm(sample.num_edges, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        reference = model(node, node_mask, edge, edge_mask, transport, structure)
        permuted = model(
            node[:, node_order],
            node_mask[:, node_order],
            edge[:, edge_order],
            edge_mask[:, edge_order],
            transport[:, edge_order],
            structure,
        )
    torch.testing.assert_close(reference, permuted, rtol=1e-6, atol=1e-6)


def test_moderate_campaign_keeps_shortcuts_below_relational_references() -> None:
    report = run_shortcut_baselines(
        ShortcutBaselineConfig(
            num_samples=300,
            data_seed=20260803,
            split_seed=404,
            training_seed=1701,
            min_vertices=24,
            max_vertices=48,
            epochs=20,
            batch_size=64,
            hidden_dim=24,
            patience=6,
            num_threads=1,
        )
    )
    baselines = report["baselines"]
    assert baselines["scalar_amplitude"]["route_accuracy"] < 0.80
    for name in (
        "constant_majority",
        "scalar_amplitude",
        "pooled_mlp",
        "permutation_invariant_deepsets",
    ):
        assert baselines[name]["test_accuracy"] < 0.70

    oracles = report["relational_oracles"]
    assert oracles["conditional_test_accuracy"]["graph"] > 0.90
    assert oracles["conditional_test_accuracy"]["cell"] > 0.95
    assert oracles["conditional_test_accuracy"]["sheaf"] > 0.95
    assert oracles["routed_test_accuracy"] > 0.95


def test_shortcut_baseline_cli_emits_json() -> None:
    process = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_shortcut_baselines.py"),
            "--samples",
            "60",
            "--max-vertices",
            "24",
            "--epochs",
            "2",
            "--hidden-dim",
            "8",
            "--patience",
            "1",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(process.stdout)
    assert report["benchmark_tier"] == "confirmatory"
    assert report["split"]["group_disjoint"] is True


def test_shortcut_config_rejects_incomplete_groups() -> None:
    with pytest.raises(ValueError, match="divisible by six"):
        ShortcutBaselineConfig(num_samples=100)
