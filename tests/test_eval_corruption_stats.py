from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch
from torch import nn

from scripts import eval_corruption as evaluation


def test_partial_spearman_is_pearson_correlation_of_rank_residuals() -> None:
    x = [1.0, 2.0, 6.0, 4.0, 5.0, 3.0]
    y = [2.0, 6.0, 3.0, 1.0, 5.0, 4.0]
    control = [3.0, 1.0, 5.0, 2.0, 6.0, 4.0]
    rank_x = evaluation._rankdata(x)
    rank_y = evaluation._rankdata(y)
    rank_control = evaluation._rankdata(control)
    r_xy = evaluation._pearson(rank_x, rank_y)
    r_xc = evaluation._pearson(rank_x, rank_control)
    r_yc = evaluation._pearson(rank_y, rank_control)
    expected = (r_xy - r_xc * r_yc) / (((1.0 - r_xc**2) * (1.0 - r_yc**2)) ** 0.5)
    assert evaluation._partial_spearman(x, y, control) == pytest.approx(expected)


def test_sha256_corruption_draws_ignore_python_hash_salt() -> None:
    code = (
        "import json; from scripts.eval_corruption import _corruption_sigmas; "
        "print(json.dumps(_corruption_sigmas(['a','b'], data_seed=7, "
        "experiment_seed=11, kind='transport_rotation', severity=0.2, "
        "block_id=3, batch_start=192)))"
    )
    outputs = []
    for salt in ("1", "987654"):
        environment = {**os.environ, "PYTHONHASHSEED": salt}
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", code],
                text=True,
                env=environment,
            ).strip()
        )
    assert outputs[0] == outputs[1]
    seed, sigmas = json.loads(outputs[0])
    assert seed >= 0
    assert len(sigmas) == 2
    assert all(0.0 < value < 0.2 for value in sigmas)
    changed = evaluation._corruption_sigmas(
        ["a", "b"],
        data_seed=7,
        experiment_seed=11,
        kind="transport_rotation",
        severity=0.4,
        block_id=3,
        batch_start=192,
    )
    assert changed != (seed, sigmas)


def test_repeated_measure_inference_is_deterministic_and_block_adjusted() -> None:
    blocks = [f"block-{block}" for block in range(4) for _ in range(5)]
    severity = [value for _ in range(4) for value in (0.05, 0.1, 0.2, 0.4, 0.8)]
    reconstruction = [
        value + 0.08 * block
        for block in range(4)
        for value in (0.05, 0.1, 0.2, 0.4, 0.8)
    ]
    topology = [
        value * (1.0 + 0.1 * block) + (position % 2) * 0.01
        for block in range(4)
        for position, value in enumerate((0.05, 0.1, 0.2, 0.4, 0.8))
    ]
    damage = [
        value * (0.7 + 0.04 * block) + ((position + block) % 3) * 0.015
        for block in range(4)
        for position, value in enumerate((0.05, 0.1, 0.2, 0.4, 0.8))
    ]
    kwargs = {
        "seed": 1234,
        "bootstrap_replicates": 64,
        "permutation_replicates": 128,
    }
    first = evaluation._repeated_measure_inference(
        topology,
        damage,
        (reconstruction, severity),
        blocks,
        **kwargs,
    )
    second = evaluation._repeated_measure_inference(
        topology,
        damage,
        (reconstruction, severity),
        blocks,
        **kwargs,
    )
    assert first == second
    assert first["estimate"] == pytest.approx(
        evaluation._partial_spearman(
            topology,
            damage,
            reconstruction,
            severity,
            blocks=blocks,
        )
    )
    assert len(first["block_bootstrap_95_ci"]) == 2
    assert 0.0 <= first["within_block_permutation_pvalue_two_sided"] <= 1.0


class _TinySheaf(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transport_angle = nn.Linear(2, 1)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.graph_to_sheaf = _TinySheaf()
        self.other = nn.Linear(2, 1)


def test_checkpoint_compatibility_allows_only_legacy_transport_angle_gap() -> None:
    model = _TinyModel()
    complete = model.state_dict()
    legacy = {
        key: value
        for key, value in complete.items()
        if not key.startswith("graph_to_sheaf.transport_angle.")
    }
    evaluation._load_model_state_compatibly(model, legacy)

    missing_other = {key: value for key, value in legacy.items() if key != "other.bias"}
    with pytest.raises(RuntimeError, match="other.bias"):
        evaluation._load_model_state_compatibly(model, missing_other)

    unexpected = {**complete, "surprise.weight": torch.ones(1)}
    with pytest.raises(RuntimeError, match="surprise.weight"):
        evaluation._load_model_state_compatibly(model, unexpected)
