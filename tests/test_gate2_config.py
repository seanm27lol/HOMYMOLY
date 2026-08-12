from __future__ import annotations

from pathlib import Path

import pytest

from homymoly.config import ConfigError
from homymoly.training.config import Gate2Config, load_gate2_config

ROOT = Path(__file__).parents[1]


def test_repository_gate2_config_loads() -> None:
    config = load_gate2_config(ROOT / "configs" / "gate2.yaml")
    assert isinstance(config, Gate2Config)
    assert config.schema_version == 2
    assert config.data.num_samples == 6144
    assert config.run_dir == ROOT / "artifacts" / "gate2"
    assert config.loss.route_costs == (1.0, 1.35, 1.60)


def test_gate2_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 2\ntraining:\n  mystery: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown training field"):
        load_gate2_config(path)


def test_gate2_config_rejects_overlapping_splits(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "schema_version: 2\ndata:\n  train_fraction: 0.8\n  validation_fraction: 0.3\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="sum to less than 1"):
        load_gate2_config(path)
