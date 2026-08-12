from __future__ import annotations

import json
from pathlib import Path

import torch

from homymoly.training.io import (
    MetricLogger,
    atomic_json,
    atomic_torch_save,
    load_checkpoint,
)


def test_atomic_serialization_round_trip(tmp_path: Path) -> None:
    json_path = tmp_path / "nested" / "state.json"
    checkpoint_path = tmp_path / "nested" / "state.pt"
    atomic_json(json_path, {"status": "ok"})
    atomic_torch_save(checkpoint_path, {"schema_version": 1, "value": torch.tensor(3)})
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"status": "ok"}
    assert int(load_checkpoint(checkpoint_path)["value"]) == 3
    assert list(json_path.parent.glob("*.tmp")) == []


def test_metric_logger_writes_jsonl(tmp_path: Path) -> None:
    with MetricLogger(tmp_path) as logger:
        logger.log({"train/loss": torch.tensor(0.25), "phase": "fixed"}, step=7)
    payload = json.loads(
        (tmp_path / "metrics" / "history.jsonl").read_text(encoding="utf-8")
    )
    assert payload == {"phase": "fixed", "step": 7, "train/loss": 0.25}
