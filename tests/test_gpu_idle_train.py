from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "gpu_idle_train.py"
SPEC = importlib.util.spec_from_file_location("gpu_idle_train", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_active_training_rejects_stale_pid(tmp_path: Path) -> None:
    pid_file = tmp_path / "trainer.json"
    pid_file.write_text(json.dumps({"pid": 999_999_999}), encoding="utf-8")
    assert MODULE._active_training(pid_file) == (False, 999_999_999)


def test_atomic_json_replaces_document(tmp_path: Path) -> None:
    destination = tmp_path / "state" / "trainer.json"
    MODULE._atomic_json(destination, {"pid": 42})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"pid": 42}
    assert list(destination.parent.glob("*.tmp")) == []


def test_launch_fingerprint_changes_with_config(tmp_path: Path) -> None:
    (tmp_path / "src" / "homymoly").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    source = tmp_path / "src" / "homymoly" / "module.py"
    config = tmp_path / "config.yaml"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    config.write_text("seed: 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / "scripts" / "train_gate2.sh").write_text(
        "#!/bin/sh\n", encoding="utf-8"
    )
    first = MODULE._launch_fingerprint(tmp_path, config)
    config.write_text("seed: 2\n", encoding="utf-8")
    assert MODULE._launch_fingerprint(tmp_path, config) != first


def test_idle_samples_rejects_compute_process(monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "_compute_processes", lambda _index: ((17, "trainer"),))
    idle, utilization, processes = MODULE._idle_samples(
        gpu_index=0,
        max_utilization=10,
        samples=3,
        interval_seconds=0,
    )
    assert not idle
    assert utilization == []
    assert processes == ((17, "trainer"),)


def test_idle_samples_requires_every_sample_below_threshold(monkeypatch) -> None:
    readings = iter((2, 11, 3))
    monkeypatch.setattr(MODULE, "_compute_processes", lambda _index: ())
    monkeypatch.setattr(MODULE, "_gpu_utilization", lambda _index: next(readings))
    idle, utilization, processes = MODULE._idle_samples(
        gpu_index=0,
        max_utilization=10,
        samples=3,
        interval_seconds=0,
    )
    assert not idle
    assert utilization == [2, 11, 3]
    assert processes == ()
