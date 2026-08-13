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
    process = MODULE.ComputeProcess(17, "trainer", 256)
    monkeypatch.setattr(MODULE, "_compute_processes", lambda _index: (process,))
    idle, utilization, processes = MODULE._idle_samples(
        gpu_index=0,
        max_utilization=10,
        samples=3,
        interval_seconds=0,
    )
    assert not idle
    assert utilization == []
    assert processes == (process,)


def test_idle_samples_allows_one_bounded_background_context(monkeypatch) -> None:
    process = MODULE.ComputeProcess(17, "ui_server.py", 462)
    monkeypatch.setattr(MODULE, "_compute_processes", lambda _index: (process,))
    monkeypatch.setattr(MODULE, "_gpu_utilization", lambda _index: 4)
    idle, utilization, processes = MODULE._idle_samples(
        gpu_index=0,
        max_utilization=10,
        samples=3,
        interval_seconds=0,
        max_background_processes=1,
        max_background_memory_mib=512,
    )
    assert idle
    assert utilization == [4, 4, 4]
    assert processes == (process,)


def test_background_policy_rejects_unknown_or_aggregate_excess_memory() -> None:
    unknown = (MODULE.ComputeProcess(17, "unknown", None),)
    assert MODULE._processes_block_training(
        unknown, max_background_processes=1, max_background_memory_mib=512
    )
    two_contexts = (
        MODULE.ComputeProcess(17, "first", 300),
        MODULE.ComputeProcess(18, "second", 300),
    )
    assert MODULE._processes_block_training(
        two_contexts, max_background_processes=2, max_background_memory_mib=512
    )


def test_compute_process_parser_includes_memory(monkeypatch) -> None:
    monkeypatch.setattr(
        MODULE,
        "_run_nvidia_smi",
        lambda _arguments: "17, ui_server.py, 462\n18, unknown, [N/A]\n",
    )
    assert MODULE._compute_processes(0) == (
        MODULE.ComputeProcess(17, "ui_server.py", 462),
        MODULE.ComputeProcess(18, "unknown", None),
    )


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
