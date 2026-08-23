from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_gpu_campaign.py"
SPEC = importlib.util.spec_from_file_location("run_gpu_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "src" / "homymoly").mkdir(parents=True)
    (project / "configs").mkdir()
    (project / "artifacts").mkdir()
    producer = project / "scripts" / "produce.py"
    producer.write_text(
        """#!/usr/bin/env python3
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
counter = pathlib.Path(sys.argv[2])
output.parent.mkdir(parents=True, exist_ok=True)
count = int(counter.read_text() if counter.exists() else "0") + 1
counter.write_text(str(count))
output.write_text(json.dumps({"status": "completed", "count": count}))
print(f"produced {output}")
""",
        encoding="utf-8",
    )
    producer.chmod(0o755)
    manifest = project / "configs" / "campaign.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "test-campaign",
                "description": "one resumable test step",
                "execution_enabled": True,
                "max_attempts_per_step": 3,
                "fingerprint_inputs": ["scripts/produce.py"],
                "steps": [
                    {
                        "id": "produce",
                        "argv": [
                            "scripts/produce.py",
                            "artifacts/result.json",
                            "artifacts/count.txt",
                        ],
                        "inputs": ["scripts/produce.py"],
                        "outputs": [
                            {
                                "path": "artifacts/result.json",
                                "json_equals": {"status": "completed"},
                                "json_required_keys": ["count"],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return project, manifest


def _launch_receipt() -> dict[str, object]:
    return {
        "schema_version": 1,
        "campaign_id": "test-campaign",
        "campaign_fingerprint": "b" * 64,
        "effective_launch_policy": {"gpu_index": 0, "force": False},
        "environment": {"python": "test", "gpu": "test"},
        "launch_fingerprint": "a" * 64,
    }


def _prepare_launch(monkeypatch) -> tuple[str, dict[str, object]]:
    receipt = _launch_receipt()
    monkeypatch.setattr(MODULE, "_campaign_launch_receipt", lambda *_a, **_k: receipt)
    return str(receipt["launch_fingerprint"]), receipt


def _two_step_project(tmp_path: Path) -> tuple[Path, Path]:
    project, manifest = _project(tmp_path)
    consumer = project / "scripts" / "consume.py"
    consumer.write_text(
        """#!/usr/bin/env python3
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
counter = pathlib.Path(sys.argv[3])
count = int(counter.read_text() if counter.exists() else "0") + 1
counter.write_text(str(count))
payload = json.loads(source.read_text())
output.write_text(json.dumps({"status": "completed", "source_count": payload["count"]}))
""",
        encoding="utf-8",
    )
    consumer.chmod(0o755)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["description"] = "two dependent resumable steps"
    payload["fingerprint_inputs"].append("scripts/consume.py")
    payload["steps"].append(
        {
            "id": "consume",
            "argv": [
                "scripts/consume.py",
                "artifacts/result.json",
                "artifacts/final.json",
                "artifacts/consume-count.txt",
            ],
            "inputs": ["scripts/consume.py", "artifacts/result.json"],
            "outputs": [
                {
                    "path": "artifacts/final.json",
                    "json_equals": {"status": "completed"},
                    "json_required_keys": ["source_count"],
                }
            ],
        }
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return project, manifest


def test_campaign_runs_once_and_resumes_from_atomic_marker(
    tmp_path: Path, monkeypatch
) -> None:
    project, manifest = _project(tmp_path)
    fingerprint, receipt = _prepare_launch(monkeypatch)
    assert (
        MODULE.run_campaign(project, manifest, fingerprint, launch_receipt=receipt) == 0
    )
    assert (
        MODULE.run_campaign(project, manifest, fingerprint, launch_receipt=receipt) == 0
    )
    assert (project / "artifacts" / "count.txt").read_text() == "1"
    paths = MODULE._campaign_state_paths(project, "test-campaign", fingerprint)
    completion = json.loads(paths["complete"].read_text(encoding="utf-8"))
    assert completion["completed_steps"] == ["produce"]
    state = json.loads((paths["run"] / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    [log] = list((paths["run"] / "logs").glob("*.log"))
    assert "produced artifacts/result.json" in log.read_text(encoding="utf-8")
    assert list(log.parent.glob("*.tmp")) == []


def test_resume_marker_rejects_changed_output_content(
    tmp_path: Path, monkeypatch
) -> None:
    project, manifest = _project(tmp_path)
    fingerprint, receipt = _prepare_launch(monkeypatch)
    assert (
        MODULE.run_campaign(project, manifest, fingerprint, launch_receipt=receipt) == 0
    )
    result = project / "artifacts" / "result.json"
    result.write_text(
        json.dumps({"status": "completed", "count": 999}), encoding="utf-8"
    )
    assert (
        MODULE.run_campaign(project, manifest, fingerprint, launch_receipt=receipt) == 0
    )
    assert (project / "artifacts" / "count.txt").read_text() == "2"


def test_generated_input_change_invalidates_downstream_marker(
    tmp_path: Path, monkeypatch
) -> None:
    project, manifest = _two_step_project(tmp_path)
    fingerprint, receipt = _prepare_launch(monkeypatch)
    assert (
        MODULE.run_campaign(project, manifest, fingerprint, launch_receipt=receipt) == 0
    )
    result = project / "artifacts" / "result.json"
    result.write_text(
        json.dumps({"status": "completed", "count": 999}), encoding="utf-8"
    )
    assert (
        MODULE.run_campaign(project, manifest, fingerprint, launch_receipt=receipt) == 0
    )
    assert (project / "artifacts" / "count.txt").read_text() == "2"
    assert (project / "artifacts" / "consume-count.txt").read_text() == "2"
    paths = MODULE._campaign_state_paths(project, "test-campaign", fingerprint)
    markers = sorted((paths["run"] / "steps").glob("*.complete.json"))
    assert len(markers) == 2
    downstream = json.loads(markers[1].read_text(encoding="utf-8"))
    assert downstream["inputs"][1]["path"] == "artifacts/result.json"
    assert len(downstream["inputs"][1]["sha256"]) == 64


def test_campaign_rejects_python_code_strings(tmp_path: Path) -> None:
    project, manifest = _project(tmp_path)
    python = project / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(Path(os.sys.executable))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["steps"][0]["argv"] = [".venv/bin/python", "-c", "print('unsafe')"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="must name an in-repo script"):
        MODULE.validate_manifest(project, manifest)


def test_campaign_rejects_outputs_outside_artifacts(tmp_path: Path) -> None:
    project, manifest = _project(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["steps"][0]["outputs"][0]["path"] = "../outside.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unsafe path component"):
        MODULE.validate_manifest(project, manifest)


def test_campaign_failure_does_not_create_step_marker(
    tmp_path: Path, monkeypatch
) -> None:
    project, manifest = _project(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["steps"][0]["outputs"][0]["json_equals"] = {"status": "impossible"}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    fingerprint, receipt = _prepare_launch(monkeypatch)
    assert (
        MODULE.run_campaign(project, manifest, fingerprint, launch_receipt=receipt) == 1
    )
    paths = MODULE._campaign_state_paths(project, "test-campaign", fingerprint)
    state = json.loads((paths["run"] / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert list((paths["run"] / "steps").glob("*.complete.json")) == []


def test_failed_step_latches_after_bounded_attempts_and_retains_logs(
    tmp_path: Path, monkeypatch
) -> None:
    project, manifest = _project(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["max_attempts_per_step"] = 2
    payload["steps"][0]["outputs"][0]["json_equals"] = {"status": "impossible"}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    fingerprint, receipt = _prepare_launch(monkeypatch)
    assert (
        MODULE.run_campaign(project, manifest, fingerprint, launch_receipt=receipt) == 1
    )
    assert (
        MODULE.run_campaign(project, manifest, fingerprint, launch_receipt=receipt)
        == 78
    )
    paths = MODULE._campaign_state_paths(project, "test-campaign", fingerprint)
    logs = sorted((paths["run"] / "logs").glob("*.log"))
    attempts = sorted((paths["run"] / "attempts").glob("*.json"))
    assert len(logs) == len(attempts) == 2
    assert "attempt-001" in logs[0].name
    assert "attempt-002" in logs[1].name
    latch = json.loads(paths["failure_latch"].read_text(encoding="utf-8"))
    assert latch["status"] == "latched_failure"
    assert latch["attempts"] == 2
    assert (
        MODULE.run_campaign(project, manifest, fingerprint, launch_receipt=receipt)
        == 78
    )
    assert len(list((paths["run"] / "logs").glob("*.log"))) == 2


def test_environment_receipt_drift_stops_before_next_incomplete_step(
    tmp_path: Path, monkeypatch
) -> None:
    project, manifest = _two_step_project(tmp_path)
    receipt = _launch_receipt()
    changed = json.loads(json.dumps(receipt))
    changed["environment"]["python"] = "changed"
    changed["launch_fingerprint"] = "c" * 64
    observed = iter((receipt, changed))
    monkeypatch.setattr(
        MODULE, "_campaign_launch_receipt", lambda *_a, **_k: next(observed)
    )
    fingerprint = str(receipt["launch_fingerprint"])
    assert (
        MODULE.run_campaign(project, manifest, fingerprint, launch_receipt=receipt) == 2
    )
    assert (project / "artifacts" / "result.json").is_file()
    assert not (project / "artifacts" / "final.json").exists()
    paths = MODULE._campaign_state_paths(project, "test-campaign", fingerprint)
    state = json.loads((paths["run"] / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "stale_launch_receipt"


@pytest.mark.parametrize("unsafe_path", ("artifacts/.", "artifacts/link/result.json"))
def test_campaign_rejects_dot_or_symlink_outputs(
    tmp_path: Path, unsafe_path: str
) -> None:
    project, manifest = _project(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / "artifacts" / "link").symlink_to(outside, target_is_directory=True)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["steps"][0]["outputs"][0]["path"] = unsafe_path
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="canonical|symlink"):
        MODULE.validate_manifest(project, manifest)


def test_busy_gpu_pauses_before_step_and_next_run_resumes(
    tmp_path: Path, monkeypatch
) -> None:
    project, manifest = _project(tmp_path)
    fingerprint, receipt = _prepare_launch(monkeypatch)
    policy = MODULE.IdlePolicy(0, 10, 1, 512, 3, 0.0)
    monkeypatch.setattr(
        MODULE,
        "_idle_before_step",
        lambda _policy: (
            False,
            {
                "reason": "gpu_busy",
                "gpu_index": 0,
                "utilizations": [20, 18, 21],
                "processes": [],
            },
        ),
    )
    assert (
        MODULE.run_campaign(
            project,
            manifest,
            fingerprint,
            idle_policy=policy,
            launch_receipt=receipt,
        )
        == 75
    )
    assert not (project / "artifacts" / "result.json").exists()
    paths = MODULE._campaign_state_paths(project, "test-campaign", fingerprint)
    state = json.loads((paths["run"] / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "paused_gpu_busy"
    assert (
        MODULE.run_campaign(project, manifest, fingerprint, launch_receipt=receipt) == 0
    )
    assert (project / "artifacts" / "result.json").is_file()


def test_frozen_idle_policy_allows_measured_resident_contexts(monkeypatch) -> None:
    processes = (
        SimpleNamespace(pid=10, name="ollama", used_memory_mib=44_614),
        SimpleNamespace(pid=11, name="ui", used_memory_mib=472),
    )
    monkeypatch.setattr(
        MODULE,
        "_idle_samples",
        lambda **_kwargs: (True, [4, 3, 4], processes),
    )
    monkeypatch.setattr(MODULE, "_compute_processes", lambda _index: processes)
    monkeypatch.setattr(MODULE, "_gpu_utilization", lambda _index: 4)
    idle, details = MODULE._idle_before_step(
        MODULE.IdlePolicy(0, 10, 2, 49_152, 3, 2.0)
    )
    assert idle
    assert details["reason"] == "gpu_idle"
    assert sum(process.used_memory_mib for process in processes) == 45_086


def test_provisional_gb10_manifest_has_complete_paired_matrix() -> None:
    repository = Path(__file__).resolve().parents[1]
    path = repository / "configs" / "identifiable-maps" / "gb10-campaign.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["execution_enabled"] is False
    assert payload["max_attempts_per_step"] == 3
    assert payload["idle_policy"] == {
        "gpu_index": 0,
        "max_utilization": 10,
        "max_background_processes": 2,
        "max_background_memory_mib": 49_152,
        "samples": 3,
        "interval_seconds": 2.0,
    }
    steps = payload["steps"]
    training = [step for step in steps if step["id"].startswith("train-")]
    identifiable_benchmarks = [
        step for step in steps if step["id"].startswith("benchmark-identifiable-")
    ]
    routing_benchmarks = [
        step for step in steps if step["id"].startswith("benchmark-routing-")
    ]
    seeds = {str(value) for value in range(20260821, 20260826)}
    modes = {
        "task_only",
        "reconstruction_only",
        "task_reconstruction",
        "task_reconstruction_cone",
        "task_reconstruction_rtd",
        "cone_only",
        "rtd_only",
        "combined",
    }
    observed = {
        (
            step["argv"][step["argv"].index("--seed") + 1],
            step["argv"][step["argv"].index("--ablation") + 1],
        )
        for step in training
    }
    assert len(training) == 40
    assert observed == {(seed, mode) for seed in seeds for mode in modes}
    assert len(identifiable_benchmarks) == 10
    assert len(routing_benchmarks) == 5
    assert len(steps) == len({step["id"] for step in steps}) == 56
    summarizer = next(
        step for step in steps if step["id"] == "summarize-identifiable-campaign"
    )
    assert steps.index(summarizer) == 40
    assert all(
        steps.index(summarizer) < steps.index(step) for step in identifiable_benchmarks
    )
    immutable = set(payload["fingerprint_inputs"])
    assert {
        "scripts/summarize_identifiable_campaign.py",
        "docs/21-identifiable-typed-map-protocol.md",
        "src/homymoly/experiments/identifiable_maps.py",
    } <= immutable
    expected_training_files = {
        "effective_config.yaml",
        "provenance.json",
        "checkpoint.pt",
        "history.json",
        "test_predictions.jsonl",
        "summary.json",
        "manifest.json",
    }
    training_outputs = {
        output["path"] for step in training for output in step["outputs"]
    }
    for step in training:
        assert {Path(output["path"]).name for output in step["outputs"]} == (
            expected_training_files
        )
        assert step["inputs"]
    assert training_outputs <= set(summarizer["inputs"])
    assert (
        hashlib.sha256(
            (
                repository / "configs" / "identifiable-maps" / "gb10-full.yaml"
            ).read_bytes()
        ).hexdigest()
        == "22abb205e8a89586b38799d7f7b8d53f0c24cef45f872453533ddf34e20fad73"
    )
