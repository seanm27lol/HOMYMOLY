from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

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


def test_campaign_fingerprint_tracks_declared_inputs(tmp_path: Path) -> None:
    (tmp_path / "src" / "homymoly").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    source = tmp_path / "src" / "homymoly" / "module.py"
    config = tmp_path / "configs" / "campaign.yaml"
    manifest = tmp_path / "configs" / "campaign.json"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    config.write_text("seed: 1\n", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "test-campaign",
                "execution_enabled": True,
                "max_attempts_per_step": 3,
                "fingerprint_inputs": ["configs/campaign.yaml"],
                "steps": [{"id": "placeholder"}],
            }
        ),
        encoding="utf-8",
    )
    first = MODULE._campaign_fingerprint(tmp_path, manifest)
    config.write_text("seed: 2\n", encoding="utf-8")
    assert MODULE._campaign_fingerprint(tmp_path, manifest) != first


def test_disabled_campaign_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    config = tmp_path / "configs" / "input.yaml"
    config.write_text("seed: 1\n", encoding="utf-8")
    manifest = tmp_path / "configs" / "campaign.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "provisional",
                "execution_enabled": False,
                "fingerprint_inputs": ["configs/input.yaml"],
                "steps": [{"id": "placeholder"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="execution is disabled"):
        MODULE._load_campaign_manifest(tmp_path, manifest)


def test_campaign_missing_execution_enabled_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "input.yaml").write_text("seed: 1\n", encoding="utf-8")
    manifest = tmp_path / "configs" / "campaign.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "missing-enable",
                "max_attempts_per_step": 3,
                "fingerprint_inputs": ["configs/input.yaml"],
                "steps": [{"id": "placeholder"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="must be explicit"):
        MODULE._load_campaign_manifest(tmp_path, manifest)


def test_launch_fingerprint_binds_policy_force_and_environment(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "configs").mkdir()
    config = tmp_path / "configs" / "input.yaml"
    config.write_text("seed: 1\n", encoding="utf-8")
    manifest = tmp_path / "configs" / "campaign.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "receipt-test",
                "execution_enabled": True,
                "max_attempts_per_step": 3,
                "fingerprint_inputs": ["configs/input.yaml"],
                "steps": [{"id": "placeholder"}],
            }
        ),
        encoding="utf-8",
    )
    environment = {"git": {"head": "a" * 40}, "python": "3.12", "gpu": "0"}
    monkeypatch.setattr(MODULE, "_environment_receipt", lambda *_a, **_k: environment)
    policy = dict(MODULE.DEFAULT_IDLE_POLICY)
    first = MODULE._campaign_launch_receipt(
        tmp_path, manifest, policy=policy, force=False
    )
    changed_policy = dict(policy)
    changed_policy["gpu_index"] = 1
    second = MODULE._campaign_launch_receipt(
        tmp_path, manifest, policy=changed_policy, force=False
    )
    forced = MODULE._campaign_launch_receipt(
        tmp_path, manifest, policy=policy, force=True
    )
    environment["git"] = {"head": "b" * 40}
    changed_revision = MODULE._campaign_launch_receipt(
        tmp_path, manifest, policy=policy, force=False
    )
    assert (
        len(
            {
                first["launch_fingerprint"],
                second["launch_fingerprint"],
                forced["launch_fingerprint"],
                changed_revision["launch_fingerprint"],
            }
        )
        == 4
    )


def test_git_attestation_rejects_dirty_worktree(monkeypatch, tmp_path: Path) -> None:
    class Result:
        returncode = 0
        stderr = ""

        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def clean_run(argv, **_kwargs):
        if "--show-toplevel" in argv:
            return Result(str(tmp_path) + "\n")
        if argv[-1] == "HEAD":
            return Result("a" * 40 + "\n")
        return Result("")

    monkeypatch.setattr(MODULE.shutil, "which", lambda _name: "/usr/bin/git")
    monkeypatch.setattr(MODULE.subprocess, "run", clean_run)
    monkeypatch.setattr(
        MODULE,
        "_absolute_file_identity",
        lambda *_a, **_k: {"path": "/usr/bin/git", "sha256": "c" * 64},
    )
    receipt = MODULE._git_attestation(tmp_path)
    assert receipt["head"] == "a" * 40
    assert receipt["status_porcelain"] == []

    def dirty_run(argv, **kwargs):
        if "status" in argv:
            return Result("?? untracked.txt\n")
        return clean_run(argv, **kwargs)

    monkeypatch.setattr(MODULE.subprocess, "run", dirty_run)
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        MODULE._git_attestation(tmp_path)


def test_aggregate_completion_revalidates_output_hash(tmp_path: Path) -> None:
    output = tmp_path / "artifacts" / "result.json"
    output.parent.mkdir()
    output.write_text('{"status":"completed","value":1}\n', encoding="utf-8")
    launch_receipt = {"launch_fingerprint": "a" * 64, "environment": {}}
    manifest = {
        "campaign_id": "receipt-test",
        "steps": [
            {
                "id": "one",
                "outputs": [
                    {
                        "path": "artifacts/result.json",
                        "minimum_bytes": 1,
                        "json_equals": {"status": "completed"},
                        "json_required_keys": ["value"],
                    }
                ],
            }
        ],
    }
    completion = tmp_path / "completion.json"
    receipt = MODULE._file_receipt(
        tmp_path, "artifacts/result.json", label="test", artifacts_only=True
    )
    completion.write_text(
        json.dumps(
            {
                "status": "completed",
                "campaign_id": "receipt-test",
                "launch_fingerprint": "a" * 64,
                "launch_receipt": launch_receipt,
                "completed_steps": ["one"],
                "outputs": [receipt],
            }
        ),
        encoding="utf-8",
    )
    assert MODULE._campaign_completion_matches(
        completion,
        project_root=tmp_path,
        manifest=manifest,
        launch_receipt=launch_receipt,
    )
    output.write_text('{"status":"completed","value":2}\n', encoding="utf-8")
    assert not MODULE._campaign_completion_matches(
        completion,
        project_root=tmp_path,
        manifest=manifest,
        launch_receipt=launch_receipt,
    )


def test_failure_latch_reset_archives_evidence_and_advances_epoch(
    tmp_path: Path,
) -> None:
    launch_receipt = {
        "launch_fingerprint": "a" * 64,
        "campaign_id": "retry-test",
        "environment": {},
    }
    paths = MODULE._campaign_state_paths(tmp_path, "retry-test", "a" * 64)
    paths["run"].mkdir(parents=True)
    MODULE._atomic_json(
        paths["failure_latch"],
        {
            "status": "latched_failure",
            "launch_fingerprint": "a" * 64,
            "launch_receipt": launch_receipt,
            "step_id": "failed-step",
        },
    )
    assert MODULE._reset_failure_latch(paths, launch_receipt)
    assert not paths["failure_latch"].exists()
    assert MODULE._retry_epoch(paths["retry_epoch"], launch_receipt) == 1
    archived = list((paths["run"] / "failure-latch-history").glob("*.json"))
    assert len(archived) == 1
    assert json.loads(archived[0].read_text(encoding="utf-8"))["step_id"] == (
        "failed-step"
    )


def test_campaign_state_is_isolated_by_id_and_fingerprint(tmp_path: Path) -> None:
    first = MODULE._campaign_state_paths(tmp_path, "campaign-a", "a" * 64)
    second = MODULE._campaign_state_paths(tmp_path, "campaign-a", "b" * 64)
    other = MODULE._campaign_state_paths(tmp_path, "campaign-b", "a" * 64)
    assert first["run"] != second["run"]
    assert first["run"] != other["run"]
    assert first["runner_lock"] == second["runner_lock"]


def test_completion_match_supports_legacy_and_campaign_markers(
    tmp_path: Path,
) -> None:
    fingerprint = "a" * 64
    marker = tmp_path / "complete"
    marker.write_text(fingerprint + "\n", encoding="utf-8")
    assert MODULE._completion_matches(marker, fingerprint)
    marker.write_text(
        json.dumps(
            {
                "status": "completed",
                "launch_fingerprint": fingerprint,
            }
        ),
        encoding="utf-8",
    )
    assert MODULE._completion_matches(marker, fingerprint)
    assert not MODULE._completion_matches(marker, "b" * 64)


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
