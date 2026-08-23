#!/usr/bin/env python3
"""Run a validated, resumable sequence of in-repository GPU commands."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from gpu_idle_train import (
    CAMPAIGN_ID_PATTERN,
    _campaign_launch_receipt,
    _campaign_state_paths,
    _compute_processes,
    _failure_latch_matches,
    _file_receipt,
    _gpu_utilization,
    _idle_samples,
    _load_campaign_manifest,
    _manifest_idle_policy,
    _processes_block_training,
    _resolved_in_root,
    _retry_epoch,
    _safe_relative_path,
)

FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class IdlePolicy(NamedTuple):
    """Conservative physical-GPU policy applied before each incomplete step."""

    gpu_index: int
    max_utilization: int
    max_background_processes: int
    max_background_memory_mib: int
    samples: int
    interval_seconds: float


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_event(path: Path, event: str, **fields: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = json.dumps(
        {"created_at": _timestamp(), "event": event, **fields}, sort_keys=True
    )
    descriptor = os.open(
        path,
        os.O_APPEND
        | os.O_CREAT
        | os.O_WRONLY
        | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError(f"campaign event log is not a regular file: {path}")
        os.write(descriptor, (record + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _clean_argument(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty string")
    if any(ord(character) < 32 for character in value):
        raise RuntimeError(f"{label} contains a control character")
    return value


def _validate_output(
    raw: object, project_root: Path, *, step_id: str, index: int
) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise TypeError(f"step {step_id} output {index} must be an object")
    allowed = {"path", "minimum_bytes", "json_equals", "json_required_keys"}
    unknown = set(raw) - allowed
    if unknown:
        raise RuntimeError(
            f"step {step_id} output {index} has unknown keys: {sorted(unknown)}"
        )
    path_text = _clean_argument(
        raw.get("path"), label=f"step {step_id} output {index} path"
    )
    path = Path(path_text)
    _safe_relative_path(
        project_root,
        path_text,
        label=f"step {step_id} output {index}",
        artifacts_only=True,
    )
    if path.parts == ("artifacts",):
        raise RuntimeError(f"step {step_id} output cannot be the artifacts directory")
    minimum_bytes = raw.get("minimum_bytes", 1)
    if (
        not isinstance(minimum_bytes, int)
        or isinstance(minimum_bytes, bool)
        or minimum_bytes <= 0
    ):
        raise RuntimeError(
            f"step {step_id} output minimum_bytes must be a positive integer"
        )
    equals = raw.get("json_equals", {})
    if not isinstance(equals, dict) or any(not isinstance(key, str) for key in equals):
        raise RuntimeError(f"step {step_id} output json_equals must be an object")
    required = raw.get("json_required_keys", [])
    if not isinstance(required, list) or any(
        not isinstance(key, str) or not key for key in required
    ):
        raise RuntimeError(
            f"step {step_id} output json_required_keys must be a string list"
        )
    return {
        "path": path_text,
        "minimum_bytes": minimum_bytes,
        "json_equals": equals,
        "json_required_keys": required,
    }


def _validate_step(raw: object, project_root: Path, *, index: int) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise TypeError(f"campaign step {index} must be an object")
    allowed = {"id", "argv", "inputs", "outputs"}
    unknown = set(raw) - allowed
    if unknown:
        raise RuntimeError(f"campaign step {index} has unknown keys: {sorted(unknown)}")
    step_id = _clean_argument(raw.get("id"), label=f"campaign step {index} id")
    if not CAMPAIGN_ID_PATTERN.fullmatch(step_id):
        raise RuntimeError(f"campaign step id is invalid: {step_id}")
    argv_raw = raw.get("argv")
    if not isinstance(argv_raw, list) or not argv_raw:
        raise RuntimeError(f"step {step_id} argv must be a non-empty list")
    argv = [
        _clean_argument(value, label=f"step {step_id} argv[{position}]")
        for position, value in enumerate(argv_raw)
    ]
    executable = Path(argv[0])
    if executable.is_absolute():
        raise RuntimeError(f"step {step_id} executable must be relative")
    executable_candidate = (project_root / executable).absolute()
    try:
        executable_candidate.relative_to(project_root.absolute())
    except ValueError as exc:
        raise RuntimeError(
            f"step {step_id} executable must be lexically inside the project"
        ) from exc
    if executable.as_posix() == ".venv/bin/python":
        # A conventional venv Python is normally a symlink to the system
        # interpreter. Only this exact, explicit external target is allowed.
        executable_path = executable_candidate
    else:
        executable_path = _resolved_in_root(
            executable_candidate,
            project_root,
            label=f"step {step_id} executable",
        )
    if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
        raise RuntimeError(
            f"step {step_id} executable is missing or not executable: {argv[0]}"
        )
    if executable.name.startswith("python"):
        if len(argv) < 2 or argv[1].startswith("-"):
            raise RuntimeError(
                f"step {step_id} Python command must name an in-repo script"
            )
        script = Path(argv[1])
        if script.is_absolute():
            raise RuntimeError(f"step {step_id} Python script must be relative")
        script_path = _resolved_in_root(
            project_root / script,
            project_root,
            label=f"step {step_id} Python script",
        )
        if not script_path.is_file() or script_path.suffix != ".py":
            raise RuntimeError(f"step {step_id} Python script is invalid: {argv[1]}")
    inputs_raw = raw.get("inputs")
    if not isinstance(inputs_raw, list) or not inputs_raw:
        raise RuntimeError(f"step {step_id} must declare at least one input")
    inputs = [
        _clean_argument(value, label=f"step {step_id} input[{position}]")
        for position, value in enumerate(inputs_raw)
    ]
    if len(inputs) != len(set(inputs)):
        raise RuntimeError(f"step {step_id} input paths must be unique")
    for position, input_path in enumerate(inputs):
        _safe_relative_path(
            project_root,
            input_path,
            label=f"step {step_id} input[{position}]",
            artifacts_only=False,
        )
    outputs_raw = raw.get("outputs")
    if not isinstance(outputs_raw, list) or not outputs_raw:
        raise RuntimeError(f"step {step_id} must declare at least one output")
    outputs = [
        _validate_output(value, project_root, step_id=step_id, index=position)
        for position, value in enumerate(outputs_raw)
    ]
    return {"id": step_id, "argv": argv, "inputs": inputs, "outputs": outputs}


def validate_manifest(
    project_root: Path, manifest_path: Path
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    """Validate all executable steps beyond the scheduler's base schema checks."""

    manifest = _load_campaign_manifest(project_root, manifest_path)
    allowed = {
        "schema_version",
        "campaign_id",
        "description",
        "execution_enabled",
        "idle_policy",
        "max_attempts_per_step",
        "fingerprint_inputs",
        "steps",
    }
    unknown = set(manifest) - allowed
    if unknown:
        raise RuntimeError(f"campaign manifest has unknown keys: {sorted(unknown)}")
    steps = [
        _validate_step(value, project_root, index=index)
        for index, value in enumerate(manifest["steps"])
    ]
    identifiers = [str(step["id"]) for step in steps]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("campaign step ids must be unique")
    raw_fingerprint_inputs = manifest["fingerprint_inputs"]
    assert isinstance(raw_fingerprint_inputs, list)
    fingerprint_inputs = {str(value) for value in raw_fingerprint_inputs}
    if len(fingerprint_inputs) != len(raw_fingerprint_inputs):
        raise RuntimeError("campaign fingerprint input paths must be unique")
    for index, input_path in enumerate(fingerprint_inputs):
        _safe_relative_path(
            project_root,
            input_path,
            label=f"fingerprint input {index}",
            artifacts_only=False,
        )
    all_outputs = {
        str(output["path"])
        for step in steps
        for output in step["outputs"]  # type: ignore[union-attr]
        if isinstance(output, dict)
    }
    if len(all_outputs) != sum(
        len(step["outputs"])
        for step in steps  # type: ignore[arg-type]
    ):
        raise RuntimeError("campaign output paths must be unique")
    overlap = fingerprint_inputs & all_outputs
    if overlap:
        raise RuntimeError(
            f"campaign inputs and outputs must not overlap: {sorted(overlap)}"
        )
    available_generated: set[str] = set()
    for step in steps:
        argv = step["argv"]
        inputs = step["inputs"]
        assert isinstance(argv, list)
        assert isinstance(inputs, list)
        unavailable = {str(value) for value in inputs} - (
            fingerprint_inputs | available_generated
        )
        if unavailable:
            raise RuntimeError(
                f"step {step['id']} inputs are neither fingerprinted nor produced "
                f"by an earlier step: {sorted(unavailable)}"
            )
        executable = str(argv[0])
        if executable != ".venv/bin/python" and executable not in inputs:
            raise RuntimeError(
                f"step {step['id']} must declare its executable as an input"
            )
        if executable == ".venv/bin/python" and str(argv[1]) not in inputs:
            raise RuntimeError(
                f"step {step['id']} must declare its Python script as an input"
            )
        for flag in ("--config", "--checkpoint", "--source-config"):
            if flag not in argv:
                continue
            if argv.count(flag) != 1:
                raise RuntimeError(f"step {step['id']} repeats {flag}")
            position = argv.index(flag)
            if position + 1 >= len(argv):
                raise RuntimeError(f"step {step['id']} has no value after {flag}")
            input_path = str(argv[position + 1])
            if input_path not in inputs:
                raise RuntimeError(
                    f"step {step['id']} must explicitly declare {flag} input: "
                    f"{input_path}"
                )
        if "--output" in argv:
            if argv.count("--output") != 1:
                raise RuntimeError(f"step {step['id']} repeats --output")
            position = argv.index("--output")
            if position + 1 >= len(argv):
                raise RuntimeError(f"step {step['id']} has no value after --output")
            command_output_text = str(argv[position + 1])
            command_output = Path(command_output_text)
            _safe_relative_path(
                project_root,
                command_output_text,
                label=f"step {step['id']} --output",
                artifacts_only=True,
            )
            if command_output.parts == ("artifacts",):
                raise RuntimeError(
                    f"step {step['id']} --output cannot be the artifacts directory"
                )
            declared = [
                Path(str(output["path"]))
                for output in step["outputs"]  # type: ignore[union-attr]
                if isinstance(output, dict)
            ]
            if not any(
                path == command_output or command_output in path.parents
                for path in declared
            ):
                raise RuntimeError(
                    f"step {step['id']} --output is not covered by its outputs"
                )
        available_generated.update(
            str(output["path"])
            for output in step["outputs"]  # type: ignore[union-attr]
            if isinstance(output, dict)
        )
    return manifest, steps


def _input_records(inputs: list[str], project_root: Path) -> list[dict[str, int | str]]:
    return [
        _file_receipt(
            project_root,
            path_text,
            label=f"campaign input {path_text}",
            artifacts_only=False,
        )
        for path_text in inputs
    ]


def _output_satisfies(specification: dict[str, object], project_root: Path) -> bool:
    path_text = str(specification["path"])
    try:
        receipt_before = _file_receipt(
            project_root,
            path_text,
            label=f"campaign output {path_text}",
            artifacts_only=True,
        )
    except (OSError, RuntimeError):
        return False
    if int(receipt_before["bytes"]) < int(specification["minimum_bytes"]):
        return False
    equals = specification["json_equals"]
    required = specification["json_required_keys"]
    if not equals and not required:
        return True
    try:
        payload = json.loads((project_root / path_text).read_text(encoding="utf-8"))
        receipt_after = _file_receipt(
            project_root,
            path_text,
            label=f"campaign output {path_text}",
            artifacts_only=True,
        )
    except (OSError, RuntimeError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if receipt_before != receipt_after or not isinstance(payload, dict):
        return False
    assert isinstance(equals, dict)
    assert isinstance(required, list)
    return all(payload.get(key) == value for key, value in equals.items()) and all(
        key in payload for key in required
    )


def _output_records(
    outputs: list[dict[str, object]], project_root: Path
) -> list[dict[str, int | str]]:
    return [
        _file_receipt(
            project_root,
            str(output["path"]),
            label=f"campaign output {output['path']}",
            artifacts_only=True,
        )
        for output in outputs
    ]


def _records_fingerprint(records: list[dict[str, int | str]]) -> str:
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _marker_matches(
    path: Path,
    *,
    step_id: str,
    launch_receipt: dict[str, object],
    inputs: list[dict[str, int | str]],
    outputs: list[dict[str, object]],
    project_root: Path,
) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not (
        isinstance(payload, dict)
        and payload.get("status") == "completed"
        and payload.get("step_id") == step_id
        and payload.get("launch_fingerprint")
        == launch_receipt.get("launch_fingerprint")
        and payload.get("launch_receipt") == launch_receipt
        and payload.get("inputs") == inputs
        and payload.get("input_fingerprint") == _records_fingerprint(inputs)
    ):
        return False
    try:
        return payload.get("outputs") == _output_records(outputs, project_root)
    except OSError:
        return False


def _write_state(
    path: Path,
    *,
    campaign_id: str,
    launch_receipt: dict[str, object],
    retry_epoch: int,
    status: str,
    completed_steps: list[str],
    current_step: str | None = None,
    returncode: int | None = None,
    attempt: int | None = None,
) -> None:
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "launch_fingerprint": launch_receipt["launch_fingerprint"],
            "launch_receipt": launch_receipt,
            "retry_epoch": retry_epoch,
            "status": status,
            "updated_at": _timestamp(),
            "completed_steps": completed_steps,
            "current_step": current_step,
            "returncode": returncode,
            "attempt": attempt,
        },
    )


def _attempt_paths(
    run_directory: Path,
    *,
    index: int,
    step_id: str,
    retry_epoch: int,
    input_fingerprint: str,
    attempt: int,
) -> tuple[Path, Path]:
    stem = (
        f"{index:02d}-{step_id}.epoch-{retry_epoch:03d}."
        f"input-{input_fingerprint}.attempt-{attempt:03d}"
    )
    return (
        run_directory / "attempts" / f"{stem}.json",
        run_directory / "logs" / f"{stem}.log",
    )


def _matching_attempts(
    run_directory: Path,
    *,
    index: int,
    step_id: str,
    retry_epoch: int,
    input_fingerprint: str,
    launch_receipt: dict[str, object],
) -> list[dict[str, object]]:
    directory = run_directory / "attempts"
    pattern = (
        f"{index:02d}-{step_id}.epoch-{retry_epoch:03d}."
        f"input-{input_fingerprint}.attempt-*.json"
    )
    receipts: list[dict[str, object]] = []
    for path in sorted(directory.glob(pattern)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid attempt receipt {path}: {exc}") from exc
        if not (
            isinstance(payload, dict)
            and payload.get("step_id") == step_id
            and payload.get("retry_epoch") == retry_epoch
            and payload.get("input_fingerprint") == input_fingerprint
            and payload.get("launch_receipt") == launch_receipt
            and isinstance(payload.get("attempt"), int)
            and int(payload["attempt"]) > 0
        ):
            raise RuntimeError(f"invalid attempt receipt contents: {path}")
        receipts.append(payload)
    attempts = [int(receipt["attempt"]) for receipt in receipts]
    if attempts != list(range(1, len(attempts) + 1)):
        raise RuntimeError(f"attempt receipts are not contiguous for step {step_id}")
    return receipts


def _failure_latch_payload(
    *,
    campaign_id: str,
    launch_receipt: dict[str, object],
    step_id: str,
    retry_epoch: int,
    attempts: int,
    inputs: list[dict[str, int | str]],
    returncode: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "latched_failure",
        "latched_at": _timestamp(),
        "campaign_id": campaign_id,
        "launch_fingerprint": launch_receipt["launch_fingerprint"],
        "launch_receipt": launch_receipt,
        "step_id": step_id,
        "retry_epoch": retry_epoch,
        "attempts": attempts,
        "input_fingerprint": _records_fingerprint(inputs),
        "inputs": inputs,
        "returncode": returncode,
    }


def _execute_step(
    step: dict[str, object],
    *,
    project_root: Path,
    log_path: Path,
) -> int:
    step_id = str(step["id"])
    argv = [str(value) for value in step["argv"]]  # type: ignore[union-attr]
    # Keep the lexical venv path: resolving its interpreter symlink would lose
    # venv discovery and execute against the system environment.
    argv[0] = str((project_root / argv[0]).absolute())
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise RuntimeError(f"refusing to overwrite retained attempt log: {log_path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{log_path.name}.", suffix=".tmp", dir=log_path.parent
    )
    temporary = Path(temporary_name)
    environment = os.environ.copy()
    source_path = str(project_root / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not environment.get("PYTHONPATH")
        else source_path + os.pathsep + environment["PYTHONPATH"]
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"started_at": _timestamp(), "step_id": step_id, "argv": argv},
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            process = subprocess.run(
                argv,
                cwd=project_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
            handle.write(
                json.dumps(
                    {
                        "finished_at": _timestamp(),
                        "step_id": step_id,
                        "returncode": process.returncode,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, log_path)
        _fsync_directory(log_path.parent)
        return process.returncode
    finally:
        # A killed process leaves a temporary log for post-mortem inspection.
        # A normal exception before subprocess launch does not need it.
        if temporary.exists() and temporary.stat().st_size == 0:
            temporary.unlink()


def _remove_own_pid_file(pid_path: Path, fingerprint: str) -> None:
    try:
        payload = json.loads(pid_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if (
        isinstance(payload, dict)
        and payload.get("pid") == os.getpid()
        and payload.get("launch_fingerprint") == fingerprint
    ):
        pid_path.unlink(missing_ok=True)


def _process_records(processes: object) -> list[dict[str, object]]:
    return [
        {
            "pid": process.pid,
            "name": process.name,
            "used_memory_mib": process.used_memory_mib,
        }
        for process in processes  # type: ignore[union-attr]
    ]


def _idle_before_step(policy: IdlePolicy) -> tuple[bool, dict[str, object]]:
    """Sample and immediately recheck the physical GPU without opening CUDA."""

    try:
        idle, utilizations, processes = _idle_samples(
            gpu_index=policy.gpu_index,
            max_utilization=policy.max_utilization,
            samples=policy.samples,
            interval_seconds=policy.interval_seconds,
            max_background_processes=policy.max_background_processes,
            max_background_memory_mib=policy.max_background_memory_mib,
        )
        details: dict[str, object] = {
            "reason": "gpu_busy",
            "gpu_index": policy.gpu_index,
            "utilizations": utilizations,
            "processes": _process_records(processes),
        }
        if not idle:
            return False, details
        final_processes = _compute_processes(policy.gpu_index)
        final_utilization = _gpu_utilization(policy.gpu_index)
        details["final_utilization"] = final_utilization
        details["final_processes"] = _process_records(final_processes)
        blocked = _processes_block_training(
            final_processes,
            max_background_processes=policy.max_background_processes,
            max_background_memory_mib=policy.max_background_memory_mib,
        )
        if blocked or final_utilization > policy.max_utilization:
            details["reason"] = "gpu_became_busy"
            return False, details
        details["reason"] = "gpu_idle"
        return True, details
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        return False, {"reason": "telemetry_error", "error": str(exc)}


def run_campaign(
    project_root: Path,
    manifest_path: Path,
    fingerprint: str,
    *,
    idle_policy: IdlePolicy | None = None,
    launch_policy: dict[str, int | float] | None = None,
    force: bool = False,
    launch_receipt: dict[str, object] | None = None,
) -> int:
    manifest, steps = validate_manifest(project_root, manifest_path)
    campaign_id = str(manifest["campaign_id"])
    effective_policy = (
        dict(launch_policy)
        if launch_policy is not None
        else (
            {
                "gpu_index": idle_policy.gpu_index,
                "max_utilization": idle_policy.max_utilization,
                "max_background_processes": idle_policy.max_background_processes,
                "max_background_memory_mib": idle_policy.max_background_memory_mib,
                "samples": idle_policy.samples,
                "interval_seconds": idle_policy.interval_seconds,
            }
            if idle_policy is not None
            else _manifest_idle_policy(manifest)
        )
    )
    expected_receipt = launch_receipt or _campaign_launch_receipt(
        project_root,
        manifest_path,
        policy=effective_policy,
        force=force,
    )
    if expected_receipt.get("launch_fingerprint") != fingerprint:
        raise RuntimeError(
            "launch receipt does not match the scheduled launch fingerprint"
        )
    paths = _campaign_state_paths(project_root, campaign_id, fingerprint)
    run_directory = paths["run"]
    markers = run_directory / "steps"
    events = run_directory / "events.jsonl"
    state = run_directory / "state.json"
    run_directory.mkdir(parents=True, exist_ok=True)
    paths["runner_lock"].parent.mkdir(parents=True, exist_ok=True)
    with paths["runner_lock"].open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        retry_epoch = _retry_epoch(paths["retry_epoch"], expected_receipt)

        def event(name: str, **fields: object) -> None:
            payload: dict[str, object] = {
                "campaign_id": campaign_id,
                "launch_fingerprint": fingerprint,
                "launch_receipt": expected_receipt,
                "retry_epoch": retry_epoch,
            }
            payload.update(fields)
            _append_event(events, name, **payload)

        if paths["failure_latch"].exists():
            if _failure_latch_matches(paths["failure_latch"], expected_receipt):
                event("campaign_failure_latched")
                return 78
            raise RuntimeError("failure latch does not match this launch receipt")
        completed: list[str] = []
        _write_state(
            state,
            campaign_id=campaign_id,
            launch_receipt=expected_receipt,
            retry_epoch=retry_epoch,
            status="running",
            completed_steps=completed,
        )
        event("campaign_started")
        for index, step in enumerate(steps, start=1):
            step_id = str(step["id"])
            marker = markers / f"{index:02d}-{step_id}.complete.json"
            inputs = step["inputs"]
            outputs = step["outputs"]
            assert isinstance(inputs, list)
            assert isinstance(outputs, list)
            typed_inputs = [str(value) for value in inputs]
            typed_outputs = [output for output in outputs if isinstance(output, dict)]
            assert len(typed_outputs) == len(outputs)
            try:
                input_records = _input_records(typed_inputs, project_root)
            except (OSError, RuntimeError) as exc:
                _write_state(
                    state,
                    campaign_id=campaign_id,
                    launch_receipt=expected_receipt,
                    retry_epoch=retry_epoch,
                    status="invalid_inputs",
                    completed_steps=completed,
                    current_step=step_id,
                    returncode=2,
                )
                event("step_inputs_invalid", step_id=step_id, error=str(exc))
                return 2
            output_valid = all(
                _output_satisfies(output, project_root) for output in typed_outputs
            )
            if (
                _marker_matches(
                    marker,
                    step_id=step_id,
                    launch_receipt=expected_receipt,
                    inputs=input_records,
                    outputs=typed_outputs,
                    project_root=project_root,
                )
                and output_valid
            ):
                completed.append(step_id)
                event("step_resumed", step_id=step_id, inputs=input_records)
                continue
            try:
                current_receipt = _campaign_launch_receipt(
                    project_root,
                    manifest_path,
                    policy=effective_policy,
                    force=force,
                )
            except (
                OSError,
                RuntimeError,
                TypeError,
                subprocess.SubprocessError,
            ) as exc:
                _write_state(
                    state,
                    campaign_id=campaign_id,
                    launch_receipt=expected_receipt,
                    retry_epoch=retry_epoch,
                    status="environment_check_failed",
                    completed_steps=completed,
                    current_step=step_id,
                    returncode=2,
                )
                event("environment_check_failed", step_id=step_id, error=str(exc))
                return 2
            if current_receipt != expected_receipt:
                _write_state(
                    state,
                    campaign_id=campaign_id,
                    launch_receipt=expected_receipt,
                    retry_epoch=retry_epoch,
                    status="stale_launch_receipt",
                    completed_steps=completed,
                    current_step=step_id,
                    returncode=2,
                )
                event(
                    "launch_receipt_changed",
                    step_id=step_id,
                    observed_launch_receipt=current_receipt,
                )
                return 2
            if idle_policy is not None:
                idle, idle_details = _idle_before_step(idle_policy)
                if not idle:
                    reason = str(idle_details["reason"])
                    _write_state(
                        state,
                        campaign_id=campaign_id,
                        launch_receipt=expected_receipt,
                        retry_epoch=retry_epoch,
                        status=f"paused_{reason}",
                        completed_steps=completed,
                        current_step=step_id,
                        returncode=75,
                    )
                    event(
                        "campaign_paused",
                        step_id=step_id,
                        **idle_details,
                    )
                    return 75
            input_fingerprint = _records_fingerprint(input_records)
            attempt_receipts = _matching_attempts(
                run_directory,
                index=index,
                step_id=step_id,
                retry_epoch=retry_epoch,
                input_fingerprint=input_fingerprint,
                launch_receipt=expected_receipt,
            )
            for prior in attempt_receipts:
                if prior.get("status") != "running":
                    continue
                prior_attempt = int(prior["attempt"])
                prior_path, _ = _attempt_paths(
                    run_directory,
                    index=index,
                    step_id=step_id,
                    retry_epoch=retry_epoch,
                    input_fingerprint=input_fingerprint,
                    attempt=prior_attempt,
                )
                prior["status"] = "interrupted"
                prior["interrupted_at"] = _timestamp()
                _atomic_json(prior_path, prior)
                event(
                    "step_attempt_interrupted",
                    step_id=step_id,
                    attempt=prior_attempt,
                    inputs=input_records,
                )
            max_attempts = int(manifest["max_attempts_per_step"])
            if len(attempt_receipts) >= max_attempts:
                last_returncode = int(attempt_receipts[-1].get("returncode") or 1)
                latch = _failure_latch_payload(
                    campaign_id=campaign_id,
                    launch_receipt=expected_receipt,
                    step_id=step_id,
                    retry_epoch=retry_epoch,
                    attempts=len(attempt_receipts),
                    inputs=input_records,
                    returncode=last_returncode,
                )
                _atomic_json(paths["failure_latch"], latch)
                _write_state(
                    state,
                    campaign_id=campaign_id,
                    launch_receipt=expected_receipt,
                    retry_epoch=retry_epoch,
                    status="latched_failure",
                    completed_steps=completed,
                    current_step=step_id,
                    returncode=78,
                    attempt=len(attempt_receipts),
                )
                event("campaign_failure_latched", **latch)
                return 78
            attempt = len(attempt_receipts) + 1
            attempt_path, log_path = _attempt_paths(
                run_directory,
                index=index,
                step_id=step_id,
                retry_epoch=retry_epoch,
                input_fingerprint=input_fingerprint,
                attempt=attempt,
            )
            attempt_receipt: dict[str, object] = {
                "schema_version": 1,
                "status": "running",
                "started_at": _timestamp(),
                "campaign_id": campaign_id,
                "launch_fingerprint": fingerprint,
                "launch_receipt": expected_receipt,
                "step_id": step_id,
                "retry_epoch": retry_epoch,
                "attempt": attempt,
                "input_fingerprint": input_fingerprint,
                "inputs": input_records,
                "argv": step["argv"],
                "log": str(log_path.relative_to(run_directory)),
            }
            _atomic_json(attempt_path, attempt_receipt)
            _write_state(
                state,
                campaign_id=campaign_id,
                launch_receipt=expected_receipt,
                retry_epoch=retry_epoch,
                status="running",
                completed_steps=completed,
                current_step=step_id,
                attempt=attempt,
            )
            event(
                "step_started",
                step_id=step_id,
                attempt=attempt,
                inputs=input_records,
            )
            try:
                returncode = _execute_step(
                    step,
                    project_root=project_root,
                    log_path=log_path,
                )
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                attempt_receipt.update(
                    {
                        "status": "failed_to_execute",
                        "finished_at": _timestamp(),
                        "returncode": 2,
                        "error": str(exc),
                    }
                )
                _atomic_json(attempt_path, attempt_receipt)
                if attempt >= max_attempts:
                    latch = _failure_latch_payload(
                        campaign_id=campaign_id,
                        launch_receipt=expected_receipt,
                        step_id=step_id,
                        retry_epoch=retry_epoch,
                        attempts=attempt,
                        inputs=input_records,
                        returncode=2,
                    )
                    _atomic_json(paths["failure_latch"], latch)
                _write_state(
                    state,
                    campaign_id=campaign_id,
                    launch_receipt=expected_receipt,
                    retry_epoch=retry_epoch,
                    status=(
                        "latched_failure"
                        if attempt >= max_attempts
                        else "failed_to_execute"
                    ),
                    completed_steps=completed,
                    current_step=step_id,
                    returncode=78 if attempt >= max_attempts else 2,
                    attempt=attempt,
                )
                event(
                    "step_execution_error",
                    step_id=step_id,
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt >= max_attempts:
                    event("campaign_failure_latched", **latch)
                    return 78
                return 2
            output_valid = all(
                _output_satisfies(output, project_root) for output in typed_outputs
            )
            if returncode != 0 or not output_valid:
                attempt_receipt.update(
                    {
                        "status": "failed",
                        "finished_at": _timestamp(),
                        "returncode": returncode,
                        "outputs_valid": output_valid,
                    }
                )
                _atomic_json(attempt_path, attempt_receipt)
                if attempt >= max_attempts:
                    latch = _failure_latch_payload(
                        campaign_id=campaign_id,
                        launch_receipt=expected_receipt,
                        step_id=step_id,
                        retry_epoch=retry_epoch,
                        attempts=attempt,
                        inputs=input_records,
                        returncode=returncode if returncode else 1,
                    )
                    _atomic_json(paths["failure_latch"], latch)
                _write_state(
                    state,
                    campaign_id=campaign_id,
                    launch_receipt=expected_receipt,
                    retry_epoch=retry_epoch,
                    status=("latched_failure" if attempt >= max_attempts else "failed"),
                    completed_steps=completed,
                    current_step=step_id,
                    returncode=78 if attempt >= max_attempts else returncode,
                    attempt=attempt,
                )
                event(
                    "step_failed",
                    step_id=step_id,
                    attempt=attempt,
                    returncode=returncode,
                    outputs_valid=output_valid,
                )
                if attempt >= max_attempts:
                    event("campaign_failure_latched", **latch)
                    return 78
                return returncode if 0 < returncode < 126 else 1
            completed.append(step_id)
            output_records = _output_records(typed_outputs, project_root)
            _atomic_json(
                marker,
                {
                    "schema_version": 1,
                    "status": "completed",
                    "step_id": step_id,
                    "completed_at": _timestamp(),
                    "launch_fingerprint": fingerprint,
                    "launch_receipt": expected_receipt,
                    "retry_epoch": retry_epoch,
                    "attempt": attempt,
                    "argv": step["argv"],
                    "input_fingerprint": input_fingerprint,
                    "inputs": input_records,
                    "outputs": output_records,
                },
            )
            attempt_receipt.update(
                {
                    "status": "completed",
                    "finished_at": _timestamp(),
                    "returncode": 0,
                    "outputs_valid": True,
                    "outputs": output_records,
                }
            )
            _atomic_json(attempt_path, attempt_receipt)
            event(
                "step_completed",
                step_id=step_id,
                attempt=attempt,
                inputs=input_records,
                outputs=output_records,
            )
        try:
            current_receipt = _campaign_launch_receipt(
                project_root,
                manifest_path,
                policy=effective_policy,
                force=force,
            )
        except (OSError, RuntimeError, TypeError, subprocess.SubprocessError) as exc:
            _write_state(
                state,
                campaign_id=campaign_id,
                launch_receipt=expected_receipt,
                retry_epoch=retry_epoch,
                status="environment_check_failed",
                completed_steps=completed,
                returncode=2,
            )
            event("environment_check_failed", error=str(exc))
            return 2
        if current_receipt != expected_receipt:
            _write_state(
                state,
                campaign_id=campaign_id,
                launch_receipt=expected_receipt,
                retry_epoch=retry_epoch,
                status="stale_launch_receipt",
                completed_steps=completed,
                returncode=2,
            )
            event("launch_receipt_changed", observed_launch_receipt=current_receipt)
            return 2
        all_output_records: list[dict[str, int | str]] = []
        for step in steps:
            outputs = step["outputs"]
            assert isinstance(outputs, list)
            typed_outputs = [output for output in outputs if isinstance(output, dict)]
            if len(typed_outputs) != len(outputs) or not all(
                _output_satisfies(output, project_root) for output in typed_outputs
            ):
                _write_state(
                    state,
                    campaign_id=campaign_id,
                    launch_receipt=expected_receipt,
                    retry_epoch=retry_epoch,
                    status="outputs_changed_before_completion",
                    completed_steps=completed,
                    returncode=2,
                )
                event("outputs_changed_before_completion", step_id=str(step["id"]))
                return 2
            all_output_records.extend(_output_records(typed_outputs, project_root))
        _write_state(
            state,
            campaign_id=campaign_id,
            launch_receipt=expected_receipt,
            retry_epoch=retry_epoch,
            status="completed",
            completed_steps=completed,
        )
        _atomic_json(
            paths["complete"],
            {
                "schema_version": 1,
                "status": "completed",
                "campaign_id": campaign_id,
                "completed_at": _timestamp(),
                "launch_fingerprint": fingerprint,
                "launch_receipt": expected_receipt,
                "retry_epoch": retry_epoch,
                "completed_steps": completed,
                "outputs": all_output_records,
            },
        )
        event("campaign_completed", outputs=all_output_records)
        return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--launch-fingerprint", required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument("--max-background-processes", type=int, default=1)
    parser.add_argument("--max-background-memory-mib", type=int, default=512)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument(
        "--skip-idle-checks",
        action="store_true",
        help="used only when the parent scheduler was explicitly forced",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    if not FINGERPRINT_PATTERN.fullmatch(args.launch_fingerprint):
        raise SystemExit("--launch-fingerprint must be a lowercase SHA-256 digest")
    if args.gpu_index < 0:
        raise SystemExit("--gpu-index must be nonnegative")
    if not 0 <= args.max_utilization <= 100:
        raise SystemExit("--max-utilization must lie between 0 and 100")
    if args.max_background_processes < 0:
        raise SystemExit("--max-background-processes must be nonnegative")
    if args.max_background_memory_mib < 0:
        raise SystemExit("--max-background-memory-mib must be nonnegative")
    if (
        args.samples <= 0
        or args.interval_seconds < 0
        or not math.isfinite(args.interval_seconds)
    ):
        raise SystemExit(
            "--samples must be positive and --interval-seconds finite and nonnegative"
        )
    fingerprint = str(args.launch_fingerprint)
    launch_policy: dict[str, int | float] = {
        "gpu_index": args.gpu_index,
        "max_utilization": args.max_utilization,
        "max_background_processes": args.max_background_processes,
        "max_background_memory_mib": args.max_background_memory_mib,
        "samples": args.samples,
        "interval_seconds": args.interval_seconds,
    }
    idle_policy = (
        None
        if args.skip_idle_checks
        else IdlePolicy(
            gpu_index=args.gpu_index,
            max_utilization=args.max_utilization,
            max_background_processes=args.max_background_processes,
            max_background_memory_mib=args.max_background_memory_mib,
            samples=args.samples,
            interval_seconds=args.interval_seconds,
        )
    )
    try:
        launch_receipt = _campaign_launch_receipt(
            project_root,
            manifest_path,
            policy=launch_policy,
            force=args.skip_idle_checks,
        )
        if launch_receipt.get("launch_fingerprint") != fingerprint:
            raise RuntimeError(
                "runtime launch receipt differs from the scheduler fingerprint"
            )
        return run_campaign(
            project_root,
            manifest_path,
            fingerprint,
            idle_policy=idle_policy,
            launch_policy=launch_policy,
            force=args.skip_idle_checks,
            launch_receipt=launch_receipt,
        )
    except (OSError, RuntimeError, TypeError, subprocess.SubprocessError) as exc:
        print(f"campaign runner error: {exc}", file=sys.stderr)
        return 2
    finally:
        try:
            manifest = _load_campaign_manifest(project_root, manifest_path)
            paths = _campaign_state_paths(
                project_root, str(manifest["campaign_id"]), fingerprint
            )
            _remove_own_pid_file(paths["pid"], fingerprint)
        except (OSError, RuntimeError, TypeError):
            pass


if __name__ == "__main__":
    raise SystemExit(main())
