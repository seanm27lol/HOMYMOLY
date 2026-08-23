#!/usr/bin/env python3
"""Launch Gate-2 training or a declared campaign once an NVIDIA GPU is idle."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

CAMPAIGN_SCHEMA_VERSION = 1
CAMPAIGN_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
DEFAULT_IDLE_POLICY: dict[str, int | float] = {
    "gpu_index": 0,
    "max_utilization": 10,
    "max_background_processes": 1,
    "max_background_memory_mib": 512,
    "samples": 3,
    "interval_seconds": 2.0,
}
RELEVANT_DISTRIBUTIONS = (
    "homymoly",
    "networkx",
    "numpy",
    "PyYAML",
    "torch",
)


class ComputeProcess(NamedTuple):
    """One NVIDIA compute context reported by ``nvidia-smi``."""

    pid: int
    name: str
    used_memory_mib: int | None


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _append_event(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"created_at": _timestamp(), "event": event, **fields}
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
            raise RuntimeError(f"scheduler event log is not a regular file: {path}")
        os.write(
            descriptor,
            (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while block := os.read(descriptor, 1024 * 1024):
        digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(
    project_root: Path,
    path_text: str,
    *,
    label: str,
    artifacts_only: bool,
) -> Path:
    """Resolve a canonical relative path without accepting symlink traversal."""

    if not path_text or any(ord(character) < 32 for character in path_text):
        raise RuntimeError(f"{label} must be a non-empty control-free path")
    if "\\" in path_text:
        raise RuntimeError(f"{label} must use POSIX path separators")
    path = Path(path_text)
    if path.is_absolute() or path_text != path.as_posix():
        raise RuntimeError(f"{label} must be a canonical relative path: {path_text}")
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"{label} contains an unsafe path component: {path_text}")
    root = project_root.resolve()
    lexical = root.joinpath(*path.parts)
    confinement = root / "artifacts" if artifacts_only else root
    try:
        lexical.relative_to(confinement)
    except ValueError as exc:
        target = "artifacts" if artifacts_only else "the project"
        raise RuntimeError(f"{label} must remain under {target}: {path_text}") from exc
    current = root
    for part in path.parts:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"{label} traverses a symlink: {current}")
    resolved = lexical.resolve(strict=False)
    try:
        resolved.relative_to(confinement.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes confinement: {path_text}") from exc
    return lexical


def _file_receipt(
    project_root: Path,
    path_text: str,
    *,
    label: str,
    artifacts_only: bool = False,
) -> dict[str, int | str]:
    """Return a stable size/SHA receipt for a nonsymlink regular file."""

    path = _safe_relative_path(
        project_root,
        path_text,
        label=label,
        artifacts_only=artifacts_only,
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(
            f"{label} cannot be opened safely: {path_text}: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} is not a regular file: {path_text}")
        sha256 = _sha256_descriptor(descriptor)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after:
            raise RuntimeError(f"{label} changed while hashing: {path_text}")
        return {"path": path_text, "bytes": after.st_size, "sha256": sha256}
    finally:
        os.close(descriptor)


def _absolute_file_identity(path: Path, *, label: str) -> dict[str, int | str]:
    resolved = path.resolve(strict=True)
    descriptor = os.open(
        resolved,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"{label} is not a regular file: {resolved}")
        return {
            "path": str(path.absolute()),
            "resolved_path": str(resolved),
            "bytes": metadata.st_size,
            "sha256": _sha256_descriptor(descriptor),
        }
    finally:
        os.close(descriptor)


def _resolved_in_root(path: Path, project_root: Path, *, label: str) -> Path:
    resolved_root = project_root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"{label} must be inside the project root: {path}") from exc
    return resolved


def _load_campaign_manifest(project_root: Path, manifest_path: Path) -> dict[str, Any]:
    """Load the scheduler-facing subset of a checked-in campaign manifest."""

    manifest_path = _resolved_in_root(
        manifest_path, project_root, label="campaign manifest"
    )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read campaign manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError("campaign manifest must be a JSON object")
    if payload.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        raise RuntimeError(f"campaign schema_version must be {CAMPAIGN_SCHEMA_VERSION}")
    if "execution_enabled" not in payload:
        raise RuntimeError("campaign execution_enabled must be explicit")
    execution_enabled = payload["execution_enabled"]
    if not isinstance(execution_enabled, bool):
        raise TypeError("campaign execution_enabled must be a boolean")
    if not execution_enabled:
        raise RuntimeError("campaign execution is disabled by its manifest")
    campaign_id = payload.get("campaign_id")
    if not isinstance(campaign_id, str) or not CAMPAIGN_ID_PATTERN.fullmatch(
        campaign_id
    ):
        raise RuntimeError(
            "campaign_id must contain only lowercase letters, digits, '.', '_', "
            "or '-' and start with a letter or digit"
        )
    inputs = payload.get("fingerprint_inputs")
    if not isinstance(inputs, list) or not inputs:
        raise RuntimeError("fingerprint_inputs must be a non-empty list")
    for index, value in enumerate(inputs):
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"fingerprint_inputs[{index}] must be a path string")
        _file_receipt(
            project_root,
            value,
            label=f"fingerprint_inputs[{index}]",
            artifacts_only=False,
        )
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise RuntimeError("campaign steps must be a non-empty list")
    max_attempts = payload.get("max_attempts_per_step")
    if (
        not isinstance(max_attempts, int)
        or isinstance(max_attempts, bool)
        or not 1 <= max_attempts <= 10
    ):
        raise RuntimeError("max_attempts_per_step must be an integer in [1,10]")
    _manifest_idle_policy(payload)
    return payload


def _manifest_idle_policy(payload: dict[str, Any]) -> dict[str, int | float]:
    raw = payload.get("idle_policy")
    if raw is None:
        return dict(DEFAULT_IDLE_POLICY)
    if not isinstance(raw, dict):
        raise TypeError("campaign idle_policy must be an object")
    if set(raw) != set(DEFAULT_IDLE_POLICY):
        raise RuntimeError(
            "campaign idle_policy must declare exactly: "
            + ", ".join(sorted(DEFAULT_IDLE_POLICY))
        )
    integer_fields = (
        "gpu_index",
        "max_utilization",
        "max_background_processes",
        "max_background_memory_mib",
        "samples",
    )
    if any(
        not isinstance(raw[name], int) or isinstance(raw[name], bool)
        for name in integer_fields
    ):
        raise TypeError("campaign idle_policy integer fields must be integers")
    interval = raw["interval_seconds"]
    if (
        not isinstance(interval, (int, float))
        or isinstance(interval, bool)
        or not math.isfinite(float(interval))
    ):
        raise TypeError("campaign idle_policy.interval_seconds must be finite")
    policy: dict[str, int | float] = {name: int(raw[name]) for name in integer_fields}
    policy["interval_seconds"] = float(interval)
    if int(policy["gpu_index"]) < 0:
        raise RuntimeError("campaign idle_policy.gpu_index must be nonnegative")
    if not 0 <= int(policy["max_utilization"]) <= 100:
        raise RuntimeError("campaign idle_policy.max_utilization must be in [0,100]")
    if int(policy["max_background_processes"]) < 0:
        raise RuntimeError(
            "campaign idle_policy.max_background_processes must be nonnegative"
        )
    if int(policy["max_background_memory_mib"]) < 0:
        raise RuntimeError(
            "campaign idle_policy.max_background_memory_mib must be nonnegative"
        )
    if int(policy["samples"]) <= 0 or float(policy["interval_seconds"]) < 0:
        raise RuntimeError(
            "campaign idle_policy samples must be positive and interval nonnegative"
        )
    return policy


def _hash_paths(digest: Any, paths: list[Path], project_root: Path) -> None:
    for path in sorted({candidate.resolve() for candidate in paths}):
        if not path.is_file():
            raise RuntimeError(f"launch input does not exist: {path}")
        try:
            label = path.relative_to(project_root).as_posix()
        except ValueError:
            label = str(path)
        digest.update(label.encode("utf-8"))
        digest.update(b"\x00")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\x00")


def _launch_fingerprint(project_root: Path, config_path: Path) -> str:
    """Fingerprint executable source plus the exact launch configuration."""

    digest = hashlib.sha256()
    candidates = list((project_root / "src" / "homymoly").rglob("*.py"))
    candidates.extend(
        (
            config_path,
            project_root / "pyproject.toml",
            project_root / "scripts" / "train_gate2.sh",
        )
    )
    for path in sorted({candidate.resolve() for candidate in candidates}):
        if not path.is_file():
            raise RuntimeError(f"launch input does not exist: {path}")
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def _campaign_fingerprint(project_root: Path, manifest_path: Path) -> str:
    """Fingerprint campaign code, declared immutable inputs, and the manifest."""

    manifest = _load_campaign_manifest(project_root, manifest_path)
    candidates = list((project_root / "src" / "homymoly").rglob("*.py"))
    candidates.extend(
        path
        for path in (
            project_root / "pyproject.toml",
            project_root / "uv.lock",
            project_root / "scripts" / "gpu_idle_train.py",
            project_root / "scripts" / "run_gpu_campaign.py",
            manifest_path,
        )
        if path.is_file()
    )
    candidates.extend(project_root / value for value in manifest["fingerprint_inputs"])
    for step in manifest["steps"]:
        if not isinstance(step, dict):
            continue
        argv = step.get("argv")
        if not isinstance(argv, list) or not argv:
            continue
        executable = argv[0]
        if isinstance(executable, str) and executable != ".venv/bin/python":
            candidate = project_root / executable
            if candidate.is_file():
                candidates.append(candidate)
        if (
            executable == ".venv/bin/python"
            and len(argv) > 1
            and isinstance(argv[1], str)
        ):
            script = project_root / argv[1]
            if script.is_file():
                candidates.append(script)
    digest = hashlib.sha256()
    _hash_paths(digest, candidates, project_root)
    return digest.hexdigest()


def _git_attestation(project_root: Path) -> dict[str, object]:
    """Require an exact clean Git worktree and attest its committed revision."""

    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is unavailable")

    def run(arguments: list[str]) -> str:
        result = subprocess.run(
            [git, "-C", str(project_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
        return result.stdout.strip()

    top_level = Path(run(["rev-parse", "--show-toplevel"])).resolve()
    if top_level != project_root.resolve():
        raise RuntimeError(
            f"project root is not the Git worktree root: {project_root} != {top_level}"
        )
    head = run(["rev-parse", "HEAD"])
    if not re.fullmatch(r"[0-9a-f]{40,64}", head):
        raise RuntimeError(f"Git returned an invalid HEAD revision: {head!r}")
    status = run(["status", "--porcelain=v1", "--untracked-files=all"])
    if status:
        lines = status.splitlines()
        preview = "; ".join(lines[:5])
        suffix = " ..." if len(lines) > 5 else ""
        raise RuntimeError(
            "campaign execution requires a clean Git worktree; " + preview + suffix
        )
    return {
        "head": head,
        "status_porcelain": [],
        "top_level": str(top_level),
        "git_executable": _absolute_file_identity(Path(git), label="git executable"),
    }


def _environment_receipt(project_root: Path, *, gpu_index: int) -> dict[str, object]:
    """Capture the interpreter, relevant distributions, and physical GPU."""

    python = project_root / ".venv" / "bin" / "python"
    interpreter = _absolute_file_identity(python, label="campaign interpreter")
    package_code = """
import importlib.metadata as metadata
import json
import platform
import sys

names = json.loads(sys.argv[1])
versions = {}
for name in names:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps({
    "base_prefix": sys.base_prefix,
    "executable": sys.executable,
    "implementation": platform.python_implementation(),
    "packages": versions,
    "prefix": sys.prefix,
    "python_version": platform.python_version(),
}, sort_keys=True))
"""
    result = subprocess.run(
        [str(python), "-c", package_code, json.dumps(RELEVANT_DISTRIBUTIONS)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"cannot inspect campaign Python environment: {detail}")
    try:
        python_environment = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "campaign interpreter returned invalid receipt JSON"
        ) from exc
    if not isinstance(python_environment, dict):
        raise TypeError("campaign interpreter receipt must be an object")
    pyvenv = project_root / ".venv" / "pyvenv.cfg"
    pyvenv_identity = _absolute_file_identity(pyvenv, label="pyvenv configuration")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("nvidia-smi is unavailable")
    gpu_row = _run_nvidia_smi(
        [
            f"--id={gpu_index}",
            "--query-gpu=index,uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ]
    ).strip()
    if not gpu_row or "\n" in gpu_row:
        raise RuntimeError(f"expected one physical GPU identity row, got {gpu_row!r}")
    return {
        "schema_version": 1,
        "interpreter": interpreter,
        "pyvenv": pyvenv_identity,
        "python_environment": python_environment,
        "nvidia_smi": _absolute_file_identity(
            Path(nvidia_smi), label="nvidia-smi executable"
        ),
        "physical_gpu": gpu_row,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "git": _git_attestation(project_root),
    }


def _canonical_launch_policy(
    policy: dict[str, int | float], *, force: bool
) -> dict[str, int | float | bool]:
    return {
        "gpu_index": int(policy["gpu_index"]),
        "max_utilization": int(policy["max_utilization"]),
        "max_background_processes": int(policy["max_background_processes"]),
        "max_background_memory_mib": int(policy["max_background_memory_mib"]),
        "samples": int(policy["samples"]),
        "interval_seconds": float(policy["interval_seconds"]),
        "force": bool(force),
        "runner_skip_idle_checks": bool(force),
    }


def _campaign_launch_receipt(
    project_root: Path,
    manifest_path: Path,
    *,
    policy: dict[str, int | float],
    force: bool,
) -> dict[str, object]:
    manifest = _load_campaign_manifest(project_root, manifest_path)
    payload: dict[str, object] = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "campaign_fingerprint": _campaign_fingerprint(project_root, manifest_path),
        "effective_launch_policy": _canonical_launch_policy(policy, force=force),
        "environment": _environment_receipt(
            project_root, gpu_index=int(policy["gpu_index"])
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["launch_fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _run_nvidia_smi(arguments: list[str]) -> str:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        raise RuntimeError("nvidia-smi is unavailable")
    result = subprocess.run(
        [binary, *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"nvidia-smi failed: {detail}")
    return result.stdout


def _compute_processes(gpu_index: int) -> tuple[ComputeProcess, ...]:
    output = _run_nvidia_smi(
        [
            f"--id={gpu_index}",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    processes: list[ComputeProcess] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("no running"):
            continue
        fields = [field.strip() for field in line.split(",", maxsplit=2)]
        if len(fields) != 3:
            raise RuntimeError(f"unexpected nvidia-smi process row: {line!r}")
        memory_text = fields[2]
        memory_mib = (
            None
            if memory_text.lower()
            in {"n/a", "[n/a]", "not supported", "[not supported]"}
            else int(memory_text)
        )
        processes.append(ComputeProcess(int(fields[0]), fields[1], memory_mib))
    return tuple(processes)


def _processes_block_training(
    processes: tuple[ComputeProcess, ...],
    *,
    max_background_processes: int,
    max_background_memory_mib: int,
) -> bool:
    """Conservatively classify existing CUDA contexts.

    A small, bounded context (for example a persistent desktop/UI process) may
    coexist with training. Unknown memory is always blocking, as are excess
    process count or aggregate memory. GPU utilization is checked separately
    across multiple samples and immediately before launch.
    """

    if not processes:
        return False
    if len(processes) > max_background_processes:
        return True
    memories = [process.used_memory_mib for process in processes]
    if any(memory is None for memory in memories):
        return True
    return (
        sum(memory for memory in memories if memory is not None)
        > max_background_memory_mib
    )


def _process_payload(process: ComputeProcess) -> dict[str, int | str | None]:
    return {
        "pid": process.pid,
        "name": process.name,
        "used_memory_mib": process.used_memory_mib,
    }


def _gpu_utilization(gpu_index: int) -> int:
    output = _run_nvidia_smi(
        [
            f"--id={gpu_index}",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected one GPU utilization row, received {rows!r}")
    return int(rows[0])


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _active_training(
    pid_file: Path, *, expected_token: str = "train_gate2.sh"
) -> tuple[bool, int | None]:
    if not pid_file.is_file():
        return False, None
    try:
        payload = json.loads(pid_file.read_text(encoding="utf-8"))
        pid = int(payload["pid"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False, None
    if not _process_is_alive(pid):
        return False, pid
    command_line = Path(f"/proc/{pid}/cmdline")
    try:
        command = (
            command_line.read_bytes().replace(b"\x00", b" ").decode(errors="replace")
        )
    except OSError:
        return True, pid
    return expected_token in command, pid


def _completion_matches(path: Path, launch_fingerprint: str) -> bool:
    """Accept the legacy plain marker and the campaign JSON marker."""

    text = path.read_text(encoding="utf-8").strip()
    if text == launch_fingerprint:
        return True
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text == launch_fingerprint
    return (
        isinstance(payload, dict)
        and payload.get("status") == "completed"
        and payload.get("launch_fingerprint") == launch_fingerprint
    )


def _output_specification_satisfied(
    project_root: Path, specification: object
) -> tuple[bool, dict[str, int | str] | None]:
    if not isinstance(specification, dict):
        return False, None
    path_text = specification.get("path")
    minimum_bytes = specification.get("minimum_bytes", 1)
    if (
        not isinstance(path_text, str)
        or not isinstance(minimum_bytes, int)
        or isinstance(minimum_bytes, bool)
        or minimum_bytes <= 0
    ):
        return False, None
    try:
        receipt = _file_receipt(
            project_root,
            path_text,
            label="campaign output",
            artifacts_only=True,
        )
    except (OSError, RuntimeError):
        return False, None
    if int(receipt["bytes"]) < minimum_bytes:
        return False, None
    equals = specification.get("json_equals", {})
    required = specification.get("json_required_keys", [])
    if not isinstance(equals, dict) or not isinstance(required, list):
        return False, None
    if equals or required:
        try:
            payload = json.loads((project_root / path_text).read_text(encoding="utf-8"))
            receipt_after = _file_receipt(
                project_root,
                path_text,
                label="campaign output",
                artifacts_only=True,
            )
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return False, None
        except RuntimeError:
            return False, None
        if receipt != receipt_after or not isinstance(payload, dict):
            return False, None
        if not all(payload.get(key) == value for key, value in equals.items()):
            return False, None
        if not all(isinstance(key, str) and key in payload for key in required):
            return False, None
    return True, receipt


def _campaign_completion_matches(
    completion_path: Path,
    *,
    project_root: Path,
    manifest: dict[str, Any],
    launch_receipt: dict[str, object],
) -> bool:
    """Revalidate every declared output before accepting campaign completion."""

    try:
        payload = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not (
        isinstance(payload, dict)
        and payload.get("status") == "completed"
        and payload.get("campaign_id") == manifest.get("campaign_id")
        and payload.get("launch_fingerprint")
        == launch_receipt.get("launch_fingerprint")
        and payload.get("launch_receipt") == launch_receipt
    ):
        return False
    expected_steps = [
        step.get("id") for step in manifest.get("steps", []) if isinstance(step, dict)
    ]
    if payload.get("completed_steps") != expected_steps:
        return False
    expected: list[dict[str, int | str]] = []
    steps = manifest.get("steps")
    if not isinstance(steps, list):
        return False
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("outputs"), list):
            return False
        for specification in step["outputs"]:
            satisfied, receipt = _output_specification_satisfied(
                project_root, specification
            )
            if not satisfied or receipt is None:
                return False
            expected.append(receipt)
    return payload.get("outputs") == expected


def _failure_latch_matches(latch_path: Path, launch_receipt: dict[str, object]) -> bool:
    try:
        payload = json.loads(latch_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("status") == "latched_failure"
        and payload.get("launch_fingerprint")
        == launch_receipt.get("launch_fingerprint")
        and payload.get("launch_receipt") == launch_receipt
    )


def _retry_epoch(path: Path, launch_receipt: dict[str, object]) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read retry epoch: {exc}") from exc
    if not (
        isinstance(payload, dict)
        and payload.get("launch_fingerprint")
        == launch_receipt.get("launch_fingerprint")
        and payload.get("launch_receipt") == launch_receipt
        and isinstance(payload.get("retry_epoch"), int)
        and int(payload["retry_epoch"]) >= 0
    ):
        raise RuntimeError("retry epoch receipt is invalid")
    return int(payload["retry_epoch"])


def _reset_failure_latch(
    paths: dict[str, Path], launch_receipt: dict[str, object]
) -> bool:
    latch = paths["failure_latch"]
    if not latch.exists():
        return False
    if not _failure_latch_matches(latch, launch_receipt):
        raise RuntimeError("failure latch does not match this launch receipt")
    epoch = _retry_epoch(paths["retry_epoch"], launch_receipt) + 1
    history = paths["run"] / "failure-latch-history"
    history.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    archived = history / f"failure-latch-{timestamp}.json"
    os.replace(latch, archived)
    _atomic_json(
        paths["retry_epoch"],
        {
            "schema_version": 1,
            "retry_epoch": epoch,
            "reset_at": _timestamp(),
            "archived_latch": str(archived.relative_to(paths["run"])),
            "launch_fingerprint": launch_receipt["launch_fingerprint"],
            "launch_receipt": launch_receipt,
        },
    )
    return True


def _campaign_state_paths(
    project_root: Path, campaign_id: str, launch_fingerprint: str
) -> dict[str, Path]:
    base = project_root / "artifacts" / "scheduler" / campaign_id
    run = base / "runs" / launch_fingerprint
    return {
        "base": base,
        "run": run,
        "launcher_lock": base / "launcher.lock",
        "runner_lock": base / "campaign.lock",
        "pid": base / "runner.json",
        "complete": run / "campaign.complete.json",
        "failure_latch": run / "failure-latch.json",
        "retry_epoch": run / "retry-epoch.json",
        "events": run / "scheduler-events.jsonl",
        "output": run / "scheduler-launch.log",
    }


def _idle_samples(
    *,
    gpu_index: int,
    max_utilization: int,
    samples: int,
    interval_seconds: float,
    max_background_processes: int = 0,
    max_background_memory_mib: int = 0,
) -> tuple[bool, list[int], tuple[ComputeProcess, ...]]:
    processes = _compute_processes(gpu_index)
    if _processes_block_training(
        processes,
        max_background_processes=max_background_processes,
        max_background_memory_mib=max_background_memory_mib,
    ):
        return False, [], processes
    utilizations: list[int] = []
    for sample_index in range(samples):
        utilizations.append(_gpu_utilization(gpu_index))
        if sample_index + 1 < samples:
            time.sleep(interval_seconds)
    return (
        all(value <= max_utilization for value in utilizations),
        utilizations,
        processes,
    )


def _parser() -> argparse.ArgumentParser:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_root)
    launch = parser.add_mutually_exclusive_group()
    launch.add_argument("--config", type=Path)
    launch.add_argument(
        "--campaign-manifest",
        type=Path,
        help="validated JSON manifest for a resumable in-repo campaign",
    )
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--max-utilization", type=int)
    parser.add_argument(
        "--max-background-processes",
        type=int,
        help="maximum existing low-memory CUDA contexts allowed",
    )
    parser.add_argument(
        "--max-background-memory-mib",
        type=int,
        help="maximum aggregate memory for allowed CUDA contexts",
    )
    parser.add_argument("--samples", type=int)
    parser.add_argument("--interval-seconds", type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="bypass only the GPU-idle check"
    )
    parser.add_argument("--print-fingerprint", action="store_true")
    parser.add_argument(
        "--reset-failure-latch",
        action="store_true",
        help="archive the matching failure latch and begin a new retry epoch",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.reset_failure_latch and args.campaign_manifest is None:
        raise SystemExit("--reset-failure-latch requires --campaign-manifest")
    if args.reset_failure_latch and (
        args.force or args.dry_run or args.print_fingerprint
    ):
        raise SystemExit(
            "--reset-failure-latch cannot be combined with force, dry-run, or "
            "print-fingerprint"
        )
    project_root = args.project_root.expanduser().resolve()
    campaign_manifest: Path | None = None
    campaign_id: str | None = None
    campaign: dict[str, Any] | None = None
    launch_receipt: dict[str, object] | None = None
    config_path: Path | None = None
    try:
        if args.campaign_manifest is not None:
            campaign_manifest = _resolved_in_root(
                args.campaign_manifest.expanduser(),
                project_root,
                label="campaign manifest",
            )
            campaign = _load_campaign_manifest(project_root, campaign_manifest)
            campaign_id = str(campaign["campaign_id"])
        else:
            config_path = (
                args.config.expanduser().resolve()
                if args.config is not None
                else project_root / "configs" / "gate2.yaml"
            )
            launch_fingerprint = _launch_fingerprint(project_root, config_path)
    except (OSError, RuntimeError, TypeError) as exc:
        print(f"cannot fingerprint training launch: {exc}", file=sys.stderr)
        return 2
    policy = (
        _manifest_idle_policy(campaign)
        if campaign is not None
        else dict(DEFAULT_IDLE_POLICY)
    )
    for name in DEFAULT_IDLE_POLICY:
        if getattr(args, name) is None:
            setattr(args, name, policy[name])
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
    if campaign_manifest is not None:
        try:
            launch_receipt = _campaign_launch_receipt(
                project_root,
                campaign_manifest,
                policy={
                    "gpu_index": args.gpu_index,
                    "max_utilization": args.max_utilization,
                    "max_background_processes": args.max_background_processes,
                    "max_background_memory_mib": args.max_background_memory_mib,
                    "samples": args.samples,
                    "interval_seconds": args.interval_seconds,
                },
                force=args.force,
            )
            launch_fingerprint = str(launch_receipt["launch_fingerprint"])
        except (OSError, RuntimeError, TypeError, subprocess.SubprocessError) as exc:
            print(f"cannot build campaign launch receipt: {exc}", file=sys.stderr)
            return 2
    if args.print_fingerprint:
        print(launch_fingerprint)
        return 0
    if campaign_manifest is not None:
        assert campaign_id is not None
        paths = _campaign_state_paths(project_root, campaign_id, launch_fingerprint)
        state_dir = paths["run"]
        lock_path = paths["launcher_lock"]
        pid_path = paths["pid"]
        complete_path = paths["complete"]
        events_path = paths["events"]
        output_path = paths["output"]
        python = project_root / ".venv" / "bin" / "python"
        runner = project_root / "scripts" / "run_gpu_campaign.py"
        launch_command = [
            str(python),
            str(runner),
            "--project-root",
            str(project_root),
            "--manifest",
            str(campaign_manifest),
            "--launch-fingerprint",
            launch_fingerprint,
            "--gpu-index",
            str(args.gpu_index),
            "--max-utilization",
            str(args.max_utilization),
            "--max-background-processes",
            str(args.max_background_processes),
            "--max-background-memory-mib",
            str(args.max_background_memory_mib),
            "--samples",
            str(args.samples),
            "--interval-seconds",
            str(args.interval_seconds),
        ]
        if args.force:
            launch_command.append("--skip-idle-checks")
        expected_token = "run_gpu_campaign.py"
        launch_label = f"campaign {campaign_id}"
    else:
        assert config_path is not None
        launcher = project_root / "scripts" / "train_gate2.sh"
        state_dir = project_root / "artifacts" / "gate2" / "scheduler"
        lock_path = state_dir / "launcher.lock"
        pid_path = state_dir / "trainer.json"
        complete_path = state_dir / "training.complete"
        events_path = state_dir / "events.jsonl"
        output_path = state_dir / "training.log"
        launch_command = [str(launcher)]
        expected_token = "train_gate2.sh"
        launch_label = "Gate-2 training"
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0

        if args.reset_failure_latch:
            assert campaign is not None and launch_receipt is not None
            active, _ = _active_training(pid_path, expected_token="run_gpu_campaign.py")
            if active:
                print(
                    "cannot reset while the campaign runner is active", file=sys.stderr
                )
                return 2
            try:
                reset = _reset_failure_latch(paths, launch_receipt)
            except (OSError, RuntimeError) as exc:
                print(f"cannot reset failure latch: {exc}", file=sys.stderr)
                return 2
            _append_event(
                events_path,
                "failure_latch_reset" if reset else "failure_latch_reset_noop",
                launch_fingerprint=launch_fingerprint,
                launch_receipt=launch_receipt,
            )
            print("failure latch reset" if reset else "no matching failure latch")
            return 0

        if complete_path.exists():
            try:
                completion_matches = (
                    _campaign_completion_matches(
                        complete_path,
                        project_root=project_root,
                        manifest=campaign,
                        launch_receipt=launch_receipt,
                    )
                    if campaign is not None and launch_receipt is not None
                    else _completion_matches(complete_path, launch_fingerprint)
                )
            except (OSError, RuntimeError) as exc:
                _append_event(events_path, "completion_read_error", error=str(exc))
                return 2
            if completion_matches:
                return 0
            _append_event(
                events_path,
                "stale_completion_marker",
                requested_fingerprint=launch_fingerprint,
            )

        if (
            campaign is not None
            and launch_receipt is not None
            and paths["failure_latch"].exists()
            and _failure_latch_matches(paths["failure_latch"], launch_receipt)
        ):
            return 0

        active, prior_pid = _active_training(pid_path, expected_token=expected_token)
        if active:
            return 0
        if pid_path.exists():
            pid_path.unlink()
            _append_event(events_path, "removed_stale_pid", pid=prior_pid)

        try:
            idle, utilizations, processes = _idle_samples(
                gpu_index=args.gpu_index,
                max_utilization=args.max_utilization,
                samples=args.samples,
                interval_seconds=args.interval_seconds,
                max_background_processes=args.max_background_processes,
                max_background_memory_mib=args.max_background_memory_mib,
            )
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
            _append_event(events_path, "telemetry_error", error=str(exc))
            print(f"GPU telemetry error: {exc}", file=sys.stderr)
            return 2

        if not idle and not args.force:
            _append_event(
                events_path,
                "gpu_busy",
                gpu_index=args.gpu_index,
                utilizations=utilizations,
                processes=[_process_payload(process) for process in processes],
            )
            return 0

        if not args.force:
            try:
                final_processes = _compute_processes(args.gpu_index)
                final_utilization = _gpu_utilization(args.gpu_index)
            except (
                OSError,
                RuntimeError,
                subprocess.SubprocessError,
                ValueError,
            ) as exc:
                _append_event(events_path, "telemetry_recheck_error", error=str(exc))
                return 2
            processes_block = _processes_block_training(
                final_processes,
                max_background_processes=args.max_background_processes,
                max_background_memory_mib=args.max_background_memory_mib,
            )
            if processes_block or final_utilization > args.max_utilization:
                _append_event(
                    events_path,
                    "gpu_became_busy",
                    gpu_index=args.gpu_index,
                    utilization=final_utilization,
                    processes=[
                        _process_payload(process) for process in final_processes
                    ],
                )
                return 0

        if args.dry_run:
            _append_event(
                events_path,
                "dry_run_idle",
                gpu_index=args.gpu_index,
                utilizations=utilizations,
            )
            print(json.dumps({"idle": True, "utilizations": utilizations}))
            return 0

        executable = Path(launch_command[0])
        if not executable.is_file() or not os.access(executable, os.X_OK):
            message = f"launch executable is missing or not executable: {executable}"
            _append_event(events_path, "launch_error", error=message)
            print(message, file=sys.stderr)
            return 2
        if campaign_manifest is not None:
            runner = Path(launch_command[1])
            if not runner.is_file():
                message = f"campaign runner is missing: {runner}"
                _append_event(events_path, "launch_error", error=message)
                print(message, file=sys.stderr)
                return 2

        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        if campaign_manifest is None:
            assert config_path is not None
            environment["HOMYMOLY_GATE2_CONFIG"] = str(config_path)
            environment["HOMYMOLY_GATE2_STATE_DIR"] = str(state_dir)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_handle = output_path.open("a", encoding="utf-8")
            try:
                try:
                    process = subprocess.Popen(
                        launch_command,
                        cwd=project_root,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=output_handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        close_fds=True,
                    )
                except OSError as exc:
                    _append_event(events_path, "launch_error", error=str(exc))
                    print(f"launch error: {exc}", file=sys.stderr)
                    return 2
            finally:
                output_handle.close()
        else:
            try:
                process = subprocess.Popen(
                    launch_command,
                    cwd=project_root,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    close_fds=True,
                )
            except OSError as exc:
                _append_event(events_path, "launch_error", error=str(exc))
                print(f"launch error: {exc}", file=sys.stderr)
                return 2
        _atomic_json(
            pid_path,
            {
                "pid": process.pid,
                "launched_at": _timestamp(),
                "gpu_index": args.gpu_index,
                "utilizations": utilizations,
                "background_processes": [
                    _process_payload(process) for process in processes
                ],
                "launch_command": launch_command,
                "launcher": launch_command[0],
                "campaign_id": campaign_id,
                "campaign_manifest": (
                    str(campaign_manifest) if campaign_manifest is not None else None
                ),
                "launch_fingerprint": launch_fingerprint,
                "launch_receipt": launch_receipt,
            },
        )
        _append_event(
            events_path,
            (
                "campaign_launched"
                if campaign_manifest is not None
                else "training_launched"
            ),
            pid=process.pid,
            launch_label=launch_label,
            gpu_index=args.gpu_index,
            utilizations=utilizations,
            background_processes=[_process_payload(process) for process in processes],
            launch_fingerprint=launch_fingerprint,
            launch_receipt=launch_receipt,
        )
        print(f"launched {launch_label} as PID {process.pid}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
