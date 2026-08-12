#!/usr/bin/env python3
"""Launch the Gate-2 trainer once the selected NVIDIA GPU is genuinely idle."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _append_event(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"created_at": _timestamp(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


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


def _compute_processes(gpu_index: int) -> tuple[tuple[int, str], ...]:
    output = _run_nvidia_smi(
        [
            f"--id={gpu_index}",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader,nounits",
        ]
    )
    processes: list[tuple[int, str]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("no running"):
            continue
        pid_text, separator, process_name = line.partition(",")
        if not separator:
            raise RuntimeError(f"unexpected nvidia-smi process row: {line!r}")
        processes.append((int(pid_text.strip()), process_name.strip()))
    return tuple(processes)


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


def _active_training(pid_file: Path) -> tuple[bool, int | None]:
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
    return "train_gate2.sh" in command, pid


def _idle_samples(
    *,
    gpu_index: int,
    max_utilization: int,
    samples: int,
    interval_seconds: float,
) -> tuple[bool, list[int], tuple[tuple[int, str], ...]]:
    processes = _compute_processes(gpu_index)
    if processes:
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
    parser.add_argument("--config", type=Path)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="bypass only the GPU-idle check"
    )
    parser.add_argument("--print-fingerprint", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.gpu_index < 0:
        raise SystemExit("--gpu-index must be nonnegative")
    if not 0 <= args.max_utilization <= 100:
        raise SystemExit("--max-utilization must lie between 0 and 100")
    if args.samples <= 0 or args.interval_seconds < 0:
        raise SystemExit(
            "--samples must be positive and --interval-seconds nonnegative"
        )

    project_root = args.project_root.expanduser().resolve()
    config_path = (
        args.config.expanduser().resolve()
        if args.config is not None
        else project_root / "configs" / "gate2.yaml"
    )
    try:
        launch_fingerprint = _launch_fingerprint(project_root, config_path)
    except (OSError, RuntimeError) as exc:
        print(f"cannot fingerprint training launch: {exc}", file=sys.stderr)
        return 2
    if args.print_fingerprint:
        print(launch_fingerprint)
        return 0
    launcher = project_root / "scripts" / "train_gate2.sh"
    state_dir = project_root / "artifacts" / "gate2" / "scheduler"
    lock_path = state_dir / "launcher.lock"
    pid_path = state_dir / "trainer.json"
    complete_path = state_dir / "training.complete"
    events_path = state_dir / "events.jsonl"
    output_path = state_dir / "training.log"
    state_dir.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0

        if complete_path.exists():
            try:
                completed_fingerprint = complete_path.read_text(
                    encoding="utf-8"
                ).strip()
            except OSError as exc:
                _append_event(events_path, "completion_read_error", error=str(exc))
                return 2
            if completed_fingerprint == launch_fingerprint:
                return 0
            _append_event(
                events_path,
                "stale_completion_marker",
                completed_fingerprint=completed_fingerprint,
                requested_fingerprint=launch_fingerprint,
            )

        active, prior_pid = _active_training(pid_path)
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
                processes=[{"pid": pid, "name": name} for pid, name in processes],
            )
            return 0

        if not args.force:
            try:
                final_processes = _compute_processes(args.gpu_index)
                final_utilization = _gpu_utilization(args.gpu_index)
            except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
                _append_event(events_path, "telemetry_recheck_error", error=str(exc))
                return 2
            if final_processes or final_utilization > args.max_utilization:
                _append_event(
                    events_path,
                    "gpu_became_busy",
                    gpu_index=args.gpu_index,
                    utilization=final_utilization,
                    processes=[
                        {"pid": pid, "name": name} for pid, name in final_processes
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

        if not launcher.is_file() or not os.access(launcher, os.X_OK):
            message = f"training launcher is missing or not executable: {launcher}"
            _append_event(events_path, "launch_error", error=message)
            print(message, file=sys.stderr)
            return 2

        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        environment["HOMYMOLY_GATE2_CONFIG"] = str(config_path)
        environment["HOMYMOLY_GATE2_STATE_DIR"] = str(state_dir)
        with output_path.open("a", encoding="utf-8") as output_handle:
            process = subprocess.Popen(
                [str(launcher)],
                cwd=project_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=output_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        _atomic_json(
            pid_path,
            {
                "pid": process.pid,
                "launched_at": _timestamp(),
                "gpu_index": args.gpu_index,
                "utilizations": utilizations,
                "launcher": str(launcher),
            },
        )
        _append_event(
            events_path,
            "training_launched",
            pid=process.pid,
            gpu_index=args.gpu_index,
            utilizations=utilizations,
        )
        print(f"launched Gate-2 training as PID {process.pid}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
