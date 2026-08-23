#!/usr/bin/env python3
"""Manage guarded HOMYMOLY Gate-2 or campaign entries in the user crontab."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import shlex
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

BEGIN = "# HOMYMOLY_GATE2_BEGIN"
END = "# HOMYMOLY_GATE2_END"
CAMPAIGN_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
DEFAULT_IDLE_POLICY: dict[str, int | float] = {
    "gpu_index": 0,
    "max_utilization": 10,
    "max_background_processes": 1,
    "max_background_memory_mib": 512,
    "samples": 3,
    "sample_interval_seconds": 2.0,
}


def _resolved_in_root(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the project root: {path}") from exc
    return resolved


def _campaign_identity(
    manifest_path: Path, root: Path, *, allow_disabled: bool = False
) -> tuple[Path, str, dict[str, int | float]]:
    manifest_path = _resolved_in_root(manifest_path, root, label="campaign manifest")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read campaign manifest: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("campaign manifest must be a schema-version 1 object")
    if "execution_enabled" not in payload:
        raise ValueError("campaign manifest execution_enabled must be explicit")
    execution_enabled = payload["execution_enabled"]
    if not isinstance(execution_enabled, bool):
        raise TypeError("campaign manifest execution_enabled must be a boolean")
    if not execution_enabled and not allow_disabled:
        raise ValueError("campaign execution is disabled by its manifest")
    if execution_enabled:
        max_attempts = payload.get("max_attempts_per_step")
        if (
            not isinstance(max_attempts, int)
            or isinstance(max_attempts, bool)
            or not 1 <= max_attempts <= 10
        ):
            raise ValueError(
                "campaign max_attempts_per_step must be an integer in [1,10]"
            )
    campaign_id = payload.get("campaign_id")
    if not isinstance(campaign_id, str) or not CAMPAIGN_ID_PATTERN.fullmatch(
        campaign_id
    ):
        raise ValueError("campaign manifest has an invalid campaign_id")
    raw_policy = payload.get("idle_policy")
    if raw_policy is None:
        policy = dict(DEFAULT_IDLE_POLICY)
    else:
        if not isinstance(raw_policy, dict):
            raise ValueError("campaign manifest idle_policy must be an object")
        manifest_keys = {
            "gpu_index",
            "max_utilization",
            "max_background_processes",
            "max_background_memory_mib",
            "samples",
            "interval_seconds",
        }
        if set(raw_policy) != manifest_keys:
            raise ValueError("campaign manifest idle_policy has invalid keys")
        integer_fields = manifest_keys - {"interval_seconds"}
        if any(
            not isinstance(raw_policy[name], int) or isinstance(raw_policy[name], bool)
            for name in integer_fields
        ):
            raise ValueError(
                "campaign manifest idle_policy integer fields must be integers"
            )
        interval = raw_policy["interval_seconds"]
        if (
            not isinstance(interval, (int, float))
            or isinstance(interval, bool)
            or not math.isfinite(float(interval))
        ):
            raise ValueError(
                "campaign manifest idle_policy.interval_seconds must be finite"
            )
        policy = {
            "gpu_index": raw_policy["gpu_index"],
            "max_utilization": raw_policy["max_utilization"],
            "max_background_processes": raw_policy["max_background_processes"],
            "max_background_memory_mib": raw_policy["max_background_memory_mib"],
            "samples": raw_policy["samples"],
            "sample_interval_seconds": raw_policy["interval_seconds"],
        }
    return manifest_path, campaign_id, policy


def _campaign_markers(campaign_id: str) -> tuple[str, str]:
    return (
        f"# HOMYMOLY_CAMPAIGN_{campaign_id}_BEGIN",
        f"# HOMYMOLY_CAMPAIGN_{campaign_id}_END",
    )


def _existing_crontab() -> str:
    result = subprocess.run(
        ["crontab", "-l"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip() or "unable to read crontab")
    return result.stdout if result.returncode == 0 else ""


def _without_managed_block(
    document: str, *, begin: str = BEGIN, end: str = END
) -> list[str]:
    result: list[str] = []
    inside = False
    blocks = 0
    for line in document.splitlines():
        if line.strip() == begin:
            if inside:
                raise ValueError("nested HOMYMOLY crontab begin marker")
            inside = True
            blocks += 1
            if blocks > 1:
                raise ValueError("multiple HOMYMOLY crontab blocks found")
            continue
        if line.strip() == end:
            if not inside:
                raise ValueError("unmatched HOMYMOLY crontab end marker")
            inside = False
            continue
        if not inside:
            result.append(line)
    if inside:
        raise ValueError("unmatched HOMYMOLY crontab begin marker")
    return result


def _write_crontab(document: str, *, original: str, backup_directory: Path) -> None:
    if _existing_crontab() != original:
        raise RuntimeError("crontab changed concurrently; refusing to overwrite it")
    backup_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup = backup_directory / f"crontab-{timestamp}.backup"
    with backup.open("x", encoding="utf-8") as handle:
        handle.write(original)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(backup_directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    subprocess.run(["crontab", "-"], input=document, text=True, check=True)
    installed = _existing_crontab()
    if installed != document:
        raise RuntimeError(
            "crontab post-write verification failed; inspect the retained backup"
        )


def _cron_safe(value: str, *, label: str) -> str:
    if not value or any(ord(character) < 32 for character in value):
        raise ValueError(f"{label} must be non-empty and contain no control characters")
    if "%" in value:
        raise ValueError(f"{label} cannot contain '%' because cron rewrites it")
    return value


def _global_crontab_lock_path() -> Path:
    return Path("/tmp") / f"homymoly-crontab-{os.getuid()}.lock"


def _global_crontab_lock() -> TextIO:
    path = _global_crontab_lock_path()
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
    ):
        os.close(descriptor)
        raise RuntimeError(f"unsafe global crontab lock: {path}")
    return os.fdopen(descriptor, "a+", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument(
        "--campaign-manifest",
        type=Path,
        help="manage a campaign-specific block instead of the legacy Gate-2 block",
    )
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--max-utilization", type=int)
    parser.add_argument("--max-background-processes", type=int)
    parser.add_argument("--max-background-memory-mib", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--sample-interval-seconds", type=float)
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.interval_minutes <= 59:
        parser.error("--interval-minutes must lie between 1 and 59")
    root = args.project_root.expanduser().resolve()
    try:
        _cron_safe(str(root), label="project root")
    except ValueError as exc:
        parser.error(str(exc))
    campaign_manifest: Path | None = None
    campaign_id: str | None = None
    policy = dict(DEFAULT_IDLE_POLICY)
    if args.campaign_manifest is not None:
        try:
            campaign_manifest, campaign_id, policy = _campaign_identity(
                args.campaign_manifest.expanduser(),
                root,
                allow_disabled=args.remove,
            )
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
        assert campaign_id is not None
        begin, end = _campaign_markers(campaign_id)
        backup_directory = root / "artifacts" / "scheduler" / campaign_id
        label = f"HOMYMOLY campaign {campaign_id}"
    else:
        begin, end = BEGIN, END
        backup_directory = root / "artifacts" / "gate2" / "scheduler"
        label = "HOMYMOLY Gate-2"
    for name in DEFAULT_IDLE_POLICY:
        if getattr(args, name) is None:
            setattr(args, name, policy[name])
    if args.gpu_index < 0:
        parser.error("--gpu-index must be nonnegative")
    if not 0 <= args.max_utilization <= 100:
        parser.error("--max-utilization must lie between 0 and 100")
    if args.max_background_processes < 0:
        parser.error("--max-background-processes must be nonnegative")
    if args.max_background_memory_mib < 0:
        parser.error("--max-background-memory-mib must be nonnegative")
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.sample_interval_seconds < 0 or not math.isfinite(
        args.sample_interval_seconds
    ):
        parser.error("--sample-interval-seconds must be finite and nonnegative")
    with _global_crontab_lock() as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        original = _existing_crontab()
        try:
            lines = _without_managed_block(original, begin=begin, end=end)
        except ValueError as exc:
            parser.error(str(exc))
        while lines and not lines[-1].strip():
            lines.pop()
        if args.remove:
            document = "\n".join(lines) + ("\n" if lines else "")
            if args.print_only:
                print(document, end="")
                return 0
            _write_crontab(
                document,
                original=original,
                backup_directory=backup_directory,
            )
            print(f"removed {label} cron entry")
            return 0

        python = root / ".venv" / "bin" / "python"
        monitor = root / "scripts" / "gpu_idle_train.py"
        if not python.is_file():
            parser.error(f"virtual-environment Python does not exist: {python}")
        if not monitor.is_file():
            parser.error(f"GPU monitor does not exist: {monitor}")
        try:
            for path, path_label in (
                (python, "virtual-environment Python"),
                (monitor, "GPU monitor"),
                (
                    campaign_manifest
                    if campaign_manifest is not None
                    else root / "configs" / "gate2.yaml",
                    "launch configuration",
                ),
            ):
                assert path is not None
                _cron_safe(str(path), label=path_label)
        except ValueError as exc:
            parser.error(str(exc))

        schedule = f"*/{args.interval_minutes} * * * *"
        launch_selector = (
            ("--campaign-manifest", shlex.quote(str(campaign_manifest)))
            if campaign_manifest is not None
            else (
                "--config",
                shlex.quote(str(root / "configs" / "gate2.yaml")),
            )
        )
        command_tokens = (
            "cd",
            shlex.quote(str(root)),
            "&&",
            shlex.quote(str(python)),
            shlex.quote(str(monitor)),
            "--project-root",
            shlex.quote(str(root)),
            *launch_selector,
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
            str(args.sample_interval_seconds),
        )
        command = " ".join(command_tokens)
        try:
            _cron_safe(command, label="cron command")
        except ValueError as exc:
            parser.error(str(exc))
        managed = [begin, f"{schedule} {command}", end]
        if lines:
            lines.append("")
        lines.extend(managed)
        document = "\n".join(lines) + "\n"
        if args.print_only:
            print(document, end="")
            return 0

        _write_crontab(
            document,
            original=original,
            backup_directory=backup_directory,
        )
        print(managed[1])
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
