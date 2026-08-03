#!/usr/bin/env python3
"""Install or replace the user crontab entry for guarded Gate-2 training."""

from __future__ import annotations

import argparse
import fcntl
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

BEGIN = "# HOMYMOLY_GATE2_BEGIN"
END = "# HOMYMOLY_GATE2_END"


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


def _without_managed_block(document: str) -> list[str]:
    result: list[str] = []
    inside = False
    blocks = 0
    for line in document.splitlines():
        if line.strip() == BEGIN:
            if inside:
                raise ValueError("nested HOMYMOLY crontab begin marker")
            inside = True
            blocks += 1
            if blocks > 1:
                raise ValueError("multiple HOMYMOLY crontab blocks found")
            continue
        if line.strip() == END:
            if not inside:
                raise ValueError("unmatched HOMYMOLY crontab end marker")
            inside = False
            continue
        if not inside:
            result.append(line)
    if inside:
        raise ValueError("unmatched HOMYMOLY crontab begin marker")
    return result


def _write_crontab(document: str, *, original: str, root: Path) -> None:
    if _existing_crontab() != original:
        raise RuntimeError("crontab changed concurrently; refusing to overwrite it")
    backup_dir = root / "artifacts" / "gate2" / "scheduler"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup = backup_dir / f"crontab-{timestamp}.backup"
    backup.write_text(original, encoding="utf-8")
    subprocess.run(["crontab", "-"], input=document, text=True, check=True)


def main() -> int:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.interval_minutes <= 59:
        parser.error("--interval-minutes must lie between 1 and 59")
    if not 0 <= args.max_utilization <= 100:
        parser.error("--max-utilization must lie between 0 and 100")

    root = args.project_root.expanduser().resolve()
    lock_path = Path("/tmp/homymoly-gate2-crontab.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        original = _existing_crontab()
        try:
            lines = _without_managed_block(original)
        except ValueError as exc:
            parser.error(str(exc))
        while lines and not lines[-1].strip():
            lines.pop()
        if args.remove:
            document = "\n".join(lines) + ("\n" if lines else "")
            if args.print_only:
                print(document, end="")
                return 0
            _write_crontab(document, original=original, root=root)
            print("removed HOMYMOLY Gate-2 cron entry")
            return 0

        python = root / ".venv" / "bin" / "python"
        monitor = root / "scripts" / "gpu_idle_train.py"
        if not python.is_file():
            parser.error(f"virtual-environment Python does not exist: {python}")
        if not monitor.is_file():
            parser.error(f"GPU monitor does not exist: {monitor}")

        schedule = f"*/{args.interval_minutes} * * * *"
        command = " ".join(
            (
                "cd",
                shlex.quote(str(root)),
                "&&",
                shlex.quote(str(python)),
                shlex.quote(str(monitor)),
                "--project-root",
                shlex.quote(str(root)),
                "--config",
                shlex.quote(str(root / "configs" / "gate2.yaml")),
                "--max-utilization",
                str(args.max_utilization),
            )
        )
        managed = [BEGIN, f"{schedule} {command}", END]
        if lines:
            lines.append("")
        lines.extend(managed)
        document = "\n".join(lines) + "\n"
        if args.print_only:
            print(document, end="")
            return 0

        _write_crontab(document, original=original, root=root)
        print(managed[1])
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
