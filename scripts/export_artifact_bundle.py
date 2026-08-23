#!/usr/bin/env python3
"""Export a compact, checksummed HOMYMOLY reproducibility bundle.

Large checkpoints and datasets are inventoried by path and size but are never
read or copied. Small textual result/configuration files are copied under the
bundle's ``files/`` directory and receive SHA-256 checksums in ``manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPORTABLE_SUFFIXES = frozenset(
    {".csv", ".json", ".jsonl", ".log", ".tsv", ".txt", ".yaml", ".yml"}
)
REPOSITORY_METADATA_NAMES = (
    "Dockerfile",
    "docker-compose.yml",
    "pyproject.toml",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_metadata(project_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, arguments in (
        ("commit", ("git", "rev-parse", "HEAD")),
        ("status", ("git", "status", "--short")),
    ):
        try:
            process = subprocess.run(
                arguments,
                cwd=project_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            result[f"{key}_error"] = str(exc)
        else:
            if process.returncode == 0:
                result[key] = process.stdout.strip()
            else:
                result[f"{key}_error"] = process.stderr.strip()
    return result


def _distribution_inventory() -> list[dict[str, str]]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.casefold()] = distribution.version
    return [{"name": name, "version": packages[name]} for name in sorted(packages)]


def _repository_metadata_files(project_root: Path) -> list[Path]:
    paths = [project_root / name for name in REPOSITORY_METADATA_NAMES]
    for directory, patterns in (
        (project_root / "configs", ("*.yaml", "*.yml")),
        (project_root / "constraints", ("*.txt",)),
    ):
        if directory.is_dir():
            for pattern in patterns:
                paths.extend(directory.rglob(pattern))
    return sorted(path for path in paths if path.is_file())


def _execution_snapshot(project_root: Path) -> list[dict[str, Any]]:
    paths = _repository_metadata_files(project_root)
    for directory, patterns in (
        (project_root / "src" / "homymoly", ("*.py",)),
        (project_root / "scripts", ("*.py", "*.sh")),
    ):
        if directory.is_dir():
            for pattern in patterns:
                paths.extend(directory.rglob(pattern))
    return [
        {
            "path": path.relative_to(project_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(set(paths))
    ]


def _copy_checked(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return {
        "bytes": source.stat().st_size,
        "sha256": _sha256(destination),
    }


def _relative_display(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.name


def export_bundle(
    *,
    project_root: Path,
    artifact_root: Path,
    output: Path,
    max_file_bytes: int,
    max_total_bytes: int,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    artifact_root = artifact_root.resolve()
    output = output.resolve()
    if not artifact_root.is_dir():
        raise ValueError(f"artifact root does not exist: {artifact_root}")
    if max_file_bytes <= 0 or max_total_bytes <= 0:
        raise ValueError("file and total byte limits must be positive")
    if output == artifact_root or output.is_relative_to(artifact_root):
        raise ValueError("output must be outside the artifact tree")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    exported_bytes = 0
    entries: list[dict[str, Any]] = []
    try:
        temporary.mkdir()
        for source in _repository_metadata_files(project_root):
            relative = source.relative_to(project_root)
            destination = temporary / "files" / "repository" / relative
            copied = _copy_checked(source, destination)
            exported_bytes += copied["bytes"]

        for source in sorted(artifact_root.rglob("*")):
            relative = source.relative_to(artifact_root)
            entry: dict[str, Any] = {
                "path": relative.as_posix(),
                "exported": False,
            }
            if source.is_symlink():
                entry["reason"] = "symlink"
                entries.append(entry)
                continue
            if not source.is_file():
                continue
            size = source.stat().st_size
            entry["bytes"] = size
            if source.suffix.casefold() not in EXPORTABLE_SUFFIXES:
                entry["reason"] = "unsupported-extension"
            elif size > max_file_bytes:
                entry["reason"] = "file-size-limit"
            elif exported_bytes + size > max_total_bytes:
                entry["reason"] = "bundle-size-limit"
            else:
                destination = temporary / "files" / "artifacts" / relative
                entry.update(_copy_checked(source, destination))
                entry["exported"] = True
                exported_bytes += size
            entries.append(entry)

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "source_artifact_root": _relative_display(artifact_root, project_root),
            "limits": {
                "max_file_bytes": max_file_bytes,
                "max_total_bytes": max_total_bytes,
            },
            "summary": {
                "artifact_files": sum("bytes" in entry for entry in entries),
                "exported_artifact_files": sum(
                    bool(entry["exported"]) for entry in entries
                ),
                "exported_bytes": exported_bytes,
            },
            "git": _git_metadata(project_root),
            "runtime": {
                "machine": platform.machine(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "distributions": _distribution_inventory(),
            },
            "execution_snapshot": _execution_snapshot(project_root),
            "artifacts": entries,
        }
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-file-mib", type=float, default=8.0)
    parser.add_argument("--max-total-mib", type=float, default=256.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    artifact_root = (
        args.artifact_root.expanduser().resolve()
        if args.artifact_root is not None
        else project_root / "artifacts"
    )
    try:
        manifest = export_bundle(
            project_root=project_root,
            artifact_root=artifact_root,
            output=args.output.expanduser(),
            max_file_bytes=int(args.max_file_mib * 1024 * 1024),
            max_total_bytes=int(args.max_total_mib * 1024 * 1024),
        )
    except (OSError, ValueError) as exc:
        print(f"artifact export failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
