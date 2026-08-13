from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "export_artifact_bundle.py"
SPEC = importlib.util.spec_from_file_location("export_artifact_bundle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    artifacts = project / "artifacts"
    (project / "configs").mkdir(parents=True)
    (project / "src" / "homymoly").mkdir(parents=True)
    (project / "scripts").mkdir()
    artifacts.mkdir()
    (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (project / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (project / "configs" / "run.yaml").write_text("seed: 1\n", encoding="utf-8")
    (project / "src" / "homymoly" / "model.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    return project, artifacts


def test_export_copies_and_hashes_only_bounded_text_evidence(tmp_path: Path) -> None:
    project, artifacts = _project(tmp_path)
    run = artifacts / "run"
    run.mkdir()
    summary = run / "summary.json"
    summary.write_text('{"accuracy": 0.75}\n', encoding="utf-8")
    (run / "checkpoint.pt").write_bytes(b"checkpoint")
    (run / "oversized.log").write_text("x" * 128, encoding="utf-8")
    output = tmp_path / "bundle"

    manifest = MODULE.export_bundle(
        project_root=project,
        artifact_root=artifacts,
        output=output,
        max_file_bytes=64,
        max_total_bytes=4096,
    )

    exported = output / "files" / "artifacts" / "run" / "summary.json"
    assert exported.read_bytes() == summary.read_bytes()
    entries = {entry["path"]: entry for entry in manifest["artifacts"]}
    assert (
        entries["run/summary.json"]["sha256"]
        == hashlib.sha256(summary.read_bytes()).hexdigest()
    )
    assert entries["run/checkpoint.pt"]["reason"] == "unsupported-extension"
    assert entries["run/oversized.log"]["reason"] == "file-size-limit"
    assert (output / "files" / "repository" / "configs" / "run.yaml").is_file()
    stored = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert stored["schema_version"] == 1
    assert stored["execution_snapshot"]
    assert stored["runtime"]["distributions"]


def test_export_rejects_destination_inside_artifacts(tmp_path: Path) -> None:
    project, artifacts = _project(tmp_path)
    with pytest.raises(ValueError, match="outside the artifact tree"):
        MODULE.export_bundle(
            project_root=project,
            artifact_root=artifacts,
            output=artifacts / "bundle",
            max_file_bytes=64,
            max_total_bytes=4096,
        )


def test_export_refuses_to_overwrite_existing_destination(tmp_path: Path) -> None:
    project, artifacts = _project(tmp_path)
    output = tmp_path / "bundle"
    output.mkdir()
    with pytest.raises(FileExistsError):
        MODULE.export_bundle(
            project_root=project,
            artifact_root=artifacts,
            output=output,
            max_file_bytes=64,
            max_total_bytes=4096,
        )
