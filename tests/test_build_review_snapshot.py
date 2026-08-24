from __future__ import annotations

import importlib.util
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "build_review_snapshot.py"
SPEC = importlib.util.spec_from_file_location("build_review_snapshot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _repository(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "results").mkdir(parents=True)
    (project / "scripts").mkdir()
    (project / "LICENSE").write_text("Proprietary.\n", encoding="utf-8")
    (project / "scripts" / "thing.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / "results" / "MANIFEST.json").write_text(
        json.dumps({"files": [{"path": "a.json"}, {"path": "b.json"}]}),
        encoding="utf-8",
    )
    (project / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    (project / "artifacts").mkdir()
    (project / "artifacts" / "huge.pt").write_bytes(b"x" * 4096)
    for command in (
        ("git", "init", "-q"),
        ("git", "config", "user.email", "a@b.c"),
        ("git", "config", "user.name", "t"),
        ("git", "add", "-A"),
        ("git", "commit", "-qm", "initial"),
    ):
        subprocess.run(command, cwd=project, check=True, capture_output=True)
    return project


def test_snapshot_contains_the_source_and_the_evidence_bundle(tmp_path: Path) -> None:
    project = _repository(tmp_path)
    output = tmp_path / "snapshot.tar.gz"

    summary = MODULE.build_snapshot(project_root=project, output=output)

    assert output.is_file()
    assert summary["evidence_files"] == 2
    assert summary["sha256"]
    with tarfile.open(output, "r:gz") as archive:
        names = {Path(name).relative_to(Path(name).parts[0]).as_posix()
                 for name in archive.getnames() if "/" in name}
    assert "results/MANIFEST.json" in names
    assert "scripts/thing.py" in names
    assert "LICENSE" in names
    assert "REVIEW.md" in names


def test_the_untracked_artifact_tree_is_excluded(tmp_path: Path) -> None:
    """The 8.8 GB artifacts tree must never travel with a review snapshot."""

    project = _repository(tmp_path)
    output = tmp_path / "snapshot.tar.gz"

    MODULE.build_snapshot(project_root=project, output=output)

    with tarfile.open(output, "r:gz") as archive:
        assert not [name for name in archive.getnames() if "artifacts/" in name]


def test_review_note_records_the_commit_and_the_manifest_hash(tmp_path: Path) -> None:
    project = _repository(tmp_path)
    output = tmp_path / "snapshot.tar.gz"

    summary = MODULE.build_snapshot(project_root=project, output=output)

    with tarfile.open(output, "r:gz") as archive:
        member = next(n for n in archive.getnames() if n.endswith("REVIEW.md"))
        note = archive.extractfile(member).read().decode()
    assert summary["revision"] in note
    assert summary["manifest_sha256"] in note
    assert "peer review only" in note
    assert "--verify-only" in note


def test_a_dirty_worktree_is_refused(tmp_path: Path) -> None:
    """A snapshot that cannot be tied to a commit cannot be re-derived."""

    project = _repository(tmp_path)
    (project / "scripts" / "thing.py").write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="worktree is dirty"):
        MODULE.build_snapshot(project_root=project, output=tmp_path / "s.tar.gz")

    summary = MODULE.build_snapshot(
        project_root=project, output=tmp_path / "s.tar.gz", allow_dirty=True
    )
    assert summary["output"].endswith("s.tar.gz")


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    project = _repository(tmp_path)
    output = tmp_path / "snapshot.tar.gz"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        MODULE.build_snapshot(project_root=project, output=output)


def test_a_missing_manifest_is_reported(tmp_path: Path) -> None:
    project = _repository(tmp_path)
    (project / "results" / "MANIFEST.json").unlink()
    subprocess.run(("git", "add", "-A"), cwd=project, check=True, capture_output=True)
    subprocess.run(
        ("git", "commit", "-qm", "drop"), cwd=project, check=True, capture_output=True
    )

    with pytest.raises(FileNotFoundError, match="MANIFEST.json is missing"):
        MODULE.build_snapshot(project_root=project, output=tmp_path / "s.tar.gz")
