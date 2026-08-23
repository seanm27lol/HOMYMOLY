from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "install_training_cron.py"
SPEC = importlib.util.spec_from_file_location("install_training_cron", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_managed_block_is_replaced_without_touching_other_jobs() -> None:
    document = (
        f"MAILTO=user@example.com\n{MODULE.BEGIN}\n*/5 * * * * old-command\n"
        f"{MODULE.END}\n0 2 * * * backup"
    )
    assert MODULE._without_managed_block(document) == [
        "MAILTO=user@example.com",
        "0 2 * * * backup",
    ]


@pytest.mark.parametrize(
    "document",
    (
        f"keep\n{MODULE.BEGIN}\nunsafe",
        f"keep\n{MODULE.END}\nunsafe",
        f"{MODULE.BEGIN}\n{MODULE.BEGIN}\n{MODULE.END}\n{MODULE.END}",
    ),
)
def test_malformed_managed_blocks_are_rejected(document: str) -> None:
    with pytest.raises(ValueError):
        MODULE._without_managed_block(document)


def test_main_rejects_negative_background_threshold() -> None:
    with pytest.raises(SystemExit):
        MODULE.main(["--max-background-processes", "-1", "--print-only"])


def _campaign_project(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project with spaces"
    python = root / ".venv" / "bin" / "python"
    monitor = root / "scripts" / "gpu_idle_train.py"
    manifest = root / "configs" / "campaign.json"
    python.parent.mkdir(parents=True)
    monitor.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    monitor.write_text("", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "identifiable-gb10-v1",
                "execution_enabled": True,
                "max_attempts_per_step": 3,
                "idle_policy": {
                    "gpu_index": 0,
                    "max_utilization": 10,
                    "max_background_processes": 2,
                    "max_background_memory_mib": 49152,
                    "samples": 3,
                    "interval_seconds": 2.0,
                },
                "fingerprint_inputs": ["placeholder"],
                "steps": [{"id": "placeholder"}],
            }
        ),
        encoding="utf-8",
    )
    return root, manifest


def test_campaign_preview_preserves_gate2_and_unrelated_jobs(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    root, manifest = _campaign_project(tmp_path)
    original = (
        "MAILTO=user@example.com\n"
        f"{MODULE.BEGIN}\n*/5 * * * * legacy-gate2\n{MODULE.END}\n"
        "0 2 * * * unrelated-backup\n"
    )
    monkeypatch.setattr(MODULE, "_existing_crontab", lambda: original)
    assert (
        MODULE.main(
            [
                "--project-root",
                str(root),
                "--campaign-manifest",
                str(manifest),
                "--print-only",
            ]
        )
        == 0
    )
    preview = capsys.readouterr().out
    assert "legacy-gate2" in preview
    assert "unrelated-backup" in preview
    begin, end = MODULE._campaign_markers("identifiable-gb10-v1")
    assert begin in preview and end in preview
    assert "--campaign-manifest" in preview
    assert "--max-background-processes 2" in preview
    assert "--max-background-memory-mib 49152" in preview
    assert "'" + str(root) + "'" in preview


def test_removing_one_campaign_block_preserves_everything_else() -> None:
    first_begin, first_end = MODULE._campaign_markers("first")
    second_begin, second_end = MODULE._campaign_markers("second")
    document = (
        f"{MODULE.BEGIN}\ngate2\n{MODULE.END}\n"
        f"{first_begin}\nfirst-command\n{first_end}\n"
        f"{second_begin}\nsecond-command\n{second_end}\n"
        "unrelated"
    )
    result = MODULE._without_managed_block(document, begin=first_begin, end=first_end)
    joined = "\n".join(result)
    assert "first-command" not in joined
    assert "gate2" in joined
    assert "second-command" in joined
    assert "unrelated" in joined


def test_disabled_campaign_can_only_be_resolved_for_removal(tmp_path: Path) -> None:
    root, manifest = _campaign_project(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["execution_enabled"] = False
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="execution is disabled"):
        MODULE._campaign_identity(manifest, root)
    _, campaign_id, _ = MODULE._campaign_identity(manifest, root, allow_disabled=True)
    assert campaign_id == "identifiable-gb10-v1"


def test_campaign_identity_requires_explicit_execution_enabled(tmp_path: Path) -> None:
    root, manifest = _campaign_project(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["execution_enabled"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must be explicit"):
        MODULE._campaign_identity(manifest, root, allow_disabled=True)


@pytest.mark.parametrize("value", ("line\nbreak", "percent%path", "tab\tpath"))
def test_cron_safe_rejects_control_and_percent_hazards(value: str) -> None:
    with pytest.raises(ValueError):
        MODULE._cron_safe(value, label="test value")


def test_all_managed_blocks_share_one_per_user_lock() -> None:
    assert MODULE._global_crontab_lock_path() == (
        Path("/tmp") / f"homymoly-crontab-{os.getuid()}.lock"
    )


def test_crontab_write_revalidates_installed_document(
    tmp_path: Path, monkeypatch
) -> None:
    observed = iter(("old-job\n", "different-job\n"))
    monkeypatch.setattr(MODULE, "_existing_crontab", lambda: next(observed))
    monkeypatch.setattr(MODULE.subprocess, "run", lambda *_a, **_k: None)
    with pytest.raises(RuntimeError, match="post-write verification failed"):
        MODULE._write_crontab(
            "new-job\n",
            original="old-job\n",
            backup_directory=tmp_path / "backups",
        )
    [backup] = list((tmp_path / "backups").glob("*.backup"))
    assert backup.read_text(encoding="utf-8") == "old-job\n"
