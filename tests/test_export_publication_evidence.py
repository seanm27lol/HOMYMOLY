from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "export_publication_evidence.py"
SPEC = importlib.util.spec_from_file_location("export_publication_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


COMMIT = "8021292e97abfec91768f1b5437c883a42c29c60"


def _corruption_report() -> dict[str, object]:
    return {
        "schema_version": 3,
        "checkpoint_sha256": "a" * 64,
        "git": {"commit": COMMIT, "status": ""},
        "analysis": {"transport_rotation": {"batch_observations": 2}},
        "per_batch": [
            {"block_id": "transport_rotation:0000", "damage_rate": 0.0},
            {"block_id": "transport_rotation:0001", "damage_rate": 0.5},
        ],
        "per_example": [{"sample_id": index} for index in range(64)],
    }


def _project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    (project / "artifacts" / "gate3g" / "run").mkdir(parents=True)
    (project / "results" / "summaries").mkdir(parents=True)
    (project / "artifacts" / "gate3g" / "run" / "corruption_report_final.json").write_text(
        json.dumps(_corruption_report(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (project / "artifacts" / "summary.json").write_text(
        json.dumps({"mean": 0.5, "shared_git_revision": COMMIT}, sort_keys=True),
        encoding="utf-8",
    )
    (project / "results" / "summaries" / "generated.json").write_text(
        json.dumps(
            {
                "validated_pairs": 8,
                "analysis_provenance": {"git_commit": COMMIT},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return project, project / "results"


def _specs() -> list[object]:
    return [
        MODULE.Spec(
            "results/summaries/generated.json",
            "summaries/generated.json",
            "compact-summary",
            "in-place",
            "Summary generated before export.",
        ),
        MODULE.Spec(
            "artifacts/summary.json",
            "summaries/copied.json",
            "endpoint-table",
            "copy",
            "Copied endpoint table.",
        ),
        MODULE.Spec(
            "artifacts/gate3g/run/corruption_report_final.json",
            "gate3g/run/corruption_report_final.compact.json",
            "corruption-report-derivative",
            "derive-corruption-report",
            "Compact corruption derivative.",
        ),
    ]


def test_compact_derivative_drops_per_example_and_keeps_the_analysis_unit() -> None:
    compact = MODULE.compact_corruption_report(_corruption_report())

    assert "per_example" not in compact
    assert len(compact["per_batch"]) == 2
    assert compact["analysis"] == {"transport_rotation": {"batch_observations": 2}}
    assert compact["checkpoint_sha256"] == "a" * 64
    assert compact["_derivative"]["dropped_keys"] == {"per_example": {"rows": 64}}
    assert compact["_derivative"]["retained_per_batch_rows"] == 2


def test_compact_derivative_refuses_a_report_without_batch_rows() -> None:
    document = _corruption_report()
    del document["per_batch"]

    with pytest.raises(ValueError, match="no per_batch rows"):
        MODULE.compact_corruption_report(document)


def test_export_writes_a_checksummed_manifest_for_every_allowlisted_file(
    tmp_path: Path,
) -> None:
    project, output = _project(tmp_path)

    manifest = MODULE.export(
        project_root=project, output_root=output, specs=_specs()
    )

    assert manifest["schema_version"] == MODULE.SCHEMA_VERSION
    assert manifest["summary"]["files"] == 3
    assert manifest["summary"]["by_kind"] == {
        "compact-summary": 1,
        "corruption-report-derivative": 1,
        "endpoint-table": 1,
    }
    assert manifest["generating_command"][1].endswith("export_publication_evidence.py")
    # Provenance is read out of the evidence, never from the exporter's own HEAD.
    assert manifest["source_commits"] == {COMMIT: 3}

    entries = {entry["path"]: entry for entry in manifest["files"]}
    derivative = output / "gate3g" / "run" / "corruption_report_final.compact.json"
    assert "per_example" not in json.loads(derivative.read_text(encoding="utf-8"))
    recorded = entries["gate3g/run/corruption_report_final.compact.json"]
    assert recorded["sha256"] == hashlib.sha256(derivative.read_bytes()).hexdigest()
    assert recorded["bytes"] == derivative.stat().st_size
    assert recorded["derivation"] == "per-batch-lossless-v1"
    # The untruncated source stays pinned even though it is not exported.
    source = project / "artifacts" / "gate3g" / "run" / "corruption_report_final.json"
    assert recorded["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert recorded["source_bytes"] == source.stat().st_size

    # Files already sitting at their destination are hashed, never rewritten.
    assert entries["summaries/generated.json"]["source"] == (
        "results/summaries/generated.json"
    )


def test_export_is_byte_identical_when_the_evidence_is_unchanged(
    tmp_path: Path,
) -> None:
    project, output = _project(tmp_path)

    MODULE.export(project_root=project, output_root=output, specs=_specs())
    first = (output / MODULE.MANIFEST_NAME).read_bytes()
    MODULE.export(project_root=project, output_root=output, specs=_specs())
    second = (output / MODULE.MANIFEST_NAME).read_bytes()

    assert first == second


def test_export_refuses_a_checkpoint_even_when_a_specification_asks_for_it(
    tmp_path: Path,
) -> None:
    project, output = _project(tmp_path)
    (project / "artifacts" / "model.pt").write_bytes(b"weights")
    specs = [
        MODULE.Spec(
            "artifacts/model.pt",
            "checkpoints/model.pt",
            "benchmark-summary",
            "copy",
            "Should never be exportable.",
        )
    ]

    with pytest.raises(ValueError, match="excluded suffix"):
        MODULE.export(project_root=project, output_root=output, specs=specs)


def test_export_refuses_a_source_under_an_excluded_directory(tmp_path: Path) -> None:
    project, output = _project(tmp_path)
    (project / "artifacts" / "run" / "checkpoints").mkdir(parents=True)
    (project / "artifacts" / "run" / "checkpoints" / "note.json").write_text(
        "{}", encoding="utf-8"
    )
    specs = [
        MODULE.Spec(
            "artifacts/run/checkpoints/note.json",
            "gate3/note.json",
            "gate-decision",
            "copy",
            "Should never be exportable.",
        )
    ]

    with pytest.raises(ValueError, match="excluded directory"):
        MODULE.export(project_root=project, output_root=output, specs=specs)


def test_export_refuses_a_file_over_the_per_file_cap(tmp_path: Path) -> None:
    project, output = _project(tmp_path)

    with pytest.raises(ValueError, match="per-file cap"):
        MODULE.export(
            project_root=project,
            output_root=output,
            specs=_specs(),
            max_file_bytes=8,
        )


def test_export_fails_loudly_when_required_evidence_is_missing(tmp_path: Path) -> None:
    project, output = _project(tmp_path)
    (project / "artifacts" / "summary.json").unlink()

    with pytest.raises(FileNotFoundError, match="required evidence is missing"):
        MODULE.export(project_root=project, output_root=output, specs=_specs())


def test_export_refuses_evidence_with_no_recorded_generating_commit(
    tmp_path: Path,
) -> None:
    project, output = _project(tmp_path)
    (project / "artifacts" / "summary.json").write_text(
        json.dumps({"mean": 0.5}, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="no recorded generating commit"):
        MODULE.export(project_root=project, output_root=output, specs=_specs())


def test_export_records_more_than_one_generating_commit_when_campaigns_differ(
    tmp_path: Path,
) -> None:
    project, output = _project(tmp_path)
    earlier = "e69b07707950b6abe332366c51fe8c94254899f3"
    (project / "artifacts" / "summary.json").write_text(
        json.dumps({"mean": 0.5, "shared_git_revision": earlier}, sort_keys=True),
        encoding="utf-8",
    )

    manifest = MODULE.export(project_root=project, output_root=output, specs=_specs())

    assert manifest["source_commits"] == {earlier: 1, COMMIT: 2}


def test_verify_detects_a_tampered_bundle_file(tmp_path: Path) -> None:
    project, output = _project(tmp_path)
    MODULE.export(project_root=project, output_root=output, specs=_specs())
    (output / "summaries" / "copied.json").write_text('{"mean": 0.9}', encoding="utf-8")

    problems = MODULE.verify(project, output)

    assert any("hash mismatch: summaries/copied.json" in problem for problem in problems)


def test_verify_detects_an_unlisted_file_and_a_changed_source(tmp_path: Path) -> None:
    project, output = _project(tmp_path)
    MODULE.export(project_root=project, output_root=output, specs=_specs())
    (output / "stray.json").write_text("{}", encoding="utf-8")
    (project / "artifacts" / "summary.json").write_text(
        json.dumps({"mean": 0.9}, sort_keys=True), encoding="utf-8"
    )

    problems = MODULE.verify(project, output)

    assert any("unlisted file in bundle: stray.json" in problem for problem in problems)
    assert any("source changed since export" in problem for problem in problems)


def test_the_real_allowlist_names_only_publishable_evidence() -> None:
    specs = MODULE.specifications()

    assert len(specs) == 48
    for spec in specs:
        assert Path(spec.source).suffix == ".json"
        assert Path(spec.destination).suffix == ".json"
        MODULE._reject_denied(Path(spec.source), "source")
        MODULE._reject_denied(Path(spec.destination), "destination")
    kinds = {spec.kind for spec in specs}
    assert kinds == {
        "benchmark-summary",
        "compact-summary",
        "corruption-report-derivative",
        "endpoint-table",
        "gate-decision",
    }
    # Every gauge seed contributes both trained runs and exactly one comparison.
    derivatives = [spec for spec in specs if spec.destination.startswith("gate3g/")]
    assert sum(spec.kind == "corruption-report-derivative" for spec in derivatives) == 16
    assert sum(spec.kind == "gate-decision" for spec in derivatives) == 8
