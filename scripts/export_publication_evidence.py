#!/usr/bin/env python3
"""Export the curated, tracked publication evidence bundle under ``results/``.

This exporter is deliberately different from ``scripts/export_artifact_bundle.py``.
That script inventories an entire artifact tree and copies whatever small text
files it finds; it is a reproducibility snapshot, not journal evidence. This
script instead works from an explicit allowlist: every exported file is named by
a specification below, every specification declares what kind of evidence it is,
and anything matching the denylist is refused even if a specification asks for it.

Large raw artifacts never enter the bundle. Checkpoints, per-example prediction
dumps, training histories, scheduler logs, caches, and environments are excluded
by construction. Corruption reports are exported as compact derivatives that drop
the ``per_example`` array and keep the ``per_batch`` array, which is the unit of
analysis: every published corruption statistic is recomputable from the retained
rows, and the SHA-256 of the untruncated source report is recorded so the dropped
detail remains pinned.

The manifest carries no timestamp on purpose. Re-running this exporter over
unchanged evidence produces a byte-identical bundle, so a tracked ``results/``
directory only changes when the evidence changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, NamedTuple

SCHEMA_VERSION = 1
BUNDLE_ROOT = "results"
MANIFEST_NAME = "MANIFEST.json"

# Suffixes and directory names that must never reach a tracked results bundle.
DENIED_SUFFIXES = frozenset(
    {
        ".bin",
        ".ckpt",
        ".gz",
        ".jsonl",
        ".log",
        ".npy",
        ".npz",
        ".pb",
        ".pt",
        ".pth",
        ".safetensors",
        ".tar",
        ".zip",
    }
)
DENIED_PARTS = frozenset(
    {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "checkpoints",
        "data",
        "logs",
        "metrics",
        "profiles",
        "steps",
        "tensorboard",
        "attempts",
    }
)
# A compact derivative must stay far below this; the cap is a tripwire, not a target.
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 48 * 1024 * 1024

DROPPED_CORRUPTION_KEYS = ("per_example",)


class Spec(NamedTuple):
    """One allowlisted evidence export."""

    source: str
    destination: str
    kind: str
    mode: str
    description: str


def specifications() -> list[Spec]:
    """The complete allowlist of publishable evidence."""

    specs: list[Spec] = [
        Spec(
            "results/summaries/gauge-corruption-campaign.json",
            "summaries/gauge-corruption-campaign.json",
            "compact-summary",
            "in-place",
            "Strict eight-seed gauge corruption summary (fixed-expert diagnostic).",
        ),
        Spec(
            "results/summaries/compute-campaign.json",
            "summaries/compute-campaign.json",
            "compact-summary",
            "in-place",
            "Strict trained GB10 compute-benchmark summary.",
        ),
        Spec(
            "artifacts/identifiable-maps/campaign-summary.json",
            "summaries/identifiable-campaign-summary.json",
            "compact-summary",
            "copy",
            "Strict 40-run identifiable typed-map campaign summary.",
        ),
        Spec(
            "artifacts/routing-confirmatory-v2-summary.json",
            "summaries/routing-confirmatory-v2-summary.json",
            "endpoint-table",
            "copy",
            "Frozen five-seed routing confirmatory endpoint table.",
        ),
        Spec(
            "artifacts/gate3/paired_comparison_final.json",
            "gate3/paired_comparison_final.json",
            "gate-decision",
            "copy",
            "Corrected Gate-3 base paired comparison across three candidate kinds.",
        ),
    ]
    for run in ("full", "plus-chain", "plus-recon", "task-only"):
        specs.append(
            Spec(
                f"artifacts/gate3/{run}/corruption_report_final.json",
                f"gate3/{run}/corruption_report_final.compact.json",
                "corruption-report-derivative",
                "derive-corruption-report",
                f"Gate-3 base final corruption report for {run}, per-batch rows only.",
            )
        )
    gauge_runs = ["gauge-task-only", "gauge-plus-chain"]
    gauge_runs += [
        f"{prefix}-s{index:02d}"
        for index in range(4, 11)
        for prefix in ("gauge-task-only", "gauge-plus-chain")
    ]
    for run in gauge_runs:
        specs.append(
            Spec(
                f"artifacts/gate3g/{run}/corruption_report_final.json",
                f"gate3g/{run}/corruption_report_final.compact.json",
                "corruption-report-derivative",
                "derive-corruption-report",
                f"Gauge final corruption report for {run}, per-batch rows only.",
            )
        )
    for index in ["", *[f"-s{value:02d}" for value in range(4, 11)]]:
        run = f"gauge-task-only{index}"
        specs.append(
            Spec(
                f"artifacts/gate3g/{run}/paired_comparison_final.json",
                f"gate3g/{run}/paired_comparison_final.json",
                "gate-decision",
                "copy",
                f"Seed-matched gauge paired comparison anchored at {run}.",
            )
        )
    for seed in range(1, 6):
        for ablation in ("combined", "task_reconstruction"):
            specs.append(
                Spec(
                    f"artifacts/identifiable-maps/benchmarks/gb10-s{seed}-{ablation}.json",
                    f"benchmarks/identifiable/gb10-s{seed}-{ablation}.json",
                    "benchmark-summary",
                    "copy",
                    f"Trained identifiable-map inference benchmark, seed {seed}, {ablation}.",
                )
            )
        specs.append(
            Spec(
                f"artifacts/benchmarks/routing-confirmatory-v2-s{seed}-compute.json",
                f"benchmarks/routing/routing-confirmatory-v2-s{seed}-compute.json",
                "benchmark-summary",
                "copy",
                f"Trained routing compute benchmark, seed {seed}.",
            )
        )
    return specs


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_denied(relative: Path, label: str) -> None:
    if relative.suffix.casefold() in DENIED_SUFFIXES:
        raise ValueError(f"{label} has an excluded suffix: {relative}")
    denied = DENIED_PARTS.intersection(relative.parts)
    if denied:
        raise ValueError(f"{label} lives under an excluded directory {sorted(denied)}: {relative}")


# Where each evidence shape records the revision that generated it. The first
# path that resolves wins; every exported file must agree on the result.
REVISION_PATHS = (
    ("analysis_provenance", "git_commit"),
    ("analysis_provenance", "shared_git_revision"),
    ("shared_provenance", "git_revision"),
    ("shared_git_revision",),
    ("provenance", "git_revision"),
    ("environment", "git_revision"),
    ("git", "commit"),
    ("inputs", "baseline", "git", "commit"),
)


def evidence_revision(document: Any) -> str | None:
    """Read the generating git revision out of one piece of evidence."""

    for path in REVISION_PATHS:
        cursor: Any = document
        for key in path:
            if not isinstance(cursor, dict) or key not in cursor:
                cursor = None
                break
            cursor = cursor[key]
        if isinstance(cursor, str) and cursor:
            return cursor
    return None


def compact_corruption_report(document: dict[str, Any]) -> dict[str, Any]:
    """Drop the per-example array and keep everything the analysis consumes.

    The published corruption statistics are computed over ``per_batch`` rows, so
    removing ``per_example`` changes no reported number. The derivative records
    which keys were dropped and how many rows each held.
    """

    if "per_batch" not in document:
        raise ValueError("corruption report has no per_batch rows to retain")
    compact = {key: value for key, value in document.items() if key not in DROPPED_CORRUPTION_KEYS}
    compact["_derivative"] = {
        "derivation": "per-batch-lossless-v1",
        "dropped_keys": {
            key: {"rows": len(document[key])}
            for key in DROPPED_CORRUPTION_KEYS
            if key in document
        },
        "retained_per_batch_rows": len(document["per_batch"]),
        "guarantee": (
            "Every published corruption statistic in this report is computed from "
            "the retained per_batch rows; the dropped arrays are per-example detail "
            "that no reported number depends on."
        ),
    }
    return compact


def export(
    *,
    project_root: Path,
    output_root: Path,
    specs: list[Spec] | None = None,
    max_file_bytes: int = MAX_FILE_BYTES,
    max_total_bytes: int = MAX_TOTAL_BYTES,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()
    specs = specifications() if specs is None else specs

    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for spec in specs:
        source = project_root / spec.source
        destination_relative = Path(spec.destination)
        _reject_denied(Path(spec.source), "source")
        _reject_denied(destination_relative, "destination")
        if not source.is_file():
            raise FileNotFoundError(f"required evidence is missing: {spec.source}")

        destination = output_root / destination_relative
        source_sha256 = _sha256(source)
        entry: dict[str, Any] = {
            "path": destination_relative.as_posix(),
            "kind": spec.kind,
            "description": spec.description,
            "source": spec.source,
            "source_sha256": source_sha256,
            "derivation": None,
        }

        if spec.mode == "in-place":
            if destination.resolve() != source.resolve():
                raise ValueError(
                    f"in-place evidence must already sit at its destination: {spec.source}"
                )
            payload = source.read_bytes()
        elif spec.mode == "copy":
            payload = source.read_bytes()
        elif spec.mode == "derive-corruption-report":
            document = json.loads(source.read_text(encoding="utf-8"))
            compact = compact_corruption_report(document)
            payload = (json.dumps(compact, indent=2, sort_keys=True) + "\n").encode()
            entry["derivation"] = compact["_derivative"]["derivation"]
            entry["source_bytes"] = source.stat().st_size
        else:
            raise ValueError(f"unknown export mode: {spec.mode}")

        entry["evidence_revision"] = evidence_revision(json.loads(payload))

        if len(payload) > max_file_bytes:
            raise ValueError(
                f"exported evidence exceeds the per-file cap ({len(payload)} bytes): "
                f"{spec.destination}"
            )
        total_bytes += len(payload)
        if total_bytes > max_total_bytes:
            raise ValueError("exported bundle exceeds the total byte cap")

        if spec.mode != "in-place":
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
            try:
                temporary.write_bytes(payload)
                temporary.replace(destination)
            finally:
                if temporary.exists():
                    temporary.unlink()

        entry["bytes"] = len(payload)
        entry["sha256"] = _sha256_bytes(payload)
        entries.append(entry)

    entries.sort(key=lambda item: item["path"])
    kinds: dict[str, int] = {}
    for entry in entries:
        kinds[entry["kind"]] = kinds.get(entry["kind"], 0) + 1

    unattributed = [
        entry["path"] for entry in entries if entry["evidence_revision"] is None
    ]
    if unattributed:
        raise ValueError(
            f"exported evidence has no recorded generating commit: {unattributed}"
        )
    revision_counts: dict[str, int] = {}
    for entry in entries:
        revision = str(entry["evidence_revision"])
        revision_counts[revision] = revision_counts.get(revision, 0) + 1

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "bundle": BUNDLE_ROOT,
        # Commits are read from the evidence itself, never from the exporter's
        # own HEAD: a manifest that named its own commit could not be committed
        # without dangling. Every exported file must name one, and the campaigns
        # were frozen at different times, so more than one commit is expected.
        "source_commits": dict(sorted(revision_counts.items())),
        "source_commit_note": (
            "Generating revision recorded inside each exported artifact, with the "
            "number of exported files attributed to it. The routing confirmatory "
            "campaign was frozen before the identifiable-map campaign, so the two "
            "families legitimately carry different commits."
        ),
        "generating_command": [
            "python",
            "scripts/export_publication_evidence.py",
            "--output-root",
            BUNDLE_ROOT,
        ],
        "generating_pipeline": [
            [
                "python",
                "scripts/summarize_gauge_corruption_campaign.py",
                "--output",
                "results/summaries/gauge-corruption-campaign.json",
            ],
            [
                "python",
                "scripts/summarize_compute_campaign.py",
                "--output",
                "results/summaries/compute-campaign.json",
            ],
            [
                "python",
                "scripts/export_publication_evidence.py",
                "--output-root",
                BUNDLE_ROOT,
            ],
        ],
        "exclusions": {
            "denied_suffixes": sorted(DENIED_SUFFIXES),
            "denied_directories": sorted(DENIED_PARTS),
            "policy": (
                "Checkpoints, per-example prediction dumps, training histories, "
                "scheduler logs, caches, and environments are excluded by "
                "construction and are not journal evidence. The untracked "
                "/artifacts/ tree remains the only home for those files."
            ),
        },
        "determinism": (
            "This manifest carries no timestamp; re-exporting unchanged evidence "
            "reproduces it byte for byte."
        ),
        "summary": {
            "files": len(entries),
            "bytes": total_bytes,
            "by_kind": dict(sorted(kinds.items())),
        },
        "files": entries,
    }

    manifest_path = output_root / MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    temporary = manifest_path.with_suffix(manifest_path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(manifest_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest


def verify(project_root: Path, output_root: Path) -> list[str]:
    """Recheck an exported bundle against its own manifest."""

    manifest = json.loads((output_root / MANIFEST_NAME).read_text(encoding="utf-8"))
    problems: list[str] = []
    for entry in manifest["files"]:
        path = output_root / entry["path"]
        if not path.is_file():
            problems.append(f"missing: {entry['path']}")
            continue
        if _sha256(path) != entry["sha256"]:
            problems.append(f"hash mismatch: {entry['path']}")
        if path.stat().st_size != entry["bytes"]:
            problems.append(f"byte count mismatch: {entry['path']}")
        source = project_root / entry["source"]
        if source.is_file() and _sha256(source) != entry["source_sha256"]:
            problems.append(f"source changed since export: {entry['source']}")
    tracked = {entry["path"] for entry in manifest["files"]} | {MANIFEST_NAME}
    for path in sorted(output_root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(output_root).as_posix()
            if relative not in tracked:
                problems.append(f"unlisted file in bundle: {relative}")
    return problems


def _parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--output-root", type=Path, default=project_root / BUNDLE_ROOT)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="recheck an existing bundle against its manifest without rewriting it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    try:
        if args.verify_only:
            problems = verify(project_root, output_root)
            for problem in problems:
                print(f"publication evidence problem: {problem}", file=sys.stderr)
            if problems:
                return 2
            print(json.dumps({"verified": True, "bundle": str(output_root)}))
            return 0
        manifest = export(project_root=project_root, output_root=output_root)
    except (OSError, ValueError) as exc:
        print(f"publication evidence export failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
