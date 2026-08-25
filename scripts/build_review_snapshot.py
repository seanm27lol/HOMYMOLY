#!/usr/bin/env python3
"""Build the read-only snapshot promised to editors and reviewers.

Section 11 of the paper states that the complete source, configurations, frozen
protocols, and evidence bundle are supplied to the handling editor as a read-only
snapshot at the reviewed commit. This produces exactly that, locally: nothing is
uploaded, published, or sent anywhere.

The snapshot is a ``git archive`` of one commit plus a ``REVIEW.md`` describing
how to verify it. The tracked ``results/`` bundle is already under version
control, so it travels with the archive; the untracked multi-gigabyte
``artifacts/`` tree does not, which is intended.

The build refuses a dirty worktree by default, because a snapshot that cannot be
tied to a commit cannot be re-derived by a reviewer.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REVIEW_NOTE = """# Review snapshot

Source, configurations, frozen protocols, and the tracked evidence bundle for
the manuscript, captured at a single commit.

- commit: `{revision}`
- evidence manifest SHA-256: `{manifest_sha256}`
- tracked evidence files: {evidence_files}

## Licence

Proprietary. Copyright (c) 2026 Sean Mahdavian, all rights reserved. See
`LICENSE`. **This snapshot is provided for peer review only** and confers no
licence to redistribute or reuse the material.

## What this snapshot can verify and rerun

Every reported number can be verified against the compact, checksummed evidence
in `results/`. The snapshot also contains the source, configurations, and frozen
protocols needed to inspect the analyses and rerun the CPU conversion campaign.
Integrity verification itself requires neither network access nor a GPU. The
tests and CPU campaign also require no GPU, but do require a compatible Python
environment; installing that environment may require network access.

```bash
uv sync --frozen --extra dev --python 3.12.3
.venv/bin/python -m pytest -q
.venv/bin/python scripts/export_publication_evidence.py --verify-only
```

The builder also re-hashes every manifest entry from the archived commit's git
objects before packaging and refuses a mismatch, so this snapshot's `results/`
bundle passed manifest verification at the recorded commit.

The last command re-hashes every tracked evidence file and its retained source
where available, then checks the bundle against `results/MANIFEST.json`. It
prints `{{"verified": true, ...}}` on success. This is an integrity check, not a
claim that all training runs can be reconstructed from the compact bundle.

To regenerate the confirmatory campaign from scratch, on CPU:

```bash
env CUDA_VISIBLE_DEVICES=-1 .venv/bin/python \
  scripts/run_conversion_campaign.py --output /tmp/campaign.json
```

The runner fails before fitting unless `uv.lock`, Python, NetworkX, NumPy, the
base Torch version, the generator, and the protocol match the recorded campaign
environment. `CUDA_VISIBLE_DEVICES=-1` keeps this CPU campaign independent of a
busy accelerator.

## What requires separately supplied raw artifacts

The untracked `artifacts/` tree, roughly 8.8 GB of checkpoints, per-example
prediction dumps, training histories, and scheduler logs. Those artifacts are
pinned by SHA-256 from the tracked bundle but are not distributed. Consequently,
the completed GB10 training campaigns and checkpoint-dependent benchmarks cannot
be rerun from this snapshot alone; reproducing those runs requires the separately
supplied raw artifacts and compatible GB10 hardware/software. Their reported
outputs remain auditable in the compact evidence included here.
"""


def _git(project_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_file(project_root: Path, revision: str, relative_path: str) -> bytes:
    """Read a tracked file from the exact revision being archived."""

    result = subprocess.run(
        ("git", "show", f"{revision}:{relative_path}"),
        cwd=project_root,
        check=False,
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise FileNotFoundError(
            f"{relative_path} is missing from archived revision {revision}"
        )
    return result.stdout


def _verify_archived_bundle(
    project_root: Path, revision: str, manifest_bytes: bytes
) -> None:
    """Refuse to ship a snapshot whose archived evidence fails its manifest.

    The exporter's ``--verify-only`` rechecks a working tree; a snapshot is
    built from an arbitrary revision, so every manifest entry is instead
    re-hashed straight from that revision's git objects. Sources under the
    untracked ``artifacts/`` tree are absent from git by design and are skipped.
    """

    manifest = json.loads(manifest_bytes)
    problems: list[str] = []
    for entry in manifest.get("files", []):
        relative = entry.get("path")
        try:
            blob = _git_file(project_root, revision, f"results/{relative}")
        except FileNotFoundError:
            problems.append(f"missing at {revision[:8]}: results/{relative}")
            continue
        digest = hashlib.sha256(blob).hexdigest()
        if digest != entry.get("sha256"):
            problems.append(f"hash mismatch at {revision[:8]}: results/{relative}")
        if len(blob) != entry.get("bytes"):
            problems.append(f"byte count mismatch at {revision[:8]}: results/{relative}")
        source = entry.get("source")
        source_sha256 = entry.get("source_sha256")
        if source and source_sha256:
            try:
                source_blob = _git_file(project_root, revision, source)
            except FileNotFoundError:
                continue  # untracked raw-artifact source, excluded by design
            if hashlib.sha256(source_blob).hexdigest() != source_sha256:
                problems.append(f"source mismatch at {revision[:8]}: {source}")
    if problems:
        raise RuntimeError(
            "the archived evidence fails manifest verification: "
            + "; ".join(problems[:10])
        )


def build_snapshot(
    *,
    project_root: Path,
    output: Path,
    revision: str = "HEAD",
    allow_dirty: bool = False,
) -> dict[str, object]:
    project_root = project_root.resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    if not allow_dirty and _git(project_root, "status", "--short"):
        raise RuntimeError(
            "the worktree is dirty; a snapshot that cannot be tied to a commit "
            "cannot be re-derived by a reviewer. Commit first, or pass "
            "--allow-dirty and accept that the archive is not reproducible."
        )
    resolved = _git(project_root, "rev-parse", revision)

    manifest_bytes = _git_file(project_root, resolved, "results/MANIFEST.json")
    _verify_archived_bundle(project_root, resolved, manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    evidence_files = len(json.loads(manifest_bytes)["files"])

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".review-", dir=output.parent) as staging:
        inner = Path(staging) / "source.tar"
        subprocess.run(
            (
                "git",
                "archive",
                "--format=tar",
                f"--prefix=review-{resolved[:8]}/",
                "-o",
                str(inner),
                resolved,
            ),
            cwd=project_root,
            check=True,
            timeout=300,
        )
        note = Path(staging) / "REVIEW.md"
        note.write_text(
            REVIEW_NOTE.format(
                revision=resolved,
                manifest_sha256=manifest_sha256,
                evidence_files=evidence_files,
            ),
            encoding="utf-8",
        )
        with tarfile.open(inner, "a") as archive:
            archive.add(note, arcname=f"review-{resolved[:8]}/REVIEW.md")
        # Compress the assembled tar by streaming, so a large archive never has
        # to be held in memory.
        with inner.open("rb") as source, gzip.open(output, "wb") as target:
            shutil.copyfileobj(source, target)

    return {
        "output": str(output),
        "revision": resolved,
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "manifest_sha256": manifest_sha256,
        "evidence_files": evidence_files,
    }


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = build_snapshot(
            project_root=args.project_root,
            output=args.output,
            revision=args.revision,
            allow_dirty=args.allow_dirty,
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"review snapshot failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
