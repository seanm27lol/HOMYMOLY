#!/usr/bin/env python3
"""Validate and summarize the frozen identifiable-map campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

SCHEMA_VERSION = 1
PROTOCOL = "docs/21-identifiable-typed-map-protocol.md"
FROZEN_SOURCE_CONFIG_SHA256 = (
    "22abb205e8a89586b38799d7f7b8d53f0c24cef45f872453533ddf34e20fad73"
)
FROZEN_SEEDS = tuple(range(20260821, 20260826))
FROZEN_ABLATIONS = (
    "task_only",
    "reconstruction_only",
    "task_reconstruction",
    "task_reconstruction_cone",
    "task_reconstruction_rtd",
    "cone_only",
    "rtd_only",
    "combined",
)
FROZEN_SPLIT_SAMPLES = {"train": 4800, "validation": 1200, "test": 1200}
FROZEN_MAP_TOLERANCE = 1e-5
FROZEN_RTD_TRAINING_ENTITIES = 48
FROZEN_BASE_BETTI = [1, 1, 0]
ANALYTIC_MARKER_ACCURACY = 1.0
CHANCE_BASELINES = {
    "transformation_accuracy": 1.0 / 12.0,
    "cell_face_accuracy": 1.0 / 6.0,
}
ENGINEERING_GATE = {
    "transformation_accuracy_min": 0.95,
    "cell_face_accuracy_min": 0.95,
    "map_mse_max": 1e-3,
    "chain_residual_max": 1e-5,
    "hard_cone_acyclic_fraction_min": 1.0,
}

ENDPOINTS = {
    "transformation_accuracy": {
        "direction": "higher_is_better",
        "family": "identification",
    },
    "cell_face_accuracy": {
        "direction": "higher_is_better",
        "family": "cell_recovery",
    },
    "map_mse": {"direction": "lower_is_better", "family": "map_error"},
    "degree_zero_mse": {
        "direction": "lower_is_better",
        "family": "typed_reconstruction_error",
    },
    "degree_one_mse": {
        "direction": "lower_is_better",
        "family": "typed_reconstruction_error",
    },
    "degree_two_mse": {
        "direction": "lower_is_better",
        "family": "typed_reconstruction_error",
    },
    "sheaf_transport_frobenius_mse": {
        "direction": "lower_is_better",
        "family": "prespecified_descriptive_reconstruction_error",
    },
}

CONTRASTS = (
    {
        "name": "combined_minus_task_reconstruction",
        "candidate": "combined",
        "control": "task_reconstruction",
        "role": "descriptive_unadjusted_structural_plus_cone_plus_rtd",
    },
    {
        "name": "task_reconstruction_cone_minus_task_reconstruction",
        "candidate": "task_reconstruction_cone",
        "control": "task_reconstruction",
        "role": "descriptive_unadjusted_structural_plus_cone",
    },
    {
        "name": "task_reconstruction_rtd_minus_task_reconstruction",
        "candidate": "task_reconstruction_rtd",
        "control": "task_reconstruction",
        "role": "descriptive_unadjusted_structural_plus_rtd",
    },
)

_T_975_DF4 = 2.776445105
_REQUIRED_ARTIFACTS = {
    "effective_config.yaml",
    "provenance.json",
    "checkpoint.pt",
    "test_predictions.jsonl",
    "history.json",
    "summary.json",
}
_SUMMARY_KEYS = {
    "schema_version",
    "status",
    "experiment",
    "scope",
    "ablation",
    "loss_weights",
    "rtd_training_entities",
    "seed",
    "device",
    "best_epoch",
    "epochs_completed",
    "best_validation_objective",
    "engineering_recovery_gate",
    "elapsed_seconds",
    "dataset",
    "declared_chain_map_equations",
    "basis_chain_residual_max",
    "test",
    "environment",
}
_TEST_KEYS = {
    "examples",
    "transformation_accuracy",
    "analytic_marker_decoder_accuracy",
    "chance_baselines",
    "cell_face_accuracy",
    "map_mse",
    "degree_zero_mse",
    "degree_one_mse",
    "degree_two_mse",
    "sheaf_transport_frobenius_mse",
    "soft_chain_residual_max",
    "hard_chain_residual_max",
    "map_tolerance",
    "cone_rank_oracle",
    "soft_cone_betti_histogram",
    "hard_cone_betti_histogram",
    "exact_rtd",
}
_PREDICTION_KEYS = {
    "sample_id",
    "target_transformation",
    "predicted_transformation",
    "analytic_marker_transformation",
    "correct",
    "confidence",
    "soft_chain_residual_max",
    "hard_chain_residual_max",
    "soft_cone_betti",
    "hard_cone_betti",
    "map_mse",
    "cell_face_correct",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise RuntimeError(
            f"{path} schema mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid JSON in {path}: {error}") from error


def _read_yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))


def _finite(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{path} must be finite")
    return result


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer")
    return value


def _close(actual: float, expected: float, path: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-12):
        raise RuntimeError(f"{path} mismatch: recorded={actual}, recomputed={expected}")


def _resolve_recorded_path(value: object, working_directory: Path) -> Path:
    path = Path(str(value))
    return (
        (working_directory / path).resolve()
        if not path.is_absolute()
        else path.resolve()
    )


def _code_fingerprint(project_root: Path, runner: Path) -> str:
    """Mirror the training runner's executable-source fingerprint."""

    digest = hashlib.sha256()
    candidates = [runner.resolve()]
    candidates.extend((project_root / "src" / "homymoly").rglob("*.py"))
    for path in sorted(
        candidates, key=lambda item: item.relative_to(project_root).as_posix()
    ):
        relative = path.relative_to(project_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _require_git_commit(project_root: Path, revision: str) -> None:
    result = subprocess.run(
        ("git", "cat-file", "-e", f"{revision}^{{commit}}"),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"shared recorded Git revision is not a repository commit: {revision}"
        )


def _option(command: Sequence[object], name: str) -> str:
    values: list[str] = []
    command_text = [str(part) for part in command]
    for index, part in enumerate(command_text):
        if part == name:
            if index + 1 >= len(command_text):
                raise RuntimeError(f"recorded command has no value for {name}")
            values.append(command_text[index + 1])
        elif part.startswith(name + "="):
            values.append(part.split("=", 1)[1])
    if len(values) != 1:
        raise RuntimeError(
            f"recorded command must contain {name} exactly once; found {len(values)}"
        )
    return values[0]


def _validate_grid(campaign_root: Path) -> list[tuple[int, str, Path]]:
    if not campaign_root.is_dir() or campaign_root.is_symlink():
        raise FileNotFoundError(
            f"campaign root is not a real directory: {campaign_root}"
        )
    expected_seed_names = {f"seed-{seed}" for seed in FROZEN_SEEDS}
    actual_seed_names = {path.name for path in campaign_root.iterdir() if path.is_dir()}
    if actual_seed_names != expected_seed_names:
        raise RuntimeError(
            "campaign seed grid mismatch: "
            f"missing={sorted(expected_seed_names - actual_seed_names)}, "
            f"unexpected={sorted(actual_seed_names - expected_seed_names)}"
        )
    grid: list[tuple[int, str, Path]] = []
    for seed in FROZEN_SEEDS:
        seed_directory = campaign_root / f"seed-{seed}"
        if seed_directory.is_symlink():
            raise RuntimeError(
                f"campaign seed directory may not be a symlink: {seed_directory}"
            )
        actual_ablations = {
            path.name for path in seed_directory.iterdir() if path.is_dir()
        }
        expected_ablations = set(FROZEN_ABLATIONS)
        if actual_ablations != expected_ablations:
            raise RuntimeError(
                f"ablation grid mismatch for seed {seed}: "
                f"missing={sorted(expected_ablations - actual_ablations)}, "
                f"unexpected={sorted(actual_ablations - expected_ablations)}"
            )
        for ablation in FROZEN_ABLATIONS:
            run_directory = seed_directory / ablation
            if run_directory.is_symlink():
                raise RuntimeError(
                    f"run directory may not be a symlink: {run_directory}"
                )
            grid.append((seed, ablation, run_directory))
    if len(grid) != 40:
        raise AssertionError("internal error: frozen grid must contain exactly 40 runs")
    return grid


def _validate_manifest(run_directory: Path) -> tuple[dict[str, Any], str]:
    manifest_path = run_directory / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(f"missing real manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    _exact_keys(
        manifest,
        {"schema_version", "source_git_revision", "artifacts"},
        str(manifest_path),
    )
    if manifest["schema_version"] != 1:
        raise RuntimeError(f"unsupported manifest schema in {manifest_path}")
    artifacts = _mapping(manifest["artifacts"], f"{manifest_path}.artifacts")
    if set(artifacts) != _REQUIRED_ARTIFACTS:
        raise RuntimeError(
            f"manifest artifact set mismatch in {manifest_path}: "
            f"missing={sorted(_REQUIRED_ARTIFACTS - set(artifacts))}, "
            f"unexpected={sorted(set(artifacts) - _REQUIRED_ARTIFACTS)}"
        )
    for name in sorted(_REQUIRED_ARTIFACTS):
        path = run_directory / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"manifested artifact is missing or a symlink: {path}"
            )
        entry = _mapping(artifacts[name], f"{manifest_path}.artifacts.{name}")
        _exact_keys(entry, {"sha256", "bytes"}, f"{manifest_path}.artifacts.{name}")
        if entry["sha256"] != _sha256(path):
            raise RuntimeError(f"manifest SHA-256 mismatch: {path}")
        if (
            _integer(entry["bytes"], f"{manifest_path}.{name}.bytes")
            != path.stat().st_size
        ):
            raise RuntimeError(f"manifest byte count mismatch: {path}")
    revision = str(manifest["source_git_revision"])
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise RuntimeError(
            f"manifest has no full lowercase Git revision: {manifest_path}"
        )
    return manifest, revision


def _expected_loss_weights(
    source: Mapping[str, Any], ablation: str
) -> dict[str, float]:
    combined = {
        key: float(value)
        for key, value in _mapping(
            _mapping(source["loss"], "source.loss")["combined_weights"],
            "source.loss.combined_weights",
        ).items()
    }
    zero = {name: 0.0 for name in combined}
    if ablation == "combined":
        return combined
    retained = {
        "task_only": {"task"},
        "reconstruction_only": {"reconstruction", "cell", "sheaf"},
        "task_reconstruction": {"task", "reconstruction", "cell", "sheaf"},
        "task_reconstruction_cone": {
            "task",
            "reconstruction",
            "cell",
            "sheaf",
            "cone",
        },
        "task_reconstruction_rtd": {
            "task",
            "reconstruction",
            "cell",
            "sheaf",
            "rtd",
        },
        "cone_only": {"cone"},
        "rtd_only": {"rtd"},
    }[ablation]
    for name in retained:
        zero[name] = combined[name]
    return zero


def _validate_effective_config(
    path: Path,
    source: Mapping[str, Any],
    *,
    seed: int,
    ablation: str,
    run_directory: Path,
) -> dict[str, Any]:
    effective = _read_yaml(path)
    expected = copy.deepcopy(dict(source))
    expected_experiment = _mapping(expected["experiment"], "source.experiment")
    expected_experiment["seed"] = seed
    expected["experiment"] = expected_experiment
    expected_loss = _mapping(expected["loss"], "source.loss")
    expected_loss["ablation"] = ablation
    expected["loss"] = expected_loss
    actual_output = _mapping(effective.get("output"), f"{path}.output")
    expected_output = _mapping(expected.get("output"), "source.output")
    recorded_output = Path(str(actual_output.get("directory")))
    if recorded_output.resolve() != run_directory.resolve():
        raise RuntimeError(
            f"effective output directory mismatch in {path}: {recorded_output}"
        )
    normalized_effective = copy.deepcopy(effective)
    actual_output["directory"] = "<RUN_DIRECTORY>"
    expected_output["directory"] = "<RUN_DIRECTORY>"
    normalized_effective["output"] = actual_output
    expected["output"] = expected_output
    if normalized_effective != expected:
        raise RuntimeError(
            f"effective config differs from the frozen source beyond seed, ablation, "
            f"and output: {path}"
        )
    if _mapping(effective["experiment"], f"{path}.experiment")["device"] != "cuda":
        raise RuntimeError(f"effective config is not CUDA: {path}")
    if (
        _mapping(effective["experiment"], f"{path}.experiment")["deterministic"]
        is not True
    ):
        raise RuntimeError(f"effective config is not deterministic: {path}")
    return effective


def _validate_provenance(
    path: Path,
    *,
    source_config: Path,
    source_sha256: str,
    train_runner: Path,
    module_path: Path,
    current_code_fingerprint: str,
    seed: int,
    ablation: str,
    run_directory: Path,
    manifest_revision: str,
) -> dict[str, Any]:
    provenance = _read_json(path)
    if provenance.get("schema_version") != 1:
        raise RuntimeError(f"unsupported provenance schema: {path}")
    working_directory = Path(str(provenance.get("working_directory")))
    if not working_directory.is_absolute():
        raise RuntimeError(f"provenance working directory is not absolute: {path}")
    command = provenance.get("command")
    if not isinstance(command, list) or len(command) < 2:
        raise RuntimeError(f"provenance command is incomplete: {path}")
    if Path(str(command[1])).name != train_runner.name:
        raise RuntimeError(f"unexpected campaign runner in {path}")
    if "--smoke" in command:
        raise RuntimeError(f"smoke override found in frozen campaign: {path}")
    if int(_option(command, "--seed")) != seed:
        raise RuntimeError(f"recorded command seed mismatch: {path}")
    if _option(command, "--ablation") != ablation:
        raise RuntimeError(f"recorded command ablation mismatch: {path}")
    if (
        _resolve_recorded_path(_option(command, "--config"), working_directory)
        != source_config.resolve()
    ):
        raise RuntimeError(f"recorded command config mismatch: {path}")
    if (
        _resolve_recorded_path(_option(command, "--output"), working_directory)
        != run_directory.resolve()
    ):
        raise RuntimeError(f"recorded command output mismatch: {path}")
    if "--device" in command and _option(command, "--device") != "cuda":
        raise RuntimeError(f"recorded device override is not CUDA: {path}")
    if (
        _resolve_recorded_path(provenance.get("output_directory"), working_directory)
        != run_directory.resolve()
    ):
        raise RuntimeError(f"provenance output directory mismatch: {path}")

    git = _mapping(provenance.get("git"), f"{path}.git")
    if git.get("revision") != manifest_revision:
        raise RuntimeError(f"manifest/provenance Git revision mismatch: {path}")
    if git.get("status_porcelain") != []:
        raise RuntimeError(f"campaign run began from a dirty Git worktree: {path}")
    if _integer(provenance.get("seed"), f"{path}.seed") != seed:
        raise RuntimeError(f"provenance seed mismatch: {path}")
    if provenance.get("device") != "cuda":
        raise RuntimeError(f"run did not execute on CUDA: {path}")
    if "gb10" not in str(provenance.get("gpu", "")).lower():
        raise RuntimeError(f"run did not record an NVIDIA GB10 device: {path}")
    if not provenance.get("cuda") or "+cu" not in str(provenance.get("torch", "")):
        raise RuntimeError(f"run did not record a CUDA-enabled Torch runtime: {path}")
    if provenance.get("deterministic_algorithms") is not True:
        raise RuntimeError(f"deterministic algorithms were not enabled: {path}")
    if provenance.get("cublas_workspace_config") not in {":4096:8", ":16:8"}:
        raise RuntimeError(f"deterministic cuBLAS workspace was not recorded: {path}")

    files = _mapping(provenance.get("files"), f"{path}.files")
    _exact_keys(files, {"input_config", "runner", "module"}, f"{path}.files")
    input_config = _mapping(files.get("input_config"), f"{path}.files.input_config")
    _exact_keys(input_config, {"path", "sha256"}, f"{path}.files.input_config")
    if input_config.get("sha256") != source_sha256:
        raise RuntimeError(f"source-config provenance hash mismatch: {path}")
    if (
        _resolve_recorded_path(input_config.get("path"), working_directory)
        != source_config.resolve()
    ):
        raise RuntimeError(f"source-config provenance path mismatch: {path}")
    runner_record = _mapping(files.get("runner"), f"{path}.files.runner")
    module_record = _mapping(files.get("module"), f"{path}.files.module")
    _exact_keys(runner_record, {"path", "sha256"}, f"{path}.files.runner")
    _exact_keys(module_record, {"path", "sha256"}, f"{path}.files.module")
    if (
        _resolve_recorded_path(runner_record.get("path"), working_directory)
        != train_runner.resolve()
    ):
        raise RuntimeError(f"runner provenance path mismatch: {path}")
    if (
        _resolve_recorded_path(module_record.get("path"), working_directory)
        != module_path.resolve()
    ):
        raise RuntimeError(f"module provenance path mismatch: {path}")
    if runner_record.get("sha256") != _sha256(train_runner):
        raise RuntimeError(f"runner fingerprint mismatch: {path}")
    if module_record.get("sha256") != _sha256(module_path):
        raise RuntimeError(f"identifiable-map module fingerprint mismatch: {path}")
    if provenance.get("code_fingerprint") != current_code_fingerprint:
        raise RuntimeError(f"executable-source fingerprint mismatch: {path}")
    return provenance


def _validate_checkpoint(
    path: Path,
    *,
    effective: Mapping[str, Any],
    ablation: str,
    best_epoch: int,
) -> None:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise RuntimeError(f"checkpoint is not safely readable: {path}") from error
    checkpoint = _mapping(checkpoint, str(path))
    _exact_keys(
        checkpoint,
        {"schema_version", "model_state_dict", "config", "best_epoch", "ablation"},
        str(path),
    )
    if checkpoint["schema_version"] != 1:
        raise RuntimeError(f"unsupported checkpoint schema: {path}")
    if checkpoint["ablation"] != ablation:
        raise RuntimeError(f"checkpoint ablation mismatch: {path}")
    if _integer(checkpoint["best_epoch"], f"{path}.best_epoch") != best_epoch:
        raise RuntimeError(f"checkpoint best epoch mismatch: {path}")
    if checkpoint["config"] != effective:
        raise RuntimeError(f"checkpoint effective config mismatch: {path}")
    state = checkpoint["model_state_dict"]
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError(f"checkpoint has no model state: {path}")


def _signature(value: object, path: str) -> str:
    if not isinstance(value, list) or len(value) != 4:
        raise RuntimeError(f"{path} must be a four-degree Betti list")
    entries = [_integer(item, path) for item in value]
    if any(item < 0 for item in entries):
        raise RuntimeError(f"{path} contains a negative Betti number")
    return "[" + ",".join(str(item) for item in entries) + "]"


def _validate_predictions(
    path: Path,
    *,
    seed: int,
    transformations: int,
    expected_examples: int,
    map_tolerance: float,
    test: Mapping[str, Any],
) -> tuple[list[str], list[int]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"blank prediction line {line_number}: {path}")
            try:
                record = _mapping(json.loads(line), f"{path}:{line_number}")
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid prediction JSON {path}:{line_number}"
                ) from error
            _exact_keys(record, _PREDICTION_KEYS, f"{path}:{line_number}")
            records.append(record)
    if len(records) != expected_examples:
        raise RuntimeError(
            f"prediction denominator mismatch in {path}: "
            f"expected={expected_examples}, actual={len(records)}"
        )
    split_seed = seed + 3011
    expected_ids = [
        f"identifiable-{split_seed}-{index:07d}" for index in range(expected_examples)
    ]
    sample_ids = [str(record["sample_id"]) for record in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError(f"duplicate sample IDs in {path}")
    if sample_ids != expected_ids:
        raise RuntimeError(f"missing, reordered, or replaced sample IDs in {path}")

    correct_count = 0
    analytic_correct_count = 0
    cell_correct_count = 0
    map_errors: list[float] = []
    soft_residuals: list[float] = []
    hard_residuals: list[float] = []
    soft_histogram: Counter[str] = Counter()
    hard_histogram: Counter[str] = Counter()
    targets: list[int] = []
    for index, record in enumerate(records):
        location = f"{path}:{index + 1}"
        target = _integer(record["target_transformation"], f"{location}.target")
        predicted = _integer(
            record["predicted_transformation"], f"{location}.predicted"
        )
        analytic = _integer(
            record["analytic_marker_transformation"], f"{location}.analytic_marker"
        )
        if (
            not 0 <= target < transformations
            or not 0 <= predicted < transformations
            or not 0 <= analytic < transformations
        ):
            raise RuntimeError(f"transformation index out of range at {location}")
        if type(record["correct"]) is not bool or record["correct"] != (
            predicted == target
        ):
            raise RuntimeError(f"inconsistent correctness flag at {location}")
        if type(record["cell_face_correct"]) is not bool:
            raise TypeError(f"cell_face_correct must be Boolean at {location}")
        confidence = _finite(record["confidence"], f"{location}.confidence")
        if not 0.0 <= confidence <= 1.0:
            raise RuntimeError(f"confidence outside [0,1] at {location}")
        map_error = _finite(record["map_mse"], f"{location}.map_mse")
        soft_residual = _finite(
            record["soft_chain_residual_max"], f"{location}.soft_chain_residual_max"
        )
        hard_residual = _finite(
            record["hard_chain_residual_max"], f"{location}.hard_chain_residual_max"
        )
        if map_error < 0 or soft_residual < 0 or hard_residual < 0:
            raise RuntimeError(f"negative error or residual at {location}")
        if soft_residual > map_tolerance or hard_residual > map_tolerance:
            raise RuntimeError(f"held-out chain-map tolerance exceeded at {location}")
        soft_histogram[_signature(record["soft_cone_betti"], location)] += 1
        hard_histogram[_signature(record["hard_cone_betti"], location)] += 1
        targets.append(target)
        correct_count += int(record["correct"])
        analytic_correct_count += int(analytic == target)
        cell_correct_count += int(record["cell_face_correct"])
        map_errors.append(map_error)
        soft_residuals.append(soft_residual)
        hard_residuals.append(hard_residual)

    _close(
        _finite(test["transformation_accuracy"], f"{path}.transformation_accuracy"),
        correct_count / expected_examples,
        f"{path}.transformation_accuracy",
    )
    _close(
        _finite(
            test["analytic_marker_decoder_accuracy"],
            f"{path}.analytic_marker_decoder_accuracy",
        ),
        analytic_correct_count / expected_examples,
        f"{path}.analytic_marker_decoder_accuracy",
    )
    _close(
        _finite(test["cell_face_accuracy"], f"{path}.cell_face_accuracy"),
        cell_correct_count / expected_examples,
        f"{path}.cell_face_accuracy",
    )
    _close(
        _finite(test["map_mse"], f"{path}.map_mse"),
        statistics.fmean(map_errors),
        f"{path}.map_mse",
    )
    _close(
        _finite(test["soft_chain_residual_max"], f"{path}.soft_chain_residual_max"),
        max(soft_residuals),
        f"{path}.soft_chain_residual_max",
    )
    _close(
        _finite(test["hard_chain_residual_max"], f"{path}.hard_chain_residual_max"),
        max(hard_residuals),
        f"{path}.hard_chain_residual_max",
    )
    if _mapping(test["soft_cone_betti_histogram"], f"{path}.soft_histogram") != dict(
        sorted(soft_histogram.items())
    ):
        raise RuntimeError(f"soft cone histogram does not match predictions: {path}")
    if _mapping(test["hard_cone_betti_histogram"], f"{path}.hard_histogram") != dict(
        sorted(hard_histogram.items())
    ):
        raise RuntimeError(f"hard cone histogram does not match predictions: {path}")
    if hard_histogram != Counter({"[0,0,0,0]": expected_examples}):
        raise RuntimeError(f"decoded basis maps are not all acyclic on test: {path}")
    return sample_ids, targets


def _validate_summary(
    path: Path,
    effective: Mapping[str, Any],
    *,
    seed: int,
    ablation: str,
    expected_split_samples: Mapping[str, int],
    run_directory: Path,
) -> tuple[dict[str, Any], list[str], list[int]]:
    summary = _read_json(path)
    _exact_keys(summary, _SUMMARY_KEYS, str(path))
    if summary["schema_version"] != 1 or summary["status"] != "completed":
        raise RuntimeError(f"run did not complete under summary schema 1: {path}")
    if summary["experiment"] != "identifiable-graph-only-typed-maps":
        raise RuntimeError(f"unexpected experiment type: {path}")
    if (
        summary["ablation"] != ablation
        or _integer(summary["seed"], f"{path}.seed") != seed
    ):
        raise RuntimeError(f"summary grid identity mismatch: {path}")
    if summary["device"] != "cuda":
        raise RuntimeError(f"summary does not record CUDA execution: {path}")
    scope = str(summary["scope"]).lower()
    if "finite dihedral" not in scope or "cellular annulus" not in scope:
        raise RuntimeError(f"summary is missing its finite annulus scope: {path}")

    expected_weights = _expected_loss_weights(effective, ablation)
    actual_weights = {
        key: _finite(value, f"{path}.loss_weights.{key}")
        for key, value in _mapping(
            summary["loss_weights"], f"{path}.loss_weights"
        ).items()
    }
    if actual_weights != expected_weights:
        raise RuntimeError(f"loss weights do not match ablation {ablation}: {path}")
    if (
        _integer(summary["rtd_training_entities"], f"{path}.rtd_training_entities")
        != FROZEN_RTD_TRAINING_ENTITIES
    ):
        raise RuntimeError(f"training RTD entity bound changed: {path}")

    epochs = _integer(summary["epochs_completed"], f"{path}.epochs_completed")
    best_epoch = _integer(summary["best_epoch"], f"{path}.best_epoch")
    maximum_epochs = _integer(
        _mapping(effective["training"], f"{path}.training")["epochs"],
        f"{path}.training.epochs",
    )
    if not 1 <= best_epoch <= epochs <= maximum_epochs:
        raise RuntimeError(f"invalid training epoch counts: {path}")
    if _finite(summary["elapsed_seconds"], f"{path}.elapsed_seconds") <= 0:
        raise RuntimeError(f"nonpositive training time: {path}")
    _finite(summary["best_validation_objective"], f"{path}.best_validation_objective")

    dataset = _mapping(summary["dataset"], f"{path}.dataset")
    split_samples = _mapping(dataset.get("split_samples"), f"{path}.split_samples")
    if split_samples != dict(expected_split_samples):
        raise RuntimeError(f"split denominator mismatch: {path}")
    split_seeds = _mapping(dataset.get("split_seeds"), f"{path}.split_seeds")
    expected_split_seeds = {
        "train": seed + 1009,
        "validation": seed + 2017,
        "test": seed + 3011,
    }
    if split_seeds != expected_split_seeds:
        raise RuntimeError(f"split seed mismatch: {path}")
    for key, expected in {
        "sectors": 6,
        "vertices": 12,
        "edges": 18,
        "faces": 6,
        "transformations": 12,
    }.items():
        if dataset.get(key) != expected:
            raise RuntimeError(f"dataset topology mismatch for {key}: {path}")
    if dataset.get("topology") != "cellular_annulus":
        raise RuntimeError(f"dataset topology name mismatch: {path}")
    if dataset.get("betti_numbers") != FROZEN_BASE_BETTI:
        raise RuntimeError(f"base annulus Betti numbers changed: {path}")

    tolerance = _finite(
        _mapping(effective["evaluation"], f"{path}.evaluation")["map_tolerance"],
        f"{path}.evaluation.map_tolerance",
    )
    if tolerance != FROZEN_MAP_TOLERANCE:
        raise RuntimeError(f"map tolerance is not frozen at 1e-5: {path}")
    if (
        _finite(summary["basis_chain_residual_max"], f"{path}.basis_residual")
        > tolerance
    ):
        raise RuntimeError(f"basis chain-map residual exceeds tolerance: {path}")
    if summary["declared_chain_map_equations"] != [
        "B1 @ F1 = F0 @ B1",
        "B2 @ F2 = F1 @ B2",
    ]:
        raise RuntimeError(f"declared chain-map equations changed: {path}")

    test = _mapping(summary["test"], f"{path}.test")
    _exact_keys(test, _TEST_KEYS, f"{path}.test")
    test_examples = _integer(test["examples"], f"{path}.test.examples")
    if test_examples != expected_split_samples["test"]:
        raise RuntimeError(f"test denominator mismatch: {path}")
    if _finite(test["map_tolerance"], f"{path}.test.map_tolerance") != tolerance:
        raise RuntimeError(f"test/config map tolerance mismatch: {path}")
    for name in ENDPOINTS:
        value = _finite(test[name], f"{path}.test.{name}")
        if name == "transformation_accuracy":
            if not 0 <= value <= 1:
                raise RuntimeError(f"accuracy outside [0,1]: {path}")
        elif value < 0:
            raise RuntimeError(f"negative error endpoint {name}: {path}")
    cell_accuracy = _finite(test["cell_face_accuracy"], f"{path}.cell_face_accuracy")
    if not 0 <= cell_accuracy <= 1:
        raise RuntimeError(f"cell accuracy outside [0,1]: {path}")
    for residual_name in ("soft_chain_residual_max", "hard_chain_residual_max"):
        residual = _finite(test[residual_name], f"{path}.{residual_name}")
        if residual < 0 or residual > tolerance:
            raise RuntimeError(f"{residual_name} exceeds frozen tolerance: {path}")
    analytic_accuracy = _finite(
        test["analytic_marker_decoder_accuracy"],
        f"{path}.analytic_marker_decoder_accuracy",
    )
    if analytic_accuracy != ANALYTIC_MARKER_ACCURACY:
        raise RuntimeError(f"analytic marker decoder is not exact: {path}")
    chance = {
        key: _finite(value, f"{path}.chance_baselines.{key}")
        for key, value in _mapping(
            test["chance_baselines"], f"{path}.chance_baselines"
        ).items()
    }
    if chance != CHANCE_BASELINES:
        raise RuntimeError(f"chance baselines changed: {path}")
    cone_oracle = _mapping(test["cone_rank_oracle"], f"{path}.cone_rank_oracle")
    if cone_oracle != {
        "method": "fixed-tolerance-float64-numerical-rank",
        "rank_atol": _mapping(effective["evaluation"], f"{path}.evaluation")[
            "rank_atol"
        ],
        "map_atol": tolerance,
    }:
        raise RuntimeError(f"cone rank oracle settings changed: {path}")

    exact_rtd = _mapping(test["exact_rtd"], f"{path}.test.exact_rtd")
    _exact_keys(
        exact_rtd,
        {
            "entities",
            "normalization",
            "max_dim",
            "half_symmetric_rtd_by_degree",
            "srtd_by_degree",
        },
        f"{path}.test.exact_rtd",
    )
    expected_entities = min(
        _integer(
            _mapping(effective["evaluation"], f"{path}.evaluation")[
                "exact_rtd_entities"
            ],
            f"{path}.evaluation.exact_rtd_entities",
        ),
        test_examples,
    )
    if exact_rtd["entities"] != expected_entities:
        raise RuntimeError(f"exact RTD denominator mismatch: {path}")
    max_dim = _integer(exact_rtd["max_dim"], f"{path}.exact_rtd.max_dim")
    if (
        max_dim
        != _mapping(effective["evaluation"], f"{path}.evaluation")["exact_rtd_max_dim"]
    ):
        raise RuntimeError(f"exact RTD dimension mismatch: {path}")
    if exact_rtd["normalization"] != "full-matrix-q0.9":
        raise RuntimeError(f"exact RTD normalization mismatch: {path}")
    for key in ("half_symmetric_rtd_by_degree", "srtd_by_degree"):
        values = exact_rtd[key]
        if not isinstance(values, list) or len(values) != max_dim + 1:
            raise RuntimeError(f"degree-specific RTD vector mismatch for {key}: {path}")
        if any(_finite(value, f"{path}.{key}") < 0 for value in values):
            raise RuntimeError(f"negative RTD value for {key}: {path}")

    environment = _mapping(summary["environment"], f"{path}.environment")
    if (
        _integer(
            environment.get("peak_cuda_memory_bytes"), f"{path}.peak_cuda_memory_bytes"
        )
        <= 0
    ):
        raise RuntimeError(f"run has no positive CUDA memory evidence: {path}")
    predictions_path = run_directory / "test_predictions.jsonl"
    sample_ids, targets = _validate_predictions(
        predictions_path,
        seed=seed,
        transformations=int(dataset["transformations"]),
        expected_examples=test_examples,
        map_tolerance=tolerance,
        test=test,
    )
    hard_histogram = _mapping(
        test["hard_cone_betti_histogram"], f"{path}.hard_cone_betti_histogram"
    )
    hard_acyclic_fraction = int(hard_histogram.get("[0,0,0,0]", 0)) / test_examples
    expected_checks = {
        "transformation_accuracy": float(test["transformation_accuracy"])
        >= ENGINEERING_GATE["transformation_accuracy_min"],
        "cell_face_accuracy": float(test["cell_face_accuracy"])
        >= ENGINEERING_GATE["cell_face_accuracy_min"],
        "map_mse": float(test["map_mse"]) <= ENGINEERING_GATE["map_mse_max"],
        "soft_chain_residual": float(test["soft_chain_residual_max"])
        <= ENGINEERING_GATE["chain_residual_max"],
        "hard_chain_residual": float(test["hard_chain_residual_max"])
        <= ENGINEERING_GATE["chain_residual_max"],
        "hard_cone_acyclic_fraction": hard_acyclic_fraction
        >= ENGINEERING_GATE["hard_cone_acyclic_fraction_min"],
    }
    gate = _mapping(
        summary["engineering_recovery_gate"], f"{path}.engineering_recovery_gate"
    )
    _exact_keys(
        gate,
        {
            "applicable",
            "thresholds",
            "checks",
            "passed",
            "hard_cone_acyclic_fraction",
            "status",
        },
        f"{path}.engineering_recovery_gate",
    )
    applicable = ablation in {"task_reconstruction", "combined"}
    if gate["applicable"] is not applicable:
        raise RuntimeError(f"engineering gate applicability mismatch: {path}")
    if gate["thresholds"] != ENGINEERING_GATE or gate["checks"] != expected_checks:
        raise RuntimeError(f"engineering gate thresholds or checks mismatch: {path}")
    expected_passed = all(expected_checks.values()) if applicable else None
    if gate["passed"] is not expected_passed:
        raise RuntimeError(f"engineering gate decision mismatch: {path}")
    _close(
        _finite(
            gate["hard_cone_acyclic_fraction"],
            f"{path}.hard_cone_acyclic_fraction",
        ),
        hard_acyclic_fraction,
        f"{path}.hard_cone_acyclic_fraction",
    )
    if gate["status"] != "pre-specified development-informed engineering gate":
        raise RuntimeError(f"engineering gate status changed: {path}")
    history = json.loads((run_directory / "history.json").read_text(encoding="utf-8"))
    if not isinstance(history, list) or len(history) != epochs:
        raise RuntimeError(
            f"history denominator does not match epochs_completed: {path}"
        )
    if [entry.get("epoch") for entry in history if isinstance(entry, Mapping)] != list(
        range(1, epochs + 1)
    ):
        raise RuntimeError(
            f"history epochs are missing, duplicated, or reordered: {path}"
        )
    return summary, sample_ids, targets


def _estimate(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != 5:
        raise ValueError("the frozen campaign requires exactly five seed values")
    return {
        "n_seeds": 5,
        "values_in_seed_order": list(values),
        "mean": statistics.fmean(values),
        "sample_standard_deviation": statistics.stdev(values),
    }


def _sign_test(differences: Sequence[float], direction: str) -> dict[str, Any]:
    nonzero = [difference for difference in differences if difference != 0.0]
    positive = sum(difference > 0 for difference in nonzero)
    negative = sum(difference < 0 for difference in nonzero)
    n = len(nonzero)
    tail = min(positive, negative)
    probability = (
        min(1.0, 2.0 * sum(math.comb(n, index) for index in range(tail + 1)) / 2**n)
        if n
        else 1.0
    )
    if direction == "higher_is_better":
        favorable, unfavorable = positive, negative
    else:
        favorable, unfavorable = negative, positive
    return {
        "definition": "exact two-sided paired sign test; exact-zero differences omitted",
        "nonzero_pairs": n,
        "ties": len(differences) - n,
        "positive_raw_differences": positive,
        "negative_raw_differences": negative,
        "candidate_favorable": favorable,
        "candidate_unfavorable": unfavorable,
        "two_sided_pvalue": probability,
    }


def _paired_estimate(
    rows_by_key: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    candidate: str,
    control: str,
    endpoint: str,
) -> dict[str, Any]:
    pairs = []
    differences = []
    for seed in FROZEN_SEEDS:
        candidate_value = float(rows_by_key[(seed, candidate)]["endpoints"][endpoint])
        control_value = float(rows_by_key[(seed, control)]["endpoints"][endpoint])
        difference = candidate_value - control_value
        differences.append(difference)
        pairs.append(
            {
                "seed": seed,
                "candidate": candidate_value,
                "control": control_value,
                "candidate_minus_control": difference,
            }
        )
    mean = statistics.fmean(differences)
    standard_deviation = statistics.stdev(differences)
    half_width = _T_975_DF4 * standard_deviation / math.sqrt(5)
    direction = str(ENDPOINTS[endpoint]["direction"])
    return {
        "endpoint": endpoint,
        "direction": direction,
        "difference_definition": "candidate minus control",
        "n_paired_seeds": 5,
        "pairs": pairs,
        "mean_difference": mean,
        "sample_standard_deviation_of_differences": standard_deviation,
        "student_t_95_ci_df4": [mean - half_width, mean + half_width],
        "sensitivity_sign_test": _sign_test(differences, direction),
    }


def _aggregate_histograms(
    rows: Sequence[Mapping[str, Any]], key: str
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(
            {
                str(signature): int(count)
                for signature, count in _mapping(row[key], key).items()
            }
        )
    return dict(sorted(counts.items()))


def summarize(
    campaign_root: Path,
    source_config: Path,
    *,
    expected_source_sha256: str = FROZEN_SOURCE_CONFIG_SHA256,
    expected_split_samples: Mapping[str, int] = FROZEN_SPLIT_SAMPLES,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Validate all frozen runs and return the prespecified paired analysis."""

    project_root = (
        Path(__file__).resolve().parents[1]
        if project_root is None
        else project_root.resolve()
    )
    train_runner = project_root / "scripts" / "train_identifiable_maps.py"
    module_path = (
        project_root / "src" / "homymoly" / "experiments" / "identifiable_maps.py"
    )
    protocol_path = project_root / PROTOCOL
    summarizer_path = Path(__file__).resolve()
    for path in (
        source_config,
        train_runner,
        module_path,
        protocol_path,
        summarizer_path,
    ):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"required frozen source is missing or a symlink: {path}"
            )
    source_sha256 = _sha256(source_config)
    if source_sha256 != expected_source_sha256:
        raise RuntimeError(
            "frozen source config SHA-256 mismatch: "
            f"expected={expected_source_sha256}, actual={source_sha256}"
        )
    source = _read_yaml(source_config)
    source_splits = {
        split: _integer(
            _mapping(source.get("data"), "source.data").get(f"{split}_samples"),
            f"source.data.{split}_samples",
        )
        for split in ("train", "validation", "test")
    }
    if source_splits != dict(expected_split_samples):
        raise RuntimeError(
            f"frozen split denominators changed: expected={dict(expected_split_samples)}, "
            f"actual={source_splits}"
        )
    if (
        _mapping(source.get("evaluation"), "source.evaluation").get("map_tolerance")
        != FROZEN_MAP_TOLERANCE
    ):
        raise RuntimeError("source map tolerance is not frozen at 1e-5")
    source_data = _mapping(source.get("data"), "source.data")
    if source_data.get("sectors") != 6:
        raise RuntimeError("source annulus is not frozen at six sectors")
    source_loss = _mapping(source.get("loss"), "source.loss")
    if source_loss.get("rtd_training_entities") != FROZEN_RTD_TRAINING_ENTITIES:
        raise RuntimeError("source RTD training entity bound is not frozen at 48")
    combined_weights = _mapping(
        source_loss.get("combined_weights"), "source.loss.combined_weights"
    )
    if combined_weights.get("cone") != 0.1:
        raise RuntimeError("source cone weight is not frozen at 0.1")

    grid = _validate_grid(campaign_root)
    current_code_fingerprint = _code_fingerprint(project_root, train_runner)
    rows: list[dict[str, Any]] = []
    rows_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    shared_revisions: set[str] = set()
    shared_code_fingerprints: set[str] = set()
    shared_runner_fingerprints: set[str] = set()
    shared_module_fingerprints: set[str] = set()
    shared_environments: set[tuple[str, ...]] = set()
    paired_ids: dict[int, list[str]] = {}
    paired_targets: dict[int, list[int]] = {}

    for seed, ablation, run_directory in grid:
        _, revision = _validate_manifest(run_directory)
        effective = _validate_effective_config(
            run_directory / "effective_config.yaml",
            source,
            seed=seed,
            ablation=ablation,
            run_directory=run_directory,
        )
        provenance = _validate_provenance(
            run_directory / "provenance.json",
            source_config=source_config,
            source_sha256=source_sha256,
            train_runner=train_runner,
            module_path=module_path,
            current_code_fingerprint=current_code_fingerprint,
            seed=seed,
            ablation=ablation,
            run_directory=run_directory,
            manifest_revision=revision,
        )
        summary, sample_ids, targets = _validate_summary(
            run_directory / "summary.json",
            effective,
            seed=seed,
            ablation=ablation,
            expected_split_samples=expected_split_samples,
            run_directory=run_directory,
        )
        _validate_checkpoint(
            run_directory / "checkpoint.pt",
            effective=effective,
            ablation=ablation,
            best_epoch=int(summary["best_epoch"]),
        )
        if seed in paired_ids and paired_ids[seed] != sample_ids:
            raise RuntimeError(
                f"paired sample IDs differ across ablations for seed {seed}"
            )
        if seed in paired_targets and paired_targets[seed] != targets:
            raise RuntimeError(
                f"paired target labels differ across ablations for seed {seed}"
            )
        paired_ids.setdefault(seed, sample_ids)
        paired_targets.setdefault(seed, targets)

        test = _mapping(summary["test"], f"{run_directory}.test")
        files = _mapping(provenance["files"], f"{run_directory}.files")
        row = {
            "seed": seed,
            "ablation": ablation,
            "run_directory": run_directory.as_posix(),
            "status": summary["status"],
            "test_examples": int(test["examples"]),
            "best_epoch": int(summary["best_epoch"]),
            "epochs_completed": int(summary["epochs_completed"]),
            "elapsed_seconds": float(summary["elapsed_seconds"]),
            "peak_cuda_memory_bytes": int(
                _mapping(summary["environment"], "summary.environment")[
                    "peak_cuda_memory_bytes"
                ]
            ),
            "endpoints": {name: float(test[name]) for name in ENDPOINTS},
            "cell_face_accuracy": float(test["cell_face_accuracy"]),
            "analytic_marker_decoder_accuracy": float(
                test["analytic_marker_decoder_accuracy"]
            ),
            "chance_baselines": test["chance_baselines"],
            "soft_chain_residual_max": float(test["soft_chain_residual_max"]),
            "hard_chain_residual_max": float(test["hard_chain_residual_max"]),
            "soft_cone_betti_histogram": test["soft_cone_betti_histogram"],
            "hard_cone_betti_histogram": test["hard_cone_betti_histogram"],
            "exact_rtd": test["exact_rtd"],
            "engineering_recovery_gate": summary["engineering_recovery_gate"],
            "git_revision": revision,
            "git_status_porcelain": _mapping(provenance["git"], "provenance.git").get(
                "status_porcelain"
            ),
            "code_fingerprint": provenance["code_fingerprint"],
            "runner_sha256": _mapping(files["runner"], "files.runner")["sha256"],
            "module_sha256": _mapping(files["module"], "files.module")["sha256"],
        }
        rows.append(row)
        rows_by_key[(seed, ablation)] = row
        shared_revisions.add(revision)
        shared_code_fingerprints.add(str(provenance["code_fingerprint"]))
        shared_runner_fingerprints.add(str(row["runner_sha256"]))
        shared_module_fingerprints.add(str(row["module_sha256"]))
        shared_environments.add(
            tuple(
                str(provenance.get(name))
                for name in (
                    "python",
                    "platform",
                    "torch",
                    "cuda",
                    "gpu",
                    "numpy",
                    "pyyaml",
                )
            )
        )

    for values, description in (
        (shared_revisions, "Git revision"),
        (shared_code_fingerprints, "executable-source fingerprint"),
        (shared_runner_fingerprints, "runner fingerprint"),
        (shared_module_fingerprints, "module fingerprint"),
        (shared_environments, "runtime environment"),
    ):
        if len(values) != 1:
            raise RuntimeError(
                f"campaign does not share one {description}: {sorted(values)}"
            )
    _require_git_commit(project_root, next(iter(shared_revisions)))

    by_ablation: dict[str, Any] = {}
    for ablation in FROZEN_ABLATIONS:
        ablation_rows = [rows_by_key[(seed, ablation)] for seed in FROZEN_SEEDS]
        by_ablation[ablation] = {
            "n_seeds": 5,
            "endpoints": {
                endpoint: _estimate(
                    [float(row["endpoints"][endpoint]) for row in ablation_rows]
                )
                for endpoint in ENDPOINTS
            },
            "cell_face_accuracy": _estimate(
                [float(row["cell_face_accuracy"]) for row in ablation_rows]
            ),
            "maximum_soft_chain_residual": max(
                float(row["soft_chain_residual_max"]) for row in ablation_rows
            ),
            "maximum_hard_chain_residual": max(
                float(row["hard_chain_residual_max"]) for row in ablation_rows
            ),
        }

    contrasts: dict[str, Any] = {}
    for specification in CONTRASTS:
        name = str(specification["name"])
        contrasts[name] = {
            "candidate": specification["candidate"],
            "control": specification["control"],
            "role": specification["role"],
            "inference_scope": (
                "descriptive, unadjusted paired estimation across five frozen seeds; "
                "no run or seed excluded and no inferential benefit claim"
            ),
            "endpoints": {
                endpoint: _paired_estimate(
                    rows_by_key,
                    candidate=str(specification["candidate"]),
                    control=str(specification["control"]),
                    endpoint=endpoint,
                )
                for endpoint in ENDPOINTS
            },
        }

    qualitative_controls: dict[str, Any] = {}
    for ablation in ("cone_only", "rtd_only"):
        control_rows = [rows_by_key[(seed, ablation)] for seed in FROZEN_SEEDS]
        qualitative_controls[ablation] = {
            "n_seeds": 5,
            "test_examples_across_runs": sum(
                int(row["test_examples"]) for row in control_rows
            ),
            "transformation_accuracy": by_ablation[ablation]["endpoints"][
                "transformation_accuracy"
            ],
            "map_mse": by_ablation[ablation]["endpoints"]["map_mse"],
            "aggregate_soft_cone_betti_histogram": _aggregate_histograms(
                control_rows, "soft_cone_betti_histogram"
            ),
            "aggregate_hard_cone_betti_histogram": _aggregate_histograms(
                control_rows, "hard_cone_betti_histogram"
            ),
            "interpretation": (
                "descriptive identifiability control only; this objective alone does not "
                "test a general categorical or representation-equivalence claim"
            ),
        }

    environment_names = (
        "python",
        "platform",
        "torch",
        "cuda",
        "gpu",
        "numpy",
        "pyyaml",
    )
    environment_values = next(iter(shared_environments))
    gate_rows = [
        rows_by_key[(seed, ablation)]
        for seed in FROZEN_SEEDS
        for ablation in ("task_reconstruction", "combined")
    ]
    gate_failures = [
        {"seed": row["seed"], "ablation": row["ablation"]}
        for row in gate_rows
        if not _mapping(
            row["engineering_recovery_gate"], "row.engineering_recovery_gate"
        )["passed"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scope": (
            "n=5 paired seeds on a synthetic finite dihedral action over one cellular "
            "annulus; no categorical-equivalence, real-data, or architecture-wide claim"
        ),
        "frozen_design": {
            "seeds": list(FROZEN_SEEDS),
            "ablations": list(FROZEN_ABLATIONS),
            "expected_run_count": 40,
            "source_config": source_config.as_posix(),
            "source_config_sha256": source_sha256,
            "split_samples_per_run": dict(expected_split_samples),
            "map_tolerance": FROZEN_MAP_TOLERANCE,
            "rtd_training_entities": FROZEN_RTD_TRAINING_ENTITIES,
            "annulus": {
                "sectors": 6,
                "vertices": 12,
                "edges": 18,
                "faces": 6,
                "betti_numbers": FROZEN_BASE_BETTI,
            },
            "analytic_marker_decoder_accuracy": ANALYTIC_MARKER_ACCURACY,
            "chance_baselines": CHANCE_BASELINES,
            "engineering_gate_thresholds": ENGINEERING_GATE,
            "endpoint_registry": ENDPOINTS,
            "contrast_registry": list(CONTRASTS),
        },
        "validation": {
            "status": "passed",
            "expected_runs": 40,
            "included_runs": len(rows),
            "excluded_runs": 0,
            "missing_runs": [],
            "replaced_runs": [],
            "manifest_hashes_and_sizes_verified": True,
            "effective_configs_verified": True,
            "source_config_hash_verified": True,
            "paired_sample_ids_and_targets_verified": True,
            "per_run_sample_ids_unique_and_complete": True,
            "cuda_gb10_execution_verified": True,
            "fixed_map_tolerance_verified": True,
            "checkpoint_identity_verified": True,
            "clean_git_status_verified": True,
            "committed_revision_verified": True,
        },
        "engineering_recovery_gate": {
            "role": "primary development-informed implementation recovery gate",
            "applicable_runs": 10,
            "passed_runs": 10 - len(gate_failures),
            "failed_runs": gate_failures,
            "passed": not gate_failures,
            "interpretation": (
                "an absolute implementation gate, not a structural-loss superiority test"
            ),
        },
        "shared_provenance": {
            "git_revision": next(iter(shared_revisions)),
            "code_fingerprint": next(iter(shared_code_fingerprints)),
            "runner_sha256": next(iter(shared_runner_fingerprints)),
            "module_sha256": next(iter(shared_module_fingerprints)),
            "environment": dict(
                zip(environment_names, environment_values, strict=True)
            ),
        },
        "analysis_provenance": {
            "protocol": {
                "path": protocol_path.relative_to(project_root).as_posix(),
                "sha256": _sha256(protocol_path),
            },
            "summarizer": {
                "path": summarizer_path.relative_to(project_root).as_posix(),
                "sha256": _sha256(summarizer_path),
            },
            "assumptions": {
                "paired_seed_count": 5,
                "student_t_degrees_of_freedom": 4,
                "minimum_attainable_two_sided_sign_test_pvalue_without_ties": 0.0625,
                "multiplicity_adjustment": "none",
                "inferential_structural_benefit_claim": False,
            },
        },
        "runs": rows,
        "by_ablation": by_ablation,
        "paired_contrasts": contrasts,
        "qualitative_identifiability_controls": qualitative_controls,
        "interpretation_guardrail": (
            "All estimates include all 40 prespecified runs. Student-t intervals are "
            "descriptive summaries of five-seed paired variation; exact sign tests have "
            "minimum two-sided p=0.0625 without ties and are sensitivity descriptions. "
            "No multiplicity correction is applied, so no structural-loss benefit is inferred. "
            "Cone-only and RTD-only results are qualitative controls and cannot establish "
            "map identification or general representation equivalence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=Path("artifacts/identifiable-maps/campaign"),
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        default=Path("configs/identifiable-maps/gb10-full.yaml"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.campaign_root, args.source_config)
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
