#!/usr/bin/env python3
"""Run the sealed independent edge-to-cycle lifting replication.

This runner must not be executed on the sealed seed block until the complete
protocol, this runner, and their fingerprints have been committed.  It keeps
the seed block inert at import time; datasets are instantiated only after all
preflight checks pass in :func:`run`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from homymoly.data.conversion import ConversionDataset

PROTOCOL = "docs/31-independent-lifting-replication-protocol.md"
RUNNER_SOURCE = "scripts/run_lifting_replication_v2.py"
GENERATOR_SOURCE = "src/homymoly/data/conversion.py"
LOCKFILE = "uv.lock"
SEAL_RECORD = "docs/32-independent-lifting-replication-seal.json"
SEAL_SCHEMA = "homymoly-lifting-replication-seal/1"
FROZEN_PROTOCOL_SHA256 = (
    "6288eade4755aa188299760303b389ce13acd42659b8b9bc340cb9d4024afec0"
)
FROZEN_GENERATOR_SHA256 = (
    "c37ab1c725aa2101e88c1a0ad8fa3b279d72330feba35077e23fec930a4df69d"
)
FROZEN_LOCKFILE_SHA256 = (
    "05c6a5ad02db5b1651d426d157add170a8542634260ce8c265a3ee32693073bf"
)
EXPECTED_ENVIRONMENT = {
    "python": "3.12.3",
    "torch_base": "2.13.0",
    "networkx": "3.6.1",
    "numpy": "2.5.2",
}

SCHEMA_VERSION = 1
RECORD_ID = "independent-lifting-replication-v2"
SEALED_SEEDS = tuple(range(20270101, 20270137))
MIN_FACES = 3
MIN_ELIGIBLE = 30
N_TRAIN = 16
N_TEST = 3072
NOISE_SD = 0.02
STEPS = 2500
LEARNING_RATE = 0.05
SOFT_BOUNDARY_WEIGHT = 3.0
SINGULAR_VALUE_WEIGHT = 0.01
RTD_INSPIRED_WEIGHT = 0.1
RIDGE_FOLDS = 4
RIDGE_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
LSTSQ_RCOND = 1e-12
PINV_RCOND = 1e-12
BASIS_TOLERANCE = 1e-10
STATIONARITY_RTOL = 1e-10
C1_REPLICATES = 12
PRIMARY_FAMILY_SIZE = 7
PRIMARY_ALPHA = 0.05
RTD_NO_BENEFIT_MARGIN = -0.045757490560675115

# t.ppf(1 - 0.05 / 7, df), computed once with SciPy for the frozen design.
_T_ONE_SIDED_BONFERRONI = {
    29: 2.606750672048818,
    30: 2.601227904110613,
    31: 2.5960807947257787,
    32: 2.5912722991315227,
    33: 2.586770085672467,
    34: 2.5825458097369376,
    35: 2.5785745178415116,
}

# t.ppf(0.975, df), used only for the unadjusted two-sided C1 interval.
_T975 = {
    29: 2.0452296421327034,
    30: 2.0422724563012378,
    31: 2.039513446396408,
    32: 2.036933343460102,
    33: 2.0345152974493383,
    34: 2.0322445093177186,
    35: 2.030107928250343,
}

ARM_NAMES = (
    "ambient_adam",
    "ambient_min_norm_ls",
    "soft_boundary_lambda3",
    "soft_boundary_closed_form_lambda3",
    "hard_cycle_ls",
    "hard_random_subspace_ls",
    "inner_cv_ridge",
    "singular_value_surrogate",
    "rtd_inspired_distance_surrogate",
    "generator_cycle_basis_oracle",
)

# The frozen seven-claim confirmatory family. The design seal must list exactly
# these identifiers; the inference and null-decision paths share this table.
_PRIMARY_CLAIM_DEFINITIONS = (
    {
        "id": "h1-soft-vs-ambient-adam",
        "description": (
            "soft boundary compatibility improves over the v1 ambient Adam reference"
        ),
        "numerator": "soft_boundary_lambda3",
        "denominator": "ambient_adam",
        "direction": "less",
        "threshold": 0.0,
    },
    {
        "id": "h2-hard-cycle-vs-ambient-ls",
        "description": (
            "hard cycle-subspace least squares improves over ambient minimum-norm "
            "least squares"
        ),
        "numerator": "hard_cycle_ls",
        "denominator": "ambient_min_norm_ls",
        "direction": "less",
        "threshold": 0.0,
    },
    {
        "id": "h3-hard-cycle-vs-soft-closed-form",
        "description": (
            "hard cycle-subspace least squares improves over the closed-form "
            "frozen soft objective"
        ),
        "numerator": "hard_cycle_ls",
        "denominator": "soft_boundary_closed_form_lambda3",
        "direction": "less",
        "threshold": 0.0,
    },
    {
        "id": "h4-hard-cycle-vs-hard-random",
        "description": (
            "the true cycle subspace improves over a dimension-matched random subspace"
        ),
        "numerator": "hard_cycle_ls",
        "denominator": "hard_random_subspace_ls",
        "direction": "less",
        "threshold": 0.0,
    },
    {
        "id": "h5-ridge-vs-ambient-ls",
        "description": (
            "training-only-selected ridge improves over ambient minimum-norm "
            "least squares"
        ),
        "numerator": "inner_cv_ridge",
        "denominator": "ambient_min_norm_ls",
        "direction": "less",
        "threshold": 0.0,
    },
    {
        "id": "h6-singular-surrogate-harm",
        "description": "the singular-value surrogate harms versus ambient Adam",
        "numerator": "singular_value_surrogate",
        "denominator": "ambient_adam",
        "direction": "greater",
        "threshold": 0.0,
    },
    {
        "id": "h7-rtd-bounded-benefit-futility",
        "description": (
            "futility bound: rule out an RTD-inspired benefit of 10% or more"
        ),
        "numerator": "rtd_inspired_distance_surrogate",
        "denominator": "ambient_adam",
        "direction": "greater",
        "threshold": RTD_NO_BENEFIT_MARGIN,
    },
)
PRIMARY_CLAIM_IDS = tuple(definition["id"] for definition in _PRIMARY_CLAIM_DEFINITIONS)


def _seal_theta(definition: dict[str, Any]) -> str:
    return (
        "mean over eligible seeds of log10(MSE_"
        f"{definition['numerator']} / MSE_{definition['denominator']})"
    )


# The exact seven claim objects the design seal must carry. The runner refuses
# any seal whose claims drift in direction, threshold, null, reference arm, or
# support rule from this frozen family -- a seal that binds only claim IDs
# would let the estimand or decision rule be edited after sealing.
_SEAL_PRIMARY_FAMILY: tuple[dict[str, Any], ...] = tuple(
    {
        "id": definition["id"],
        "theta": _seal_theta(definition),
        "null": null,
        "alternative": alternative,
        "reference_arm": definition["denominator"],
        "bound_direction": definition["direction"],
        "threshold": definition["threshold"],
        "support_rule": support_rule,
    }
    for definition, null, alternative, support_rule in zip(
        _PRIMARY_CLAIM_DEFINITIONS,
        (
            "theta >= 0",
            "theta >= 0",
            "theta >= 0",
            "theta >= 0",
            "theta >= 0",
            "theta <= 0",
            "theta <= log10(0.90)",
        ),
        (
            "theta < 0",
            "theta < 0",
            "theta < 0",
            "theta < 0",
            "theta < 0",
            "theta > 0",
            "theta > log10(0.90) is rejected as futile for a 10% benefit",
        ),
        (
            (
                "supported iff the one-sided Bonferroni upper bound (alpha = 0.05/7, "
                "Student-t critical value for the eligible n) is below 0"
            ),
            "supported iff the one-sided Bonferroni upper bound is below 0",
            "supported iff the one-sided Bonferroni upper bound is below 0",
            "supported iff the one-sided Bonferroni upper bound is below 0",
            "supported iff the one-sided Bonferroni upper bound is below 0",
            "supported iff the one-sided Bonferroni lower bound is above 0",
            (
                "bounded-benefit/futility supported iff the one-sided Bonferroni "
                "lower bound exceeds log10(0.90) = -0.045757490560675115, ruling "
                "out a benefit of 10% or more; this is not an equivalence test and "
                "not noninferiority"
            ),
        ),
        strict=True,
    )
)

# The exact stop conditions the design seal must carry, in order.
_SEAL_STOP_RULES: tuple[str, ...] = (
    (
        "Fewer than 30 eligible seeds (connected, F >= 3) among "
        "20270101..20270136: frozen design failure, status "
        "design_failure_insufficient_eligible, no fits, no confirmatory claims."
    ),
    (
        "Any generator exception: campaign failure, status design_failure, before "
        "any fit; the failing seed is recorded, never excluded."
    ),
    (
        "Any rank, dimension, orthogonality, nullspace-membership, or stationarity "
        "validation failure: whole-campaign failure, status design_failure; never "
        "delete only the offending seed."
    ),
    (
        "Any nonfinite or nonpositive C1 defect or held-out MSE: whole-campaign "
        "failure, status design_failure; never add an epsilon after outcomes."
    ),
    (
        "No outcome-dependent stopping: all 36 candidate seeds are attempted and "
        "all failures preserved regardless of intermediate results."
    ),
    "Never rerun with another seed block because the result is surprising or weak.",
    (
        "Unexpected exception: status execution_failure with completed rows "
        "preserved; KeyboardInterrupt: status interrupted."
    ),
    (
        "Runner refuses to start on a dirty worktree, an existing output path, a "
        "mismatched environment/lock/generator/protocol/runner fingerprint, a seal "
        "record not committed at HEAD, available CUDA, more than one PyTorch "
        "thread, or non-CPU non-float64 execution."
    ),
)


class DesignFailureError(RuntimeError):
    """A frozen-design validation failure: the whole campaign is a design failure."""

    def __init__(
        self,
        message: str,
        *,
        seed: int | None = None,
        arm: str | None = None,
    ) -> None:
        super().__init__(message)
        self.seed = seed
        self.arm = arm


@dataclass(frozen=True)
class RegressionData:
    """Paired train/test tensors for one topology."""

    train_x: Tensor
    train_y: Tensor
    test_x: Tensor
    test_y: Tensor


@dataclass(frozen=True)
class FittedMatrix:
    """One fitted matrix returned by a firewall-compliant fitting API.

    Fitting APIs see only their declared training tensors plus, where
    applicable, ``boundary_1`` or the seeded random basis -- never ``B2``,
    never held-out tensors. Held-out evaluation happens outside them.
    """

    matrix: Tensor  # F x E, detached
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FitResult:
    """One fitted matrix and its held-out endpoint."""

    matrix: Tensor  # F x E
    held_out_mse: float
    boundary_defect: float
    random_subspace_defect: float
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_sha256(value: str, *, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RuntimeError(f"stop condition: {label} must be a lowercase SHA-256")


def _verified_sha256(
    project_root: Path, relative_path: str, expected: str, *, label: str
) -> str:
    _validate_sha256(expected, label=f"expected {label} fingerprint")
    path = project_root / relative_path
    try:
        actual = _sha256(path)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"stop condition: {label} is missing at {relative_path}"
        ) from error
    if actual != expected:
        raise RuntimeError(
            f"stop condition: {label} SHA-256 is {actual}, expected {expected}"
        )
    return actual


def _git_checked(project_root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown git error"
        raise RuntimeError(f"stop condition: git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def _git_require_design_ancestry(project_root: Path, design_commit: str) -> None:
    """Require the sealed design commit to be a strict ancestor of HEAD.

    The seal is committed immediately after the design commit, so a design
    commit that is missing from HEAD's ancestry -- or equals HEAD -- means the
    recorded lineage does not match the repository being executed.
    """

    head = _git_checked(project_root, "rev-parse", "HEAD")
    if design_commit == head:
        raise RuntimeError(
            "stop condition: design seal design_commit must be a strict ancestor "
            "of HEAD; the seal record is committed after the design commit"
        )
    result = subprocess.run(
        ("git", "merge-base", "--is-ancestor", design_commit, "HEAD"),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode == 1:
        raise RuntimeError(
            "stop condition: design seal design_commit is not an ancestor of HEAD"
        )
    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown git error"
        raise RuntimeError(f"stop condition: git merge-base failed: {detail}")


def _environment_provenance(project_root: Path) -> dict[str, Any]:
    lock_hash = _verified_sha256(
        project_root,
        LOCKFILE,
        FROZEN_LOCKFILE_SHA256,
        label="dependency lockfile",
    )
    actual = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_base": torch.__version__.split("+", maxsplit=1)[0],
        "networkx": importlib.metadata.version("networkx"),
        "numpy": importlib.metadata.version("numpy"),
    }
    mismatches = {
        key: {"expected": value, "actual": actual[key]}
        for key, value in EXPECTED_ENVIRONMENT.items()
        if actual[key] != value
    }
    if mismatches:
        raise RuntimeError(
            "stop condition: environment does not match the frozen design: "
            + json.dumps(mismatches, sort_keys=True)
        )
    uname = platform.uname()
    return {
        "actual": actual,
        "expected": EXPECTED_ENVIRONMENT,
        "matches_expected": True,
        "platform": {
            "system": uname.system,
            "node": uname.node,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "processor": uname.processor,
        },
        "torch_build_config": torch.__config__.show(),
        "torch_cuda_build": torch.version.cuda,
        "lockfile": {
            "path": LOCKFILE,
            "sha256": lock_hash,
            "frozen_sha256": FROZEN_LOCKFILE_SHA256,
        },
    }


def _execution_environment() -> dict[str, Any]:
    """Pin single-thread CPU float64 execution and refuse a visible CUDA device."""

    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        raise RuntimeError(
            "stop condition: PyTorch must run with exactly one thread, got "
            f"{torch.get_num_threads()}"
        )
    if torch.cuda.is_available():
        raise RuntimeError(
            "stop condition: CUDA must be unavailable or hidden "
            "(run with CUDA_VISIBLE_DEVICES=-1)"
        )
    probe = torch.tensor([1.0, -2.0], dtype=torch.float64, device="cpu")
    reduced = float((probe * 3.0).sum())
    if (
        probe.dtype is not torch.float64
        or probe.device.type != "cpu"
        or reduced != -3.0
    ):
        raise RuntimeError("stop condition: CPU float64 tensor operation failed")
    return {
        "tensor_device": "cpu",
        "tensor_dtype": "float64",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "argv": list(sys.argv),
    }


def _load_seal(project_root: Path, seal_relative: str) -> dict[str, Any]:
    """Parse the committed design seal and validate its frozen structure."""

    path = project_root / seal_relative
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise RuntimeError(
            f"stop condition: design seal is missing at {seal_relative}"
        ) from error
    try:
        seal = json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"stop condition: design seal is not valid JSON: {error}"
        ) from error
    if not isinstance(seal, dict):
        raise RuntimeError(  # noqa: TRY004 - stop conditions always use RuntimeError
            "stop condition: design seal must be a JSON object"
        )
    if seal.get("schema") != SEAL_SCHEMA:
        raise RuntimeError(
            f"stop condition: design seal schema must be {SEAL_SCHEMA!r}, got "
            f"{seal.get('schema')!r}"
        )
    required = (
        "schema",
        "design_commit",
        "protocol_sha256",
        "runner_sha256",
        "generator_sha256",
        "lock_sha256",
        "seed_interval",
        "no_preview_declaration",
        "primary_family",
        "stop_rules",
        "output_path",
    )
    missing = [key for key in required if key not in seal]
    if missing:
        raise RuntimeError(f"stop condition: design seal is missing keys: {missing}")
    commit = seal["design_commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise RuntimeError(
            "stop condition: design seal design_commit must be a full lowercase "
            "commit hash"
        )
    interval = seal["seed_interval"]
    if (
        not isinstance(interval, dict)
        or interval.get("first") != SEALED_SEEDS[0]
        or interval.get("last") != SEALED_SEEDS[-1]
    ):
        raise RuntimeError(
            "stop condition: design seal seed_interval must be exactly "
            f"{{'first': {SEALED_SEEDS[0]}, 'last': {SEALED_SEEDS[-1]}}}"
        )
    declaration = seal["no_preview_declaration"]
    if not isinstance(declaration, str) or not declaration.strip():
        raise RuntimeError(
            "stop condition: design seal no_preview_declaration must be a "
            "nonempty string"
        )
    if not isinstance(seal["stop_rules"], list) or not seal["stop_rules"]:
        raise RuntimeError(
            "stop condition: design seal stop_rules must be a nonempty list"
        )
    if seal["stop_rules"] != list(_SEAL_STOP_RULES):
        raise RuntimeError(
            "stop condition: design seal stop_rules differ from the runner's "
            "frozen stop conditions"
        )
    family = seal["primary_family"]
    if not isinstance(family, list) or len(family) != PRIMARY_FAMILY_SIZE:
        raise RuntimeError(
            "stop condition: design seal primary_family must list exactly the "
            "seven frozen claims"
        )
    for claim in family:
        if not isinstance(claim, dict) or not isinstance(claim.get("id"), str):
            raise RuntimeError(  # noqa: TRY004 - stop conditions use RuntimeError
                "stop condition: every design seal primary_family entry needs a "
                "string id"
            )
    # The seal binds the complete frozen claim objects -- theta, null,
    # alternative, reference arm, bound direction, threshold, and support rule --
    # not merely the claim identifiers.
    if family != list(_SEAL_PRIMARY_FAMILY):
        raise RuntimeError(
            "stop condition: design seal primary_family differs from the "
            "runner's seven frozen claim objects"
        )
    if not isinstance(seal["output_path"], str) or not seal["output_path"]:
        raise RuntimeError("stop condition: design seal output_path must be a string")
    for key in ("protocol_sha256", "runner_sha256", "generator_sha256", "lock_sha256"):
        _validate_sha256(seal[key], label=f"design seal {key}")
    return seal


def _preflight(
    project_root: Path,
    output: Path,
    *,
    seal: str,
) -> dict[str, Any]:
    """Fail closed before constructing any dataset from a sealed seed."""

    if output.exists():
        raise RuntimeError(f"stop condition: output already exists: {output}")
    try:
        output_relative = output.relative_to(project_root).as_posix()
    except ValueError as error:
        raise RuntimeError(
            "stop condition: output must lie inside the project root"
        ) from error
    status = _git_checked(project_root, "status", "--short", "--untracked-files=all")
    if status:
        raise RuntimeError("stop condition: working tree is dirty; no seed was opened")

    seal_path = Path(seal)
    if seal_path.is_absolute():
        try:
            seal_path = seal_path.relative_to(project_root)
        except ValueError as error:
            raise RuntimeError(
                "stop condition: the design seal must lie inside the project root"
            ) from error
    seal_relative = seal_path.as_posix()
    seal_record = _load_seal(project_root, seal_relative)
    # The seal itself must be committed at HEAD; the clean-worktree check above
    # then guarantees the on-disk seal is exactly the committed seal.
    _git_checked(project_root, "cat-file", "-e", f"HEAD:{seal_relative}")
    if seal_record["output_path"] != output_relative:
        raise RuntimeError(
            "stop condition: design seal output_path "
            f"{seal_record['output_path']!r} does not match --output "
            f"{output_relative!r}"
        )

    if seal_record["generator_sha256"] != FROZEN_GENERATOR_SHA256:
        raise RuntimeError(
            "stop condition: embedded generator fingerprint differs from the "
            "design seal"
        )
    if seal_record["lock_sha256"] != FROZEN_LOCKFILE_SHA256:
        raise RuntimeError(
            "stop condition: embedded lockfile fingerprint differs from the design seal"
        )
    if seal_record["protocol_sha256"] != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "stop condition: embedded protocol fingerprint differs from the design seal"
        )
    protocol_hash = _verified_sha256(
        project_root,
        PROTOCOL,
        FROZEN_PROTOCOL_SHA256,
        label="sealed v2 protocol",
    )
    runner_hash = _verified_sha256(
        project_root,
        RUNNER_SOURCE,
        seal_record["runner_sha256"],
        label="sealed v2 runner",
    )
    running_hash = _sha256(Path(__file__).resolve())
    if running_hash != runner_hash:
        raise RuntimeError(
            "stop condition: the running file's SHA-256 differs from the sealed "
            f"runner at {RUNNER_SOURCE}"
        )
    generator_hash = _verified_sha256(
        project_root,
        GENERATOR_SOURCE,
        FROZEN_GENERATOR_SHA256,
        label="conversion generator",
    )
    environment = _environment_provenance(project_root)
    execution = _execution_environment()
    revision = _git_checked(project_root, "rev-parse", "HEAD")
    # Lineage, not just content: the sealed design commit must sit in HEAD's
    # ancestry so the seal cannot name an unrelated or fabricated commit.
    _git_require_design_ancestry(project_root, seal_record["design_commit"])
    return {
        "git_revision": revision,
        "git_status": status,
        "design_commit": seal_record["design_commit"],
        "execution_revision": revision,
        "seal": {
            "path": seal_relative,
            "schema": SEAL_SCHEMA,
            "sha256": _sha256(project_root / seal_relative),
            "design_commit": seal_record["design_commit"],
            "committed_at_head": True,
        },
        "protocol": {"path": PROTOCOL, "sha256": protocol_hash},
        "runner": {"path": RUNNER_SOURCE, "sha256": runner_hash},
        "generator": {
            "class": "homymoly.data.conversion.ConversionDataset",
            "path": GENERATOR_SOURCE,
            "sha256": generator_hash,
        },
        "environment": environment,
        "execution": execution,
    }


def _atomic_json_new(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish JSON without ever replacing an existing result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError(
                f"stop condition: output appeared during execution: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _subseed(topology_seed: int, component: str, replicate: int = 0) -> int:
    """Derive a stable 63-bit seed without consuming shared RNG state."""

    if not component or ":" in component:
        raise ValueError("component must be a nonempty colon-free label")
    if replicate < 0:
        raise ValueError("replicate must be nonnegative")
    message = f"homymoly-lifting-v2:{topology_seed}:{component}:{replicate}"
    digest = hashlib.sha256(message.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) & ((1 << 63) - 1)


def _normal(shape: tuple[int, ...], seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float64)


def _regression_data(
    sample: Any, topology_seed: int, *, namespace: str
) -> RegressionData:
    edges = int(sample.num_edges)
    faces = int(sample.num_faces)
    truth = sample.boundary_2.mT
    train_x = _normal(
        (N_TRAIN, edges), _subseed(topology_seed, f"{namespace}-train-inputs")
    )
    train_noise = _normal(
        (N_TRAIN, faces), _subseed(topology_seed, f"{namespace}-training-noise")
    )
    test_x = _normal(
        (N_TEST, edges), _subseed(topology_seed, f"{namespace}-test-inputs")
    )
    return RegressionData(
        train_x=train_x,
        train_y=train_x @ truth.mT + NOISE_SD * train_noise,
        test_x=test_x,
        test_y=test_x @ truth.mT,
    )


def _c1_regression_data(
    sample: Any, topology_seed: int, replicate: int, test_x: Tensor
) -> RegressionData:
    edges = int(sample.num_edges)
    faces = int(sample.num_faces)
    truth = sample.boundary_2.mT
    train_x = _normal(
        (N_TRAIN, edges), _subseed(topology_seed, "c1-train-inputs", replicate)
    )
    noise = _normal(
        (N_TRAIN, faces), _subseed(topology_seed, "c1-training-noise", replicate)
    )
    return RegressionData(
        train_x=train_x,
        train_y=train_x @ truth.mT + NOISE_SD * noise,
        test_x=test_x,
        test_y=test_x @ truth.mT,
    )


def _canonicalize_columns(matrix: Tensor) -> Tensor:
    """Make each orthonormal column's largest-magnitude entry positive."""

    if matrix.ndim != 2:
        raise ValueError("matrix must have two dimensions")
    if matrix.shape[1] == 0:
        return matrix.clone()
    pivots = matrix.abs().argmax(dim=0)
    signs = torch.sign(matrix[pivots, torch.arange(matrix.shape[1])])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return matrix * signs


def _cycle_nullspace_basis(
    boundary_1: Tensor, expected_faces: int
) -> tuple[Tensor, dict[str, Any]]:
    """Return a deterministic-sign orthonormal basis for ``ker(B1)``.

    The returned certificate records the observed rank, the rank tolerance,
    and the asserted nullspace/orthonormality residuals. Any rank, dimension,
    orthogonality, or nullspace violation raises and fails the whole campaign.
    """

    if boundary_1.ndim != 2:
        raise ValueError("boundary_1 must be a matrix")
    edges = int(boundary_1.shape[1])
    vertices = int(boundary_1.shape[0])
    expected_rank = vertices - 1
    expected_cycle_rank = edges - vertices + 1
    if expected_cycle_rank != expected_faces:
        raise DesignFailureError(
            "connected-graph cycle-rank mismatch: "
            f"E-V+1={expected_cycle_rank}, declared faces={expected_faces}"
        )
    _, singular_values, vh = torch.linalg.svd(boundary_1, full_matrices=True)
    tolerance = (
        max(boundary_1.shape)
        * torch.finfo(boundary_1.dtype).eps
        * float(singular_values.max())
    )
    observed_rank = int((singular_values > tolerance).sum().item())
    if observed_rank != expected_rank:
        raise DesignFailureError(
            "cycle-nullspace rank mismatch: "
            f"observed rank {observed_rank}, expected {expected_rank}"
        )
    # For a connected incidence matrix rank(B1)=V-1, so the full right singular
    # vectors Vh[V-1:] span the cycle space. The rank check above fails closed
    # if the generated graph is not connected or has an unexpected incidence.
    basis = _canonicalize_columns(vh[vertices - 1 :].mT.contiguous())
    if basis.shape != (edges, expected_faces):
        raise DesignFailureError("cycle-nullspace basis has the wrong shape")
    defect = float(torch.linalg.matrix_norm(boundary_1 @ basis))
    if defect > BASIS_TOLERANCE:
        raise DesignFailureError(f"cycle-nullspace basis defect is {defect}")
    orthonormality = float(
        torch.linalg.matrix_norm(
            basis.mT @ basis - torch.eye(expected_faces, dtype=basis.dtype)
        )
    )
    if orthonormality > BASIS_TOLERANCE:
        raise DesignFailureError(
            f"cycle-nullspace basis orthonormality defect is {orthonormality}"
        )
    certificate = {
        "observed_rank": observed_rank,
        "expected_rank": expected_rank,
        "rank_tolerance": tolerance,
        "boundary_defect_frobenius": defect,
        "orthonormality_defect_frobenius": orthonormality,
        "asserted_tolerance": BASIS_TOLERANCE,
    }
    return basis, certificate


def _matched_random_basis(
    edges: int, faces: int, seed: int
) -> tuple[Tensor, dict[str, Any]]:
    """Return a deterministic Haar-like dimension-matched orthonormal basis.

    Every diagonal entry of the reduced QR factor must be finite and nonzero,
    and orthonormality is asserted at the frozen tolerance; any violation
    raises and fails the whole campaign.
    """

    if not 0 < faces <= edges:
        raise ValueError("random-subspace dimensions must satisfy 0 < F <= E")
    gaussian = _normal((edges, faces), seed)
    q, r = torch.linalg.qr(gaussian, mode="reduced")
    diagonal = torch.diagonal(r)
    if not bool(torch.all(torch.isfinite(diagonal))) or bool(torch.any(diagonal == 0)):
        raise DesignFailureError(
            "random-subspace QR diagonal entries must be finite and nonzero"
        )
    min_abs_diagonal = float(diagonal.abs().min())
    signs = torch.where(
        diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal)
    )
    basis = q * signs
    orthonormality = float(
        torch.linalg.matrix_norm(basis.mT @ basis - torch.eye(faces, dtype=basis.dtype))
    )
    if orthonormality > BASIS_TOLERANCE:
        raise DesignFailureError(
            f"random-subspace basis orthonormality defect is {orthonormality}"
        )
    certificate = {
        "min_abs_diagonal_r": min_abs_diagonal,
        "orthonormality_defect_frobenius": orthonormality,
        "asserted_tolerance": BASIS_TOLERANCE,
    }
    return basis, certificate


def _held_out_mse(matrix: Tensor, test_x: Tensor, test_y: Tensor) -> float:
    value = float(((test_x @ matrix.mT) - test_y).pow(2).mean())
    if not math.isfinite(value) or value < 0:
        raise DesignFailureError(f"invalid held-out MSE: {value}")
    return value


def _defects(
    matrix: Tensor, boundary_1: Tensor, random_basis: Tensor
) -> tuple[float, float]:
    edge_to_output = matrix.mT
    boundary = float(torch.linalg.matrix_norm(boundary_1 @ edge_to_output))
    random_residual = edge_to_output - random_basis @ (random_basis.mT @ edge_to_output)
    random = float(torch.linalg.matrix_norm(random_residual))
    if not math.isfinite(boundary) or not math.isfinite(random):
        raise DesignFailureError("non-finite structural defect")
    return boundary, random


def _evaluate_fit(
    matrix: Tensor,
    test_x: Tensor,
    test_y: Tensor,
    boundary_1: Tensor,
    random_basis: Tensor,
    metadata: dict[str, Any],
) -> FitResult:
    """Evaluate an already fitted matrix; this stays outside every fit API.

    The fitted matrix is detached before any diagnostic is computed so the
    evaluation graph never touches the autograd tape of a learned arm.
    """

    detached = matrix.detach()
    boundary, random = _defects(detached, boundary_1, random_basis)
    return FitResult(
        matrix=detached,
        held_out_mse=_held_out_mse(detached, test_x, test_y),
        boundary_defect=boundary,
        random_subspace_defect=random,
        metadata=dict(metadata),
    )


def _lstsq_gelsd(design: Tensor, target: Tensor) -> tuple[Tensor, dict[str, Any]]:
    """Run the frozen gelsd least-squares convention and record its rank."""

    result = torch.linalg.lstsq(design, target, rcond=LSTSQ_RCOND, driver="gelsd")
    min_singular_value = float(result.singular_values.min())
    if not math.isfinite(min_singular_value) or min_singular_value <= 0:
        raise DesignFailureError(
            "gelsd returned a non-finite or nonpositive smallest singular value: "
            f"{min_singular_value}"
        )
    if int(result.rank) < 1:
        raise DesignFailureError(
            f"gelsd returned a numerical rank below one: {int(result.rank)}"
        )
    metadata = {
        "driver": "gelsd",
        "rcond": LSTSQ_RCOND,
        "gelsd_returned_rank": int(result.rank),
        "gelsd_min_singular_value": min_singular_value,
    }
    return result.solution, metadata


def _fit_adam(
    train_x: Tensor,
    train_y: Tensor,
    *,
    term: str | None,
    weight: float,
    boundary_1: Tensor | None = None,
) -> FittedMatrix:
    """Fit a learned Adam arm from training tensors only.

    Only the soft boundary arm may receive ``boundary_1``; the graph-blind
    arms (ambient, singular-value, RTD-inspired) never see it.
    """

    if term == "boundary":
        if boundary_1 is None:
            raise ValueError("the boundary penalty term requires boundary_1")
    elif boundary_1 is not None:
        raise ValueError("graph-blind Adam arms must not receive boundary_1")
    faces = int(train_y.shape[1])
    edges = int(train_x.shape[1])
    matrix = torch.zeros((faces, edges), dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.Adam([matrix], lr=LEARNING_RATE)
    source_distances = None
    if term == "rtd_inspired":
        raw_source = torch.cdist(train_x, train_x)
        source_distances = raw_source / (raw_source.mean() + 1e-12)

    def objective() -> Tensor:
        predicted = train_x @ matrix.mT
        loss = (predicted - train_y).pow(2).mean()
        if term == "boundary":
            assert boundary_1 is not None
            loss = loss + weight * (boundary_1 @ matrix.mT).pow(2).mean()
        elif term == "singular_value":
            loss = loss + weight * torch.exp(-2.0 * torch.linalg.svdvals(matrix).min())
        elif term == "rtd_inspired":
            mapped = torch.cdist(predicted, predicted)
            normalised_mapped = mapped / (mapped.mean() + 1e-12)
            assert source_distances is not None
            loss = loss + weight * (normalised_mapped - source_distances).pow(2).mean()
        elif term is not None:
            raise ValueError(f"unknown Adam term: {term}")
        return loss

    for _ in range(STEPS):
        optimiser.zero_grad()
        objective().backward()
        optimiser.step()
    optimiser.zero_grad()
    objective().backward()
    final_gradient_norm = float(torch.linalg.vector_norm(matrix.grad))
    if not math.isfinite(final_gradient_norm):
        raise DesignFailureError("non-finite final full-batch gradient norm")
    return FittedMatrix(
        matrix=matrix.detach().clone(),
        metadata={
            "estimator": "Adam",
            "term": term,
            "weight": weight,
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "final_full_batch_gradient_norm": final_gradient_norm,
        },
    )


def _ambient_min_norm_ls(train_x: Tensor, train_y: Tensor) -> FittedMatrix:
    """Graph-blind full-space minimum-norm least squares (training data only)."""

    solution, lstsq_metadata = _lstsq_gelsd(train_x, train_y)
    return FittedMatrix(
        matrix=solution.mT.detach().clone(),
        metadata={
            "estimator": "torch.linalg.lstsq",
            "constraint": "ambient minimum-norm least squares",
            **lstsq_metadata,
        },
    )


def _subspace_ls(
    train_x: Tensor,
    train_y: Tensor,
    fit_basis: Tensor,
    *,
    name: str,
) -> FittedMatrix:
    """Least squares restricted to a precomputed orthonormal subspace basis."""

    reduced = train_x @ fit_basis
    solution, lstsq_metadata = _lstsq_gelsd(reduced, train_y)
    matrix = (fit_basis @ solution).mT
    return FittedMatrix(
        matrix=matrix.detach().clone(),
        metadata={
            "estimator": "torch.linalg.lstsq",
            "constraint": name,
            "subspace_dimension": int(fit_basis.shape[1]),
            **lstsq_metadata,
        },
    )


def _ridge_matrix(train_x: Tensor, train_y: Tensor, alpha: float) -> Tensor:
    gram = train_x.mT @ train_x
    right = train_x.mT @ train_y
    identity = torch.eye(gram.shape[0], dtype=train_x.dtype)
    return torch.linalg.solve(gram + alpha * identity, right).mT


def _fit_inner_cv_ridge(train_x: Tensor, train_y: Tensor) -> FittedMatrix:
    """Graph-blind ridge with deterministic training-only four-fold CV."""

    train_rows = int(train_x.shape[0])
    candidates = []
    for alpha in RIDGE_GRID:
        fold_losses = []
        for fold in range(RIDGE_FOLDS):
            validation_mask = torch.arange(train_rows) % RIDGE_FOLDS == fold
            train_mask = ~validation_mask
            matrix = _ridge_matrix(train_x[train_mask], train_y[train_mask], alpha)
            validation_error = (
                train_x[validation_mask] @ matrix.mT - train_y[validation_mask]
            )
            fold_losses.append(float(validation_error.pow(2).mean()))
        candidates.append((statistics.fmean(fold_losses), alpha, fold_losses))
    # Grid order and the tuple's second component make exact ties choose the
    # smaller alpha. No held-out endpoint is involved in selection.
    _, selected, _ = min(candidates, key=lambda row: (row[0], row[1]))
    matrix = _ridge_matrix(train_x, train_y, selected)
    return FittedMatrix(
        matrix=matrix.detach().clone(),
        metadata={
            "estimator": "closed-form ridge",
            "selected_alpha": selected,
            "grid": list(RIDGE_GRID),
            "folds": RIDGE_FOLDS,
            "fold_assignment": "row index modulo 4",
            "train_rows_per_fold": 12,
            "validation_rows_per_fold": 4,
            "selection": (
                "minimum mean four-fold validation MSE; exact ties choose smaller alpha"
            ),
            "cross_validation_by_alpha": [
                {
                    "alpha": alpha,
                    "mean_validation_mse": value,
                    "fold_validation_mse": fold_losses,
                }
                for value, alpha, fold_losses in candidates
            ],
        },
    )


def _soft_boundary_closed_form_lambda3(
    train_x: Tensor, train_y: Tensor, boundary_1: Tensor
) -> FittedMatrix:
    """Solve the exact mean-normalised lambda=3 quadratic objective."""

    train_rows = int(train_x.shape[0])
    scale = SOFT_BOUNDARY_WEIGHT * train_rows / int(boundary_1.shape[0])
    system = train_x.mT @ train_x + scale * (boundary_1.mT @ boundary_1)
    right_hand_side = train_x.mT @ train_y
    coefficients = torch.linalg.pinv(system, rcond=PINV_RCOND) @ right_hand_side
    singular_values = torch.linalg.svdvals(system)
    cutoff = PINV_RCOND * float(singular_values.max())
    effective_rank = int((singular_values > cutoff).sum().item())
    # Frozen stationarity assertion of protocol section 7.3: the normal equation
    # is consistent by construction, so a residual above the frozen relative
    # tolerance indicates an implementation fault and fails the whole design.
    stationarity_residual = float(
        torch.linalg.matrix_norm(system @ coefficients - right_hand_side)
    )
    stationarity_bound = STATIONARITY_RTOL * float(
        torch.linalg.matrix_norm(right_hand_side)
    )
    if not math.isfinite(stationarity_residual) or (
        stationarity_residual > stationarity_bound
    ):
        raise DesignFailureError(
            "soft closed-form stationarity residual "
            f"{stationarity_residual} exceeds the frozen bound {stationarity_bound}"
        )
    return FittedMatrix(
        matrix=coefficients.mT.detach().clone(),
        metadata={
            "estimator": "closed-form quadratic solution via torch.linalg.pinv",
            "objective": ("mean((X @ W.T - Y)^2) + 3.0 * mean((B1 @ W.T)^2)"),
            "normal_equation_boundary_scale": scale,
            "rcond": PINV_RCOND,
            "pinv_effective_rank": effective_rank,
            "pinv_rank_cutoff": cutoff,
            "pinv_min_singular_value": float(singular_values.min()),
            "stationarity_residual_frobenius": stationarity_residual,
            "stationarity_bound_frobenius": stationarity_bound,
            "stationarity_relative_tolerance": STATIONARITY_RTOL,
        },
    )


_ORACLE_METADATA = {
    "estimator": "analytic generator cycle-basis oracle",
    "uses_withheld_generator_basis": True,
    "inference_role": "descriptive attainability ceiling only",
}


def _serialise_fit(fit: FitResult) -> dict[str, Any]:
    return {
        "held_out_mse": fit.held_out_mse,
        "boundary_compatibility_defect_frobenius": fit.boundary_defect,
        "matched_random_subspace_defect_frobenius": fit.random_subspace_defect,
        "metadata": fit.metadata,
    }


def _evaluate_primary(sample: Any, topology_seed: int) -> dict[str, Any]:
    boundary_1 = sample.boundary_1.to(dtype=torch.float64)
    data = _regression_data(sample, topology_seed, namespace="primary")
    cycle_basis, cycle_certificate = _cycle_nullspace_basis(
        boundary_1, int(sample.num_faces)
    )
    random_basis, random_certificate = _matched_random_basis(
        int(sample.num_edges),
        int(sample.num_faces),
        _subseed(topology_seed, "matched-random-subspace"),
    )

    # Information-flow firewall: each fit receives only its declared training
    # tensors plus, where applicable, boundary_1 or the seeded random basis.
    # The graph-blind arms (ambient_adam, ambient_min_norm_ls, inner_cv_ridge)
    # never receive boundary_1; no fit ever receives B2 or held-out tensors.
    # Held-out MSE and defects are evaluated here, outside the fit APIs.
    fitters: dict[str, Callable[[], FittedMatrix]] = {
        "ambient_adam": lambda: _fit_adam(
            data.train_x, data.train_y, term=None, weight=0.0
        ),
        "ambient_min_norm_ls": lambda: _ambient_min_norm_ls(data.train_x, data.train_y),
        "soft_boundary_lambda3": lambda: _fit_adam(
            data.train_x,
            data.train_y,
            term="boundary",
            weight=SOFT_BOUNDARY_WEIGHT,
            boundary_1=boundary_1,
        ),
        "soft_boundary_closed_form_lambda3": (
            lambda: _soft_boundary_closed_form_lambda3(
                data.train_x, data.train_y, boundary_1
            )
        ),
        "hard_cycle_ls": lambda: _subspace_ls(
            data.train_x, data.train_y, cycle_basis, name="ker(B1)"
        ),
        "hard_random_subspace_ls": lambda: _subspace_ls(
            data.train_x,
            data.train_y,
            random_basis,
            name="SHA256-seeded dimension-matched random subspace",
        ),
        "inner_cv_ridge": lambda: _fit_inner_cv_ridge(data.train_x, data.train_y),
        "singular_value_surrogate": lambda: _fit_adam(
            data.train_x,
            data.train_y,
            term="singular_value",
            weight=SINGULAR_VALUE_WEIGHT,
        ),
        "rtd_inspired_distance_surrogate": lambda: _fit_adam(
            data.train_x,
            data.train_y,
            term="rtd_inspired",
            weight=RTD_INSPIRED_WEIGHT,
        ),
    }
    fits: dict[str, FitResult] = {}
    for name, fitter in fitters.items():
        try:
            fitted = fitter()
            fits[name] = _evaluate_fit(
                fitted.matrix,
                data.test_x,
                data.test_y,
                boundary_1,
                random_basis,
                fitted.metadata,
            )
        except BaseException as error:
            # Attach the frozen arm identifier to any failure or interruption.
            if getattr(error, "arm", None) is None:
                error.arm = name
            raise
    # Truth-access oracle: isolated from every fitted arm and from inference.
    try:
        oracle_matrix = sample.boundary_2.mT
        fits["generator_cycle_basis_oracle"] = _evaluate_fit(
            oracle_matrix,
            data.test_x,
            data.test_y,
            boundary_1,
            random_basis,
            _ORACLE_METADATA,
        )
    except BaseException as error:
        if getattr(error, "arm", None) is None:
            error.arm = "generator_cycle_basis_oracle"
        raise
    if tuple(fits) != ARM_NAMES:
        raise RuntimeError("primary arms differ from the frozen order")
    test_second_moment = float(data.test_y.pow(2).mean())
    if not math.isfinite(test_second_moment) or test_second_moment <= 0:
        raise DesignFailureError(
            "held-out target second moment must be finite and strictly positive"
        )
    oracle = fits["generator_cycle_basis_oracle"]
    oracle_relative_error = oracle.held_out_mse / test_second_moment
    if not math.isfinite(oracle_relative_error) or oracle_relative_error < 0:
        raise DesignFailureError(
            f"invalid oracle relative error: {oracle_relative_error}"
        )
    solution_gap = float(
        torch.linalg.matrix_norm(
            fits["soft_boundary_lambda3"].matrix
            - fits["soft_boundary_closed_form_lambda3"].matrix
        )
    )
    if not math.isfinite(solution_gap):
        raise DesignFailureError("non-finite soft Adam versus closed-form solution gap")
    arms = {name: _serialise_fit(fit) for name, fit in fits.items()}
    # Raw MSE only, plus the frozen relative error; never a log ratio.
    arms["generator_cycle_basis_oracle"]["mean_squared_test_target"] = (
        test_second_moment
    )
    arms["generator_cycle_basis_oracle"][
        "relative_error_to_mean_squared_test_target"
    ] = oracle_relative_error
    return {
        "seed": topology_seed,
        "sample_id": sample.sample_id,
        "vertices": int(sample.num_vertices),
        "edges": int(sample.num_edges),
        "faces": int(sample.num_faces),
        "subseeds": {
            label: _subseed(topology_seed, label)
            for label in (
                "primary-train-inputs",
                "primary-training-noise",
                "primary-test-inputs",
                "matched-random-subspace",
            )
        },
        "cycle_nullspace_certificate": {
            "basis_shape": list(cycle_basis.shape),
            **cycle_certificate,
        },
        "random_subspace_certificate": {
            "basis_shape": list(random_basis.shape),
            **random_certificate,
        },
        "optimizer_descriptive": {
            "soft_adam_vs_closed_form_solution_gap_frobenius": solution_gap,
            "role": "descriptive optimization audit; outside the primary family",
        },
        "arms": arms,
    }


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Pearson correlation requires equal vectors of length >= 2")
    if not all(math.isfinite(value) for value in (*left, *right)):
        raise ValueError("Pearson inputs must be finite")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_centred = [value - left_mean for value in left]
    right_centred = [value - right_mean for value in right]
    numerator = math.fsum(
        a * b for a, b in zip(left_centred, right_centred, strict=True)
    )
    denominator = math.sqrt(
        math.fsum(value * value for value in left_centred)
        * math.fsum(value * value for value in right_centred)
    )
    if denominator == 0:
        raise DesignFailureError(
            "Pearson correlation is undefined for a constant vector"
        )
    return min(1.0, max(-1.0, numerator / denominator))


def _fisher_z(correlation: float) -> float:
    if not math.isfinite(correlation):
        raise ValueError("correlation must be finite")
    limit = math.nextafter(1.0, 0.0)
    return math.atanh(min(limit, max(-limit, correlation)))


def _evaluate_c1(sample: Any, topology_seed: int) -> dict[str, Any]:
    boundary_1 = sample.boundary_1.to(dtype=torch.float64)
    cycle_basis, _ = _cycle_nullspace_basis(boundary_1, int(sample.num_faces))
    test_x = _normal(
        (N_TEST, int(sample.num_edges)), _subseed(topology_seed, "c1-test-inputs")
    )
    random_basis, _ = _matched_random_basis(
        int(sample.num_edges),
        int(sample.num_faces),
        _subseed(topology_seed, "matched-random-subspace"),
    )
    rows = []
    for replicate in range(C1_REPLICATES):
        if replicate == 0:
            primary_data = _regression_data(sample, topology_seed, namespace="primary")
            data = RegressionData(
                train_x=primary_data.train_x,
                train_y=primary_data.train_y,
                test_x=test_x,
                test_y=test_x @ sample.boundary_2.mT.mT,
            )
            train_inputs_component = "primary-train-inputs"
            training_noise_component = "primary-training-noise"
        else:
            data = _c1_regression_data(sample, topology_seed, replicate, test_x)
            train_inputs_component = "c1-train-inputs"
            training_noise_component = "c1-training-noise"
        # The C1 estimator is graph-blind: the fit sees training tensors only.
        fitted = _ambient_min_norm_ls(data.train_x, data.train_y)
        evaluated = _evaluate_fit(
            fitted.matrix,
            data.test_x,
            data.test_y,
            boundary_1,
            random_basis,
            fitted.metadata,
        )
        edge_to_output = evaluated.matrix.mT
        cycle_residual = edge_to_output - cycle_basis @ (
            cycle_basis.mT @ edge_to_output
        )
        rows.append(
            {
                "replicate": replicate,
                "reuses_primary_training_realisation": replicate == 0,
                "train_inputs_subseed": _subseed(
                    topology_seed, train_inputs_component, replicate
                ),
                "training_noise_subseed": _subseed(
                    topology_seed, training_noise_component, replicate
                ),
                "held_out_mse": evaluated.held_out_mse,
                "boundary_compatibility_defect_frobenius": evaluated.boundary_defect,
                "cycle_projector_defect_frobenius": float(
                    torch.linalg.matrix_norm(cycle_residual)
                ),
                "matched_random_subspace_defect_frobenius": (
                    evaluated.random_subspace_defect
                ),
                "metadata": evaluated.metadata,
            }
        )
    # Every held-out MSE and every defect vector -- including the legacy raw
    # boundary defect -- must be finite and strictly positive; otherwise C1 is
    # undefined and the whole campaign fails. No epsilon floor is ever
    # substituted after observing outcomes.
    vectors = {
        "held-out MSE": [row["held_out_mse"] for row in rows],
        "cycle-projector defect": [
            row["cycle_projector_defect_frobenius"] for row in rows
        ],
        "matched-random-subspace defect": [
            row["matched_random_subspace_defect_frobenius"] for row in rows
        ],
        "boundary defect": [
            row["boundary_compatibility_defect_frobenius"] for row in rows
        ],
    }
    for label, values in vectors.items():
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise DesignFailureError(
                f"C1 {label} vector must be finite and strictly positive across "
                f"all {C1_REPLICATES} replicates"
            )
    log_error = [math.log10(value) for value in vectors["held-out MSE"]]
    cycle_r = _correlation(
        [math.log10(value) for value in vectors["cycle-projector defect"]],
        log_error,
    )
    random_r = _correlation(
        [math.log10(value) for value in vectors["matched-random-subspace defect"]],
        log_error,
    )
    return {
        "seed": topology_seed,
        "sample_id": sample.sample_id,
        "shared_test_subseed": _subseed(topology_seed, "c1-test-inputs"),
        "shared_test_rows": N_TEST,
        "replicate_zero_reuses_primary_train_inputs_and_noise": True,
        "replicates": rows,
        "cycle_projector_correlation": cycle_r,
        "cycle_projector_fisher_z": _fisher_z(cycle_r),
        "matched_random_correlation": random_r,
        "matched_random_fisher_z": _fisher_z(random_r),
        "estimator": "ambient minimum-norm torch.linalg.lstsq",
        "truth_used_only_for_responses": True,
    }


def _sign_test(values: list[float]) -> dict[str, Any]:
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    ties = len(values) - positive - negative
    trials = positive + negative
    if trials == 0:
        pvalue = None
    else:
        tail = sum(
            math.comb(trials, index) for index in range(min(positive, negative) + 1)
        )
        pvalue = min(1.0, 2.0 * tail / (2.0**trials))
    return {
        "pvalue_two_sided": pvalue,
        "negative": negative,
        "positive": positive,
        "ties_discarded": ties,
        "role": "direction-neutral sensitivity analysis",
    }


def _paired_log_ratios(
    rows: list[dict[str, Any]], numerator: str, denominator: str
) -> list[float]:
    values = []
    for row in rows:
        numerator_mse = float(row["arms"][numerator]["held_out_mse"])
        denominator_mse = float(row["arms"][denominator]["held_out_mse"])
        if numerator_mse <= 0 or denominator_mse <= 0:
            raise DesignFailureError("log-ratio endpoints must be strictly positive")
        values.append(math.log10(numerator_mse / denominator_mse))
    return values


def _one_sided_claim(
    *,
    claim_id: str,
    description: str,
    numerator: str,
    denominator: str,
    values: list[float],
    direction: str,
    threshold: float = 0.0,
) -> dict[str, Any]:
    degrees = len(values) - 1
    if degrees not in _T_ONE_SIDED_BONFERRONI:
        raise RuntimeError(f"no frozen Bonferroni t critical value for df={degrees}")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    standard_error = standard_deviation / math.sqrt(len(values))
    half = _T_ONE_SIDED_BONFERRONI[degrees] * standard_error
    if direction == "less":
        bound_name = "one_sided_upper_bound"
        bound = mean + half
        supported = bound < threshold
    elif direction == "greater":
        bound_name = "one_sided_lower_bound"
        bound = mean - half
        supported = bound > threshold
    else:
        raise ValueError("claim direction must be 'less' or 'greater'")
    return {
        "id": claim_id,
        "description": description,
        "numerator_arm": numerator,
        "reference_arm": denominator,
        "endpoint": f"log10(MSE_{numerator} / MSE_{denominator})",
        "alternative": f"mean {direction} {threshold}",
        "direction": direction,
        "threshold": threshold,
        "n": len(values),
        "estimate": mean,
        "mean_log10_ratio": mean,
        "median_log10_ratio": statistics.median(values),
        "sample_standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "geometric_mean_ratio": 10.0**mean,
        "two_sided_interval_95_descriptive": _two_sided_interval(values),
        "critical_value": _T_ONE_SIDED_BONFERRONI[degrees],
        "critical_quantile": 1.0 - PRIMARY_ALPHA / PRIMARY_FAMILY_SIZE,
        bound_name: bound,
        "supported": supported,
        "per_seed_log10_ratio": values,
        "sensitivity_sign_test": _sign_test(values),
    }


def _claim_stub(definition: dict[str, Any]) -> dict[str, Any]:
    """The frozen identity of one claim, shared by decisions and null rows."""

    return {
        "id": definition["id"],
        "description": definition["description"],
        "numerator_arm": definition["numerator"],
        "reference_arm": definition["denominator"],
        "endpoint": (
            f"log10(MSE_{definition['numerator']} / MSE_{definition['denominator']})"
        ),
        "direction": definition["direction"],
        "threshold": definition["threshold"],
    }


def _primary_inference(rows: list[dict[str, Any]]) -> dict[str, Any]:
    claims = []
    for definition in _PRIMARY_CLAIM_DEFINITIONS:
        claims.append(
            _one_sided_claim(
                claim_id=definition["id"],
                description=definition["description"],
                numerator=definition["numerator"],
                denominator=definition["denominator"],
                values=_paired_log_ratios(
                    rows, definition["numerator"], definition["denominator"]
                ),
                direction=definition["direction"],
                threshold=definition["threshold"],
            )
        )
    return {
        "family_size": PRIMARY_FAMILY_SIZE,
        "familywise_alpha": PRIMARY_ALPHA,
        "per_claim_alpha": PRIMARY_ALPHA / PRIMARY_FAMILY_SIZE,
        "method": "one-sided Student-t bounds with Bonferroni correction",
        "inference_unit": (
            "one eligible generator seed jointly determining topology, predictors, and noise"
        ),
        "estimand": "mean paired log10 held-out-MSE ratio over eligible seeds",
        "exchangeability_assumption": (
            "eligible seed-level joint realizations are exchangeable; topology and "
            "data/noise heterogeneity are not separated"
        ),
        "claims": claims,
    }


def _null_primary_inference() -> dict[str, Any]:
    """All seven frozen claims with a null decision after a failed campaign."""

    claims = []
    for definition in _PRIMARY_CLAIM_DEFINITIONS:
        claims.append(
            {
                **_claim_stub(definition),
                "alternative": (
                    f"mean {definition['direction']} {definition['threshold']}"
                ),
                "estimate": None,
                "standard_error": None,
                "geometric_mean_ratio": None,
                "two_sided_interval_95_descriptive": None,
                "critical_value": None,
                "supported": None,
            }
        )
    return {
        "family_size": PRIMARY_FAMILY_SIZE,
        "familywise_alpha": PRIMARY_ALPHA,
        "per_claim_alpha": PRIMARY_ALPHA / PRIMARY_FAMILY_SIZE,
        "method": "one-sided Student-t bounds with Bonferroni correction",
        "claims": claims,
        "role": "no confirmatory decision: the campaign did not complete",
    }


def _two_sided_interval(values: list[float]) -> list[float]:
    degrees = len(values) - 1
    if degrees not in _T975:
        raise RuntimeError(f"no frozen 95% t critical value for df={degrees}")
    half = _T975[degrees] * statistics.stdev(values) / math.sqrt(len(values))
    mean = statistics.fmean(values)
    return [mean - half, mean + half]


def _c1_inference(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cycle = [float(row["cycle_projector_fisher_z"]) for row in rows]
    random = [float(row["matched_random_fisher_z"]) for row in rows]

    def summary(values: list[float], *, primary: bool) -> dict[str, Any]:
        interval = _two_sided_interval(values)
        mean = statistics.fmean(values)
        return {
            "n_topologies": len(values),
            "mean_fisher_z": mean,
            "interval_95_fisher_z": interval,
            "back_transformed_mean_r": math.tanh(mean),
            "back_transformed_interval_95_r": [math.tanh(value) for value in interval],
            "role": (
                "prespecified descriptive secondary C1 endpoint; no decision"
                if primary
                else "specificity sensitivity analysis; no confirmatory decision"
            ),
            "per_topology_fisher_z": values,
        }

    delta = [left - right for left, right in zip(cycle, random, strict=True)]
    return {
        "design": (
            "12 independent ambient minimum-norm least-squares training/noise "
            "replicates per topology "
            "with one common independently generated noiseless test set"
        ),
        "estimator": (
            "within-topology conventional Pearson r, clipped inside (-1,1), "
            "then Fisher atanh and seed-level Student-t aggregation"
        ),
        "does_not_establish": ["causation", "calibration", "real-data transfer"],
        "cycle_projector_defect": summary(cycle, primary=True),
        "matched_random_subspace_defect": summary(random, primary=False),
        "paired_specificity_delta_fisher_z": {
            "definition": "cycle-projector Fisher z minus random-projector Fisher z",
            "mean": statistics.fmean(delta),
            "interval_95": _two_sided_interval(delta),
            "per_topology": delta,
            "role": "descriptive paired specificity contrast; no decision",
        },
        "per_topology": rows,
    }


def _descriptive_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = _paired_log_ratios(rows, "ambient_adam", "ambient_min_norm_ls")
    soft_gap_ratios = _paired_log_ratios(
        rows, "soft_boundary_lambda3", "soft_boundary_closed_form_lambda3"
    )
    oracle = [
        float(row["arms"]["generator_cycle_basis_oracle"]["held_out_mse"])
        for row in rows
    ]
    oracle_relative = [
        float(
            row["arms"]["generator_cycle_basis_oracle"][
                "relative_error_to_mean_squared_test_target"
            ]
        )
        for row in rows
    ]
    solution_gaps = [
        float(
            row["optimizer_descriptive"][
                "soft_adam_vs_closed_form_solution_gap_frobenius"
            ]
        )
        for row in rows
    ]
    adam_gradient_norms = {
        arm: [
            float(row["arms"][arm]["metadata"]["final_full_batch_gradient_norm"])
            for row in rows
        ]
        for arm in (
            "ambient_adam",
            "soft_boundary_lambda3",
            "singular_value_surrogate",
            "rtd_inspired_distance_surrogate",
        )
    }
    return {
        "ambient_adam_vs_min_norm_ls": {
            "mean_log10_ratio": statistics.fmean(values),
            "median_log10_ratio": statistics.median(values),
            "per_seed_log10_ratio": values,
            "sensitivity_sign_test": _sign_test(values),
            "role": "descriptive optimizer diagnostic; outside the primary family",
        },
        "soft_boundary_adam_vs_closed_form": {
            "mean_log10_ratio": statistics.fmean(soft_gap_ratios),
            "per_seed_log10_ratio": soft_gap_ratios,
            "role": "descriptive optimizer-gap diagnostic; outside the primary family",
        },
        "generator_cycle_basis_oracle": {
            "minimum_held_out_mse": min(oracle),
            "median_held_out_mse": statistics.median(oracle),
            "maximum_held_out_mse": max(oracle),
            "per_seed_held_out_mse": oracle,
            "maximum_relative_error_to_mean_squared_test_target": max(oracle_relative),
            "per_seed_relative_error_to_mean_squared_test_target": oracle_relative,
            "role": "descriptive attainability ceiling; outside efficacy inference",
        },
        "optimization_audits": {
            "soft_adam_vs_closed_form_solution_gap_frobenius": {
                "per_seed": solution_gaps,
                "mean": statistics.fmean(solution_gaps),
                "maximum": max(solution_gaps),
                "role": "descriptive solution gap ||A_adam - A_closed_form||_F",
            },
            "adam_final_full_batch_gradient_norm": {
                arm: {"per_seed": norms, "maximum": max(norms)}
                for arm, norms in adam_gradient_norms.items()
            },
            "stationarity_role": (
                "descriptive stationarity gap of each learned arm's training "
                "objective; outside the primary family"
            ),
        },
    }


def _audit_block(
    eligibility: dict[str, Any],
    primary_rows: list[dict[str, Any]],
    c1_rows: list[dict[str, Any]],
    *,
    complete: bool,
) -> dict[str, Any]:
    """Counts and raw-row-recomputable means for independent verification."""

    block: dict[str, Any] = {
        "recompute_scope": (
            "every value in this block is recomputable from the retained raw rows"
        ),
        "declared_seeds": eligibility["declared"],
        "eligible_seeds": eligibility["eligible"],
        "eligible_seed_ids": list(eligibility["eligible_seeds"]),
        "ineligible_seed_rows": len(eligibility["ineligible"]),
        "generation_failure_rows": len(eligibility["generation_failures"]),
        "raw_primary_rows": len(primary_rows),
        "raw_c1_rows": len(c1_rows),
        "c1_replicates_per_topology": C1_REPLICATES,
        "arm_names": list(ARM_NAMES),
    }
    if complete:
        block["mean_per_seed_log10_ratio_by_claim"] = {
            definition["id"]: statistics.fmean(
                _paired_log_ratios(
                    primary_rows, definition["numerator"], definition["denominator"]
                )
            )
            for definition in _PRIMARY_CLAIM_DEFINITIONS
        }
        block["mean_per_seed_log10_ratio_ambient_adam_vs_min_norm_ls"] = (
            statistics.fmean(
                _paired_log_ratios(primary_rows, "ambient_adam", "ambient_min_norm_ls")
            )
        )
    return block


def _design_record() -> dict[str, Any]:
    return {
        "declared_seeds": list(SEALED_SEEDS),
        "eligibility": "all connected generated cases with F >= 3; no replacement",
        "minimum_eligible": MIN_ELIGIBLE,
        "training_pairs": N_TRAIN,
        "held_out_pairs": N_TEST,
        "training_label_noise": {
            "distribution": "independent zero-mean Gaussian",
            "standard_deviation": NOISE_SD,
        },
        "held_out_targets": "noiseless ground-truth linear responses",
        "learned_arms": {
            "initialisation": "W = 0",
            "dtype": "float64",
            "optimiser": "Adam",
            "learning_rate": LEARNING_RATE,
            "steps": STEPS,
        },
        "paired_arm_sharing": (
            "within each seed all primary arms share topology, train inputs, label "
            "noise, test inputs, noiseless targets, and learned initialization"
        ),
        "subseed_derivation": (
            "first eight big-endian bytes of SHA256('homymoly-lifting-v2:' + "
            "topology_seed + ':' + component + ':' + replicate), masked to 63 bits"
        ),
        "arms": list(ARM_NAMES),
        "inner_cv_ridge": {
            "grid": list(RIDGE_GRID),
            "folds": RIDGE_FOLDS,
            "fold_assignment": "row index modulo 4",
            "train_rows_per_fold": 12,
            "validation_rows_per_fold": 4,
            "selection": "minimum mean fold MSE; exact ties choose smaller alpha",
            "refit": "closed-form ridge on all 16 training rows",
            "held_out_endpoint_used_for_selection": False,
        },
        "rtd_margin_log10": RTD_NO_BENEFIT_MARGIN,
        "linear_algebra": {
            "lstsq_driver": "gelsd",
            "lstsq_rcond": LSTSQ_RCOND,
            "soft_closed_form_pinv_rcond": PINV_RCOND,
        },
        "h5_present": False,
        "same_generator_family_only": True,
        "does_not_test": [
            "unseen generator families",
            "one shared model across topologies",
            "sheaf conversion",
            "mapping-cone homology",
            "published RTD or SRTD",
            "real data",
        ],
    }


def run(
    project_root: Path,
    output: Path,
    *,
    seal: str = SEAL_RECORD,
    dataset_factory: Callable[..., Any] = ConversionDataset,
) -> dict[str, Any]:
    """Run the complete sealed design after fail-closed preflight checks.

    The terminal status is one of ``complete``, ``design_failure``,
    ``design_failure_insufficient_eligible``, ``execution_failure``, or
    ``interrupted``. Every failure path preserves all completed raw rows, the
    failing seed and arm, and emits all seven claims with ``supported: null``.
    """

    provenance = _preflight(project_root, output, seal=seal)
    eligible: list[tuple[int, Any]] = []
    ineligible = []
    generation_failures = []
    generation_failure = None
    for seed in SEALED_SEEDS:
        try:
            sample = dataset_factory(1, seed=seed, dtype=torch.float64)[0]
        except Exception as error:  # noqa: BLE001 - a generator fault stops the design
            generation_failure = {
                "seed": seed,
                "type": type(error).__name__,
                "message": str(error),
            }
            generation_failures.append(generation_failure)
            break
        if int(sample.num_faces) >= MIN_FACES:
            eligible.append((seed, sample))
        else:
            ineligible.append(
                {
                    "seed": seed,
                    "sample_id": sample.sample_id,
                    "reason": f"num_faces={sample.num_faces} < {MIN_FACES}",
                    "vertices": int(sample.num_vertices),
                    "edges": int(sample.num_edges),
                    "faces": int(sample.num_faces),
                }
            )

    eligibility = {
        "declared": len(SEALED_SEEDS),
        "eligible": len(eligible),
        "eligible_seeds": [seed for seed, _ in eligible],
        "ineligible": ineligible,
        "generation_failures": generation_failures,
    }
    base: dict[str, Any] = {
        "schema": {
            "name": "homymoly.independent-lifting-replication-result",
            "version": SCHEMA_VERSION,
            "record_id": RECORD_ID,
        },
        "campaign": "independent-lifting-replication-v2",
        "provenance": provenance,
        "design": _design_record(),
        "eligibility": eligibility,
    }
    if generation_failure is not None:
        # A generator exception is a campaign failure, never an exclusion: the
        # campaign stops before any seed is fitted and the offending seed is
        # never deleted from the record.
        base["status"] = "design_failure"
        base["failure"] = {
            "seed": generation_failure["seed"],
            "arm": None,
            "phase": "generation",
            "type": generation_failure["type"],
            "message": generation_failure["message"],
        }
        base["stop_condition"] = (
            "generator failure is a whole-campaign design failure; no seed was "
            "fitted, excluded, or replaced"
        )
        base["raw_primary"] = []
        base["raw_c1"] = []
        base["primary"] = _null_primary_inference()
        base["audit"] = _audit_block(eligibility, [], [], complete=False)
        return base
    if len(eligible) < MIN_ELIGIBLE:
        base["status"] = "design_failure_insufficient_eligible"
        base["stop_condition"] = (
            f"only {len(eligible)} seeds were eligible; minimum is {MIN_ELIGIBLE}; "
            "no efficacy fits were run and no seeds were replaced"
        )
        base["raw_primary"] = []
        base["raw_c1"] = []
        base["primary"] = _null_primary_inference()
        base["audit"] = _audit_block(eligibility, [], [], complete=False)
        return base

    primary_rows: list[dict[str, Any]] = []
    c1_rows: list[dict[str, Any]] = []
    failure: tuple[str, BaseException] | None = None
    try:
        for seed, sample in eligible:
            try:
                primary_rows.append(_evaluate_primary(sample, seed))
                c1_rows.append(_evaluate_c1(sample, seed))
            except BaseException as error:
                if getattr(error, "seed", None) is None:
                    error.seed = seed
                raise
        base["primary"] = _primary_inference(primary_rows)
        base["c1"] = _c1_inference(c1_rows)
        base["descriptive"] = _descriptive_diagnostics(primary_rows)
    except DesignFailureError as error:
        failure = ("design_failure", error)
    except KeyboardInterrupt as error:
        failure = ("interrupted", error)
    except Exception as error:  # noqa: BLE001 - unexpected faults are preserved
        failure = ("execution_failure", error)

    base["raw_primary"] = primary_rows
    base["raw_c1"] = c1_rows
    if failure is not None:
        status, error = failure
        base["status"] = status
        base["failure"] = {
            "seed": getattr(error, "seed", None),
            "arm": getattr(error, "arm", None),
            "phase": "campaign",
            "type": type(error).__name__,
            "message": str(error) or type(error).__name__,
        }
        base["stop_condition"] = (
            "the campaign stopped before completion; all completed raw rows are "
            "preserved and every claim carries a null decision"
        )
        base["primary"] = _null_primary_inference()
        base["audit"] = _audit_block(eligibility, primary_rows, c1_rows, complete=False)
        return base

    base["audit"] = _audit_block(eligibility, primary_rows, c1_rows, complete=True)
    base["status"] = "complete"
    return base


def main(argv: list[str] | None = None) -> int:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seal",
        default=SEAL_RECORD,
        help=(
            "repository-relative path of the committed design-seal JSON "
            f"(default: {SEAL_RECORD})"
        ),
    )
    args = parser.parse_args(argv)
    root = args.project_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    report = run(root, output, seal=args.seal)
    _atomic_json_new(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "eligible": report["eligibility"]["eligible"],
                "output": str(output),
                "supported_primary_claims": (
                    [
                        claim["id"]
                        for claim in report.get("primary", {}).get("claims", [])
                        if claim["supported"]
                    ]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
