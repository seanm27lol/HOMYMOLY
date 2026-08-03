"""Typed configuration loading and artifact-path resolution for Stage 1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, TypeVar

import yaml


class ConfigError(ValueError):
    """Raised when a HOMYMOLY configuration violates its schema."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(name: str, value: object) -> None:
    if not _is_int(value) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer; got {value!r}")


def _nonnegative_int(name: str, value: object) -> None:
    if not _is_int(value) or value < 0:
        raise ConfigError(f"{name} must be a nonnegative integer; got {value!r}")


def _nonempty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a nonempty string; got {value!r}")


def _relative_child(name: str, value: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ConfigError(f"{name} must be a nonempty relative path without '..'; got {value!r}")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Identity and reproducibility settings for a run."""

    name: str = "stage1-foundation"
    seed: int = 20260802

    def __post_init__(self) -> None:
        _nonempty_string("experiment.name", self.name)
        _nonnegative_int("experiment.seed", self.seed)


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Small synthetic-data envelope used by the Stage-1 smoke path."""

    num_samples: int = 6144
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    min_vertices: int = 24
    max_vertices: int = 96
    node_feature_dim: int = 4
    edge_feature_dim: int = 2
    num_classes: int = 2
    seed: int = 20260802

    def __post_init__(self) -> None:
        for name in ("num_samples", "min_vertices", "max_vertices"):
            _positive_int(f"data.{name}", getattr(self, name))
        if self.num_samples < 18 or self.num_samples % 6 != 0:
            raise ConfigError(
                "data.num_samples must be at least 18 and divisible by 6"
            )
        for name in ("train_fraction", "validation_fraction"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 < value < 1
            ):
                raise ConfigError(f"data.{name} must lie strictly between 0 and 1")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ConfigError(
                "data.train_fraction + data.validation_fraction must be less than 1"
            )
        if self.node_feature_dim < 4:
            raise ConfigError("data.node_feature_dim must be at least 4")
        if self.edge_feature_dim < 2:
            raise ConfigError("data.edge_feature_dim must be at least 2")
        if self.num_classes != 2:
            raise ConfigError("data.num_classes must be 2 for the binary Stage-1 task")
        _nonnegative_int("data.seed", self.seed)
        if not (24 <= self.min_vertices <= self.max_vertices <= 96):
            raise ConfigError(
                "data vertex bounds must satisfy 24 <= min_vertices <= "
                "max_vertices <= 96"
            )


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Device policy shared by smoke tests and profiling."""

    device: str = "auto"
    precision: str = "bfloat16"
    deterministic: bool = True
    allow_tf32: bool = True
    compile: bool = False
    num_workers: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or self.device not in {"auto", "cpu", "cuda"}:
            raise ConfigError("runtime.device must be one of: auto, cpu, cuda")
        if not isinstance(self.precision, str) or self.precision not in {
            "float32",
            "float16",
            "bfloat16",
        }:
            raise ConfigError(
                "runtime.precision must be one of: float32, float16, bfloat16"
            )
        for name in ("deterministic", "allow_tf32", "compile"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"runtime.{name} must be a boolean")
        _nonnegative_int("runtime.num_workers", self.num_workers)


@dataclass(frozen=True, slots=True)
class ArtifactConfig:
    """Portable artifact layout; relative roots resolve from the repository root."""

    root: str = "artifacts"
    run_name: str = "stage1"
    checkpoints_dir: str = "checkpoints"
    tensorboard_dir: str = "tensorboard"
    profiles_dir: str = "profiles"
    metrics_dir: str = "metrics"

    def __post_init__(self) -> None:
        _nonempty_string("artifacts.root", self.root)
        _nonempty_string("artifacts.run_name", self.run_name)
        _relative_child("artifacts.run_name", self.run_name)
        for name in (
            "checkpoints_dir",
            "tensorboard_dir",
            "profiles_dir",
            "metrics_dir",
        ):
            value = getattr(self, name)
            _nonempty_string(f"artifacts.{name}", value)
            _relative_child(f"artifacts.{name}", value)


@dataclass(frozen=True, slots=True)
class ProfilingConfig:
    """Bounded GEMM probe settings suitable for a GB10 smoke profile."""

    warmup_steps: int = 5
    active_steps: int = 20
    repetitions: int = 3
    matrix_sizes: tuple[int, ...] = (2048, 4096)

    def __post_init__(self) -> None:
        _nonnegative_int("profiling.warmup_steps", self.warmup_steps)
        _positive_int("profiling.active_steps", self.active_steps)
        _positive_int("profiling.repetitions", self.repetitions)
        if not isinstance(self.matrix_sizes, tuple) or not self.matrix_sizes:
            raise ConfigError("profiling.matrix_sizes must be a nonempty sequence")
        for size in self.matrix_sizes:
            _positive_int("profiling.matrix_sizes[]", size)
        if len(set(self.matrix_sizes)) != len(self.matrix_sizes):
            raise ConfigError("profiling.matrix_sizes must not contain duplicates")


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    """Resolved filesystem locations for one run."""

    root: Path
    run: Path
    checkpoints: Path
    tensorboard: Path
    profiles: Path
    metrics: Path

    def create(self) -> ArtifactPaths:
        for path in (
            self.root,
            self.run,
            self.checkpoints,
            self.tensorboard,
            self.profiles,
            self.metrics,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def as_dict(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class Stage1Config:
    """Complete, intentionally small Stage-1 configuration."""

    schema_version: int = 1
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    source_path: Path = field(default=Path("configs/stage1.yaml"), repr=False, compare=False)
    project_root: Path = field(default=Path.cwd(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ConfigError(
                f"unsupported schema_version {self.schema_version!r}; expected 1"
            )

    def artifact_paths(self) -> ArtifactPaths:
        root = Path(self.artifacts.root).expanduser()
        if not root.is_absolute():
            root = self.project_root / root
        root = root.resolve()
        run = root / self.artifacts.run_name
        return ArtifactPaths(
            root=root,
            run=run,
            checkpoints=run / self.artifacts.checkpoints_dir,
            tensorboard=run / self.artifacts.tensorboard_dir,
            profiles=run / self.artifacts.profiles_dir,
            metrics=run / self.artifacts.metrics_dir,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment": asdict(self.experiment),
            "data": asdict(self.data),
            "runtime": asdict(self.runtime),
            "artifacts": asdict(self.artifacts),
            "profiling": asdict(self.profiling),
            "source_path": str(self.source_path),
            "project_root": str(self.project_root),
        }


ConfigSection = TypeVar(
    "ConfigSection",
    ExperimentConfig,
    DataConfig,
    RuntimeConfig,
    ArtifactConfig,
    ProfilingConfig,
)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping; got {type(value).__name__}")
    if not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{name} keys must be strings")
    return value


def _section(
    cls: type[ConfigSection], value: object, name: str
) -> ConfigSection:
    values = dict(_mapping(value, name))
    allowed = set(cls.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigError(f"unknown {name} field(s): {', '.join(unknown)}")
    if cls is ProfilingConfig and "matrix_sizes" in values:
        sizes = values["matrix_sizes"]
        if isinstance(sizes, (str, bytes)) or not isinstance(sizes, Sequence):
            raise ConfigError("profiling.matrix_sizes must be a sequence of integers")
        values["matrix_sizes"] = tuple(sizes)
    try:
        return cls(**values)
    except TypeError as exc:
        raise ConfigError(f"invalid {name} configuration: {exc}") from exc


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate.resolve()
    return start.resolve()


def load_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> Stage1Config:
    """Load and validate a Stage-1 YAML file without creating directories."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise ConfigError(f"configuration file does not exist: {source_path}")
    try:
        document = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {source_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read configuration {source_path}: {exc}") from exc
    root = _mapping(document, "configuration")
    allowed = {
        "schema_version",
        "experiment",
        "data",
        "runtime",
        "artifacts",
        "profiling",
    }
    unknown = sorted(set(root) - allowed)
    if unknown:
        raise ConfigError(f"unknown top-level field(s): {', '.join(unknown)}")

    schema_version = root.get("schema_version", 1)
    if not _is_int(schema_version):
        raise ConfigError("schema_version must be an integer")

    artifacts = _section(ArtifactConfig, root.get("artifacts"), "artifacts")
    if artifact_root is not None:
        artifacts = replace(artifacts, root=str(artifact_root))

    resolved_project_root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else _find_project_root(source_path.parent)
    )
    return Stage1Config(
        schema_version=schema_version,
        experiment=_section(ExperimentConfig, root.get("experiment"), "experiment"),
        data=_section(DataConfig, root.get("data"), "data"),
        runtime=_section(RuntimeConfig, root.get("runtime"), "runtime"),
        artifacts=artifacts,
        profiling=_section(ProfilingConfig, root.get("profiling"), "profiling"),
        source_path=source_path,
        project_root=resolved_project_root,
    )
