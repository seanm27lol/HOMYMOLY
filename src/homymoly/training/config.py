"""Strict configuration schema for the Gate-2 HOMYMOLY experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, TypeVar

import yaml

from homymoly.config import ArtifactConfig, ConfigError, RuntimeConfig


def _positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer; got {value!r}")


def _nonnegative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{name} must be a nonnegative integer; got {value!r}")


def _bounded_float(name: str, value: object, lower: float, upper: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric; got {value!r}")
    if not lower <= float(value) <= upper:
        raise ConfigError(f"{name} must lie in [{lower}, {upper}]; got {value!r}")


@dataclass(frozen=True, slots=True)
class Gate2ExperimentConfig:
    name: str = "gate2-confirmatory"
    seed: int = 20260803

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigError("experiment.name must be a nonempty string")
        _nonnegative_int("experiment.seed", self.seed)


@dataclass(frozen=True, slots=True)
class Gate2DataConfig:
    num_samples: int = 6144
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    min_vertices: int = 24
    max_vertices: int = 64
    node_feature_dim: int = 4
    edge_feature_dim: int = 2
    seed: int = 20260803
    stalk_mode: str = "independent"
    gauge_noise_std: float = 0.3

    def __post_init__(self) -> None:
        _positive_int("data.num_samples", self.num_samples)
        if self.num_samples < 18 or self.num_samples % 6:
            raise ConfigError("data.num_samples must be at least 18 and divisible by 6")
        _bounded_float("data.train_fraction", self.train_fraction, 0.01, 0.98)
        _bounded_float("data.validation_fraction", self.validation_fraction, 0.01, 0.98)
        if self.train_fraction + self.validation_fraction >= 1:
            raise ConfigError(
                "training and validation fractions must sum to less than 1"
            )
        if not 24 <= self.min_vertices <= self.max_vertices <= 96:
            raise ConfigError("data vertices must satisfy 24 <= min <= max <= 96")
        if self.node_feature_dim < 4 or self.edge_feature_dim < 2:
            raise ConfigError("data requires at least four node and two edge features")
        _nonnegative_int("data.seed", self.seed)
        if self.stalk_mode not in ("independent", "gauge"):
            raise ConfigError("data.stalk_mode must be 'independent' or 'gauge'")
        if self.gauge_noise_std < 0:
            raise ConfigError("data.gauge_noise_std must be nonnegative")


@dataclass(frozen=True, slots=True)
class Gate2ModelConfig:
    hidden_dim: int = 128
    embedding_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.10
    router_hidden_dim: int = 96
    router_temperature: float = 1.0

    def __post_init__(self) -> None:
        for name in ("hidden_dim", "embedding_dim", "num_layers", "router_hidden_dim"):
            _positive_int(f"model.{name}", getattr(self, name))
        _bounded_float("model.dropout", self.dropout, 0.0, 0.9)
        _bounded_float("model.router_temperature", self.router_temperature, 0.05, 10.0)


@dataclass(frozen=True, slots=True)
class Gate2TrainingConfig:
    batch_size: int = 64
    fixed_expert_epochs: int = 16
    translator_epochs: int = 8
    router_warmup_epochs: int = 6
    joint_finetune_epochs: int = 10
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.05
    early_stopping_patience: int = 6
    checkpoint_every: int = 1
    max_steps_per_epoch: int = 0
    pin_memory: bool = True

    def __post_init__(self) -> None:
        _positive_int("training.batch_size", self.batch_size)
        for name in (
            "fixed_expert_epochs",
            "translator_epochs",
            "router_warmup_epochs",
            "joint_finetune_epochs",
        ):
            _nonnegative_int(f"training.{name}", getattr(self, name))
        if (
            sum(
                (
                    self.fixed_expert_epochs,
                    self.translator_epochs,
                    self.router_warmup_epochs,
                    self.joint_finetune_epochs,
                )
            )
            == 0
        ):
            raise ConfigError(
                "at least one training phase must have a positive epoch count"
            )
        for name in ("learning_rate", "min_learning_rate", "weight_decay"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or value < 0
            ):
                raise ConfigError(f"training.{name} must be nonnegative")
        if self.learning_rate <= 0 or self.min_learning_rate > self.learning_rate:
            raise ConfigError("training learning-rate bounds are inconsistent")
        if not isfinite(self.grad_clip_norm) or self.grad_clip_norm <= 0:
            raise ConfigError("training.grad_clip_norm must be positive")
        _bounded_float("training.label_smoothing", self.label_smoothing, 0.0, 0.5)
        _positive_int("training.early_stopping_patience", self.early_stopping_patience)
        _positive_int("training.checkpoint_every", self.checkpoint_every)
        _nonnegative_int("training.max_steps_per_epoch", self.max_steps_per_epoch)
        if not isinstance(self.pin_memory, bool):
            raise ConfigError("training.pin_memory must be boolean")


@dataclass(frozen=True, slots=True)
class Gate2LossConfig:
    expert_weight: float = 1.0
    router_supervision_weight: float = 0.25
    translator_task_weight: float = 0.50
    translator_weight: float = 0.10
    chain_weight: float = 0.05
    rtd_weight: float = 0.02
    compute_cost_weight: float = 0.01
    entropy_weight: float = 0.005
    route_costs: tuple[float, float, float] = (1.0, 1.35, 1.60)
    rtd_max_points: int = 24
    oracle_cost_weight: float = 0.02

    def __post_init__(self) -> None:
        for name in (
            "expert_weight",
            "router_supervision_weight",
            "translator_task_weight",
            "translator_weight",
            "chain_weight",
            "rtd_weight",
            "compute_cost_weight",
            "entropy_weight",
            "oracle_cost_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or value < 0
            ):
                raise ConfigError(f"loss.{name} must be nonnegative")
        if len(self.route_costs) != 3 or any(
            not isfinite(cost) or cost <= 0 for cost in self.route_costs
        ):
            raise ConfigError("loss.route_costs must contain three positive values")
        _positive_int("loss.rtd_max_points", self.rtd_max_points)


@dataclass(frozen=True, slots=True)
class Gate2GateConfig:
    """Automatic engineering gates before more complex phases are allowed."""

    enforce: bool = True
    minimum_specialized_routes: int = 2
    specialization_margin: float = 0.0
    translator_relative_improvement: float = 0.02

    def __post_init__(self) -> None:
        if not isinstance(self.enforce, bool):
            raise ConfigError("gates.enforce must be boolean")
        if self.minimum_specialized_routes not in (1, 2):
            raise ConfigError("gates.minimum_specialized_routes must be 1 or 2")
        _bounded_float("gates.specialization_margin", self.specialization_margin, 0.0, 1.0)
        _bounded_float(
            "gates.translator_relative_improvement",
            self.translator_relative_improvement,
            0.0,
            1.0,
        )


@dataclass(frozen=True, slots=True)
class Gate2Config:
    schema_version: int = 2
    experiment: Gate2ExperimentConfig = field(default_factory=Gate2ExperimentConfig)
    data: Gate2DataConfig = field(default_factory=Gate2DataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: Gate2ModelConfig = field(default_factory=Gate2ModelConfig)
    training: Gate2TrainingConfig = field(default_factory=Gate2TrainingConfig)
    loss: Gate2LossConfig = field(default_factory=Gate2LossConfig)
    gates: Gate2GateConfig = field(default_factory=Gate2GateConfig)
    artifacts: ArtifactConfig = field(
        default_factory=lambda: ArtifactConfig(run_name="gate2")
    )
    source_path: Path = field(
        default=Path("configs/gate2.yaml"), repr=False, compare=False
    )
    project_root: Path = field(default=Path.cwd(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ConfigError("Gate-2 schema_version must be 2")
        if self.runtime.precision == "float16" and self.runtime.device == "cpu":
            raise ConfigError("float16 Gate-2 training is not supported on CPU")

    @property
    def run_dir(self) -> Path:
        root = Path(self.artifacts.root).expanduser()
        if not root.is_absolute():
            root = self.project_root / root
        return (root / self.artifacts.run_name).resolve()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment": asdict(self.experiment),
            "data": asdict(self.data),
            "runtime": asdict(self.runtime),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "loss": {**asdict(self.loss), "route_costs": list(self.loss.route_costs)},
            "gates": asdict(self.gates),
            "artifacts": asdict(self.artifacts),
            "source_path": str(self.source_path),
            "project_root": str(self.project_root),
            "run_dir": str(self.run_dir),
        }


SectionT = TypeVar(
    "SectionT",
    Gate2ExperimentConfig,
    Gate2DataConfig,
    RuntimeConfig,
    Gate2ModelConfig,
    Gate2TrainingConfig,
    Gate2LossConfig,
    Gate2GateConfig,
    ArtifactConfig,
)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{name} must be a string-keyed mapping")
    return value


def _section(cls: type[SectionT], value: object, name: str) -> SectionT:
    values = dict(_mapping(value, name))
    unknown = sorted(set(values) - set(cls.__dataclass_fields__))
    if unknown:
        raise ConfigError(f"unknown {name} field(s): {', '.join(unknown)}")
    if cls is Gate2LossConfig and "route_costs" in values:
        costs = values["route_costs"]
        if isinstance(costs, (str, bytes)) or not isinstance(costs, (list, tuple)):
            raise ConfigError("loss.route_costs must be a sequence")
        values["route_costs"] = tuple(float(value) for value in costs)
    try:
        return cls(**values)
    except TypeError as exc:
        raise ConfigError(f"invalid {name} configuration: {exc}") from exc


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate.resolve()
    return start.resolve()


def load_gate2_config(path: str | Path) -> Gate2Config:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise ConfigError(f"configuration file does not exist: {source_path}")
    try:
        document = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot load Gate-2 configuration: {exc}") from exc
    root = _mapping(document, "configuration")
    allowed = {
        "schema_version",
        "experiment",
        "data",
        "runtime",
        "model",
        "training",
        "loss",
        "gates",
        "artifacts",
    }
    unknown = sorted(set(root) - allowed)
    if unknown:
        raise ConfigError(f"unknown Gate-2 section(s): {', '.join(unknown)}")
    schema_version = root.get("schema_version", 2)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ConfigError("schema_version must be an integer")
    return Gate2Config(
        schema_version=schema_version,
        experiment=_section(
            Gate2ExperimentConfig, root.get("experiment"), "experiment"
        ),
        data=_section(Gate2DataConfig, root.get("data"), "data"),
        runtime=_section(RuntimeConfig, root.get("runtime"), "runtime"),
        model=_section(Gate2ModelConfig, root.get("model"), "model"),
        training=_section(Gate2TrainingConfig, root.get("training"), "training"),
        loss=_section(Gate2LossConfig, root.get("loss"), "loss"),
        gates=_section(Gate2GateConfig, root.get("gates"), "gates"),
        artifacts=_section(ArtifactConfig, root.get("artifacts"), "artifacts"),
        source_path=source_path,
        project_root=_find_project_root(source_path.parent),
    )


__all__ = [
    "Gate2Config",
    "Gate2DataConfig",
    "Gate2ExperimentConfig",
    "Gate2GateConfig",
    "Gate2LossConfig",
    "Gate2ModelConfig",
    "Gate2TrainingConfig",
    "load_gate2_config",
]
