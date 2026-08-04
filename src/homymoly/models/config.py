"""Validated, code-local configuration for the Gate-2 model stack."""

from __future__ import annotations

from dataclasses import dataclass, field


def _positive(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class ExpertConfig:
    """Shared capacity contract for graph, cell, and sheaf experts."""

    node_feature_dim: int = 4
    edge_feature_dim: int = 2
    hidden_dim: int = 32
    embedding_dim: int = 32
    num_classes: int = 2
    num_layers: int = 2
    dropout: float = 0.0
    molecular_mode: bool = False

    def __post_init__(self) -> None:
        for name in (
            "node_feature_dim",
            "edge_feature_dim",
            "hidden_dim",
            "embedding_dim",
            "num_classes",
            "num_layers",
        ):
            _positive(name, getattr(self, name))
        if self.node_feature_dim < 2:
            raise ValueError(
                "node_feature_dim must be at least two for rank-2 sheaf data"
            )
        if self.hidden_dim % 2:
            raise ValueError("hidden_dim must be even for rank-2 transport messages")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if not isinstance(self.molecular_mode, bool):
            raise TypeError("molecular_mode must be boolean")


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Small cost- and diagnostic-aware router configuration."""

    hidden_dim: int = 32
    diagnostic_dim: int = 2
    temperature: float = 1.0
    route_costs: tuple[float, float, float] = (1.0, 1.35, 1.6)
    cost_strength: float = 0.1
    straight_through: bool = True

    def __post_init__(self) -> None:
        _positive("hidden_dim", self.hidden_dim)
        _positive("diagnostic_dim", self.diagnostic_dim)
        if self.diagnostic_dim != 2:
            raise ValueError("Gate-2 routing exposes two cheap diagnostics per route")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if len(self.route_costs) != 3 or any(cost <= 0 for cost in self.route_costs):
            raise ValueError("route_costs must contain three positive values")
        if self.cost_strength < 0:
            raise ValueError("cost_strength must be non-negative")
        if not isinstance(self.straight_through, bool):
            raise TypeError("straight_through must be boolean")


@dataclass(frozen=True, slots=True)
class TranslatorConfig:
    """Graph-hub translator capacity and numerical stabilization settings."""

    hidden_dim: int = 32
    stalk_rank: int = 2
    eps: float = 1e-6

    def __post_init__(self) -> None:
        _positive("hidden_dim", self.hidden_dim)
        if self.stalk_rank != 2:
            raise ValueError("Gate-2 data exposes rank-2 connection transports")
        if self.eps <= 0:
            raise ValueError("eps must be positive")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Complete Gate-2 model configuration.

    This remains separate from the Stage-1 runtime YAML schema so model work
    does not silently widen an already-frozen configuration contract.
    """

    expert: ExpertConfig = field(default_factory=ExpertConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    translator: TranslatorConfig = field(default_factory=TranslatorConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.expert, ExpertConfig):
            raise TypeError("expert must be ExpertConfig")
        if not isinstance(self.router, RouterConfig):
            raise TypeError("router must be RouterConfig")
        if not isinstance(self.translator, TranslatorConfig):
            raise TypeError("translator must be TranslatorConfig")


__all__ = ["ExpertConfig", "ModelConfig", "RouterConfig", "TranslatorConfig"]
