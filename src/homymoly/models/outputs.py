"""Result containers shared across experts, translators, and routing."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from homymoly.data import SignalRegime


@dataclass(slots=True)
class ExpertOutput:
    """Common output contract for each fixed expert."""

    route: SignalRegime
    embedding: Tensor
    logits: Tensor
    diagnostics: Tensor


@dataclass(slots=True)
class EnsembleOutput:
    """Stacked outputs plus an optional fixed or uniform combination."""

    embeddings: Tensor
    expert_logits: Tensor
    embedding: Tensor
    logits: Tensor
    expert_outputs: tuple[ExpertOutput, ...]


@dataclass(slots=True)
class RouterOutput:
    """Soft or straight-through hard routing decision."""

    logits: Tensor
    weights: Tensor
    selected_routes: Tensor
    expected_cost: Tensor
    entropy: Tensor


@dataclass(slots=True)
class TranslationOutput:
    """Learned lift/reconstruction and explicitly surrogate diagnostics."""

    task_embedding: Tensor
    task_logits: Tensor
    structure_logits: Tensor
    node_latent: Tensor
    edge_latent: Tensor
    higher_latent: Tensor
    node_reconstruction: Tensor
    edge_reconstruction: Tensor
    reconstruction_loss: Tensor
    consistency_surrogate: Tensor
    map_reconstruction_loss: Tensor
    supervision_loss: Tensor
    per_sample_diagnostics: Tensor


@dataclass(slots=True)
class ModelOutput:
    """Stable public result returned by :class:`HomologicalRouterSystem`."""

    route_logits: Tensor
    route_weights: Tensor
    expert_logits: Tensor
    mixed_logits: Tensor
    embeddings: Tensor
    diagnostics: Tensor
    auxiliary_losses: dict[str, Tensor]
    selected_routes: Tensor
    evaluated_routes: Tensor
    translated_embeddings: Tensor
    translated_logits: Tensor
    translation_diagnostics: Tensor
    evaluated_translators: Tensor


__all__ = [
    "EnsembleOutput",
    "ExpertOutput",
    "ModelOutput",
    "RouterOutput",
    "TranslationOutput",
]
