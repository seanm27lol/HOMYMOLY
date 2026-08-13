"""Integrated fixed-expert, translator, and routing system for Gate 2."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from homymoly.data import (
    SignalRegime,
    StructuredBatch,
    StructuredObservations,
)

from .config import ModelConfig
from .experts import ROUTE_ORDER, FixedExpertEnsemble
from .ops import apply_mask, masked_feature_energy, masked_mean, safe_gather_nodes
from .outputs import ModelOutput
from .router import DiagnosticCostRouter
from .translators import GraphToCellTranslator, GraphToSheafTranslator

DIAGNOSTIC_NAMES = (
    "graph_edge_density",
    "graph_edge_feature_energy",
    "cell_candidate_face_density",
    "cell_active_face_fraction",
    "sheaf_connection_residual_energy",
    "sheaf_transport_deviation",
    "route_entropy",
    "expected_route_cost",
)

TRANSLATOR_ORDER = ("graph_to_cell", "graph_to_sheaf")


def _slice_batch(batch: StructuredBatch, indices: Tensor) -> StructuredBatch:
    """Select samples while retaining the validated padded-batch contract."""

    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("indices must be a nonempty rank-1 tensor")
    selected = [int(index) for index in indices.detach().cpu().tolist()]

    def take(values: Tensor) -> Tensor:
        return values.index_select(0, indices)

    return StructuredBatch(
        observations=StructuredObservations(
            node_features=take(batch.node_features),
            edge_features=take(batch.edge_features),
        ),
        node_mask=take(batch.node_mask),
        edge_index=take(batch.edge_index),
        edge_mask=take(batch.edge_mask),
        face_index=take(batch.face_index),
        face_mask=take(batch.face_mask),
        face_active=take(batch.face_active),
        transport=take(batch.transport),
        labels=take(batch.labels),
        regimes=tuple(batch.regimes[index] for index in selected),
        sample_ids=tuple(batch.sample_ids[index] for index in selected),
        metadata=tuple(batch.metadata[index] for index in selected),
        num_vertices=take(batch.num_vertices),
        num_edges=take(batch.num_edges),
        num_faces=take(batch.num_faces),
        face_boundary=(
            None if batch.face_boundary is None else take(batch.face_boundary)
        ),
        face_vertices=(
            None if batch.face_vertices is None else take(batch.face_vertices)
        ),
    )


def _masked_fraction(numerator: Tensor, denominator: Tensor) -> Tensor:
    return numerator.to(torch.float32) / denominator.to(torch.float32).clamp_min(1.0)


def _raw_routing_features(batch: StructuredBatch) -> tuple[Tensor, Tensor]:
    """Build cheap, label-independent features without invoking any expert.

    The context pairs per-channel means with per-channel max-abs amplitudes
    and the mean stalk norm.  The confirmatory generator makes route
    reliability observable through overlapping amplitude ranges (see the
    experimental protocol: regime selection must be statistically possible
    from label-independent cues), and the amplitudes live in maxima that
    mean pooling dilutes away — measured F-statistics across regimes on the
    shipped diagnostics were ~0 (regime-blind) while the max-abs cues reach
    9-37.
    """

    graph = batch.model_inputs(SignalRegime.GRAPH)
    cell = batch.model_inputs(SignalRegime.CELL)
    sheaf = batch.model_inputs(SignalRegime.SHEAF)
    context = torch.cat(
        (
            masked_mean(graph["node_features"], graph["node_mask"]),
            masked_mean(graph["edge_features"], graph["edge_mask"]),
            apply_mask(graph["node_features"].abs(), graph["node_mask"]).amax(dim=1),
            apply_mask(graph["edge_features"].abs(), graph["edge_mask"]).amax(dim=1),
            masked_mean(
                apply_mask(
                    graph["node_features"][..., -2:].square().sum(dim=-1).sqrt(),
                    graph["node_mask"],
                ),
                graph["node_mask"],
            ).unsqueeze(-1),
        ),
        dim=-1,
    )

    vertices = graph["node_mask"].sum(dim=1).to(torch.float32)
    edges = graph["edge_mask"].sum(dim=1).to(torch.float32)
    possible_edges = (vertices * (vertices - 1.0) * 0.5).clamp_min(1.0)
    graph_diagnostics = torch.stack(
        (
            edges / possible_edges,
            masked_feature_energy(graph["edge_features"], graph["edge_mask"]),
        ),
        dim=-1,
    )

    candidate_faces = cell["face_mask"].sum(dim=1).to(torch.float32)
    active_faces = (cell["face_mask"] & cell["face_active"]).sum(dim=1)
    possible_faces = (vertices * (vertices - 1.0) * (vertices - 2.0) / 6.0).clamp_min(
        1.0
    )
    cell_diagnostics = torch.stack(
        (
            candidate_faces / possible_faces,
            _masked_fraction(active_faces, candidate_faces),
        ),
        dim=-1,
    )

    vectors = apply_mask(graph["node_features"][..., -2:], graph["node_mask"])
    tails = safe_gather_nodes(vectors, graph["edge_index"][:, 0])
    heads = safe_gather_nodes(vectors, graph["edge_index"][:, 1])
    connection = apply_mask(sheaf["transport"], graph["edge_mask"])
    transported = torch.einsum("beij,bej->bei", connection, tails)
    residual = apply_mask(heads - transported, graph["edge_mask"])
    residual_energy = masked_feature_energy(residual, graph["edge_mask"])
    stalk_energy = masked_feature_energy(vectors, graph["node_mask"]).clamp_min(1e-6)
    identity = torch.eye(
        2,
        dtype=sheaf["transport"].dtype,
        device=sheaf["transport"].device,
    ).reshape(1, 1, 2, 2)
    transport_deviation = masked_feature_energy(
        sheaf["transport"] - identity,
        graph["edge_mask"],
    )
    sheaf_diagnostics = torch.stack(
        (residual_energy / stalk_energy, transport_deviation),
        dim=-1,
    )
    return context, torch.stack(
        (graph_diagnostics, cell_diagnostics, sheaf_diagnostics),
        dim=1,
    )


class HomologicalRouterSystem(nn.Module):
    """Raw-feature routing followed by experts and graph-hub translators.

    Translator quantities remain reconstruction and consistency *surrogates*;
    this module does not compute or claim exact mapping-cone homology.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be ModelConfig")
        self.config = config
        self.fixed_experts = FixedExpertEnsemble(config.expert)
        raw_context_dim = (
            2 * (config.expert.node_feature_dim + config.expert.edge_feature_dim) + 1
        )
        self.router = DiagnosticCostRouter(raw_context_dim, config.router)
        self.graph_to_cell = GraphToCellTranslator(config.expert, config.translator)
        self.graph_to_sheaf = GraphToSheafTranslator(config.expert, config.translator)
        # Regime-conditional expert accuracies used as the routing-oracle
        # utility, fitted on validation when the router-warmup phase begins
        # (see engine); persisted so resume restores the fitted table.
        self.register_buffer(
            "oracle_conditional_accuracy",
            torch.zeros(len(ROUTE_ORDER), len(ROUTE_ORDER), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "oracle_table_ready",
            torch.zeros((), dtype=torch.uint8),
            persistent=True,
        )

    def _run_experts(
        self,
        batch: StructuredBatch,
        selected_routes: Tensor,
        *,
        selective: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not selective:
            ensemble = self.fixed_experts(batch)
            evaluated = torch.ones(
                (len(batch), len(ROUTE_ORDER)),
                dtype=torch.bool,
                device=batch.labels.device,
            )
            return ensemble.embeddings, ensemble.expert_logits, evaluated

        embedding_routes: list[Tensor] = []
        logit_routes: list[Tensor] = []
        evaluated_routes = torch.nn.functional.one_hot(
            selected_routes,
            num_classes=len(ROUTE_ORDER),
        ).to(torch.bool)
        for route_index, route in enumerate(ROUTE_ORDER):
            indices = torch.nonzero(
                selected_routes == route_index,
                as_tuple=False,
            ).squeeze(-1)
            route_embeddings = torch.zeros(
                (len(batch), self.config.expert.embedding_dim),
                dtype=torch.float32,
                device=batch.labels.device,
            )
            route_logits = torch.zeros(
                (len(batch), self.config.expert.num_classes),
                dtype=torch.float32,
                device=batch.labels.device,
            )
            if indices.numel():
                subset = _slice_batch(batch, indices)
                output = self.fixed_experts.experts[route.value](subset)
                route_embeddings = route_embeddings.index_copy(
                    0,
                    indices,
                    output.embedding.to(route_embeddings.dtype),
                )
                route_logits = route_logits.index_copy(
                    0,
                    indices,
                    output.logits.to(route_logits.dtype),
                )
            embedding_routes.append(route_embeddings)
            logit_routes.append(route_logits)
        return (
            torch.stack(embedding_routes, dim=1),
            torch.stack(logit_routes, dim=1),
            evaluated_routes,
        )

    def _run_translators(
        self,
        batch: StructuredBatch,
        *,
        skip: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        if skip:
            translated_embeddings = torch.zeros(
                (len(batch), len(TRANSLATOR_ORDER), self.config.expert.embedding_dim),
                dtype=torch.float32,
                device=batch.labels.device,
            )
            translated_logits = torch.zeros(
                (len(batch), len(TRANSLATOR_ORDER), self.config.expert.num_classes),
                dtype=torch.float32,
                device=batch.labels.device,
            )
            translation_diagnostics = torch.zeros(
                (len(batch), len(TRANSLATOR_ORDER), 3),
                dtype=torch.float32,
                device=batch.labels.device,
            )
            evaluated = torch.zeros(
                (len(batch), len(TRANSLATOR_ORDER)),
                dtype=torch.bool,
                device=batch.labels.device,
            )
            zero = translated_logits.sum()
            losses = {
                "cell_reconstruction": zero,
                "cell_chain_consistency_surrogate": zero,
                "cell_boundary_map_reconstruction": zero,
                "cell_face_gate_supervision": zero,
                "sheaf_reconstruction": zero,
                "sheaf_cochain_consistency_surrogate": zero,
                "sheaf_transport_map_reconstruction": zero,
            }
            return (
                translated_embeddings,
                translated_logits,
                translation_diagnostics,
                evaluated,
                losses,
            )

        cell = self.graph_to_cell(batch)
        sheaf = self.graph_to_sheaf(batch)
        translated_embeddings = torch.stack(
            (cell.task_embedding, sheaf.task_embedding),
            dim=1,
        )
        translated_logits = torch.stack((cell.task_logits, sheaf.task_logits), dim=1)
        translation_diagnostics = torch.stack(
            (cell.per_sample_diagnostics, sheaf.per_sample_diagnostics),
            dim=1,
        )
        evaluated = torch.ones(
            (len(batch), len(TRANSLATOR_ORDER)),
            dtype=torch.bool,
            device=batch.labels.device,
        )
        losses = {
            "cell_reconstruction": cell.reconstruction_loss,
            "cell_chain_consistency_surrogate": cell.consistency_surrogate,
            "cell_boundary_map_reconstruction": cell.map_reconstruction_loss,
            "cell_face_gate_supervision": cell.supervision_loss,
            "sheaf_reconstruction": sheaf.reconstruction_loss,
            "sheaf_cochain_consistency_surrogate": sheaf.consistency_surrogate,
            "sheaf_transport_map_reconstruction": sheaf.map_reconstruction_loss,
        }
        return (
            translated_embeddings,
            translated_logits,
            translation_diagnostics,
            evaluated,
            losses,
        )

    def forward(self, batch: StructuredBatch, *, hard: bool = False) -> ModelOutput:
        if not isinstance(batch, StructuredBatch):
            raise TypeError("batch must be StructuredBatch")

        raw_context, route_diagnostics = _raw_routing_features(batch)
        router = self.router(raw_context, route_diagnostics, hard=hard)
        selective = hard and not self.training
        embeddings, expert_logits, evaluated_routes = self._run_experts(
            batch,
            router.selected_routes,
            selective=selective,
        )
        mixed_logits = torch.einsum(
            "br,brc->bc",
            router.weights,
            expert_logits.to(router.weights.dtype),
        )

        (
            translated_embeddings,
            translated_logits,
            translation_diagnostics,
            evaluated_translators,
            translation_losses,
        ) = self._run_translators(batch, skip=selective)
        diagnostics = torch.cat(
            (
                route_diagnostics.reshape(len(batch), -1).to(torch.float32),
                router.entropy.unsqueeze(-1),
                router.expected_cost.unsqueeze(-1),
            ),
            dim=-1,
        )
        average_weights = router.weights.to(torch.float32).mean(dim=0)
        uniform = torch.full_like(average_weights, 1.0 / len(ROUTE_ORDER))
        auxiliary_losses = {
            **translation_losses,
            "route_expected_cost": router.expected_cost.mean(),
            "route_load_balance": (average_weights - uniform).square().sum(),
            "route_entropy": router.entropy.mean(),
        }
        return ModelOutput(
            route_logits=router.logits,
            route_weights=router.weights,
            expert_logits=expert_logits,
            mixed_logits=mixed_logits,
            embeddings=embeddings,
            diagnostics=diagnostics,
            auxiliary_losses=auxiliary_losses,
            selected_routes=router.selected_routes,
            evaluated_routes=evaluated_routes,
            translated_embeddings=translated_embeddings,
            translated_logits=translated_logits,
            translation_diagnostics=translation_diagnostics,
            evaluated_translators=evaluated_translators,
        )


def build_model(config: ModelConfig) -> HomologicalRouterSystem:
    """Construct the complete Gate-2 system from one validated config."""

    if not isinstance(config, ModelConfig):
        raise TypeError("build_model requires ModelConfig")
    return HomologicalRouterSystem(config)


__all__ = [
    "DIAGNOSTIC_NAMES",
    "TRANSLATOR_ORDER",
    "HomologicalRouterSystem",
    "build_model",
]
