"""Learned graph-hub translators with conservative surrogate diagnostics."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from homymoly.data import SignalRegime, StructuredBatch

from .config import ExpertConfig, TranslatorConfig
from .ops import (
    MLP,
    apply_mask,
    face_boundary_aggregate,
    face_vertex_mean,
    masked_feature_energy,
    masked_mean,
    masked_mse,
    safe_gather_nodes,
)
from .outputs import TranslationOutput


def _per_sample_masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    batch, items = prediction.shape[:2]
    per_item = (prediction - target).reshape(batch, items, -1).to(torch.float32)
    per_item = per_item.square().mean(dim=-1)
    per_item = torch.where(mask, per_item, torch.zeros_like(per_item))
    numerator = per_item.sum(dim=1)
    denominator = mask.sum(dim=1).to(per_item.dtype).clamp_min(1.0)
    return numerator / denominator


def _masked_binary_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> Tensor:
    per_item = F.binary_cross_entropy_with_logits(
        logits.to(torch.float32),
        targets.to(torch.float32),
        reduction="none",
    )
    per_item = torch.where(mask, per_item, torch.zeros_like(per_item))
    return per_item.sum() / mask.sum().to(per_item.dtype).clamp_min(1.0)


class GraphToCellTranslator(nn.Module):
    """Lift graph observations to active-face features and reconstruct them.

    Candidate-face gates and task predictions use only graph observations and
    candidate incidence. ``face_active`` appears only in the separately named
    gate-supervision loss. ``consistency_surrogate`` compares a learned face
    code with the oriented aggregation of learned edge features; it is not
    asserted to be an exact chain-map residual.
    """

    def __init__(
        self,
        expert_config: ExpertConfig,
        translator_config: TranslatorConfig,
    ) -> None:
        super().__init__()
        hidden = translator_config.hidden_dim
        self.node_lift = MLP(expert_config.node_feature_dim, hidden, hidden)
        self.edge_lift = MLP(expert_config.edge_feature_dim, hidden, hidden)
        self.face_lift = MLP(2 * hidden, hidden, hidden)
        self.face_gate = MLP(2 * hidden, hidden, 1)
        self.face_boundary_prediction = nn.Linear(hidden, hidden)
        self.node_reconstruction = MLP(hidden, hidden, expert_config.node_feature_dim)
        self.edge_reconstruction = MLP(hidden, hidden, expert_config.edge_feature_dim)
        self.task_readout = MLP(
            3 * hidden,
            hidden,
            expert_config.embedding_dim,
        )
        self.task_classifier = nn.Linear(
            expert_config.embedding_dim,
            expert_config.num_classes,
        )

    def forward(self, batch: StructuredBatch) -> TranslationOutput:
        graph = batch.model_inputs(SignalRegime.GRAPH)
        cell = batch.model_inputs(SignalRegime.CELL)
        node_latent = apply_mask(
            self.node_lift(apply_mask(graph["node_features"], graph["node_mask"])),
            graph["node_mask"],
        )
        edge_latent = apply_mask(
            self.edge_lift(apply_mask(graph["edge_features"], graph["edge_mask"])),
            graph["edge_mask"],
        )
        candidate_mask = cell["face_mask"]
        boundary_latent = face_boundary_aggregate(
            edge_latent,
            graph["edge_index"],
            graph["edge_mask"],
            cell["face_index"],
            candidate_mask,
        )
        vertex_latent = face_vertex_mean(
            node_latent,
            cell["face_index"],
            candidate_mask,
        )
        face_inputs = torch.cat((vertex_latent, boundary_latent), dim=-1)
        ungated_faces = apply_mask(self.face_lift(face_inputs), candidate_mask)
        structure_logits = apply_mask(
            self.face_gate(face_inputs).squeeze(-1),
            candidate_mask,
        )
        higher_latent = apply_mask(
            ungated_faces * torch.sigmoid(structure_logits).unsqueeze(-1),
            candidate_mask,
        )

        task_embedding = self.task_readout(
            torch.cat(
                (
                    masked_mean(node_latent, graph["node_mask"]),
                    masked_mean(edge_latent, graph["edge_mask"]),
                    masked_mean(higher_latent, candidate_mask),
                ),
                dim=-1,
            )
        )
        task_logits = self.task_classifier(task_embedding)

        node_reconstruction = apply_mask(
            self.node_reconstruction(node_latent), graph["node_mask"]
        )
        edge_reconstruction = apply_mask(
            self.edge_reconstruction(edge_latent), graph["edge_mask"]
        )
        node_loss = masked_mse(
            node_reconstruction, graph["node_features"], graph["node_mask"]
        )
        edge_loss = masked_mse(
            edge_reconstruction, graph["edge_features"], graph["edge_mask"]
        )
        reconstruction_loss = 0.5 * (node_loss + edge_loss)
        predicted_boundary = apply_mask(
            self.face_boundary_prediction(higher_latent), candidate_mask
        )
        consistency_surrogate = masked_mse(
            predicted_boundary, boundary_latent, candidate_mask
        )
        supervision_loss = _masked_binary_cross_entropy(
            structure_logits,
            cell["face_active"],
            candidate_mask,
        )

        per_sample_reconstruction = 0.5 * (
            _per_sample_masked_mse(
                node_reconstruction, graph["node_features"], graph["node_mask"]
            )
            + _per_sample_masked_mse(
                edge_reconstruction, graph["edge_features"], graph["edge_mask"]
            )
        )
        per_sample_consistency = _per_sample_masked_mse(
            predicted_boundary, boundary_latent, candidate_mask
        )
        return TranslationOutput(
            task_embedding=task_embedding,
            task_logits=task_logits,
            structure_logits=structure_logits,
            node_latent=node_latent,
            edge_latent=edge_latent,
            higher_latent=higher_latent,
            node_reconstruction=node_reconstruction,
            edge_reconstruction=edge_reconstruction,
            reconstruction_loss=reconstruction_loss,
            consistency_surrogate=consistency_surrogate,
            supervision_loss=supervision_loss,
            per_sample_diagnostics=torch.stack(
                (per_sample_reconstruction, per_sample_consistency), dim=-1
            ),
        )


class GraphToSheafTranslator(nn.Module):
    """Lift node features to rank-2 stalks with connection consistency.

    The returned consistency value is the normalized energy of
    ``z_head - T z_tail`` for the learned stalk vectors.  It is a training
    surrogate, not a claim that an exact cochain map has been constructed.
    """

    def __init__(
        self,
        expert_config: ExpertConfig,
        translator_config: TranslatorConfig,
    ) -> None:
        super().__init__()
        hidden = translator_config.hidden_dim
        rank = translator_config.stalk_rank
        self.eps = translator_config.eps
        self.node_lift = MLP(expert_config.node_feature_dim, hidden, rank)
        self.edge_lift = MLP(expert_config.edge_feature_dim + rank + 1, hidden, hidden)
        self.node_reconstruction = MLP(rank, hidden, expert_config.node_feature_dim)
        self.edge_reconstruction = MLP(hidden, hidden, expert_config.edge_feature_dim)
        self.task_readout = MLP(
            hidden + 2 * rank,
            hidden,
            expert_config.embedding_dim,
        )
        self.task_classifier = nn.Linear(
            expert_config.embedding_dim,
            expert_config.num_classes,
        )

    @staticmethod
    def _residual(node_latent: Tensor, sheaf: dict[str, Tensor]) -> Tensor:
        tails = safe_gather_nodes(node_latent, sheaf["edge_index"][:, 0])
        heads = safe_gather_nodes(node_latent, sheaf["edge_index"][:, 1])
        connection = apply_mask(sheaf["transport"], sheaf["edge_mask"]).to(
            node_latent.dtype
        )
        transported = torch.einsum("beij,bej->bei", connection, tails)
        return apply_mask(heads - transported, sheaf["edge_mask"])

    def forward(self, batch: StructuredBatch) -> TranslationOutput:
        graph = batch.model_inputs(SignalRegime.GRAPH)
        sheaf = batch.model_inputs(SignalRegime.SHEAF)
        node_latent = apply_mask(
            self.node_lift(apply_mask(graph["node_features"], graph["node_mask"])),
            graph["node_mask"],
        )
        residual = self._residual(node_latent, sheaf)
        residual_norm = (
            residual.to(torch.float32).square().sum(dim=-1, keepdim=True).sqrt()
        )
        edge_inputs = torch.cat(
            (
                graph["edge_features"],
                residual.to(graph["edge_features"].dtype),
                residual_norm.to(graph["edge_features"].dtype),
            ),
            dim=-1,
        )
        edge_latent = apply_mask(
            self.edge_lift(apply_mask(edge_inputs, graph["edge_mask"])),
            graph["edge_mask"],
        )
        structure_logits = apply_mask(
            residual_norm.squeeze(-1).to(node_latent.dtype),
            graph["edge_mask"],
        )
        task_embedding = self.task_readout(
            torch.cat(
                (
                    masked_mean(node_latent, graph["node_mask"]),
                    masked_mean(edge_latent, graph["edge_mask"]),
                    masked_mean(residual.abs(), graph["edge_mask"]),
                ),
                dim=-1,
            )
        )
        task_logits = self.task_classifier(task_embedding)
        node_reconstruction = apply_mask(
            self.node_reconstruction(node_latent), graph["node_mask"]
        )
        edge_reconstruction = apply_mask(
            self.edge_reconstruction(edge_latent), graph["edge_mask"]
        )
        node_loss = masked_mse(
            node_reconstruction, graph["node_features"], graph["node_mask"]
        )
        edge_loss = masked_mse(
            edge_reconstruction, graph["edge_features"], graph["edge_mask"]
        )
        reconstruction_loss = 0.5 * (node_loss + edge_loss)

        residual_energy = masked_feature_energy(residual, graph["edge_mask"])
        stalk_energy = masked_feature_energy(node_latent, graph["node_mask"])
        per_sample_consistency = residual_energy / stalk_energy.clamp_min(self.eps)
        consistency_surrogate = per_sample_consistency.mean()
        per_sample_reconstruction = 0.5 * (
            _per_sample_masked_mse(
                node_reconstruction, graph["node_features"], graph["node_mask"]
            )
            + _per_sample_masked_mse(
                edge_reconstruction, graph["edge_features"], graph["edge_mask"]
            )
        )
        return TranslationOutput(
            task_embedding=task_embedding,
            task_logits=task_logits,
            structure_logits=structure_logits,
            node_latent=node_latent,
            edge_latent=edge_latent,
            higher_latent=residual,
            node_reconstruction=node_reconstruction,
            edge_reconstruction=edge_reconstruction,
            reconstruction_loss=reconstruction_loss,
            consistency_surrogate=consistency_surrogate,
            supervision_loss=reconstruction_loss * 0.0,
            per_sample_diagnostics=torch.stack(
                (per_sample_reconstruction, per_sample_consistency), dim=-1
            ),
        )


__all__ = ["GraphToCellTranslator", "GraphToSheafTranslator"]
