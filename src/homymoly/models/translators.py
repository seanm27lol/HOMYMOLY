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
    face_boundary_coefficients,
    face_holonomy,
    face_vertex_mean,
    masked_feature_energy,
    masked_max,
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
    """Predict a cell lift from graph observations and candidate incidence.

    By default the target activity mask is held out of the forward inputs and
    used only to supervise the predicted face gates. An explicit compatibility
    flag can restore the historical target-view encoder for old checkpoints,
    but that mode must not be described as graph-only conversion.

    ``map_reconstruction_loss`` compares the predicted soft cellular boundary
    operator against the held-out target operator. It is not a cone loss or a
    chain-map residual; it is an explicit, typed structure-reconstruction term.
    """

    def __init__(
        self,
        expert_config: ExpertConfig,
        translator_config: TranslatorConfig,
    ) -> None:
        super().__init__()
        hidden = translator_config.hidden_dim
        self.target_structure_access = translator_config.target_structure_access
        self.node_lift = MLP(expert_config.node_feature_dim, hidden, hidden)
        self.edge_lift = MLP(expert_config.edge_feature_dim, hidden, hidden)
        self.face_lift = MLP(2 * hidden + 1, hidden, hidden)
        self.face_gate = MLP(2 * hidden + 1, hidden, 1)
        self.face_boundary_prediction = nn.Linear(hidden, hidden)
        self.node_reconstruction = MLP(hidden, hidden, expert_config.node_feature_dim)
        self.edge_reconstruction = MLP(hidden, hidden, expert_config.edge_feature_dim)
        self.task_readout = MLP(
            4 * hidden,
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
            face_boundary=cell.get("face_boundary"),
        )
        vertex_latent = face_vertex_mean(
            node_latent,
            cell["face_index"],
            candidate_mask,
            face_vertices=cell.get("face_vertices"),
        )
        # Historical compatibility may expose the observed target activity.
        # In graph-only mode the corresponding input channel is identically
        # zero and face_active appears only in the supervised losses below.
        activity_input = (
            cell["face_active"].unsqueeze(-1).to(vertex_latent.dtype)
            if self.target_structure_access
            else torch.zeros_like(vertex_latent[..., :1])
        )
        face_inputs = torch.cat(
            (
                vertex_latent,
                boundary_latent,
                activity_input,
            ),
            dim=-1,
        )
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
                    masked_max(higher_latent, candidate_mask),
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
        boundary_operator = face_boundary_coefficients(
            graph["edge_index"],
            graph["edge_mask"],
            cell["face_index"],
            candidate_mask,
            dtype=torch.float32,
            face_boundary=cell.get("face_boundary"),
        )
        soft_faces = torch.sigmoid(structure_logits.float()) * candidate_mask.float()
        predicted_boundary_operator = boundary_operator * soft_faces.unsqueeze(-1)
        target_boundary_operator = boundary_operator * cell[
            "face_active"
        ].float().unsqueeze(-1)
        per_sample_map_reconstruction = _per_sample_masked_mse(
            predicted_boundary_operator,
            target_boundary_operator,
            candidate_mask,
        )
        map_reconstruction_loss = per_sample_map_reconstruction.mean()

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
            map_reconstruction_loss=map_reconstruction_loss,
            supervision_loss=supervision_loss,
            per_sample_diagnostics=torch.stack(
                (
                    per_sample_reconstruction,
                    per_sample_consistency,
                    per_sample_map_reconstruction,
                ),
                dim=-1,
            ),
        )


class GraphToSheafTranslator(nn.Module):
    """Lift node features to rank-2 stalks with connection consistency.

    The returned consistency value is the normalized energy of
    ``z_head - T z_tail`` for the learned stalk vectors.  It is a training
    surrogate, not a claim that an exact cochain map has been constructed.

    The task readout additionally encodes every face's observed transport
    holonomy (mean and max over faces): the sheaf-regime label lives in
    cycle holonomy, which per-edge residuals cannot see — the same failure
    the sheaf expert had before its holonomy pathway.  The transports are
    observation-level structure, so this uses no supervision metadata.
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
        self.target_structure_access = translator_config.target_structure_access
        self.node_lift = MLP(expert_config.node_feature_dim, hidden, rank)
        self.transport_angle = MLP(
            expert_config.edge_feature_dim + 2 * rank,
            hidden,
            1,
        )
        self.edge_lift = MLP(expert_config.edge_feature_dim + rank + 1, hidden, hidden)
        self.face_encoder = MLP(4, hidden, hidden)
        self.node_reconstruction = MLP(rank, hidden, expert_config.node_feature_dim)
        self.edge_reconstruction = MLP(hidden, hidden, expert_config.edge_feature_dim)
        self.task_readout = MLP(
            3 * hidden + 2 * rank,
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
        if self.target_structure_access:
            predicted_transport = sheaf["transport"].to(node_latent.dtype)
        else:
            tails = safe_gather_nodes(node_latent, graph["edge_index"][:, 0])
            heads = safe_gather_nodes(node_latent, graph["edge_index"][:, 1])
            angle = apply_mask(
                self.transport_angle(
                    torch.cat((graph["edge_features"], tails, heads), dim=-1)
                ).squeeze(-1),
                graph["edge_mask"],
            )
            cosine = torch.cos(angle.float()).to(node_latent.dtype)
            sine = torch.sin(angle.float()).to(node_latent.dtype)
            predicted_transport = torch.stack(
                (
                    torch.stack((cosine, -sine), dim=-1),
                    torch.stack((sine, cosine), dim=-1),
                ),
                dim=-2,
            )
            predicted_transport = apply_mask(predicted_transport, graph["edge_mask"])
        predicted_sheaf = {**sheaf, "transport": predicted_transport}
        residual = self._residual(node_latent, predicted_sheaf)
        # The consistency loss drives residuals toward zero, where an
        # unclamped sqrt has infinite derivative.  Eps must go *inside* the
        # sqrt: clamping afterwards still chains 0 * inf = NaN on backward.
        residual_norm = (
            residual.to(torch.float32)
            .square()
            .sum(dim=-1, keepdim=True)
            .add(self.eps)
            .sqrt()
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
        face_valid = sheaf["face_mask"]
        holonomy = face_holonomy(
            predicted_transport,
            sheaf["edge_index"],
            sheaf["edge_mask"],
            sheaf["face_index"],
            face_valid,
            face_boundary=sheaf.get("face_boundary"),
        )
        identity = torch.eye(2, dtype=holonomy.dtype, device=holonomy.device)
        face_hidden = apply_mask(
            self.face_encoder((holonomy - identity).flatten(-2).to(node_latent.dtype)),
            face_valid,
        )
        task_embedding = self.task_readout(
            torch.cat(
                (
                    masked_mean(node_latent, graph["node_mask"]),
                    masked_mean(edge_latent, graph["edge_mask"]),
                    masked_mean(residual.abs(), graph["edge_mask"]),
                    masked_mean(face_hidden, face_valid),
                    masked_max(face_hidden, face_valid),
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
        per_sample_map_reconstruction = _per_sample_masked_mse(
            predicted_transport,
            sheaf["transport"],
            graph["edge_mask"],
        )
        map_reconstruction_loss = per_sample_map_reconstruction.mean()
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
            map_reconstruction_loss=map_reconstruction_loss,
            supervision_loss=reconstruction_loss * 0.0,
            per_sample_diagnostics=torch.stack(
                (
                    per_sample_reconstruction,
                    per_sample_consistency,
                    per_sample_map_reconstruction,
                ),
                dim=-1,
            ),
        )


__all__ = ["GraphToCellTranslator", "GraphToSheafTranslator"]
