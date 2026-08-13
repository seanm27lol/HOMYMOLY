"""Masked graph, cell, and rank-2 connection-sheaf experts."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from homymoly.data import SignalRegime, StructuredBatch

from .config import ExpertConfig
from .ops import (
    MLP,
    GraphMessageLayer,
    apply_mask,
    face_boundary_aggregate,
    face_boundary_coefficients,
    face_holonomy,
    face_vertex_mean,
    masked_feature_energy,
    masked_max,
    masked_mean,
    safe_gather_nodes,
    scatter_faces_to_nodes,
    scatter_mean_to_nodes,
)
from .outputs import EnsembleOutput, ExpertOutput

ROUTE_ORDER = (
    SignalRegime.GRAPH,
    SignalRegime.CELL,
    SignalRegime.SHEAF,
)


def _embedding_diagnostics(structural_energy: Tensor, embedding: Tensor) -> Tensor:
    feature_norm = embedding.to(torch.float32).square().mean(dim=-1)
    return torch.stack((structural_energy.to(torch.float32), feature_norm), dim=-1)


class _GraphBackbone(nn.Module):
    def __init__(self, config: ExpertConfig) -> None:
        super().__init__()
        self.node_encoder = MLP(
            config.node_feature_dim,
            config.hidden_dim,
            config.hidden_dim,
            dropout=config.dropout,
        )
        self.edge_encoder = MLP(
            config.edge_feature_dim,
            config.hidden_dim,
            config.hidden_dim,
            dropout=config.dropout,
        )
        self.layers = nn.ModuleList(
            GraphMessageLayer(config.hidden_dim, dropout=config.dropout)
            for _ in range(config.num_layers)
        )

    def forward(self, inputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        node_hidden = apply_mask(
            self.node_encoder(apply_mask(inputs["node_features"], inputs["node_mask"])),
            inputs["node_mask"],
        )
        edge_hidden = apply_mask(
            self.edge_encoder(apply_mask(inputs["edge_features"], inputs["edge_mask"])),
            inputs["edge_mask"],
        )
        for layer in self.layers:
            node_hidden, edge_hidden = layer(
                node_hidden,
                edge_hidden,
                inputs["edge_index"],
                inputs["node_mask"],
                inputs["edge_mask"],
            )
        return node_hidden, edge_hidden


class GraphExpert(nn.Module):
    """Edge-conditioned graph message passing on the graph-scoped view.

    The graph-regime label is a pairwise statistic (sign agreement between
    the two endpoints of a few unmarked anchor edges).  Neither the per-node
    encoder nor mean pooling can form or preserve such a product, so the
    expert adds a raw endpoint-pair pathway: each edge sees its two
    endpoints' raw features, and the readout keeps the masked max so a
    single informative edge survives aggregation.
    """

    route = SignalRegime.GRAPH

    def __init__(self, config: ExpertConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = _GraphBackbone(config)
        self.edge_pair_encoder = MLP(
            2 * config.node_feature_dim,
            config.hidden_dim,
            config.hidden_dim,
            dropout=config.dropout,
        )
        self.readout = MLP(
            3 * config.hidden_dim,
            config.hidden_dim,
            config.embedding_dim,
            dropout=config.dropout,
        )
        self.classifier = nn.Linear(config.embedding_dim, config.num_classes)

    def forward(self, batch: StructuredBatch) -> ExpertOutput:
        inputs = batch.model_inputs(self.route)
        node_hidden, edge_hidden = self.backbone(inputs)
        tail_features = safe_gather_nodes(
            inputs["node_features"], inputs["edge_index"][:, 0]
        )
        head_features = safe_gather_nodes(
            inputs["node_features"], inputs["edge_index"][:, 1]
        )
        pair_hidden = apply_mask(
            self.edge_pair_encoder(torch.cat((tail_features, head_features), dim=-1)),
            inputs["edge_mask"],
        )
        pooled = torch.cat(
            (
                masked_mean(node_hidden, inputs["node_mask"]),
                masked_mean(edge_hidden, inputs["edge_mask"]),
                masked_max(pair_hidden, inputs["edge_mask"]),
            ),
            dim=-1,
        )
        embedding = self.readout(pooled)
        logits = self.classifier(embedding)
        structural = masked_feature_energy(edge_hidden, inputs["edge_mask"])
        return ExpertOutput(
            route=self.route,
            embedding=embedding,
            logits=logits,
            diagnostics=_embedding_diagnostics(structural, embedding),
        )


class CellExpert(nn.Module):
    """Graph processing augmented by active oriented triangular 2-cells.

    ``molecular_mode`` adds two ring-motivated inputs to the face encoder:
    a per-face masked max over boundary edge features (the strongest bond in
    the ring, which the oriented sum can cancel away) and the ring size as a
    normalized scalar.  It changes only the face encoder's input width; the
    synthetic path is unchanged when the flag is off.
    """

    route = SignalRegime.CELL

    def __init__(self, config: ExpertConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = _GraphBackbone(config)
        self.face_encoder = MLP(
            (3 if config.molecular_mode else 2) * config.hidden_dim
            + (1 if config.molecular_mode else 0)
            + (config.bond_feature_dim if config.molecular_mode else 0),
            config.hidden_dim,
            config.hidden_dim,
            dropout=config.dropout,
        )
        self.node_face_update = MLP(
            2 * config.hidden_dim,
            config.hidden_dim,
            config.hidden_dim,
            dropout=config.dropout,
        )
        self.node_norm = nn.LayerNorm(config.hidden_dim)
        self.readout = MLP(
            3 * config.hidden_dim,
            config.hidden_dim,
            config.embedding_dim,
            dropout=config.dropout,
        )
        self.classifier = nn.Linear(config.embedding_dim, config.num_classes)

    def forward(self, batch: StructuredBatch) -> ExpertOutput:
        inputs = batch.model_inputs(self.route)
        node_hidden, edge_hidden = self.backbone(inputs)
        face_valid = inputs["face_mask"] & inputs["face_active"]
        boundary_hidden = face_boundary_aggregate(
            edge_hidden,
            inputs["edge_index"],
            inputs["edge_mask"],
            inputs["face_index"],
            face_valid,
            face_boundary=inputs.get("face_boundary"),
        )
        vertex_hidden = face_vertex_mean(
            node_hidden,
            inputs["face_index"],
            face_valid,
            face_vertices=inputs.get("face_vertices"),
        )
        face_inputs = torch.cat((vertex_hidden, boundary_hidden), dim=-1)
        if self.config.molecular_mode:
            coefficients = face_boundary_coefficients(
                inputs["edge_index"],
                inputs["edge_mask"],
                inputs["face_index"],
                face_valid,
                dtype=edge_hidden.dtype,
                face_boundary=inputs.get("face_boundary"),
            )
            edge_presence = coefficients.abs() > 0
            negative_fill = torch.full(
                (),
                -1.0e30,
                dtype=edge_hidden.dtype,
                device=edge_hidden.device,
            )
            boundary_max = torch.where(
                edge_presence.unsqueeze(-1), edge_hidden.unsqueeze(1), negative_fill
            ).amax(dim=2)
            # Every valid face has boundary edges; padded faces are zeroed.
            boundary_max = apply_mask(boundary_max, face_valid)
            if inputs.get("face_boundary") is not None:
                ring_size = (inputs["face_boundary"][..., 1] != 0).sum(
                    dim=-1, keepdim=True
                ).to(boundary_hidden.dtype) / 12.0
            else:
                ring_size = torch.full(
                    (*face_valid.shape, 1),
                    0.25,
                    dtype=boundary_hidden.dtype,
                    device=boundary_hidden.device,
                )
            face_inputs = torch.cat((face_inputs, boundary_max, ring_size), dim=-1)
            if self.config.bond_feature_dim > 0:
                # Bond-type histogram of each ring (the aromatic-bond share
                # is the chemically meaningful ring signal); the first
                # bond_feature_dim one-hot channels are the type block.
                type_block = inputs["edge_features"][
                    ..., : self.config.bond_feature_dim
                ]
                type_counts = torch.einsum(
                    "bfe,beh->bfh", coefficients.abs(), type_block
                )
                face_inputs = torch.cat((face_inputs, type_counts), dim=-1)
        face_hidden = apply_mask(
            self.face_encoder(face_inputs),
            face_valid,
        )
        face_messages = scatter_faces_to_nodes(
            face_hidden,
            inputs["face_index"],
            face_valid,
            num_nodes=node_hidden.shape[1],
            face_vertices=inputs.get("face_vertices"),
        )
        node_hidden = apply_mask(
            self.node_norm(
                node_hidden
                + self.node_face_update(torch.cat((node_hidden, face_messages), dim=-1))
            ),
            inputs["node_mask"],
        )
        pooled = torch.cat(
            (
                masked_mean(node_hidden, inputs["node_mask"]),
                masked_mean(edge_hidden, inputs["edge_mask"]),
                masked_mean(face_hidden, face_valid),
            ),
            dim=-1,
        )
        embedding = self.readout(pooled)
        logits = self.classifier(embedding)
        structural = masked_feature_energy(boundary_hidden, face_valid)
        return ExpertOutput(
            route=self.route,
            embedding=embedding,
            logits=logits,
            diagnostics=_embedding_diagnostics(structural, embedding),
        )


class _SheafMessageLayer(nn.Module):
    """Transport-aware tied forward/reverse messages for rank-2 connections."""

    def __init__(self, hidden_dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.to_head = MLP(3 * hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.to_tail = MLP(3 * hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.node_update = MLP(2 * hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.edge_update = MLP(3 * hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_hidden: Tensor,
        edge_hidden: Tensor,
        edge_index: Tensor,
        transport: Tensor,
        node_mask: Tensor,
        edge_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        tails = edge_index[:, 0]
        heads = edge_index[:, 1]
        tail_hidden = safe_gather_nodes(node_hidden, tails)
        head_hidden = safe_gather_nodes(node_hidden, heads)
        vector_channels = self.hidden_dim // 2
        tail_pairs = tail_hidden.reshape(*tail_hidden.shape[:-1], vector_channels, 2)
        head_pairs = head_hidden.reshape(*head_hidden.shape[:-1], vector_channels, 2)
        connection = apply_mask(transport, edge_mask).to(dtype=node_hidden.dtype)
        tail_in_head = torch.einsum("beij,bekj->beki", connection, tail_pairs).flatten(
            -2
        )
        head_in_tail = torch.einsum("beji,bekj->beki", connection, head_pairs).flatten(
            -2
        )

        forward = self.to_head(
            torch.cat((tail_in_head, head_hidden, edge_hidden), dim=-1)
        )
        reverse = self.to_tail(
            torch.cat((head_in_tail, tail_hidden, edge_hidden), dim=-1)
        )
        aggregated = scatter_mean_to_nodes(
            forward, heads, edge_mask, num_nodes=node_hidden.shape[1]
        ) + scatter_mean_to_nodes(
            reverse, tails, edge_mask, num_nodes=node_hidden.shape[1]
        )
        next_nodes = apply_mask(
            self.node_norm(
                node_hidden
                + self.node_update(torch.cat((node_hidden, aggregated), dim=-1))
            ),
            node_mask,
        )
        next_edges = apply_mask(
            self.edge_norm(
                edge_hidden
                + self.edge_update(
                    torch.cat((tail_in_head, head_hidden, edge_hidden), dim=-1)
                )
            ),
            edge_mask,
        )
        return next_nodes, next_edges


class ConnectionSheafExpert(nn.Module):
    """Rank-2 connection expert with a per-face transport-holonomy pathway.

    The per-edge tail-to-head residual remains as an input and structural
    diagnostic, but the confirmatory sheaf label lives in cycle holonomy:
    node-field angles and connection frame angles are independent draws, so
    only a product of transports around a face can see the defect.  The
    expert therefore encodes every face's holonomy and mixes it back into the
    node states before readout.
    """

    route = SignalRegime.SHEAF

    def __init__(self, config: ExpertConfig) -> None:
        super().__init__()
        self.config = config
        self.node_encoder = MLP(
            config.node_feature_dim,
            config.hidden_dim,
            config.hidden_dim,
            dropout=config.dropout,
        )
        self.edge_encoder = MLP(
            config.edge_feature_dim + 3,
            config.hidden_dim,
            config.hidden_dim,
            dropout=config.dropout,
        )
        self.layers = nn.ModuleList(
            _SheafMessageLayer(config.hidden_dim, dropout=config.dropout)
            for _ in range(config.num_layers)
        )
        self.face_encoder = MLP(
            4,
            config.hidden_dim,
            config.hidden_dim,
            dropout=config.dropout,
        )
        self.node_face_update = MLP(
            2 * config.hidden_dim,
            config.hidden_dim,
            config.hidden_dim,
            dropout=config.dropout,
        )
        self.node_norm = nn.LayerNorm(config.hidden_dim)
        self.readout = MLP(
            4 * config.hidden_dim,
            config.hidden_dim,
            config.embedding_dim,
            dropout=config.dropout,
        )
        self.classifier = nn.Linear(config.embedding_dim, config.num_classes)

    @staticmethod
    def _connection_residual(inputs: dict[str, Tensor]) -> Tensor:
        vectors = inputs["node_features"][..., -2:]
        tails = safe_gather_nodes(vectors, inputs["edge_index"][:, 0])
        heads = safe_gather_nodes(vectors, inputs["edge_index"][:, 1])
        connection = apply_mask(inputs["transport"], inputs["edge_mask"])
        transported = torch.einsum("beij,bej->bei", connection, tails)
        return apply_mask(heads - transported, inputs["edge_mask"])

    def forward(self, batch: StructuredBatch) -> ExpertOutput:
        inputs = batch.model_inputs(self.route)
        vectors = inputs["node_features"][..., -2:]
        residual = self._connection_residual(inputs)
        # Eps inside the sqrt keeps the derivative finite at zero residual
        # (clamping after the sqrt would still chain 0 * inf = NaN).
        residual_norm = (
            residual.to(torch.float32)
            .square()
            .sum(dim=-1, keepdim=True)
            .add(1e-12)
            .sqrt()
        )
        edge_inputs = torch.cat(
            (
                inputs["edge_features"],
                residual.to(inputs["edge_features"].dtype),
                residual_norm.to(inputs["edge_features"].dtype),
            ),
            dim=-1,
        )
        node_hidden = apply_mask(
            self.node_encoder(apply_mask(inputs["node_features"], inputs["node_mask"])),
            inputs["node_mask"],
        )
        edge_hidden = apply_mask(
            self.edge_encoder(apply_mask(edge_inputs, inputs["edge_mask"])),
            inputs["edge_mask"],
        )
        for layer in self.layers:
            node_hidden, edge_hidden = layer(
                node_hidden,
                edge_hidden,
                inputs["edge_index"],
                inputs["transport"],
                inputs["node_mask"],
                inputs["edge_mask"],
            )
        face_valid = inputs["face_mask"]
        holonomy = face_holonomy(
            inputs["transport"],
            inputs["edge_index"],
            inputs["edge_mask"],
            inputs["face_index"],
            face_valid,
            face_boundary=inputs.get("face_boundary"),
        )
        identity = torch.eye(2, dtype=holonomy.dtype, device=holonomy.device)
        face_features = (holonomy - identity).flatten(-2)
        face_hidden = apply_mask(
            self.face_encoder(face_features.to(node_hidden.dtype)),
            face_valid,
        )
        face_messages = scatter_faces_to_nodes(
            face_hidden,
            inputs["face_index"],
            face_valid,
            num_nodes=node_hidden.shape[1],
            face_vertices=inputs.get("face_vertices"),
        )
        node_hidden = apply_mask(
            self.node_norm(
                node_hidden
                + self.node_face_update(torch.cat((node_hidden, face_messages), dim=-1))
            ),
            inputs["node_mask"],
        )
        pooled = torch.cat(
            (
                masked_mean(node_hidden, inputs["node_mask"]),
                masked_mean(edge_hidden, inputs["edge_mask"]),
                masked_mean(face_hidden, face_valid),
                # The holonomy defect is a single-face event; mean pooling
                # dilutes it by the face count, so keep the max as well.
                masked_max(face_hidden, face_valid),
            ),
            dim=-1,
        )
        embedding = self.readout(pooled)
        logits = self.classifier(embedding)
        numerator = masked_feature_energy(residual, inputs["edge_mask"])
        denominator = masked_feature_energy(
            apply_mask(vectors, inputs["node_mask"]), inputs["node_mask"]
        ).clamp_min(1e-6)
        structural = numerator / denominator
        return ExpertOutput(
            route=self.route,
            embedding=embedding,
            logits=logits,
            diagnostics=_embedding_diagnostics(structural, embedding),
        )


class FixedExpertEnsemble(nn.Module):
    """Parameter-independent fixed experts with a common output space."""

    def __init__(self, config: ExpertConfig) -> None:
        super().__init__()
        self.config = config
        self.experts = nn.ModuleDict(
            {
                SignalRegime.GRAPH.value: GraphExpert(config),
                SignalRegime.CELL.value: CellExpert(config),
                SignalRegime.SHEAF.value: ConnectionSheafExpert(config),
            }
        )

    def forward_all(self, batch: StructuredBatch) -> tuple[ExpertOutput, ...]:
        return tuple(self.experts[route.value](batch) for route in ROUTE_ORDER)

    def forward(
        self,
        batch: StructuredBatch,
        *,
        route: SignalRegime | str | None = None,
    ) -> EnsembleOutput:
        outputs = self.forward_all(batch)
        embeddings = torch.stack([output.embedding for output in outputs], dim=1)
        expert_logits = torch.stack([output.logits for output in outputs], dim=1)
        if route is None:
            embedding = embeddings.mean(dim=1)
            logits = expert_logits.mean(dim=1)
        else:
            selected = ROUTE_ORDER.index(SignalRegime.coerce(route))
            embedding = embeddings[:, selected]
            logits = expert_logits[:, selected]
        return EnsembleOutput(
            embeddings=embeddings,
            expert_logits=expert_logits,
            embedding=embedding,
            logits=logits,
            expert_outputs=outputs,
        )


__all__ = [
    "ROUTE_ORDER",
    "CellExpert",
    "ConnectionSheafExpert",
    "FixedExpertEnsemble",
    "GraphExpert",
]
