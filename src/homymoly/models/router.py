"""Cost- and diagnostic-aware soft/hard routing."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import RouterConfig
from .ops import MLP
from .outputs import RouterOutput


class DiagnosticCostRouter(nn.Module):
    """Route from a common context, per-route diagnostics, and declared costs."""

    num_routes = 3

    def __init__(self, embedding_dim: int, config: RouterConfig) -> None:
        super().__init__()
        self.config = config
        self.context_scores = MLP(
            embedding_dim,
            config.hidden_dim,
            self.num_routes,
        )
        self.diagnostic_weight = nn.Parameter(
            torch.empty(self.num_routes, config.diagnostic_dim)
        )
        self.diagnostic_bias = nn.Parameter(torch.zeros(self.num_routes))
        nn.init.xavier_uniform_(self.diagnostic_weight)
        self.register_buffer(
            "route_costs",
            torch.tensor(config.route_costs, dtype=torch.float32),
            persistent=True,
        )

    def forward(
        self,
        context: Tensor,
        diagnostics: Tensor,
        *,
        hard: bool = False,
        temperature: float | None = None,
        route_costs: Tensor | None = None,
    ) -> RouterOutput:
        if context.ndim != 2:
            raise ValueError("context must have shape [B, D]")
        expected = (context.shape[0], self.num_routes, self.config.diagnostic_dim)
        if tuple(diagnostics.shape) != expected:
            raise ValueError(f"diagnostics must have shape {expected}")
        selected_temperature = (
            self.config.temperature if temperature is None else temperature
        )
        if selected_temperature <= 0:
            raise ValueError("temperature must be positive")

        if route_costs is None:
            costs = self.route_costs.unsqueeze(0).expand(context.shape[0], -1)
        else:
            costs = route_costs.to(device=context.device, dtype=torch.float32)
            if costs.ndim == 1:
                costs = costs.unsqueeze(0).expand(context.shape[0], -1)
            if tuple(costs.shape) != (context.shape[0], self.num_routes):
                raise ValueError("route_costs must have shape [3] or [B, 3]")

        context_logits = self.context_scores(context).to(torch.float32)
        diagnostic_logits = (
            diagnostics.to(torch.float32) * self.diagnostic_weight.unsqueeze(0)
        ).sum(dim=-1) + self.diagnostic_bias
        normalized_costs = costs / costs.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        logits = (
            context_logits
            + diagnostic_logits
            - self.config.cost_strength * normalized_costs
        )
        soft_weights = torch.softmax(logits / selected_temperature, dim=-1)
        selected_routes = soft_weights.argmax(dim=-1)

        if hard:
            discrete = F.one_hot(selected_routes, num_classes=self.num_routes).to(
                soft_weights.dtype
            )
            if self.training and self.config.straight_through:
                weights = discrete - soft_weights.detach() + soft_weights
            else:
                weights = discrete
        else:
            weights = soft_weights

        expected_cost = (weights * costs).sum(dim=-1)
        entropy = -(
            soft_weights
            * soft_weights.clamp_min(torch.finfo(soft_weights.dtype).tiny).log()
        ).sum(dim=-1)
        return RouterOutput(
            logits=logits,
            weights=weights,
            selected_routes=selected_routes,
            expected_cost=expected_cost,
            entropy=entropy,
        )


__all__ = ["DiagnosticCostRouter"]
