"""Generic aggregation for task, structural, and compute route diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Real
from typing import Literal

import torch
from torch import Tensor

DiagnosticValue = Tensor | float | int
AggregateReduction = Literal["none", "mean", "sum"]


def aggregate_route_diagnostics(
    components: Mapping[str, DiagnosticValue],
    weights: Mapping[str, float] | None = None,
    *,
    reduction: AggregateReduction = "none",
    require_finite: bool = True,
) -> Tensor:
    """Return a differentiable weighted sum of named route diagnostics.

    This utility assigns no mathematical meaning to the components: callers
    must record which terms, normalizations, and weights were used. Tensor
    components may be scalar or broadcast-compatible per-example arrays.
    Unknown weight names and negative/non-finite weights are rejected.
    """

    if not components:
        raise ValueError("at least one diagnostic component is required")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of: none, mean, sum")
    weights = {} if weights is None else dict(weights)
    unknown_weights = sorted(set(weights) - set(components))
    if unknown_weights:
        raise ValueError(
            f"weights reference unknown components: {', '.join(unknown_weights)}"
        )

    tensor_values = [
        value for value in components.values() if isinstance(value, Tensor)
    ]
    devices = {value.device for value in tensor_values}
    if len(devices) > 1:
        raise ValueError("tensor diagnostics must share a device")
    device = tensor_values[0].device if tensor_values else torch.device("cpu")
    dtype = torch.float64
    if tensor_values:
        dtype = (
            tensor_values[0].dtype
            if tensor_values[0].is_floating_point()
            else torch.float32
        )
        for value in tensor_values[1:]:
            value_dtype = value.dtype if value.is_floating_point() else torch.float32
            dtype = torch.promote_types(dtype, value_dtype)

    weighted: list[Tensor] = []
    for name, component in components.items():
        if not isinstance(name, str) or not name:
            raise ValueError("diagnostic component names must be nonempty strings")
        if not isinstance(component, (Tensor, Real)):
            raise TypeError(f"diagnostic {name!r} must be a tensor or real number")
        weight = float(weights.get(name, 1.0))
        if not isfinite(weight) or weight < 0:
            raise ValueError(f"weight for {name!r} must be finite and nonnegative")
        value = torch.as_tensor(component, dtype=dtype, device=device)
        if require_finite and not torch.isfinite(value).all():
            raise ValueError(f"diagnostic {name!r} contains non-finite values")
        weighted.append(value * weight)

    broadcast = torch.broadcast_tensors(*weighted)
    total = torch.stack(broadcast, dim=0).sum(dim=0)
    if reduction == "mean":
        return total.mean()
    if reduction == "sum":
        return total.sum()
    return total


__all__ = ["aggregate_route_diagnostics"]
