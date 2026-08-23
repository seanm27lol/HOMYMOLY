"""Deterministic CPU shortcut baselines for the confirmatory benchmark.

These models deliberately avoid graph adjacency, face-edge incidence, and cycle
composition.  They measure how far cheap amplitudes, pooled unary statistics,
or separately permutation-invariant node/edge/transport sets can get before a
structured model is credited with a Gate-2 result.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW

from homymoly.data.collate import collate_structured
from homymoly.data.confirmatory import (
    ConfirmatoryConfig,
    ConfirmatoryStructuredSignal,
)
from homymoly.data.types import SignalRegime, StructuredBatch, StructuredSample

ROUTES = tuple(SignalRegime)
ROUTE_TO_INDEX = {route: index for index, route in enumerate(ROUTES)}


@dataclass(frozen=True, slots=True)
class ShortcutBaselineConfig:
    """Small, fully deterministic baseline-campaign configuration."""

    num_samples: int = 600
    data_seed: int = 20260803
    split_seed: int = 404
    training_seed: int = 1701
    min_vertices: int = 24
    max_vertices: int = 64
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    epochs: int = 40
    batch_size: int = 64
    hidden_dim: int = 32
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    patience: int = 10
    num_threads: int = 1

    def __post_init__(self) -> None:
        for name in (
            "num_samples",
            "epochs",
            "batch_size",
            "hidden_dim",
            "patience",
            "num_threads",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_samples < 18 or self.num_samples % 6:
            raise ValueError("num_samples must be at least 18 and divisible by six")
        for name in ("data_seed", "split_seed", "training_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not 24 <= self.min_vertices <= self.max_vertices <= 96:
            raise ValueError("vertices must satisfy 24 <= min <= max <= 96")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must lie strictly between zero and one")
        if not 0 < self.validation_fraction < 1:
            raise ValueError(
                "validation_fraction must lie strictly between zero and one"
            )
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("training and validation fractions must sum below one")
        for name in ("learning_rate", "weight_decay"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError(
                "learning_rate must be positive and weight_decay non-negative"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class _CampaignData:
    samples: tuple[StructuredSample, ...]
    batch: StructuredBatch
    scalar_features: Tensor
    pooled_features: Tensor
    structure_features: Tensor
    labels: Tensor
    regimes: Tensor
    splits: dict[str, tuple[int, ...]]


class _FeatureMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


class PermutationInvariantBaseline(nn.Module):
    """DeepSets-style baseline over separate node, edge, and transport sets.

    It intentionally receives no ``edge_index``, ``face_index``, or face-edge
    incidence.  Therefore it can exploit unary distributions but cannot compute
    the benchmark's intended graph relations, filled-probe relation, or cycle
    holonomy directly.
    """

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        structure_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        def encoder(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        self.node_encoder = encoder(node_dim)
        self.edge_encoder = encoder(edge_dim)
        self.transport_encoder = encoder(4)
        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden_dim + structure_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _masked_mean(encoded: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(dtype=encoded.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return (encoded * weights).sum(dim=1) / denominator

    def forward(
        self,
        node_features: Tensor,
        node_mask: Tensor,
        edge_features: Tensor,
        edge_mask: Tensor,
        transport: Tensor,
        structure_features: Tensor,
    ) -> Tensor:
        node_embedding = self._masked_mean(self.node_encoder(node_features), node_mask)
        edge_embedding = self._masked_mean(self.edge_encoder(edge_features), edge_mask)
        transport_embedding = self._masked_mean(
            self.transport_encoder(transport.flatten(start_dim=-2)), edge_mask
        )
        pooled = torch.cat(
            (
                node_embedding,
                edge_embedding,
                transport_embedding,
                structure_features,
            ),
            dim=-1,
        )
        return self.classifier(pooled).squeeze(-1)


def scalar_amplitude_features(sample: StructuredSample) -> Tensor:
    """Return the three label-independent amplitude cues audited in Gate 1."""

    top_edges = sample.edge_features[:, 1].abs().topk(min(3, sample.num_edges))
    return torch.tensor(
        (
            float(sample.node_features[:, 0].abs().max()),
            float(top_edges.values.mean()),
            float(torch.linalg.vector_norm(sample.node_features[:, -2:], dim=1).mean()),
        ),
        dtype=torch.float32,
    )


def _summary(values: Tensor) -> Tensor:
    values = values.to(dtype=torch.float32)
    return torch.cat(
        (
            values.mean(dim=0),
            values.std(dim=0, unbiased=False),
            values.amin(dim=0),
            values.amax(dim=0),
            values.abs().mean(dim=0),
        )
    )


def pooled_unary_features(sample: StructuredSample) -> Tensor:
    """Summarize unary distributions without composing structural relations."""

    transport = sample.transport.flatten(start_dim=-2)
    trace = sample.transport.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
    determinant = torch.linalg.det(sample.transport).unsqueeze(-1)
    counts = torch.log1p(
        torch.tensor(
            (
                sample.num_vertices,
                sample.num_edges,
                sample.num_faces,
                int(sample.face_active.sum()),
            ),
            dtype=torch.float32,
        )
    )
    return torch.cat(
        (
            counts,
            _summary(sample.node_features),
            _summary(sample.edge_features),
            _summary(transport),
            _summary(trace),
            _summary(determinant),
        )
    )


def _structure_features(batch: StructuredBatch) -> Tensor:
    active_faces = batch.face_active.sum(dim=1).to(dtype=torch.float32)
    counts = torch.stack(
        (
            batch.num_vertices.to(dtype=torch.float32),
            batch.num_edges.to(dtype=torch.float32),
            batch.num_faces.to(dtype=torch.float32),
            active_faces,
        ),
        dim=-1,
    )
    return torch.log1p(counts)


def _materialize(config: ShortcutBaselineConfig) -> _CampaignData:
    dataset = ConfirmatoryStructuredSignal(
        ConfirmatoryConfig(
            num_samples=config.num_samples,
            seed=config.data_seed,
            min_vertices=config.min_vertices,
            max_vertices=config.max_vertices,
        )
    )
    splits = dataset.split_indices(
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        seed=config.split_seed,
    )
    samples = tuple(dataset[index] for index in range(len(dataset)))
    batch = collate_structured(samples)
    return _CampaignData(
        samples=samples,
        batch=batch,
        scalar_features=torch.stack(
            [scalar_amplitude_features(sample) for sample in samples]
        ),
        pooled_features=torch.stack(
            [pooled_unary_features(sample) for sample in samples]
        ),
        structure_features=_structure_features(batch),
        labels=batch.labels.to(dtype=torch.float32),
        regimes=torch.tensor(
            [ROUTE_TO_INDEX[regime] for regime in batch.regimes], dtype=torch.long
        ),
        splits=splits,
    )


@contextmanager
def _deterministic_cpu(seed: int, num_threads: int):  # type: ignore[no-untyped-def]
    prior_threads = torch.get_num_threads()
    prior_rng = torch.random.get_rng_state()
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(num_threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.random.set_rng_state(prior_rng)
        torch.use_deterministic_algorithms(prior_deterministic)
        torch.set_num_threads(prior_threads)


def _standardize(features: Tensor, train_indices: tuple[int, ...]) -> Tensor:
    train = features[list(train_indices)]
    mean = train.mean(dim=0)
    scale = train.std(dim=0, unbiased=False)
    scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
    return (features - mean) / scale


def _accuracy(logits: Tensor, labels: Tensor, indices: tuple[int, ...]) -> float:
    selected = list(indices)
    predictions = (logits[selected] >= 0).to(dtype=torch.long)
    targets = labels[selected].to(dtype=torch.long)
    return float((predictions == targets).to(dtype=torch.float32).mean())


def _metrics(
    logits: Tensor,
    data: _CampaignData,
    *,
    best_epoch: int | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split_name, indices in data.splits.items():
        report[f"{split_name}_accuracy"] = _accuracy(logits, data.labels, indices)
    test_indices = data.splits["test"]
    by_regime: dict[str, float] = {}
    for route_index, route in enumerate(ROUTES):
        indices = tuple(
            index for index in test_indices if int(data.regimes[index]) == route_index
        )
        by_regime[route.value] = _accuracy(logits, data.labels, indices)
    report["test_accuracy_by_regime"] = by_regime
    if best_epoch is not None:
        report["best_epoch"] = best_epoch
    return report


def _fit_binary_model(
    model: nn.Module,
    forward: Callable[[Tensor], Tensor],
    data: _CampaignData,
    config: ShortcutBaselineConfig,
    *,
    seed_offset: int,
) -> tuple[dict[str, Any], Tensor]:
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(config.training_seed + seed_offset)
    train = torch.tensor(data.splits["train"], dtype=torch.long)
    validation = torch.tensor(data.splits["validation"], dtype=torch.long)
    best_loss = float("inf")
    best_epoch = 0
    best_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        order = train[torch.randperm(len(train), generator=generator)]
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = forward(indices)
            loss = F.binary_cross_entropy_with_logits(logits, data.labels[indices])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = float(
                F.binary_cross_entropy_with_logits(
                    forward(validation), data.labels[validation]
                )
            )
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        all_indices = torch.arange(len(data.samples), dtype=torch.long)
        logits = forward(all_indices).detach().cpu()
    return _metrics(logits, data, best_epoch=best_epoch), logits


def _nearest_centroid_route_accuracy(data: _CampaignData) -> float:
    features = _standardize(data.scalar_features, data.splits["train"])
    train = torch.tensor(data.splits["train"], dtype=torch.long)
    test = torch.tensor(data.splits["test"], dtype=torch.long)
    centroids = torch.stack(
        [
            features[train][data.regimes[train] == route_index].mean(dim=0)
            for route_index in range(len(ROUTES))
        ]
    )
    distances = ((features[test, None, :] - centroids[None, :, :]) ** 2).sum(dim=-1)
    predictions = distances.argmin(dim=-1)
    return float((predictions == data.regimes[test]).to(dtype=torch.float32).mean())


def _graph_oracle(sample: StructuredSample) -> int:
    tail, head = sample.edge_index
    anchor_mask = (sample.node_features[tail, 0].abs() > 0.5) & (
        sample.node_features[head, 0].abs() > 0.5
    )
    products = (
        sample.node_features[tail[anchor_mask], 0]
        * sample.node_features[head[anchor_mask], 0]
    )
    return int(float(products.sum()) > 0)


def _cell_oracle(sample: StructuredSample) -> int:
    edge_positions = {
        tuple(edge): position
        for position, edge in enumerate(sample.edge_index.t().tolist())
    }
    circulations = []
    for a, b, c in sample.face_index.t().tolist():
        circulations.append(
            sample.edge_features[edge_positions[(b, c)], 1]
            - sample.edge_features[edge_positions[(a, c)], 1]
            + sample.edge_features[edge_positions[(a, b)], 1]
        )
    probe_position = int(torch.stack(circulations).abs().argmax())
    return int(sample.face_active[probe_position])


def _sheaf_oracle(sample: StructuredSample) -> int:
    edge_positions = {
        tuple(edge): position
        for position, edge in enumerate(sample.edge_index.t().tolist())
    }
    identity = torch.eye(2, dtype=sample.transport.dtype)
    largest_defect = 0.0
    for a, b, c in sample.face_index.t().tolist():
        transport_ab = sample.transport[edge_positions[(a, b)]]
        transport_ac = sample.transport[edge_positions[(a, c)]]
        transport_bc = sample.transport[edge_positions[(b, c)]]
        holonomy = transport_ac.transpose(-1, -2) @ transport_bc @ transport_ab
        largest_defect = max(
            largest_defect, float(torch.linalg.matrix_norm(holonomy - identity))
        )
    return int(largest_defect > 1.0)


def _oracle_report(data: _CampaignData) -> dict[str, Any]:
    predictors = (_graph_oracle, _cell_oracle, _sheaf_oracle)
    predictions = torch.tensor(
        [[predictor(sample) for predictor in predictors] for sample in data.samples],
        dtype=torch.long,
    )
    test = data.splits["test"]
    conditional: dict[str, float] = {}
    for route_index, route in enumerate(ROUTES):
        indices = [index for index in test if int(data.regimes[index]) == route_index]
        conditional[route.value] = float(
            (
                predictions[indices, route_index]
                == data.labels[indices].to(dtype=torch.long)
            )
            .to(dtype=torch.float32)
            .mean()
        )
    routed = predictions[torch.arange(len(data.samples)), data.regimes]
    routed_logits = torch.where(
        routed.to(dtype=torch.bool),
        torch.ones_like(routed, dtype=torch.float32),
        -torch.ones_like(routed, dtype=torch.float32),
    )
    return {
        "conditional_test_accuracy": conditional,
        "routed_test_accuracy": _accuracy(routed_logits, data.labels, test),
        "uses_hidden_regime": True,
        "purpose": "reference ceiling only; not a deployable baseline",
    }


def _split_report(data: _CampaignData) -> dict[str, Any]:
    group_sets = {
        name: {int(data.samples[index].metadata["group_id"]) for index in indices}
        for name, indices in data.splits.items()
    }
    names = tuple(group_sets)
    group_disjoint = all(
        group_sets[names[left]].isdisjoint(group_sets[names[right]])
        for left in range(len(names))
        for right in range(left + 1, len(names))
    )
    return {
        "sizes": {name: len(indices) for name, indices in data.splits.items()},
        "group_counts": {name: len(groups) for name, groups in group_sets.items()},
        "group_disjoint": group_disjoint,
    }


def run_shortcut_baselines(
    config: ShortcutBaselineConfig | None = None,
) -> dict[str, Any]:
    """Run all confirmatory shortcut baselines and return a JSON-safe report."""

    selected = config or ShortcutBaselineConfig()
    with _deterministic_cpu(selected.training_seed, selected.num_threads):
        data = _materialize(selected)

        train_labels = data.labels[list(data.splits["train"])].to(dtype=torch.long)
        majority_class = int(torch.bincount(train_labels, minlength=2).argmax())
        majority_logits = torch.full(
            (len(data.samples),),
            1.0 if majority_class else -1.0,
            dtype=torch.float32,
        )
        majority_report = {
            **_metrics(majority_logits, data),
            "majority_class": majority_class,
        }

        scalar = _standardize(data.scalar_features, data.splits["train"])
        torch.manual_seed(selected.training_seed + 11)
        scalar_model = _FeatureMLP(scalar.shape[1], selected.hidden_dim)
        scalar_report, _ = _fit_binary_model(
            scalar_model,
            lambda indices: scalar_model(scalar[indices]),
            data,
            selected,
            seed_offset=11,
        )
        scalar_report["route_accuracy"] = _nearest_centroid_route_accuracy(data)
        scalar_report["feature_names"] = [
            "max_abs_graph_channel",
            "top3_abs_cell_channel",
            "mean_sheaf_field_norm",
        ]

        pooled = _standardize(data.pooled_features, data.splits["train"])
        torch.manual_seed(selected.training_seed + 23)
        pooled_model = _FeatureMLP(pooled.shape[1], selected.hidden_dim)
        pooled_report, _ = _fit_binary_model(
            pooled_model,
            lambda indices: pooled_model(pooled[indices]),
            data,
            selected,
            seed_offset=23,
        )

        torch.manual_seed(selected.training_seed + 37)
        deepsets_model = PermutationInvariantBaseline(
            node_dim=data.batch.node_features.shape[-1],
            edge_dim=data.batch.edge_features.shape[-1],
            structure_dim=data.structure_features.shape[-1],
            hidden_dim=selected.hidden_dim,
        )

        def deepsets_forward(indices: Tensor) -> Tensor:
            return deepsets_model(
                data.batch.node_features[indices],
                data.batch.node_mask[indices],
                data.batch.edge_features[indices],
                data.batch.edge_mask[indices],
                data.batch.transport[indices],
                data.structure_features[indices],
            )

        deepsets_report, _ = _fit_binary_model(
            deepsets_model,
            deepsets_forward,
            data,
            selected,
            seed_offset=37,
        )

        return {
            "schema_version": 1,
            "benchmark_tier": "confirmatory",
            "config": selected.as_dict(),
            "split": _split_report(data),
            "baselines": {
                "constant_majority": majority_report,
                "scalar_amplitude": scalar_report,
                "pooled_mlp": pooled_report,
                "permutation_invariant_deepsets": deepsets_report,
            },
            "relational_oracles": _oracle_report(data),
        }


__all__ = [
    "PermutationInvariantBaseline",
    "ShortcutBaselineConfig",
    "pooled_unary_features",
    "run_shortcut_baselines",
    "scalar_amplitude_features",
]
