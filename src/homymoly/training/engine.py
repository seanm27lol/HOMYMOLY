"""Phased, resumable Gate-2 training for experts, translators, and routing."""

from __future__ import annotations

import fcntl
import hashlib
import json
import random
import subprocess
import time
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from homymoly.data.collate import collate_structured
from homymoly.data.confirmatory import ConfirmatoryConfig, ConfirmatoryStructuredSignal
from homymoly.data.types import SignalRegime, StructuredBatch
from homymoly.metrics import (
    pairwise_euclidean_distances,
    symmetric_h0_srtd_surrogate,
)
from homymoly.models import (
    ExpertConfig,
    ModelConfig,
    ModelOutput,
    RouterConfig,
    TranslatorConfig,
    build_model,
)
from homymoly.runtime import (
    RuntimeState,
    initialize_runtime,
    maybe_compile,
    seed_worker,
)

from .config import Gate2Config
from .io import MetricLogger, atomic_json, atomic_torch_save, load_checkpoint

ROUTES = (SignalRegime.GRAPH, SignalRegime.CELL, SignalRegime.SHEAF)
PHASES = ("fixed_experts", "translators", "router_warmup", "joint_finetune")


@dataclass(slots=True)
class _RunState:
    phase_index: int = 0
    epoch_in_phase: int = 0
    global_epoch: int = 0
    global_step: int = 0
    best_score: float = float("-inf")
    bad_epochs: int = 0
    translator_baseline: float | None = None
    gate_reports: dict[str, Any] = field(default_factory=dict)


def _config_fingerprint(config: Gate2Config) -> str:
    payload = config.as_dict()
    for environment_key in ("source_path", "project_root", "run_dir"):
        payload.pop(environment_key, None)
    document = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def _code_fingerprint(project_root: Path) -> str:
    """Hash executable experiment sources when Git metadata is unavailable."""

    digest = hashlib.sha256()
    candidates = list((project_root / "src" / "homymoly").rglob("*.py"))
    candidates.extend(
        path
        for path in (
            project_root / "pyproject.toml",
            project_root / "scripts" / "train_gate2.sh",
        )
        if path.is_file()
    )
    for path in sorted(candidates):
        digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


@contextmanager
def _exclusive_run_lock(run_dir: Path):  # type: ignore[no-untyped-def]
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".training.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another training process holds {lock_path}") from exc
        yield


def _git_revision(project_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, arguments in (
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("status", ["git", "status", "--short"]),
    ):
        process = subprocess.run(
            arguments,
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode == 0:
            result[name] = process.stdout.strip()
        else:
            result[f"{name}_error"] = process.stderr.strip()
    return result


def _build_dataset(config: Gate2Config, *, smoke: bool) -> ConfirmatoryStructuredSignal:
    data = config.data
    selected_samples = 120 if smoke else data.num_samples
    selected_max_vertices = min(data.max_vertices, 32) if smoke else data.max_vertices
    dataset_config = ConfirmatoryConfig(
        num_samples=selected_samples,
        seed=data.seed,
        min_vertices=data.min_vertices,
        max_vertices=selected_max_vertices,
        node_feature_dim=data.node_feature_dim,
        edge_feature_dim=data.edge_feature_dim,
        stalk_mode=data.stalk_mode,
        gauge_noise_std=data.gauge_noise_std,
    )
    return ConfirmatoryStructuredSignal(dataset_config)


def _loader(
    dataset: ConfirmatoryStructuredSignal,
    indices: tuple[int, ...],
    *,
    batch_size: int,
    shuffle: bool,
    runtime: RuntimeState,
    pin_memory: bool,
    generator: torch.Generator,
    smoke: bool,
) -> DataLoader[StructuredBatch]:
    num_workers = 0 if smoke else runtime.num_workers
    arguments: dict[str, Any] = {
        "dataset": Subset(dataset, indices),
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_structured,
        "pin_memory": pin_memory and runtime.device.type == "cuda",
        "drop_last": False,
        "generator": generator,
        "worker_init_fn": seed_worker,
        # Recreate workers each epoch so the checkpointed loader generator
        # produces the same sampler/worker-seed sequence after interruption.
        "persistent_workers": False,
    }
    if num_workers > 0:
        arguments["prefetch_factor"] = 2
    return DataLoader(**arguments)


def _build_loaders(
    config: Gate2Config,
    runtime: RuntimeState,
    *,
    smoke: bool,
) -> tuple[
    ConfirmatoryStructuredSignal,
    dict[str, tuple[int, ...]],
    dict[str, DataLoader[StructuredBatch]],
    torch.Generator,
]:
    dataset = _build_dataset(config, smoke=smoke)
    splits = dataset.split_indices(
        train_fraction=config.data.train_fraction,
        validation_fraction=config.data.validation_fraction,
        seed=config.experiment.seed,
    )
    train_generator = torch.Generator().manual_seed(config.experiment.seed + 101)
    loaders = {
        "train": _loader(
            dataset,
            splits["train"],
            batch_size=config.training.batch_size,
            shuffle=True,
            runtime=runtime,
            pin_memory=config.training.pin_memory,
            generator=train_generator,
            smoke=smoke,
        ),
        "validation": _loader(
            dataset,
            splits["validation"],
            batch_size=config.training.batch_size,
            shuffle=False,
            runtime=runtime,
            pin_memory=config.training.pin_memory,
            generator=torch.Generator().manual_seed(config.experiment.seed + 102),
            smoke=smoke,
        ),
        "test": _loader(
            dataset,
            splits["test"],
            batch_size=config.training.batch_size,
            shuffle=False,
            runtime=runtime,
            pin_memory=config.training.pin_memory,
            generator=torch.Generator().manual_seed(config.experiment.seed + 103),
            smoke=smoke,
        ),
    }
    return dataset, splits, loaders, train_generator


def _build_model_config(config: Gate2Config) -> ModelConfig:
    return ModelConfig(
        expert=ExpertConfig(
            node_feature_dim=config.data.node_feature_dim,
            edge_feature_dim=config.data.edge_feature_dim,
            hidden_dim=config.model.hidden_dim,
            embedding_dim=config.model.embedding_dim,
            num_classes=2,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
        ),
        router=RouterConfig(
            hidden_dim=config.model.router_hidden_dim,
            diagnostic_dim=2,
            temperature=config.model.router_temperature,
            route_costs=config.loss.route_costs,
            cost_strength=config.loss.compute_cost_weight,
            straight_through=True,
        ),
        translator=TranslatorConfig(
            hidden_dim=config.model.hidden_dim,
            stalk_rank=2,
        ),
    )


def _regime_targets(batch: StructuredBatch) -> Tensor:
    lookup = {route: index for index, route in enumerate(ROUTES)}
    return torch.tensor(
        [lookup[regime] for regime in batch.regimes],
        dtype=torch.long,
        device=batch.labels.device,
    )


def _oracle_logits(output: ModelOutput, regime_targets: Tensor) -> Tensor:
    indices = torch.arange(regime_targets.shape[0], device=regime_targets.device)
    return output.expert_logits[indices, regime_targets]


def _oracle_route_targets(
    output: ModelOutput,
    batch: StructuredBatch,
    config: Gate2Config,
    conditional_accuracy: Tensor | None = None,
) -> Tensor:
    """Select the best per-example route utility with declared compute tie-breaking.

    The utility of a route is the probability that it predicts the true
    label for this example.  The fitted path uses the regime-conditional
    expert accuracies measured on the validation split (the low-variance
    estimate of that quantity for this benchmark); the unfitted fallback
    uses the per-example correct-label log-probability.  Three plug-in
    estimators were measured and rejected before this design: the raw
    log-probability utility (the accurate but underconfident graph expert
    lost its own regime 62% of the time, probe ceiling 0.536), per-route
    temperature scaling (miscalibration is regime-conditional, so a global
    temperature changes nothing: picks-own 42%, probe 0.529), and
    correctness-first utilities (lucky cross-regime guesses dominate:
    marginal 0.693 vs probe 0.712).  The regime labels are used only as a
    supervision target, never as a model input.
    """

    if conditional_accuracy is not None:
        regimes = _regime_targets(batch)
        utility = conditional_accuracy.to(batch.labels.device)[regimes]
    else:
        log_probabilities = output.expert_logits.float().log_softmax(dim=-1)
        label_index = batch.labels[:, None, None].expand(-1, len(ROUTES), 1)
        utility = log_probabilities.gather(-1, label_index).squeeze(-1)
    costs = torch.tensor(
        config.loss.route_costs,
        dtype=utility.dtype,
        device=utility.device,
    )
    normalized_costs = costs / costs.mean()
    return (utility - config.loss.oracle_cost_weight * normalized_costs).argmax(dim=-1)


def _sum_auxiliary(output: ModelOutput, names: Iterable[str]) -> Tensor:
    selected = [output.auxiliary_losses[name] for name in names]
    if not selected:
        return output.mixed_logits.sum() * 0.0
    return torch.stack([value.float() for value in selected]).sum()


def _topology_surrogate(output: ModelOutput, max_points: int) -> Tensor:
    count = min(int(output.embeddings.shape[0]), max_points)
    if count < 2:
        return output.embeddings.sum() * 0.0
    graph_embeddings = output.embeddings[:count, 0].float()
    translated_embeddings = output.translated_embeddings[:count].float()
    graph_distances = pairwise_euclidean_distances(graph_embeddings)
    cell_distances = pairwise_euclidean_distances(translated_embeddings[:, 0])
    sheaf_distances = pairwise_euclidean_distances(translated_embeddings[:, 1])
    return 0.5 * (
        symmetric_h0_srtd_surrogate(graph_distances, cell_distances)
        + symmetric_h0_srtd_surrogate(graph_distances, sheaf_distances)
    )


def _oracle_table(model: nn.Module) -> Tensor | None:
    """Return the fitted regime-conditional accuracy table, if available."""

    table = getattr(model, "oracle_conditional_accuracy", None)
    ready = getattr(model, "oracle_table_ready", None)
    if table is None or ready is None or not bool(ready):
        return None
    return table


def _loss_terms(
    output: ModelOutput,
    batch: StructuredBatch,
    config: Gate2Config,
    phase: str,
    conditional_accuracy: Tensor | None = None,
) -> dict[str, Tensor]:
    regimes = _regime_targets(batch)
    repeated_labels = batch.labels[:, None].expand(-1, len(ROUTES)).reshape(-1)
    dense_expert_ce = F.cross_entropy(
        output.expert_logits.float().reshape(-1, output.expert_logits.shape[-1]),
        repeated_labels,
        label_smoothing=config.training.label_smoothing,
    )
    oracle_ce = F.cross_entropy(
        _oracle_logits(output, regimes).float(),
        batch.labels,
        label_smoothing=config.training.label_smoothing,
    )
    mixed_ce = F.cross_entropy(
        output.mixed_logits.float(),
        batch.labels,
        label_smoothing=config.training.label_smoothing,
    )
    oracle_routes = _oracle_route_targets(
        output, batch, config, conditional_accuracy=conditional_accuracy
    )
    oracle_route_ce = F.cross_entropy(output.route_logits.float(), oracle_routes)
    repeated_translator_labels = batch.labels[:, None].expand(-1, 2).reshape(-1)
    translated_ce = F.cross_entropy(
        output.translated_logits.float().reshape(-1, output.translated_logits.shape[-1]),
        repeated_translator_labels,
        label_smoothing=config.training.label_smoothing,
    )
    reconstruction = _sum_auxiliary(
        output,
        ("cell_reconstruction", "sheaf_reconstruction"),
    )
    consistency = _sum_auxiliary(
        output,
        ("cell_chain_consistency_surrogate", "sheaf_cochain_consistency_surrogate"),
    )
    topology = _topology_surrogate(output, config.loss.rtd_max_points)
    expected_cost = output.auxiliary_losses["route_expected_cost"].float()
    entropy = output.auxiliary_losses["route_entropy"].float()
    load_balance = output.auxiliary_losses["route_load_balance"].float()

    if phase == "fixed_experts":
        total = config.loss.expert_weight * dense_expert_ce
    elif phase == "translators":
        total = (
            config.loss.translator_task_weight * translated_ce
            + config.loss.translator_weight * reconstruction
            + config.loss.chain_weight * consistency
            + config.loss.rtd_weight * topology
        )
    elif phase == "router_warmup":
        total = (
            config.loss.router_supervision_weight * oracle_route_ce
            + config.loss.compute_cost_weight * expected_cost
            - config.loss.entropy_weight * entropy
            + config.loss.entropy_weight * load_balance
        )
    elif phase == "joint_finetune":
        total = (
            config.loss.expert_weight * mixed_ce
            + 0.25 * config.loss.expert_weight * dense_expert_ce
            + config.loss.router_supervision_weight * oracle_route_ce
            + config.loss.translator_task_weight * translated_ce
            + config.loss.translator_weight * reconstruction
            + config.loss.chain_weight * consistency
            + config.loss.rtd_weight * topology
            + config.loss.compute_cost_weight * expected_cost
            - config.loss.entropy_weight * entropy
            + config.loss.entropy_weight * load_balance
        )
    else:
        raise ValueError(f"unknown training phase: {phase}")
    return {
        "loss": total,
        "dense_expert_ce": dense_expert_ce,
        "oracle_ce": oracle_ce,
        "mixed_ce": mixed_ce,
        "oracle_route_ce": oracle_route_ce,
        "translated_ce": translated_ce,
        "reconstruction": reconstruction,
        "consistency": consistency,
        "h0_rtd_style": topology,
        "expected_cost": expected_cost,
        "route_entropy": entropy,
        "load_balance": load_balance,
    }


def _autocast(runtime: RuntimeState):  # type: ignore[no-untyped-def]
    enabled = runtime.device.type == "cuda" and runtime.neural_dtype != torch.float32
    return torch.autocast(
        device_type=runtime.device.type,
        dtype=runtime.neural_dtype,
        enabled=enabled,
    )


def _set_phase_trainability(model: nn.Module, phase: str) -> None:
    """Freeze completed components and leave only the declared phase trainable."""

    root = getattr(model, "_orig_mod", model)
    if phase not in PHASES:
        raise ValueError(f"unknown training phase: {phase}")
    enabled_components = {
        "fixed_experts": {"fixed_experts"},
        "translators": {"graph_to_cell", "graph_to_sheaf"},
        "router_warmup": {"router_context", "router", "cheap_router"},
        "joint_finetune": {
            "fixed_experts",
            "graph_to_cell",
            "graph_to_sheaf",
            "router_context",
            "router",
            "cheap_router",
        },
    }[phase]
    for parameter in root.parameters():
        parameter.requires_grad_(False)
    matched = 0
    for component_name in enabled_components:
        component = getattr(root, component_name, None)
        if isinstance(component, nn.Module):
            component.train(True)
            for parameter in component.parameters():
                parameter.requires_grad_(True)
                matched += parameter.numel()
    for component_name in (
        "fixed_experts",
        "graph_to_cell",
        "graph_to_sheaf",
        "router_context",
        "router",
        "cheap_router",
    ):
        component = getattr(root, component_name, None)
        if isinstance(component, nn.Module) and component_name not in enabled_components:
            component.eval()
    if matched == 0:
        raise RuntimeError(f"phase {phase} did not select any trainable parameters")


def _train_epoch(
    model: nn.Module,
    loader: DataLoader[StructuredBatch],
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    runtime: RuntimeState,
    config: Gate2Config,
    phase: str,
    state: _RunState,
    *,
    max_steps: int,
) -> dict[str, float]:
    model.train()
    _set_phase_trainability(model, phase)
    totals: dict[str, float] = {}
    examples = 0
    for batch_index, batch in enumerate(loader):
        if max_steps and batch_index >= max_steps:
            break
        batch = batch.to(runtime.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(runtime):
            output = model(batch, hard=phase == "joint_finetune")
            terms = _loss_terms(
                output,
                batch,
                config,
                phase,
                conditional_accuracy=_oracle_table(model),
            )
        if not torch.isfinite(terms["loss"]):
            raise FloatingPointError(f"non-finite training loss in {phase}")
        scaler.scale(terms["loss"]).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.training.grad_clip_norm
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient norm in {phase}")
        scaler.step(optimizer)
        scaler.update()

        count = len(batch)
        examples += count
        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * count
        totals["gradient_norm"] = (
            totals.get("gradient_norm", 0.0) + float(gradient_norm.detach()) * count
        )
        state.global_step += 1
    if examples == 0:
        raise RuntimeError("training loader produced no examples")
    return {name: value / examples for name, value in totals.items()}


@torch.no_grad()
def _fit_oracle_conditional_accuracy(
    model: nn.Module,
    loader: DataLoader[StructuredBatch],
    runtime: RuntimeState,
    *,
    max_steps: int,
) -> Tensor:
    """Measure the regime-conditional expert accuracies on the validation split.

    The table ``C[regime, route]`` is the routing oracle's utility: the
    estimated probability that each route predicts the true label for
    examples of each regime.  Fitted once when the router-warmup phase
    begins, while the experts are frozen, so refitting on resume reproduces
    identical values; stored in model buffers so checkpoints persist it.
    """

    was_training = model.training
    model.eval()
    correct = torch.zeros((len(ROUTES), len(ROUTES)), dtype=torch.long)
    counts = torch.zeros(len(ROUTES), dtype=torch.long)
    for batch_index, batch in enumerate(loader):
        if max_steps and batch_index >= max_steps:
            break
        batch = batch.to(runtime.device, non_blocking=True)
        with _autocast(runtime):
            output = model(batch)
        predictions = output.expert_logits.float().argmax(dim=-1)
        regimes = _regime_targets(batch).cpu()
        hits = (predictions.cpu() == batch.labels.cpu().unsqueeze(1)).long()
        for regime_index in range(len(ROUTES)):
            selected = regimes == regime_index
            counts[regime_index] += int(selected.sum())
            correct[regime_index] += hits[selected].sum(dim=0)
    if int(counts.sum()) == 0:
        raise RuntimeError("oracle table fitting received no validation examples")
    table = (correct.double() / counts.double().clamp_min(1).unsqueeze(1)).float()
    model.oracle_conditional_accuracy.copy_(table.to(model.oracle_conditional_accuracy.device))
    model.oracle_table_ready.fill_(1)
    if was_training:
        model.train()
    return table


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader[StructuredBatch],
    runtime: RuntimeState,
    config: Gate2Config,
    phase: str,
    *,
    max_steps: int,
    prediction_path: Path | None = None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    examples = 0
    expert_by_regime_correct = torch.zeros((3, 3), dtype=torch.long)
    regime_total = torch.zeros(3, dtype=torch.long)
    expert_overall_correct = torch.zeros(3, dtype=torch.long)
    translated_overall_correct = torch.zeros(2, dtype=torch.long)
    translated_intended_correct = torch.zeros(2, dtype=torch.long)
    translated_intended_total = torch.zeros(2, dtype=torch.long)
    hard_elapsed_seconds = 0.0
    evaluated_expert_count = 0
    route_selection_count = torch.zeros(3, dtype=torch.long)
    route_regime_count = torch.zeros((3, 3), dtype=torch.long)
    records: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(loader):
        if max_steps and batch_index >= max_steps:
            break
        batch = batch.to(runtime.device, non_blocking=True)
        with _autocast(runtime):
            output = model(batch)
            terms = _loss_terms(
                output,
                batch,
                config,
                phase,
                conditional_accuracy=_oracle_table(model),
            )
        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)
        hard_started = time.perf_counter()
        with _autocast(runtime):
            hard_output = model(batch, hard=True)
        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)
        hard_elapsed_seconds += time.perf_counter() - hard_started

        regimes = _regime_targets(batch)
        oracle_routes = _oracle_route_targets(
            output,
            batch,
            config,
            conditional_accuracy=_oracle_table(model),
        )
        oracle_predictions = _oracle_logits(output, regimes).argmax(dim=-1)
        soft_predictions = output.mixed_logits.argmax(dim=-1)
        hard_predictions = hard_output.mixed_logits.argmax(dim=-1)
        route_predictions = output.route_logits.argmax(dim=-1)
        utility_oracle_predictions = _oracle_logits(
            output, oracle_routes
        ).argmax(dim=-1)
        dense_predictions = output.expert_logits.mean(dim=1).argmax(dim=-1)
        random_routes = torch.tensor(
            [
                int.from_bytes(
                    hashlib.sha256(sample_id.encode("utf-8")).digest()[:8], "big"
                )
                % len(ROUTES)
                for sample_id in batch.sample_ids
            ],
            dtype=torch.long,
            device=batch.labels.device,
        )
        random_predictions = _oracle_logits(output, random_routes).argmax(dim=-1)
        correct_log_probabilities = output.expert_logits.float().log_softmax(dim=-1)
        label_index = batch.labels[:, None, None].expand(-1, len(ROUTES), 1)
        correct_utilities = correct_log_probabilities.gather(
            -1, label_index
        ).squeeze(-1)
        route_costs = torch.tensor(
            config.loss.route_costs,
            device=batch.labels.device,
            dtype=correct_utilities.dtype,
        )
        correct_utilities = correct_utilities - config.loss.oracle_cost_weight * (
            route_costs / route_costs.mean()
        )
        selected_utilities = correct_utilities.gather(
            1, route_predictions.unsqueeze(1)
        ).squeeze(1)
        utility_regret = correct_utilities.max(dim=1).values - selected_utilities
        count = len(batch)
        examples += count
        evaluated_expert_count += int(hard_output.evaluated_routes.sum())
        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * count
        totals["oracle_correct"] = totals.get("oracle_correct", 0.0) + int(
            (oracle_predictions == batch.labels).sum()
        )
        totals["soft_correct"] = totals.get("soft_correct", 0.0) + int(
            (soft_predictions == batch.labels).sum()
        )
        totals["hard_correct"] = totals.get("hard_correct", 0.0) + int(
            (hard_predictions == batch.labels).sum()
        )
        totals["regime_route_correct"] = totals.get(
            "regime_route_correct", 0.0
        ) + int(
            (route_predictions == regimes).sum()
        )
        totals["oracle_route_correct"] = totals.get(
            "oracle_route_correct", 0.0
        ) + int((route_predictions == oracle_routes).sum())
        totals["utility_oracle_correct"] = totals.get(
            "utility_oracle_correct", 0.0
        ) + int((utility_oracle_predictions == batch.labels).sum())
        totals["dense_correct"] = totals.get("dense_correct", 0.0) + int(
            (dense_predictions == batch.labels).sum()
        )
        totals["random_correct"] = totals.get("random_correct", 0.0) + int(
            (random_predictions == batch.labels).sum()
        )
        totals["oracle_regret_sum"] = totals.get("oracle_regret_sum", 0.0) + float(
            utility_regret.sum()
        )
        for route_index in range(3):
            selected_mask = route_predictions == route_index
            route_selection_count[route_index] += int(selected_mask.sum())
            for regime_index in range(3):
                route_regime_count[regime_index, route_index] += int(
                    (selected_mask & (regimes == regime_index)).sum()
                )
        for expert_index in range(3):
            expert_predictions = output.expert_logits[:, expert_index].argmax(dim=-1)
            expert_overall_correct[expert_index] += int(
                (expert_predictions == batch.labels).sum()
            )
            for regime_index in range(3):
                mask = regimes == regime_index
                if torch.any(mask):
                    expert_by_regime_correct[regime_index, expert_index] += int(
                        (expert_predictions[mask] == batch.labels[mask]).sum()
                    )
        for regime_index in range(3):
            regime_total[regime_index] += int((regimes == regime_index).sum())
        for translator_index, intended_regime in enumerate((1, 2)):
            translated_predictions = output.translated_logits[:, translator_index].argmax(
                dim=-1
            )
            translated_overall_correct[translator_index] += int(
                (translated_predictions == batch.labels).sum()
            )
            intended_mask = regimes == intended_regime
            translated_intended_total[translator_index] += int(intended_mask.sum())
            if torch.any(intended_mask):
                translated_intended_correct[translator_index] += int(
                    (
                        translated_predictions[intended_mask]
                        == batch.labels[intended_mask]
                    ).sum()
                )
        if prediction_path is not None:
            for item_index, sample_id in enumerate(batch.sample_ids):
                records.append(
                    {
                        "sample_id": sample_id,
                        "label": int(batch.labels[item_index]),
                        "regime": batch.regimes[item_index].value,
                        "selected_route": ROUTES[int(route_predictions[item_index])].value,
                        "oracle_route": ROUTES[int(oracle_routes[item_index])].value,
                        "soft_prediction": int(soft_predictions[item_index]),
                        "hard_prediction": int(hard_predictions[item_index]),
                        "expert_logits": output.expert_logits[item_index]
                        .float()
                        .cpu()
                        .tolist(),
                        "translated_logits": output.translated_logits[item_index]
                        .float()
                        .cpu()
                        .tolist(),
                        "route_logits": output.route_logits[item_index]
                        .float()
                        .cpu()
                        .tolist(),
                        "route_weights": output.route_weights[item_index]
                        .float()
                        .cpu()
                        .tolist(),
                        "diagnostics": output.diagnostics[item_index]
                        .float()
                        .cpu()
                        .tolist(),
                        "translation_diagnostics": output.translation_diagnostics[
                            item_index
                        ]
                        .float()
                        .cpu()
                        .tolist(),
                        "evaluated_routes": hard_output.evaluated_routes[item_index]
                        .cpu()
                        .tolist(),
                        "oracle_regret": float(utility_regret[item_index]),
                    }
                )

    if examples == 0:
        raise RuntimeError("evaluation loader produced no examples")
    metrics = {
        name: value / examples
        for name, value in totals.items()
        if not name.endswith("_correct") and name != "oracle_regret_sum"
    }
    metrics.update(
        {
            "oracle_accuracy": totals["oracle_correct"] / examples,
            "soft_accuracy": totals["soft_correct"] / examples,
            "hard_accuracy": totals["hard_correct"] / examples,
            "route_accuracy": totals["oracle_route_correct"] / examples,
            "oracle_route_accuracy": totals["oracle_route_correct"] / examples,
            "regime_route_accuracy": totals["regime_route_correct"] / examples,
            "utility_oracle_accuracy": totals["utility_oracle_correct"] / examples,
            "dense_accuracy": totals["dense_correct"] / examples,
            "random_accuracy": totals["random_correct"] / examples,
            "oracle_regret": totals["oracle_regret_sum"] / examples,
            "hard_evaluated_experts_per_example": evaluated_expert_count / examples,
            "hard_milliseconds_per_example": 1000.0
            * hard_elapsed_seconds
            / examples,
        }
    )
    for expert_index, expert_route in enumerate(ROUTES):
        metrics[f"{expert_route.value}_expert_accuracy"] = (
            int(expert_overall_correct[expert_index]) / examples
        )
        for regime_index, regime in enumerate(ROUTES):
            denominator = max(1, int(regime_total[regime_index]))
            metrics[
                f"expert_{expert_route.value}_on_{regime.value}_accuracy"
            ] = int(expert_by_regime_correct[regime_index, expert_index]) / denominator
    for translator_index, name in enumerate(("graph_to_cell", "graph_to_sheaf")):
        metrics[f"{name}_accuracy"] = (
            int(translated_overall_correct[translator_index]) / examples
        )
        denominator = max(1, int(translated_intended_total[translator_index]))
        metrics[f"{name}_intended_accuracy"] = (
            int(translated_intended_correct[translator_index]) / denominator
        )
    probabilities = route_regime_count.to(torch.float64) / examples
    regime_probabilities = probabilities.sum(dim=1, keepdim=True)
    route_probabilities = probabilities.sum(dim=0, keepdim=True)
    independent = regime_probabilities @ route_probabilities
    positive = probabilities > 0
    metrics["regime_route_mutual_information"] = float(
        (
            probabilities[positive]
            * (probabilities[positive] / independent[positive]).log()
        ).sum()
    )
    for route_index, route in enumerate(ROUTES):
        metrics[f"route_utilization_{route.value}"] = (
            int(route_selection_count[route_index]) / examples
        )
    if prediction_path is not None:
        atomic_json(
            prediction_path,
            {
                "schema_version": 1,
                "num_examples": examples,
                "records": records,
            },
        )
    return metrics


def _phase_score(phase: str, metrics: Mapping[str, float]) -> float:
    if phase == "fixed_experts":
        return sum(metrics[f"{route.value}_expert_accuracy"] for route in ROUTES) / 3
    if phase == "translators":
        return -(
            metrics["translated_ce"]
            + metrics["reconstruction"]
            + metrics["consistency"]
            + metrics["h0_rtd_style"]
        )
    if phase == "router_warmup":
        return metrics["oracle_route_accuracy"]
    return metrics["hard_accuracy"] + 0.25 * metrics["oracle_route_accuracy"]


def _specialization_gate(
    metrics: Mapping[str, float], config: Gate2Config
) -> dict[str, Any]:
    improvements = {
        route.value: metrics[f"expert_{route.value}_on_{route.value}_accuracy"]
        - metrics[f"expert_graph_on_{route.value}_accuracy"]
        for route in (SignalRegime.CELL, SignalRegime.SHEAF)
    }
    passing = sorted(
        route
        for route, improvement in improvements.items()
        if improvement > config.gates.specialization_margin
    )
    return {
        "name": "fixed_expert_specialization",
        "passed": len(passing) >= config.gates.minimum_specialized_routes,
        "minimum_routes": config.gates.minimum_specialized_routes,
        "minimum_margin": config.gates.specialization_margin,
        "passing_routes": passing,
        "improvements_over_graph": improvements,
    }


def _translator_gate(
    metrics: Mapping[str, float], baseline: float, config: Gate2Config
) -> dict[str, Any]:
    final_value = metrics["reconstruction"] + metrics["consistency"]
    relative_improvement = (baseline - final_value) / max(abs(baseline), 1e-12)
    return {
        "name": "translator_engineering",
        "passed": relative_improvement
        >= config.gates.translator_relative_improvement,
        "baseline_reconstruction_plus_consistency": baseline,
        "final_reconstruction_plus_consistency": final_value,
        "relative_improvement": relative_improvement,
        "minimum_relative_improvement": config.gates.translator_relative_improvement,
        "claim_boundary": (
            "This engineering gate does not replace the preregistered structural-loss "
            "predictive-value and ablation gate."
        ),
    }


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _checkpoint_payload(
    *,
    config: Gate2Config,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    state: _RunState,
    train_generator: torch.Generator,
    smoke: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config_fingerprint": _config_fingerprint(config),
        "code_fingerprint": _code_fingerprint(config.project_root),
        "mode": "smoke" if smoke else "full",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "run_state": {
            "phase_index": state.phase_index,
            "epoch_in_phase": state.epoch_in_phase,
            "global_epoch": state.global_epoch,
            "global_step": state.global_step,
            "best_score": state.best_score,
            "bad_epochs": state.bad_epochs,
            "translator_baseline": state.translator_baseline,
            "gate_reports": state.gate_reports,
        },
        "rng_state": _rng_state(),
        "train_generator_state": train_generator.get_state(),
    }


def _restore_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    config: Gate2Config,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    train_generator: torch.Generator,
    smoke: bool,
) -> _RunState:
    if checkpoint.get("config_fingerprint") != _config_fingerprint(config):
        raise RuntimeError("checkpoint configuration does not match the requested run")
    if checkpoint.get("code_fingerprint") != _code_fingerprint(config.project_root):
        raise RuntimeError("checkpoint executable-code fingerprint does not match")
    if checkpoint.get("mode") != ("smoke" if smoke else "full"):
        raise RuntimeError(
            "checkpoint smoke/full mode does not match the requested run"
        )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    _restore_rng_state(checkpoint["rng_state"])
    train_generator.set_state(checkpoint["train_generator_state"])
    return _RunState(**checkpoint["run_state"])


def _epoch_counts(config: Gate2Config, *, smoke: bool) -> tuple[int, int, int, int]:
    if smoke:
        return (1, 1, 1, 1)
    return (
        config.training.fixed_expert_epochs,
        config.training.translator_epochs,
        config.training.router_warmup_epochs,
        config.training.joint_finetune_epochs,
    )


def _dry_run_report(
    model: nn.Module,
    loader: DataLoader[StructuredBatch],
    runtime: RuntimeState,
    splits: Mapping[str, tuple[int, ...]],
) -> dict[str, Any]:
    batch = next(iter(loader)).to(runtime.device)
    model.eval()
    with torch.no_grad(), _autocast(runtime):
        output = model(batch)
    return {
        "status": "dry-run-passed",
        "device": str(runtime.device),
        "dtype": str(runtime.neural_dtype),
        "batch_size": len(batch),
        "split_sizes": {name: len(indices) for name, indices in splits.items()},
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "route_logits_shape": list(output.route_logits.shape),
        "expert_logits_shape": list(output.expert_logits.shape),
        "embeddings_shape": list(output.embeddings.shape),
        "translated_embeddings_shape": list(output.translated_embeddings.shape),
        "translated_logits_shape": list(output.translated_logits.shape),
        "diagnostics_shape": list(output.diagnostics.shape),
        "auxiliary_losses": sorted(output.auxiliary_losses),
    }


def _run_training_unlocked(
    config: Gate2Config,
    *,
    resume: bool = False,
    smoke: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute or resume the canonical gated four-phase Gate-2 experiment."""

    if not isinstance(config, Gate2Config):
        raise TypeError("config must be Gate2Config")
    runtime = initialize_runtime(config.runtime, seed=config.experiment.seed)
    dataset, splits, loaders, train_generator = _build_loaders(
        config, runtime, smoke=smoke
    )
    model: nn.Module = build_model(_build_model_config(config)).to(runtime.device)
    model = maybe_compile(model, runtime)
    if dry_run:
        return _dry_run_report(model, loaders["validation"], runtime, splits)

    run_dir = (
        config.run_dir.with_name(
            config.run_dir.name + f"-smoke-{_config_fingerprint(config)[:10]}"
        )
        if smoke
        else config.run_dir
    )
    checkpoint_dir = run_dir / "checkpoints"
    last_checkpoint = checkpoint_dir / "last.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        run_dir / "config.json",
        {
            **config.as_dict(),
            "mode": "smoke" if smoke else "full",
            "config_fingerprint": _config_fingerprint(config),
        },
    )
    atomic_json(
        run_dir / "environment.json",
        {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(runtime.device),
            "device_name": (
                torch.cuda.get_device_name(runtime.device)
                if runtime.device.type == "cuda"
                else None
            ),
            "dataset_size": len(dataset),
            "split_sizes": {name: len(indices) for name, indices in splits.items()},
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "code_fingerprint": _code_fingerprint(config.project_root),
            "git": _git_revision(config.project_root),
        },
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    epoch_counts = _epoch_counts(config, smoke=smoke)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, epoch_counts[0]),
        eta_min=config.training.min_learning_rate,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=runtime.device.type == "cuda" and runtime.neural_dtype == torch.float16,
    )
    state = _RunState()
    if last_checkpoint.exists():
        if not resume:
            raise RuntimeError(
                f"checkpoint already exists at {last_checkpoint}; pass --resume"
            )
        state = _restore_checkpoint(
            load_checkpoint(last_checkpoint),
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_generator=train_generator,
            smoke=smoke,
        )

    max_steps = 2 if smoke else config.training.max_steps_per_epoch
    final_validation: dict[str, float] = {}
    failed_gate = next(
        (
            report
            for report in state.gate_reports.values()
            if isinstance(report, Mapping) and not report.get("passed", True)
        ),
        None,
    )
    atomic_json(
        run_dir / "status.json",
        {
            "status": "running",
            "phase_index": state.phase_index,
            "global_epoch": state.global_epoch,
        },
    )
    try:
        with MetricLogger(run_dir) as logger:
            for phase_index in range(state.phase_index, len(PHASES)):
                phase = PHASES[phase_index]
                phase_epochs = epoch_counts[phase_index]
                first_epoch = (
                    state.epoch_in_phase if phase_index == state.phase_index else 0
                )
                if phase_index != state.phase_index:
                    state.best_score = float("-inf")
                    state.bad_epochs = 0
                if first_epoch == 0 and phase_index > 0:
                    # Restart the LR schedule per phase: a single cosine over
                    # all phases starves late phases (the router trained at
                    # ~1e-4 falling to 1e-6 and never learned; measured route
                    # accuracy 0.32 in-engine vs 0.54 with a per-phase
                    # restart at the same configured LR).  This must key on
                    # first_epoch, not phase_index: state.phase_index is
                    # advanced at each phase end, so a phase-index check
                    # never fires.  A mid-phase resume (first_epoch > 0)
                    # keeps the restored phase scheduler.
                    phase_lr = config.training.learning_rate
                    if (
                        phase in ("router_warmup", "joint_finetune")
                        and config.training.router_learning_rate > 0
                    ):
                        # The router trains briefly at the end of a long run
                        # and can stall at uniform output on bad draws
                        # (measured: warmup stuck at route accuracy 1/3 on
                        # seed s2); it gets its own rate when configured.
                        phase_lr = config.training.router_learning_rate
                    for group in optimizer.param_groups:
                        group["lr"] = phase_lr
                    scheduler = CosineAnnealingLR(
                        optimizer,
                        T_max=max(1, phase_epochs),
                        eta_min=config.training.min_learning_rate,
                    )
                if phase == "translators" and state.translator_baseline is None:
                    baseline_metrics = _evaluate(
                        model,
                        loaders["validation"],
                        runtime,
                        config,
                        phase,
                        max_steps=max_steps,
                    )
                    state.translator_baseline = (
                        baseline_metrics["reconstruction"]
                        + baseline_metrics["consistency"]
                    )
                if phase == "router_warmup":
                    _fit_oracle_conditional_accuracy(
                        model,
                        loaders["validation"],
                        runtime,
                        max_steps=max_steps,
                    )
                for epoch in range(first_epoch, phase_epochs):
                    train_metrics = _train_epoch(
                        model,
                        loaders["train"],
                        optimizer,
                        scaler,
                        runtime,
                        config,
                        phase,
                        state,
                        max_steps=max_steps,
                    )
                    validation_metrics = _evaluate(
                        model,
                        loaders["validation"],
                        runtime,
                        config,
                        phase,
                        max_steps=max_steps,
                    )
                    final_validation = validation_metrics
                    score = _phase_score(phase, validation_metrics)
                    improved = score > state.best_score + 1e-8
                    state.bad_epochs = 0 if improved else state.bad_epochs + 1
                    state.best_score = max(state.best_score, score)
                    state.global_epoch += 1
                    state.phase_index = phase_index
                    state.epoch_in_phase = epoch + 1
                    scheduler.step()
                    metrics: dict[str, Any] = {
                        "phase": phase,
                        "epoch": epoch,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                    metrics.update(
                        {
                            f"train/{name}": value
                            for name, value in train_metrics.items()
                        }
                    )
                    metrics.update(
                        {
                            f"validation/{name}": value
                            for name, value in validation_metrics.items()
                        }
                    )
                    logger.log(metrics, step=state.global_epoch)
                    payload = _checkpoint_payload(
                        config=config,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        state=state,
                        train_generator=train_generator,
                        smoke=smoke,
                    )
                    if state.global_epoch % config.training.checkpoint_every == 0:
                        atomic_torch_save(last_checkpoint, payload)
                    if improved:
                        atomic_torch_save(checkpoint_dir / f"best-{phase}.pt", payload)
                    atomic_json(
                        run_dir / "status.json",
                        {
                            "status": "running",
                            "phase": phase,
                            "epoch": epoch,
                            "global_epoch": state.global_epoch,
                            "validation": validation_metrics,
                        },
                    )
                    if (
                        not smoke
                        and state.bad_epochs >= config.training.early_stopping_patience
                    ):
                        break
                best_path = checkpoint_dir / f"best-{phase}.pt"
                if best_path.is_file():
                    best_checkpoint = load_checkpoint(best_path)
                    model.load_state_dict(best_checkpoint["model"])
                    optimizer.load_state_dict(best_checkpoint["optimizer"])
                    scaler.load_state_dict(best_checkpoint["scaler"])
                final_validation = _evaluate(
                    model,
                    loaders["validation"],
                    runtime,
                    config,
                    phase,
                    max_steps=max_steps,
                )
                gate_report: dict[str, Any] | None = None
                if not smoke and config.gates.enforce:
                    if phase == "fixed_experts":
                        gate_report = _specialization_gate(final_validation, config)
                    elif phase == "translators":
                        if state.translator_baseline is None:
                            raise RuntimeError("translator baseline was not recorded")
                        gate_report = _translator_gate(
                            final_validation, state.translator_baseline, config
                        )
                if gate_report is not None:
                    state.gate_reports[phase] = gate_report
                    logger.log(
                        {
                            "phase": f"{phase}_gate",
                            "gate/passed": bool(gate_report["passed"]),
                        },
                        step=state.global_epoch,
                    )
                    if not gate_report["passed"]:
                        failed_gate = gate_report
                state.phase_index = (
                    len(PHASES) if failed_gate is not None else phase_index + 1
                )
                state.epoch_in_phase = 0
                state.best_score = float("-inf")
                state.bad_epochs = 0
                atomic_torch_save(
                    last_checkpoint,
                    _checkpoint_payload(
                        config=config,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        state=state,
                        train_generator=train_generator,
                        smoke=smoke,
                    ),
                )
                if failed_gate is not None:
                    break

            if not final_validation:
                final_validation = _evaluate(
                    model,
                    loaders["validation"],
                    runtime,
                    config,
                    "joint_finetune",
                    max_steps=max_steps,
                )
            final_test = _evaluate(
                model,
                loaders["test"],
                runtime,
                config,
                "joint_finetune",
                max_steps=max_steps,
                prediction_path=run_dir / "metrics" / "test_predictions.json",
            )
            logger.log(
                {
                    "phase": "final_test",
                    **{f"test/{name}": value for name, value in final_test.items()},
                },
                step=state.global_epoch + 1,
            )
    except Exception as exc:
        atomic_json(
            run_dir / "status.json",
            {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise

    summary = {
        "status": "gate-failed" if failed_gate is not None else "completed",
        "mode": "smoke" if smoke else "full",
        "device": str(runtime.device),
        "epochs_completed": state.global_epoch,
        "steps_completed": state.global_step,
        "validation": final_validation,
        "test": final_test,
        "gates": state.gate_reports,
        "failed_gate": failed_gate,
        "run_dir": str(run_dir),
    }
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(run_dir / "status.json", summary)
    return summary


def run_training(
    config: Gate2Config,
    *,
    resume: bool = False,
    smoke: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute one run while holding an exclusive run-directory lock."""

    if dry_run:
        return _run_training_unlocked(
            config, resume=resume, smoke=smoke, dry_run=True
        )
    run_dir = (
        config.run_dir.with_name(
            config.run_dir.name + f"-smoke-{_config_fingerprint(config)[:10]}"
        )
        if smoke
        else config.run_dir
    )
    with _exclusive_run_lock(run_dir):
        return _run_training_unlocked(
            config, resume=resume, smoke=smoke, dry_run=False
        )


__all__ = ["run_training"]
