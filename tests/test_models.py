from __future__ import annotations

import copy

import pytest
import torch
from torch.nn import functional as F

from homymoly.data import MixedStructuredSignal, SignalRegime, collate_structured
from homymoly.models import (
    DIAGNOSTIC_NAMES,
    DiagnosticCostRouter,
    ExpertConfig,
    ModelConfig,
    RouterConfig,
    TranslatorConfig,
    build_model,
)


def _model_config() -> ModelConfig:
    return ModelConfig(
        expert=ExpertConfig(
            hidden_dim=16,
            embedding_dim=12,
            num_layers=1,
            dropout=0.0,
        ),
        router=RouterConfig(hidden_dim=12),
        translator=TranslatorConfig(hidden_dim=12),
    )


@pytest.fixture(scope="module")
def padded_batch():  # type: ignore[no-untyped-def]
    small = MixedStructuredSignal(6, seed=101, num_vertices=24)[0]
    large = MixedStructuredSignal(6, seed=102, num_vertices=31)[1]
    return collate_structured((small, large))


def _assert_finite(values: torch.Tensor) -> None:
    assert torch.isfinite(values).all()


def _assert_expert_equal(left, right) -> None:  # type: ignore[no-untyped-def]
    torch.testing.assert_close(left.embedding, right.embedding, rtol=0, atol=1e-6)
    torch.testing.assert_close(left.logits, right.logits, rtol=0, atol=1e-6)
    torch.testing.assert_close(left.diagnostics, right.diagnostics, rtol=0, atol=1e-6)


def test_system_public_contract_and_hard_soft_routing(padded_batch) -> None:  # type: ignore[no-untyped-def]
    torch.manual_seed(3)
    config = _model_config()
    model = build_model(config).eval()

    soft = model(padded_batch)
    batch_size = len(padded_batch)
    classes = config.expert.num_classes
    embedding_dim = config.expert.embedding_dim
    assert soft.route_logits.shape == (batch_size, 3)
    assert soft.route_weights.shape == (batch_size, 3)
    assert soft.expert_logits.shape == (batch_size, 3, classes)
    assert soft.mixed_logits.shape == (batch_size, classes)
    assert soft.embeddings.shape == (batch_size, 3, embedding_dim)
    assert soft.diagnostics.shape == (batch_size, len(DIAGNOSTIC_NAMES))
    assert soft.selected_routes.shape == (batch_size,)
    assert soft.evaluated_routes.shape == (batch_size, 3)
    assert soft.translated_embeddings.shape == (batch_size, 2, embedding_dim)
    assert soft.translated_logits.shape == (batch_size, 2, classes)
    assert soft.translation_diagnostics.shape == (batch_size, 2, 3)
    assert soft.evaluated_translators.shape == (batch_size, 2)
    assert torch.all(soft.evaluated_routes)
    assert torch.all(soft.evaluated_translators)
    assert len(DIAGNOSTIC_NAMES) == 8
    torch.testing.assert_close(
        soft.route_weights.sum(dim=-1),
        torch.ones(batch_size),
    )
    for tensor in (
        soft.route_logits,
        soft.route_weights,
        soft.expert_logits,
        soft.mixed_logits,
        soft.embeddings,
        soft.diagnostics,
        soft.translated_embeddings,
        soft.translated_logits,
        soft.translation_diagnostics,
    ):
        _assert_finite(tensor)

    expected_auxiliary = {
        "cell_reconstruction",
        "cell_chain_consistency_surrogate",
        "cell_boundary_map_reconstruction",
        "cell_face_gate_supervision",
        "sheaf_reconstruction",
        "sheaf_cochain_consistency_surrogate",
        "sheaf_transport_map_reconstruction",
        "route_expected_cost",
        "route_load_balance",
        "route_entropy",
    }
    assert set(soft.auxiliary_losses) == expected_auxiliary
    for value in soft.auxiliary_losses.values():
        assert value.ndim == 0
        _assert_finite(value)

    hard = model(padded_batch, hard=True)
    assert torch.all((hard.route_weights == 0) | (hard.route_weights == 1))
    torch.testing.assert_close(
        hard.route_weights.sum(dim=-1),
        torch.ones(batch_size),
    )
    assert torch.equal(hard.route_weights.argmax(dim=-1), hard.selected_routes)
    assert torch.equal(
        hard.evaluated_routes,
        F.one_hot(hard.selected_routes, num_classes=3).bool(),
    )
    assert not torch.any(hard.evaluated_translators)
    assert torch.count_nonzero(hard.translated_logits) == 0

    ensemble = model.fixed_experts(padded_batch, route=SignalRegime.CELL)
    torch.testing.assert_close(ensemble.embedding, ensemble.embeddings[:, 1])
    torch.testing.assert_close(ensemble.logits, ensemble.expert_logits[:, 1])


def test_padding_content_is_inert_for_experts_and_system(padded_batch) -> None:  # type: ignore[no-untyped-def]
    torch.manual_seed(4)
    model = build_model(_model_config()).eval()
    baseline_experts = model.fixed_experts.forward_all(padded_batch)
    baseline = model(padded_batch)
    corrupted = copy.deepcopy(padded_batch)

    nodes = int(corrupted.num_vertices[0])
    edges = int(corrupted.num_edges[0])
    faces = int(corrupted.num_faces[0])
    with torch.no_grad():
        corrupted.node_features[0, nodes:] = 10_000.0
        corrupted.edge_features[0, edges:] = -10_000.0
        corrupted.edge_index[0, :, edges:] = 1_000_000
        corrupted.transport[0, edges:] = 10_000.0
        corrupted.face_index[0, :, faces:] = -1_000_000
        corrupted.face_active[0, faces:] = True

    changed_experts = model.fixed_experts.forward_all(corrupted)
    for original, changed in zip(baseline_experts, changed_experts, strict=True):
        _assert_expert_equal(original, changed)

    changed = model(corrupted)
    for name in (
        "route_logits",
        "route_weights",
        "expert_logits",
        "mixed_logits",
        "embeddings",
        "diagnostics",
    ):
        torch.testing.assert_close(
            getattr(baseline, name), getattr(changed, name), rtol=0, atol=1e-6
        )
    for name, original in baseline.auxiliary_losses.items():
        torch.testing.assert_close(
            original, changed.auxiliary_losses[name], rtol=0, atol=1e-6
        )


def test_route_scoped_experts_and_translators_ignore_privileged_structure(
    padded_batch,
) -> None:  # type: ignore[no-untyped-def]
    torch.manual_seed(5)
    model = build_model(_model_config()).eval()
    graph, cell, sheaf = model.fixed_experts.forward_all(padded_batch)
    cell_translation = model.graph_to_cell(padded_batch)
    sheaf_translation = model.graph_to_sheaf(padded_batch)

    changed_transport = copy.deepcopy(padded_batch)
    with torch.no_grad():
        changed_transport.transport[changed_transport.edge_mask] *= -3.0
    graph_after_transport, cell_after_transport, _ = model.fixed_experts.forward_all(
        changed_transport
    )
    _assert_expert_equal(graph, graph_after_transport)
    _assert_expert_equal(cell, cell_after_transport)
    translated_cell_after_transport = model.graph_to_cell(changed_transport)
    torch.testing.assert_close(
        cell_translation.higher_latent,
        translated_cell_after_transport.higher_latent,
        rtol=0,
        atol=1e-6,
    )

    changed_faces = copy.deepcopy(padded_batch)
    with torch.no_grad():
        changed_faces.face_active[changed_faces.face_mask] = ~changed_faces.face_active[
            changed_faces.face_mask
        ]
        changed_faces.face_index[:] = 0
    graph_after_faces, _, _ = model.fixed_experts.forward_all(changed_faces)
    _assert_expert_equal(graph, graph_after_faces)
    translated_sheaf_after_faces = model.graph_to_sheaf(changed_faces)
    torch.testing.assert_close(
        sheaf_translation.higher_latent,
        translated_sheaf_after_faces.higher_latent,
        rtol=0,
        atol=1e-6,
    )
    # The sheaf expert consumes face_index (its holonomy pathway is defined on
    # faces), so only face_active flips — which it never reads — must leave it
    # unchanged.
    changed_active_only = copy.deepcopy(padded_batch)
    with torch.no_grad():
        changed_active_only.face_active[
            changed_active_only.face_mask
        ] = ~changed_active_only.face_active[changed_active_only.face_mask]
    _, _, sheaf_after_active = model.fixed_experts.forward_all(changed_active_only)
    _assert_expert_equal(sheaf, sheaf_after_active)


def test_translator_shapes_and_surrogate_losses_are_finite(padded_batch) -> None:  # type: ignore[no-untyped-def]
    config = _model_config()
    model = build_model(config).eval()
    cell = model.graph_to_cell(padded_batch)
    sheaf = model.graph_to_sheaf(padded_batch)
    batch_size = len(padded_batch)

    assert cell.node_latent.shape == (
        batch_size,
        padded_batch.node_mask.shape[1],
        config.translator.hidden_dim,
    )
    assert cell.edge_latent.shape == (
        batch_size,
        padded_batch.edge_mask.shape[1],
        config.translator.hidden_dim,
    )
    assert cell.higher_latent.shape == (
        batch_size,
        padded_batch.face_mask.shape[1],
        config.translator.hidden_dim,
    )
    assert sheaf.node_latent.shape[-1] == config.translator.stalk_rank
    assert sheaf.edge_latent.shape[-1] == config.translator.hidden_dim
    assert sheaf.higher_latent.shape == (
        batch_size,
        padded_batch.edge_mask.shape[1],
        config.translator.stalk_rank,
    )
    assert cell.task_embedding.shape == (
        batch_size,
        config.expert.embedding_dim,
    )
    assert sheaf.task_embedding.shape == (
        batch_size,
        config.expert.embedding_dim,
    )
    assert cell.task_logits.shape == (batch_size, config.expert.num_classes)
    assert sheaf.task_logits.shape == (batch_size, config.expert.num_classes)
    assert cell.structure_logits.shape == padded_batch.face_mask.shape
    assert sheaf.structure_logits.shape == padded_batch.edge_mask.shape
    for translation in (cell, sheaf):
        assert translation.per_sample_diagnostics.shape == (batch_size, 3)
        for value in (
            translation.reconstruction_loss,
            translation.consistency_surrogate,
            translation.map_reconstruction_loss,
            translation.supervision_loss,
            translation.per_sample_diagnostics,
        ):
            _assert_finite(value)
            assert torch.all(value >= 0)


def test_sheaf_translator_backward_is_finite_at_zero_residual() -> None:
    """Regression: sqrt of a exactly-zero residual must not yield NaN grads.

    The translators phase minimizes the consistency surrogate, which drives
    stalk residuals toward zero; an unclamped (or post-clamped) sqrt has an
    infinite derivative there and killed the first full runs.
    """

    torch.manual_seed(3)
    config = _model_config()
    model = build_model(config)
    batch = collate_structured(
        tuple(MixedStructuredSignal(6, seed=105, num_vertices=24))
    )
    with torch.no_grad():
        for parameter in model.graph_to_sheaf.node_lift.parameters():
            parameter.zero_()
    output = model.graph_to_sheaf(batch)
    assert torch.all(output.higher_latent == 0)
    loss = (
        output.task_logits.float().square().mean()
        + output.reconstruction_loss
        + output.consistency_surrogate
        + output.structure_logits.float().mean()
    )
    loss.backward()
    for name, parameter in model.graph_to_sheaf.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


def test_router_uses_diagnostics_and_declared_costs() -> None:
    router = DiagnosticCostRouter(
        embedding_dim=4,
        config=RouterConfig(
            hidden_dim=8,
            route_costs=(1.0, 2.0, 4.0),
            cost_strength=2.0,
        ),
    ).eval()
    with torch.no_grad():
        for parameter in router.parameters():
            parameter.zero_()

    context = torch.zeros(1, 4)
    diagnostics = torch.zeros(1, 3, 2)
    cost_only = router(context, diagnostics)
    assert cost_only.weights[0, 0] > cost_only.weights[0, 1]
    assert cost_only.weights[0, 1] > cost_only.weights[0, 2]

    with torch.no_grad():
        router.diagnostic_weight[2, 0] = 1.0
        diagnostics[0, 2, 0] = 10.0
    diagnostic_driven = router(context, diagnostics, hard=True)
    assert diagnostic_driven.selected_routes.item() == 2
    assert diagnostic_driven.weights.tolist() == [[0.0, 0.0, 1.0]]


def test_batch_size_one_autocast_and_backward_gradients(padded_batch) -> None:  # type: ignore[no-untyped-def]
    single = collate_structured(
        (MixedStructuredSignal(6, seed=103, num_vertices=24)[2],)
    )
    model = build_model(_model_config()).eval()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast_output = model(single, hard=True)
    assert autocast_output.mixed_logits.shape == (1, 2)
    assert autocast_output.route_weights.shape == (1, 3)
    assert autocast_output.evaluated_routes.sum().item() == 1
    assert not torch.any(autocast_output.evaluated_translators)
    _assert_finite(autocast_output.mixed_logits)
    for value in autocast_output.auxiliary_losses.values():
        _assert_finite(value)

    model.train()
    model.zero_grad(set_to_none=True)
    output = model(padded_batch)
    loss = F.cross_entropy(output.mixed_logits, padded_batch.labels)
    loss = loss + 0.05 * torch.stack(tuple(output.auxiliary_losses.values())).sum()
    _assert_finite(loss)
    loss.backward()

    modules = {
        "graph expert": model.fixed_experts.experts[SignalRegime.GRAPH.value],
        "cell expert": model.fixed_experts.experts[SignalRegime.CELL.value],
        "sheaf expert": model.fixed_experts.experts[SignalRegime.SHEAF.value],
        "router": model.router,
        "graph-to-cell translator": model.graph_to_cell,
        "graph-to-sheaf translator": model.graph_to_sheaf,
    }
    for name, module in modules.items():
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        assert gradients, f"{name} received no gradients"
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_eval_hard_executes_only_selected_expert_and_skips_translators(
    padded_batch,
) -> None:  # type: ignore[no-untyped-def]
    model = build_model(_model_config())
    with torch.no_grad():
        for parameter in model.router.parameters():
            parameter.zero_()
        model.router.diagnostic_bias[0] = 20.0

    model.train()
    training_output = model(padded_batch, hard=True)
    expected_training_weights = F.one_hot(
        training_output.selected_routes,
        num_classes=3,
    ).to(training_output.route_weights.dtype)
    assert torch.equal(training_output.route_weights, expected_training_weights)
    assert torch.all(training_output.evaluated_routes)
    assert torch.all(training_output.evaluated_translators)
    F.cross_entropy(training_output.mixed_logits, padded_batch.labels).backward()
    router_gradients = [
        parameter.grad
        for parameter in model.router.parameters()
        if parameter.grad is not None
    ]
    assert router_gradients
    assert sum(float(gradient.abs().sum()) for gradient in router_gradients) > 0.0

    model.eval()

    expert_counts = {route.value: 0 for route in SignalRegime}
    translator_counts = {"cell": 0, "sheaf": 0}

    def count_expert(route):  # type: ignore[no-untyped-def]
        def hook(_module, arguments) -> None:  # type: ignore[no-untyped-def]
            expert_counts[route.value] += len(arguments[0])

        return hook

    def count_translator(name):  # type: ignore[no-untyped-def]
        def hook(_module, _arguments) -> None:  # type: ignore[no-untyped-def]
            translator_counts[name] += 1

        return hook

    handles = [
        model.fixed_experts.experts[route.value].register_forward_pre_hook(
            count_expert(route)
        )
        for route in SignalRegime
    ]
    handles.extend(
        (
            model.graph_to_cell.register_forward_pre_hook(count_translator("cell")),
            model.graph_to_sheaf.register_forward_pre_hook(count_translator("sheaf")),
        )
    )
    try:
        with torch.no_grad():
            output = model(padded_batch, hard=True)
    finally:
        for handle in handles:
            handle.remove()

    assert output.selected_routes.tolist() == [0] * len(padded_batch)
    assert expert_counts == {
        SignalRegime.GRAPH.value: len(padded_batch),
        SignalRegime.CELL.value: 0,
        SignalRegime.SHEAF.value: 0,
    }
    assert translator_counts == {"cell": 0, "sheaf": 0}
    assert output.evaluated_routes.tolist() == [[True, False, False]] * len(
        padded_batch
    )
    assert not torch.any(output.evaluated_translators)
    assert torch.count_nonzero(output.expert_logits[:, 1:]) == 0
    torch.testing.assert_close(output.mixed_logits, output.expert_logits[:, 0])


def test_router_precedes_and_is_independent_of_expert_parameters(padded_batch) -> None:  # type: ignore[no-untyped-def]
    torch.manual_seed(17)
    model = build_model(_model_config()).eval()
    events: list[str] = []
    handles = [
        model.router.register_forward_pre_hook(
            lambda _module, _arguments: events.append("router")
        )
    ]
    handles.extend(
        model.fixed_experts.experts[route.value].register_forward_pre_hook(
            lambda _module, _arguments, route=route: events.append(route.value)
        )
        for route in SignalRegime
    )
    try:
        with torch.no_grad():
            baseline = model(padded_batch)
            for parameter in model.fixed_experts.parameters():
                parameter.add_(3.0 * torch.randn_like(parameter))
            changed = model(padded_batch)
    finally:
        for handle in handles:
            handle.remove()

    assert events[0] == "router"
    assert events[4] == "router"
    torch.testing.assert_close(
        baseline.route_logits,
        changed.route_logits,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        baseline.route_weights,
        changed.route_weights,
        rtol=0,
        atol=0,
    )


def test_graph_only_translators_hold_out_target_structure(
    padded_batch,
) -> None:  # type: ignore[no-untyped-def]
    """Targets supervise typed maps but never enter graph-only forward paths."""

    torch.manual_seed(19)
    model = build_model(_model_config()).eval()
    changed_cell_target = copy.deepcopy(padded_batch)
    changed_sheaf_target = copy.deepcopy(padded_batch)
    with torch.no_grad():
        changed_cell_target.face_active[
            changed_cell_target.face_mask
        ] = ~changed_cell_target.face_active[changed_cell_target.face_mask]
        changed_sheaf_target.transport[changed_sheaf_target.edge_mask] *= -1.0
        baseline = model.graph_to_cell(padded_batch)
        changed_cell = model.graph_to_cell(changed_cell_target)
        baseline_sheaf = model.graph_to_sheaf(padded_batch)
        changed_sheaf = model.graph_to_sheaf(changed_sheaf_target)

    # Node/edge latents and reconstructions do not touch the face pathway.
    for name in (
        "node_latent",
        "edge_latent",
        "node_reconstruction",
        "edge_reconstruction",
    ):
        torch.testing.assert_close(
            getattr(baseline, name),
            getattr(changed_cell, name),
            rtol=0,
            atol=0,
        )
    for name in ("task_logits", "structure_logits", "higher_latent"):
        torch.testing.assert_close(
            getattr(baseline, name),
            getattr(changed_cell, name),
            rtol=0,
            atol=0,
        )
    assert not torch.equal(
        baseline.map_reconstruction_loss,
        changed_cell.map_reconstruction_loss,
    )
    for name in ("task_logits", "node_latent", "higher_latent", "structure_logits"):
        torch.testing.assert_close(
            getattr(baseline_sheaf, name),
            getattr(changed_sheaf, name),
            rtol=0,
            atol=0,
        )
    assert not torch.equal(
        baseline_sheaf.map_reconstruction_loss,
        changed_sheaf.map_reconstruction_loss,
    )


def test_target_view_compatibility_mode_is_explicit(padded_batch) -> None:  # type: ignore[no-untyped-def]
    base = _model_config()
    config = ModelConfig(
        expert=base.expert,
        router=base.router,
        translator=TranslatorConfig(
            hidden_dim=base.translator.hidden_dim,
            stalk_rank=base.translator.stalk_rank,
            target_structure_access=True,
        ),
    )
    torch.manual_seed(21)
    model = build_model(config).eval()
    changed = copy.deepcopy(padded_batch)
    with torch.no_grad():
        changed.face_active[changed.face_mask] = ~changed.face_active[changed.face_mask]
        changed.transport[changed.edge_mask] *= -1.0
        cell_before = model.graph_to_cell(padded_batch)
        cell_after = model.graph_to_cell(changed)
        sheaf_before = model.graph_to_sheaf(padded_batch)
        sheaf_after = model.graph_to_sheaf(changed)
    assert not torch.equal(cell_before.task_logits, cell_after.task_logits)
    assert not torch.equal(sheaf_before.task_logits, sheaf_after.task_logits)


def test_translated_task_logits_backpropagate_into_both_translators(
    padded_batch,
) -> None:  # type: ignore[no-untyped-def]
    torch.manual_seed(23)
    model = build_model(_model_config()).train()
    output = model(padded_batch)
    output.translated_logits.square().mean().backward()

    for name, translator in (
        ("graph-to-cell", model.graph_to_cell),
        ("graph-to-sheaf", model.graph_to_sheaf),
    ):
        gradients = [
            parameter.grad
            for parameter in translator.parameters()
            if parameter.grad is not None
        ]
        assert gradients, f"{name} task path received no gradients"
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_model_configuration_rejects_incompatible_dimensions() -> None:
    with pytest.raises(ValueError, match="even"):
        ExpertConfig(hidden_dim=15)
    with pytest.raises(ValueError, match="rank-2"):
        TranslatorConfig(stalk_rank=3)
    with pytest.raises(TypeError, match="ModelConfig"):
        build_model({})  # type: ignore[arg-type]
