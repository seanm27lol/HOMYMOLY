from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import re
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROJECT_ROOT = Path(__file__).parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_lifting_replication_v2.py"
SPEC = importlib.util.spec_from_file_location("run_lifting_replication_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

OLD_TEST_SEED = 20261001


def _hand_sample() -> SimpleNamespace:
    """A connected four-vertex, five-edge complex with cycle rank two."""

    boundary_1 = torch.tensor(
        [
            [-1.0, -1.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    boundary_2 = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 1.0],
            [0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    assert torch.count_nonzero(boundary_1 @ boundary_2) == 0
    return SimpleNamespace(
        sample_id="hand-connected-cycle-rank-two",
        boundary_1=boundary_1,
        boundary_2=boundary_2,
        num_vertices=4,
        num_edges=5,
        num_faces=2,
    )


class _FakeDataset:
    """Serves the hand fixture for any seed; never touches the sealed block."""

    def __init__(self, count: int, *, seed: int, dtype: object) -> None:
        assert count == 1
        assert dtype is torch.float64
        self.seed = seed

    def __getitem__(self, index: int) -> SimpleNamespace:
        assert index == 0
        return _hand_sample()


def _data(sample: SimpleNamespace) -> MODULE.RegressionData:
    train_x = MODULE._normal((MODULE.N_TRAIN, sample.num_edges), 101)
    test_x = MODULE._normal((64, sample.num_edges), 102)
    truth = sample.boundary_2.mT
    return MODULE.RegressionData(
        train_x=train_x,
        train_y=train_x @ truth.mT
        + 0.02 * MODULE._normal((MODULE.N_TRAIN, sample.num_faces), 103),
        test_x=test_x,
        test_y=test_x @ truth.mT,
    )


def _basis_pair(
    sample: SimpleNamespace,
) -> tuple[torch.Tensor, dict, torch.Tensor, dict]:
    cycle, cycle_certificate = MODULE._cycle_nullspace_basis(
        sample.boundary_1, sample.num_faces
    )
    random, random_certificate = MODULE._matched_random_basis(
        sample.num_edges, sample.num_faces, 1234
    )
    return cycle, cycle_certificate, random, random_certificate


def _fake_primary_rows(n: int = 30) -> list[dict[str, object]]:
    rows = []
    for index in range(n):
        scale = 1.0 + index / 1000.0
        gradient = {"final_full_batch_gradient_norm": 1e-8}
        rows.append(
            {
                "seed": 1000 + index,
                "arms": {
                    "ambient_adam": {
                        "held_out_mse": 1.0 * scale,
                        "metadata": dict(gradient),
                    },
                    "ambient_min_norm_ls": {"held_out_mse": 1.0 * scale},
                    "soft_boundary_lambda3": {
                        "held_out_mse": 0.52 * scale,
                        "metadata": dict(gradient),
                    },
                    "soft_boundary_closed_form_lambda3": {"held_out_mse": 0.50 * scale},
                    "hard_cycle_ls": {"held_out_mse": 0.25 * scale},
                    "hard_random_subspace_ls": {"held_out_mse": 0.90 * scale},
                    "inner_cv_ridge": {"held_out_mse": 0.70 * scale},
                    "singular_value_surrogate": {
                        "held_out_mse": 1.40 * scale,
                        "metadata": dict(gradient),
                    },
                    "rtd_inspired_distance_surrogate": {
                        "held_out_mse": 0.98 * scale,
                        "metadata": dict(gradient),
                    },
                    "generator_cycle_basis_oracle": {
                        "held_out_mse": 1e-30,
                        "mean_squared_test_target": 1.0 * scale,
                        "relative_error_to_mean_squared_test_target": 1e-30,
                    },
                },
                "optimizer_descriptive": {
                    "soft_adam_vs_closed_form_solution_gap_frobenius": 1e-6,
                },
            }
        )
    return rows


def _seal_payload(output_path: str = "result.json", **overrides: object) -> dict:
    payload: dict[str, object] = {
        "schema": MODULE.SEAL_SCHEMA,
        "design_commit": "0" * 40,
        "protocol_sha256": "1" * 64,
        "runner_sha256": "2" * 64,
        "generator_sha256": "3" * 64,
        "lock_sha256": "4" * 64,
        "seed_interval": {"first": 20270101, "last": 20270136},
        "no_preview_declaration": "no sealed seed was previewed before the seal",
        "primary_family": [{"id": claim_id} for claim_id in MODULE.PRIMARY_CLAIM_IDS],
        "stop_rules": ["any design violation stops the campaign"],
        "output_path": output_path,
    }
    payload.update(overrides)
    return payload


def _write_seal(root: Path, payload: dict, seal: str = "seal.json") -> str:
    path = root / seal
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return seal


def test_sealed_seed_block_is_declared_but_never_opened_at_import() -> None:
    assert MODULE.SEALED_SEEDS == tuple(range(20270101, 20270137))
    assert all(seed != OLD_TEST_SEED for seed in MODULE.SEALED_SEEDS)


def test_sha256_subseeds_are_stable_component_separated_and_63_bit() -> None:
    seed = MODULE._subseed(OLD_TEST_SEED, "primary-train-inputs")

    assert seed == 5392934588254313459
    assert 0 <= seed < 2**63
    assert seed == MODULE._subseed(OLD_TEST_SEED, "primary-train-inputs", 0)
    assert seed != MODULE._subseed(OLD_TEST_SEED, "primary-training-noise", 0)
    assert seed != MODULE._subseed(OLD_TEST_SEED, "primary-train-inputs", 1)


def test_cycle_basis_has_expected_shape_orthonormality_and_exact_membership() -> None:
    sample = _hand_sample()
    basis, _certificate = MODULE._cycle_nullspace_basis(
        sample.boundary_1, sample.num_faces
    )
    assert basis.shape == (sample.num_edges, sample.num_faces)
    assert sample.boundary_1 @ basis == pytest.approx(
        torch.zeros((sample.num_vertices, sample.num_faces)), abs=1e-14
    )
    assert basis.mT @ basis == pytest.approx(torch.eye(sample.num_faces), abs=1e-14)
    pivots = basis.abs().argmax(dim=0)
    assert torch.all(basis[pivots, torch.arange(sample.num_faces)] > 0)


def test_cycle_basis_certificate_is_complete_and_consistent() -> None:
    sample = _hand_sample()
    _, certificate = MODULE._cycle_nullspace_basis(sample.boundary_1, sample.num_faces)

    assert certificate["observed_rank"] == sample.num_vertices - 1
    assert certificate["expected_rank"] == sample.num_vertices - 1
    assert certificate["rank_tolerance"] > 0
    assert certificate["asserted_tolerance"] == MODULE.BASIS_TOLERANCE == 1e-10
    assert 0 <= certificate["boundary_defect_frobenius"] <= 1e-10
    assert 0 <= certificate["orthonormality_defect_frobenius"] <= 1e-10


def test_cycle_basis_rejects_disconnected_or_wrong_cycle_rank() -> None:
    sample = _hand_sample()
    # Two connected components: E=5, V=4 keeps E-V+1=2 equal to the declared
    # faces, but rank(B1)=2 differs from V-1=3, so the rank check must fire.
    disconnected = torch.tensor(
        [
            [-1.0, -1.0, -1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, -1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )

    with pytest.raises(RuntimeError, match="rank mismatch"):
        MODULE._cycle_nullspace_basis(disconnected, sample.num_faces)
    with pytest.raises(RuntimeError, match="cycle-rank mismatch"):
        MODULE._cycle_nullspace_basis(sample.boundary_1, sample.num_faces + 1)


def test_matched_random_qr_is_orthonormal_reproducible_and_sign_fixed() -> None:
    first, _ = MODULE._matched_random_basis(8, 3, 4242)
    second, _ = MODULE._matched_random_basis(8, 3, 4242)
    gaussian = MODULE._normal((8, 3), 4242)

    assert first == pytest.approx(second, abs=0.0)
    assert first.mT @ first == pytest.approx(torch.eye(3), abs=1e-14)
    # Q'G is the sign-canonicalized R, whose diagonal is nonnegative.
    assert torch.all(torch.diagonal(first.mT @ gaussian) >= 0)


def test_matched_random_certificate_records_diagonal_and_tolerance() -> None:
    _, certificate = MODULE._matched_random_basis(8, 3, 4242)

    assert math.isfinite(certificate["min_abs_diagonal_r"])
    assert certificate["min_abs_diagonal_r"] > 0
    assert certificate["asserted_tolerance"] == MODULE.BASIS_TOLERANCE == 1e-10
    assert 0 <= certificate["orthonormality_defect_frobenius"] <= 1e-10


def test_ambient_least_squares_uses_frozen_gelsd_formula() -> None:
    sample = _hand_sample()
    data = _data(sample)

    ambient = MODULE._ambient_min_norm_ls(data.train_x, data.train_y)
    reference = torch.linalg.lstsq(
        data.train_x,
        data.train_y,
        driver="gelsd",
        rcond=MODULE.LSTSQ_RCOND,
    ).solution.mT

    assert ambient.matrix == pytest.approx(reference, abs=1e-13)
    assert ambient.metadata["driver"] == "gelsd"
    assert ambient.metadata["rcond"] == 1e-12
    assert ambient.metadata["gelsd_returned_rank"] >= 1
    minimum = ambient.metadata["gelsd_min_singular_value"]
    assert math.isfinite(minimum) and minimum > 0


def test_hard_subspace_fits_record_gelsd_driver_rcond_and_rank() -> None:
    sample = _hand_sample()
    data = _data(sample)
    cycle, _, random, _ = _basis_pair(sample)

    for basis, name in ((cycle, "ker(B1)"), (random, "matched random subspace")):
        fitted = MODULE._subspace_ls(data.train_x, data.train_y, basis, name=name)
        assert fitted.metadata["driver"] == "gelsd"
        assert fitted.metadata["rcond"] == 1e-12
        assert fitted.metadata["gelsd_returned_rank"] >= 1
        minimum = fitted.metadata["gelsd_min_singular_value"]
        assert math.isfinite(minimum) and minimum > 0
        assert fitted.metadata["subspace_dimension"] == sample.num_faces

    evaluated = MODULE._evaluate_fit(
        MODULE._subspace_ls(data.train_x, data.train_y, cycle, name="ker(B1)").matrix,
        data.test_x,
        data.test_y,
        sample.boundary_1,
        random,
        {},
    )
    assert evaluated.boundary_defect < 1e-12
    assert evaluated.matrix.mT == pytest.approx(
        cycle @ (cycle.mT @ evaluated.matrix.mT), abs=1e-12
    )


def test_soft_closed_form_satisfies_the_frozen_mean_normalised_normal_equation() -> (
    None
):
    sample = _hand_sample()
    data = _data(sample)
    fit = MODULE._soft_boundary_closed_form_lambda3(
        data.train_x, data.train_y, sample.boundary_1
    )
    coefficients = fit.matrix.mT
    scale = MODULE.SOFT_BOUNDARY_WEIGHT * MODULE.N_TRAIN / sample.num_vertices
    residual = (
        (data.train_x.mT @ data.train_x) @ coefficients
        + scale * (sample.boundary_1.mT @ sample.boundary_1) @ coefficients
        - data.train_x.mT @ data.train_y
    )

    assert torch.linalg.matrix_norm(residual) < 1e-10
    assert fit.metadata["normal_equation_boundary_scale"] == scale
    assert fit.metadata["rcond"] == MODULE.PINV_RCOND
    assert fit.metadata["pinv_effective_rank"] >= 1
    assert fit.metadata["pinv_rank_cutoff"] > 0
    minimum = fit.metadata["pinv_min_singular_value"]
    assert math.isfinite(minimum) and minimum >= 0


def test_inner_cv_ridge_uses_four_training_only_folds_and_records_all_losses() -> None:
    sample = _hand_sample()
    data = _data(sample)
    fit = MODULE._fit_inner_cv_ridge(data.train_x, data.train_y)

    assert fit.metadata["folds"] == 4
    assert fit.metadata["fold_assignment"] == "row index modulo 4"
    candidates = fit.metadata["cross_validation_by_alpha"]
    assert [row["alpha"] for row in candidates] == list(MODULE.RIDGE_GRID)
    assert all(len(row["fold_validation_mse"]) == 4 for row in candidates)
    expected = min(
        candidates, key=lambda row: (row["mean_validation_mse"], row["alpha"])
    )
    assert fit.metadata["selected_alpha"] == expected["alpha"]


def test_ridge_exact_tie_selects_the_smaller_alpha(monkeypatch) -> None:
    sample = _hand_sample()
    data = _data(sample)
    tied = (MODULE.RIDGE_GRID[0], MODULE.RIDGE_GRID[1])

    def fake_ridge(train_x: torch.Tensor, train_y: torch.Tensor, alpha: float):
        del train_y
        shape = (sample.num_faces, int(train_x.shape[1]))
        if alpha in tied:
            # Identical fits produce bit-identical fold losses: an exact tie.
            return torch.zeros(shape, dtype=torch.float64)
        return torch.full(shape, 1e3, dtype=torch.float64)

    monkeypatch.setattr(MODULE, "_ridge_matrix", fake_ridge)
    fit = MODULE._fit_inner_cv_ridge(data.train_x, data.train_y)

    by_alpha = {
        row["alpha"]: row["mean_validation_mse"]
        for row in fit.metadata["cross_validation_by_alpha"]
    }
    assert by_alpha[tied[0]] == by_alpha[tied[1]]
    assert by_alpha[tied[0]] < min(
        value for alpha, value in by_alpha.items() if alpha not in tied
    )
    assert fit.metadata["selected_alpha"] == tied[0]


def test_graph_blind_fit_apis_reject_boundary_1_and_never_take_held_out_tensors() -> (
    None
):
    sample = _hand_sample()
    data = _data(sample)

    with pytest.raises(ValueError, match="must not receive boundary_1"):
        MODULE._fit_adam(
            data.train_x,
            data.train_y,
            term=None,
            weight=0.0,
            boundary_1=sample.boundary_1,
        )
    with pytest.raises(ValueError, match="requires boundary_1"):
        MODULE._fit_adam(data.train_x, data.train_y, term="boundary", weight=1.0)

    # The structural firewall: no fitting API can even name a held-out tensor.
    expected_signatures = {
        MODULE._fit_adam: ["train_x", "train_y", "term", "weight", "boundary_1"],
        MODULE._ambient_min_norm_ls: ["train_x", "train_y"],
        MODULE._subspace_ls: ["train_x", "train_y", "fit_basis", "name"],
        MODULE._fit_inner_cv_ridge: ["train_x", "train_y"],
        MODULE._soft_boundary_closed_form_lambda3: [
            "train_x",
            "train_y",
            "boundary_1",
        ],
    }
    for function, parameters in expected_signatures.items():
        signature = inspect.signature(function)
        assert list(signature.parameters) == parameters
        assert not any(
            "test" in name or "held" in name for name in signature.parameters
        )
    adam_parameters = inspect.signature(MODULE._fit_adam).parameters
    for keyword_only in ("term", "weight", "boundary_1"):
        assert adam_parameters[keyword_only].kind is inspect.Parameter.KEYWORD_ONLY

    # Held-out evaluation lives in a separate API that takes no training data.
    evaluation = inspect.signature(MODULE._evaluate_fit)
    assert list(evaluation.parameters) == [
        "matrix",
        "test_x",
        "test_y",
        "boundary_1",
        "random_basis",
        "metadata",
    ]


def test_primary_evaluator_has_every_frozen_arm_and_isolates_oracle(
    monkeypatch,
) -> None:
    sample = _hand_sample()
    monkeypatch.setattr(MODULE, "STEPS", 2)
    monkeypatch.setattr(MODULE, "N_TEST", 32)

    row = MODULE._evaluate_primary(sample, OLD_TEST_SEED)

    assert tuple(row["arms"]) == MODULE.ARM_NAMES
    oracle = row["arms"]["generator_cycle_basis_oracle"]
    assert oracle["held_out_mse"] < 1e-28
    assert oracle["metadata"] == {
        "estimator": "analytic generator cycle-basis oracle",
        "uses_withheld_generator_basis": True,
        "inference_role": "descriptive attainability ceiling only",
    }
    assert oracle["relative_error_to_mean_squared_test_target"] < 1e-28
    assert oracle["mean_squared_test_target"] > 0
    assert row["cycle_nullspace_certificate"]["boundary_defect_frobenius"] < 1e-12
    assert row["cycle_nullspace_certificate"]["basis_shape"] == [
        sample.num_edges,
        sample.num_faces,
    ]
    assert row["random_subspace_certificate"]["basis_shape"] == [
        sample.num_edges,
        sample.num_faces,
    ]
    gap = row["optimizer_descriptive"][
        "soft_adam_vs_closed_form_solution_gap_frobenius"
    ]
    assert math.isfinite(gap) and gap >= 0
    for arm in (
        "ambient_adam",
        "soft_boundary_lambda3",
        "singular_value_surrogate",
        "rtd_inspired_distance_surrogate",
    ):
        norm = row["arms"][arm]["metadata"]["final_full_batch_gradient_norm"]
        assert math.isfinite(norm) and norm >= 0


def test_off_path_c1_uses_min_norm_ls_reuses_primary_rep_zero_and_common_test(
    monkeypatch,
) -> None:
    sample = _hand_sample()
    monkeypatch.setattr(MODULE, "N_TEST", 64)

    row = MODULE._evaluate_c1(sample, OLD_TEST_SEED)

    assert row["estimator"] == "ambient minimum-norm torch.linalg.lstsq"
    assert row["replicate_zero_reuses_primary_train_inputs_and_noise"] is True
    assert len(row["replicates"]) == 12
    first = row["replicates"][0]
    second = row["replicates"][1]
    assert first["reuses_primary_training_realisation"] is True
    assert first["train_inputs_subseed"] == MODULE._subseed(
        OLD_TEST_SEED, "primary-train-inputs"
    )
    assert second["reuses_primary_training_realisation"] is False
    assert second["train_inputs_subseed"] == MODULE._subseed(
        OLD_TEST_SEED, "c1-train-inputs", 1
    )
    assert "cycle_projector_defect_frobenius" in first
    assert "boundary_compatibility_defect_frobenius" in first
    assert first["metadata"]["driver"] == "gelsd"
    assert row["shared_test_subseed"] == MODULE._subseed(
        OLD_TEST_SEED, "c1-test-inputs"
    )


def test_c1_rejects_zeroed_or_nonfinite_defects_without_epsilon_floor(
    monkeypatch,
) -> None:
    sample = _hand_sample()
    monkeypatch.setattr(MODULE, "N_TEST", 32)
    truth = sample.boundary_2.mT

    # The exact truth has zero MSE and zero cycle defect: C1 must fail closed.
    monkeypatch.setattr(
        MODULE,
        "_ambient_min_norm_ls",
        lambda train_x, train_y: MODULE.FittedMatrix(matrix=truth.clone(), metadata={}),
    )
    with pytest.raises(MODULE.DesignFailureError, match="strictly positive"):
        MODULE._evaluate_c1(sample, OLD_TEST_SEED)

    nonfinite = truth.clone()
    nonfinite[0, 0] = float("nan")
    monkeypatch.setattr(
        MODULE,
        "_ambient_min_norm_ls",
        lambda train_x, train_y: MODULE.FittedMatrix(
            matrix=nonfinite.clone(), metadata={}
        ),
    )
    with pytest.raises(MODULE.DesignFailureError, match="non-finite structural defect"):
        MODULE._evaluate_c1(sample, OLD_TEST_SEED)


def test_primary_family_has_exactly_seven_one_sided_bonferroni_claims() -> None:
    inference = MODULE._primary_inference(_fake_primary_rows())

    assert inference["family_size"] == 7
    assert inference["per_claim_alpha"] == pytest.approx(0.05 / 7)
    assert len(inference["claims"]) == 7
    assert all(claim["supported"] for claim in inference["claims"])
    assert inference["claims"][2]["numerator_arm"] == "hard_cycle_ls"
    assert (
        inference["claims"][2]["reference_arm"] == "soft_boundary_closed_form_lambda3"
    )
    assert inference["claims"][4]["numerator_arm"] == "inner_cv_ridge"
    assert inference["claims"][4]["reference_arm"] == "ambient_min_norm_ls"
    for claim in inference["claims"]:
        for key in (
            "estimate",
            "standard_error",
            "geometric_mean_ratio",
            "two_sided_interval_95_descriptive",
            "direction",
            "critical_value",
            "threshold",
            "supported",
        ):
            assert key in claim
        bound_keys = {"one_sided_upper_bound", "one_sided_lower_bound"}
        assert len(bound_keys & claim.keys()) == 1
        assert claim["geometric_mean_ratio"] == pytest.approx(10.0 ** claim["estimate"])
    h7 = inference["claims"][-1]
    assert h7["id"] == "h7-rtd-bounded-benefit-futility"
    assert h7["threshold"] == -0.045757490560675115
    assert h7["direction"] == "greater"
    assert "one_sided_lower_bound" in h7
    assert "equivalence" not in json.dumps(h7).lower()
    assert "noninferiority" not in json.dumps(h7).lower()
    assert "noninferiority" not in json.dumps(inference).lower()


def test_multiplicity_decision_uses_adjusted_one_sided_bound_not_interval() -> None:
    rows = _fake_primary_rows()
    # Adverse fixture for h4: paired log ratios alternate around a slightly
    # negative mean with enough spread that the Bonferroni-adjusted one-sided
    # upper bound crosses zero even though the descriptive two-sided 95%
    # interval still lies entirely below zero.
    low = 10.0**-0.342
    high = 10.0**0.142
    for index, row in enumerate(rows):
        ratio = low if index % 2 == 0 else high
        row["arms"]["hard_cycle_ls"]["held_out_mse"] = ratio
        row["arms"]["hard_random_subspace_ls"]["held_out_mse"] = 1.0
        # Pin the other hard_cycle contrasts so only h4 flips.
        row["arms"]["ambient_min_norm_ls"]["held_out_mse"] = ratio / 0.25
        row["arms"]["soft_boundary_closed_form_lambda3"]["held_out_mse"] = ratio / 0.25

    inference = MODULE._primary_inference(rows)
    claims = {claim["id"]: claim for claim in inference["claims"]}
    h4 = claims["h4-hard-cycle-vs-hard-random"]

    assert h4["supported"] is False
    assert h4["direction"] == "less"
    assert h4["one_sided_upper_bound"] > h4["threshold"]
    assert h4["two_sided_interval_95_descriptive"][1] < h4["threshold"]
    # The decision follows the adjusted one-sided bound, not the interval.
    assert h4["supported"] == (h4["one_sided_upper_bound"] < h4["threshold"])
    for claim_id, claim in claims.items():
        if claim_id != "h4-hard-cycle-vs-hard-random":
            assert claim["supported"] is True


def test_all_seven_bonferroni_critical_values_are_pinned() -> None:
    assert MODULE._T_ONE_SIDED_BONFERRONI == {
        29: 2.606750672048818,
        30: 2.601227904110613,
        31: 2.5960807947257787,
        32: 2.5912722991315227,
        33: 2.586770085672467,
        34: 2.5825458097369376,
        35: 2.5785745178415116,
    }


def test_direction_neutral_exact_sign_sensitivity() -> None:
    sensitivity = MODULE._sign_test([-2.0, -1.0, 0.0, 4.0])

    assert sensitivity == {
        "pvalue_two_sided": 1.0,
        "negative": 2,
        "positive": 1,
        "ties_discarded": 1,
        "role": "direction-neutral sensitivity analysis",
    }


def test_oracle_enters_no_claim_and_no_log_ratio_anywhere() -> None:
    rows = _fake_primary_rows()
    inference = MODULE._primary_inference(rows)
    descriptive = MODULE._descriptive_diagnostics(rows)

    for definition in MODULE._PRIMARY_CLAIM_DEFINITIONS:
        assert definition["numerator"] != "generator_cycle_basis_oracle"
        assert definition["denominator"] != "generator_cycle_basis_oracle"
    for claim in inference["claims"]:
        assert claim["numerator_arm"] != "generator_cycle_basis_oracle"
        assert claim["reference_arm"] != "generator_cycle_basis_oracle"
    # No log ratio anywhere is formed from the oracle: it never appears in the
    # confirmatory family, and its descriptive block carries raw errors only.
    assert "generator_cycle_basis_oracle" not in json.dumps(inference)
    oracle_block = descriptive["generator_cycle_basis_oracle"]
    assert not any("log" in key or "ratio" in key for key in oracle_block)
    assert oracle_block["role"].endswith("outside efficacy inference")


def test_c1_aggregation_is_descriptive_with_paired_delta_and_no_decision() -> None:
    rows = [
        {
            "cycle_projector_fisher_z": 0.40 + index / 1000.0,
            "matched_random_fisher_z": 0.10 + index / 2000.0,
        }
        for index in range(30)
    ]
    result = MODULE._c1_inference(rows)

    assert "supported" not in json.dumps(result)
    assert result["cycle_projector_defect"]["role"].endswith("no decision")
    assert result["paired_specificity_delta_fisher_z"]["mean"] > 0
    assert len(result["paired_specificity_delta_fisher_z"]["interval_95"]) == 2


def test_fisher_transform_clips_roundoff_only_inside_open_interval() -> None:
    assert math.isfinite(MODULE._fisher_z(1.0))
    assert math.isfinite(MODULE._fisher_z(-1.0))
    assert MODULE._fisher_z(0.0) == 0.0


def test_load_seal_accepts_a_valid_frozen_seal(tmp_path: Path) -> None:
    _write_seal(tmp_path, _seal_payload())

    seal = MODULE._load_seal(tmp_path, "seal.json")

    assert seal["schema"] == MODULE.SEAL_SCHEMA
    assert seal["output_path"] == "result.json"
    assert [entry["id"] for entry in seal["primary_family"]] == list(
        MODULE.PRIMARY_CLAIM_IDS
    )


def test_load_seal_rejects_wrong_schema_tag(tmp_path: Path) -> None:
    _write_seal(tmp_path, _seal_payload(schema="homymoly-lifting-replication-seal/0"))

    with pytest.raises(RuntimeError, match="design seal schema must be"):
        MODULE._load_seal(tmp_path, "seal.json")


def test_load_seal_rejects_wrong_seed_interval(tmp_path: Path) -> None:
    _write_seal(
        tmp_path,
        _seal_payload(seed_interval={"first": 20270101, "last": 20270135}),
    )

    with pytest.raises(RuntimeError, match="seed_interval must be exactly"):
        MODULE._load_seal(tmp_path, "seal.json")


def test_load_seal_rejects_missing_keys_and_bad_commit(tmp_path: Path) -> None:
    payload = _seal_payload()
    del payload["stop_rules"]
    _write_seal(tmp_path, payload)

    with pytest.raises(RuntimeError, match="missing keys"):
        MODULE._load_seal(tmp_path, "seal.json")

    _write_seal(tmp_path, _seal_payload(design_commit="0" * 39))
    with pytest.raises(RuntimeError, match="design_commit must be a full"):
        MODULE._load_seal(tmp_path, "seal.json")


def test_load_seal_rejects_altered_primary_family(tmp_path: Path) -> None:
    family = [{"id": claim_id} for claim_id in MODULE.PRIMARY_CLAIM_IDS]
    family[0] = {"id": "h1-renamed-after-sealing"}
    _write_seal(tmp_path, _seal_payload(primary_family=family))

    with pytest.raises(RuntimeError, match="primary_family ids differ"):
        MODULE._load_seal(tmp_path, "seal.json")


def _stage_sealed_tree(tmp_path: Path) -> dict[str, str]:
    """Create a tmp project root with sealed files and a matching seal record."""

    contents = {
        MODULE.PROTOCOL: b"sealed protocol\n",
        MODULE.RUNNER_SOURCE: b"sealed runner\n",
        MODULE.GENERATOR_SOURCE: b"sealed generator\n",
        MODULE.LOCKFILE: b"sealed lock\n",
    }
    hashes = {}
    for relative, content in contents.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        hashes[relative] = hashlib.sha256(content).hexdigest()
    return hashes


def _patch_frozen_constants(monkeypatch, hashes: dict[str, str]) -> None:
    monkeypatch.setattr(MODULE, "FROZEN_PROTOCOL_SHA256", hashes[MODULE.PROTOCOL])
    monkeypatch.setattr(
        MODULE, "FROZEN_GENERATOR_SHA256", hashes[MODULE.GENERATOR_SOURCE]
    )
    monkeypatch.setattr(MODULE, "FROZEN_LOCKFILE_SHA256", hashes[MODULE.LOCKFILE])


def test_preflight_records_seal_fingerprints_environment_and_git(
    monkeypatch, tmp_path: Path
) -> None:
    hashes = _stage_sealed_tree(tmp_path)
    _patch_frozen_constants(monkeypatch, hashes)
    seal_payload = _seal_payload(
        protocol_sha256=hashes[MODULE.PROTOCOL],
        runner_sha256=hashes[MODULE.RUNNER_SOURCE],
        generator_sha256=hashes[MODULE.GENERATOR_SOURCE],
        lock_sha256=hashes[MODULE.LOCKFILE],
    )
    _write_seal(tmp_path, seal_payload)
    # The running-file check hashes MODULE.__file__; point it at the staged copy.
    monkeypatch.setattr(MODULE, "__file__", str(tmp_path / MODULE.RUNNER_SOURCE))
    monkeypatch.setattr(
        MODULE,
        "_environment_provenance",
        lambda root: {
            "matches_expected": root == tmp_path,
            "lockfile": {
                "path": MODULE.LOCKFILE,
                "sha256": hashes[MODULE.LOCKFILE],
            },
        },
    )
    monkeypatch.setattr(
        MODULE,
        "_execution_environment",
        lambda: {"cuda_available": False, "torch_num_threads": 1},
    )

    def fake_git(_root: Path, *args: str) -> str:
        return "abc123" if args[0] == "rev-parse" else ""

    monkeypatch.setattr(MODULE, "_git_checked", fake_git)
    result = MODULE._preflight(tmp_path, tmp_path / "result.json", seal="seal.json")

    assert result["git_revision"] == "abc123"
    assert result["git_status"] == ""
    assert result["protocol"]["sha256"] == hashes[MODULE.PROTOCOL]
    assert result["runner"]["sha256"] == hashes[MODULE.RUNNER_SOURCE]
    assert result["generator"]["sha256"] == hashes[MODULE.GENERATOR_SOURCE]
    assert result["environment"]["lockfile"]["sha256"] == hashes[MODULE.LOCKFILE]
    assert result["seal"]["path"] == "seal.json"
    assert result["seal"]["committed_at_head"] is True
    assert result["seal"]["design_commit"] == "0" * 40
    assert result["execution"] == {"cuda_available": False, "torch_num_threads": 1}


def test_preflight_refuses_existing_output_before_git_or_seed_access(
    monkeypatch, tmp_path: Path
) -> None:
    output = tmp_path / "exists.json"
    output.write_text("reserved", encoding="utf-8")
    monkeypatch.setattr(
        MODULE,
        "_git_checked",
        lambda *_args: pytest.fail("git must not run after output refusal"),
    )

    with pytest.raises(RuntimeError, match="output already exists"):
        MODULE._preflight(tmp_path, output, seal="seal.json")


def test_preflight_refuses_dirty_tree_before_seal_or_fingerprint_access(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []

    def fake_git(_root: Path, *args: str) -> str:
        calls.append(args)
        return "?? uncommitted.txt"

    monkeypatch.setattr(MODULE, "_git_checked", fake_git)
    monkeypatch.setattr(
        MODULE,
        "_load_seal",
        lambda *_args, **_kwargs: pytest.fail("seal must follow clean-tree check"),
    )
    monkeypatch.setattr(
        MODULE,
        "_verified_sha256",
        lambda *_args, **_kwargs: pytest.fail("hashing must follow clean-tree check"),
    )

    with pytest.raises(RuntimeError, match="working tree is dirty"):
        MODULE._preflight(tmp_path, tmp_path / "new.json", seal="seal.json")
    assert calls == [("status", "--short", "--untracked-files=all")]


def test_preflight_rejects_seal_not_committed_at_head(
    monkeypatch, tmp_path: Path
) -> None:
    _write_seal(tmp_path, _seal_payload())

    def fake_git(_root: Path, *args: str) -> str:
        if args[0] == "cat-file":
            raise RuntimeError(
                f"stop condition: git {' '.join(args)} failed: not in HEAD"
            )
        return ""

    monkeypatch.setattr(MODULE, "_git_checked", fake_git)

    with pytest.raises(RuntimeError, match="cat-file"):
        MODULE._preflight(tmp_path, tmp_path / "result.json", seal="seal.json")


def test_preflight_rejects_output_path_mismatch_with_seal(
    monkeypatch, tmp_path: Path
) -> None:
    _write_seal(tmp_path, _seal_payload(output_path="other.json"))
    monkeypatch.setattr(MODULE, "_git_checked", lambda *_args: "")

    with pytest.raises(RuntimeError, match="does not match --output"):
        MODULE._preflight(tmp_path, tmp_path / "result.json", seal="seal.json")


def test_runner_fingerprint_mismatch_is_a_stop_condition(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(MODULE, "_git_checked", lambda *_args: "")
    protocol = tmp_path / MODULE.PROTOCOL
    protocol.parent.mkdir(parents=True)
    protocol.write_bytes(b"protocol")
    protocol_hash = hashlib.sha256(b"protocol").hexdigest()
    monkeypatch.setattr(MODULE, "FROZEN_PROTOCOL_SHA256", protocol_hash)
    runner = tmp_path / MODULE.RUNNER_SOURCE
    runner.parent.mkdir(parents=True)
    runner.write_bytes(b"runner")
    _write_seal(
        tmp_path,
        _seal_payload(
            protocol_sha256=protocol_hash,
            runner_sha256="0" * 64,
            generator_sha256=MODULE.FROZEN_GENERATOR_SHA256,
            lock_sha256=MODULE.FROZEN_LOCKFILE_SHA256,
        ),
    )

    with pytest.raises(RuntimeError, match="sealed v2 runner SHA-256"):
        MODULE._preflight(tmp_path, tmp_path / "result.json", seal="seal.json")


def test_execution_environment_refuses_a_visible_cuda_device(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(RuntimeError, match="CUDA must be unavailable"):
        MODULE._execution_environment()


def test_execution_environment_pins_single_thread_cpu_float64(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    execution = MODULE._execution_environment()

    assert execution["tensor_device"] == "cpu"
    assert execution["tensor_dtype"] == "float64"
    assert execution["torch_num_threads"] == 1
    assert execution["cuda_available"] is False


def test_cli_drops_expected_runner_sha256_and_defaults_seal(
    monkeypatch, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        MODULE.main(
            [
                "--output",
                str(tmp_path / "o.json"),
                "--expected-runner-sha256",
                "0" * 64,
            ]
        )
    assert excinfo.value.code == 2

    captured = {}

    def fake_run(
        root: Path, output: Path, *, seal: str, dataset_factory: object = None
    ):
        captured["seal"] = seal
        return {
            "status": "complete",
            "eligibility": {"eligible": 36},
            "primary": {"claims": []},
        }

    monkeypatch.setattr(MODULE, "run", fake_run)
    monkeypatch.setattr(MODULE, "_atomic_json_new", lambda *_args: None)
    code = MODULE.main(
        [
            "--project-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "o.json"),
            "--seal",
            "custom-seal.json",
        ]
    )

    assert code == 0
    assert captured["seal"] == "custom-seal.json"

    code = MODULE.main(
        ["--project-root", str(tmp_path), "--output", str(tmp_path / "p.json")]
    )
    assert code == 0
    assert (
        captured["seal"]
        == MODULE.SEAL_RECORD
        == "docs/32-independent-lifting-replication-seal.json"
    )


def test_run_signature_takes_seal_not_expected_runner_sha256() -> None:
    parameters = inspect.signature(MODULE.run).parameters
    assert "seal" in parameters
    assert parameters["seal"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "expected_runner_sha256" not in parameters


def test_atomic_writer_never_overwrites_an_existing_record(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    MODULE._atomic_json_new(output, {"status": "complete"})

    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "complete"}
    with pytest.raises(RuntimeError, match="output appeared"):
        MODULE._atomic_json_new(output, {"status": "changed"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "complete"}


def test_run_never_calls_dataset_factory_when_preflight_fails(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        MODULE,
        "_preflight",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("dirty")),
    )

    def forbidden_factory(*_args, **_kwargs):
        pytest.fail("a sealed dataset was instantiated before preflight")

    with pytest.raises(RuntimeError, match="dirty"):
        MODULE.run(
            tmp_path,
            tmp_path / "result.json",
            seal="seal.json",
            dataset_factory=forbidden_factory,
        )


def test_design_failure_uses_only_explicit_old_test_seed_and_runs_no_fits(
    monkeypatch, tmp_path: Path
) -> None:
    observed = []
    monkeypatch.setattr(MODULE, "SEALED_SEEDS", (OLD_TEST_SEED,))
    monkeypatch.setattr(MODULE, "MIN_FACES", 2)
    monkeypatch.setattr(MODULE, "MIN_ELIGIBLE", 2)
    monkeypatch.setattr(MODULE, "_preflight", lambda *_args, **_kwargs: {"clean": True})
    monkeypatch.setattr(
        MODULE,
        "_evaluate_primary",
        lambda *_args: pytest.fail("fits must not run after eligibility failure"),
    )

    class OldSeedOnlyDataset:
        def __init__(self, _count: int, *, seed: int, dtype: object) -> None:
            assert dtype is torch.float64
            assert seed == OLD_TEST_SEED
            observed.append(seed)

        def __getitem__(self, index: int) -> SimpleNamespace:
            assert index == 0
            return _hand_sample()

    report = MODULE.run(
        tmp_path,
        tmp_path / "result.json",
        seal="seal.json",
        dataset_factory=OldSeedOnlyDataset,
    )

    assert observed == [OLD_TEST_SEED]
    assert report["status"] == "design_failure_insufficient_eligible"
    assert report["eligibility"]["eligible_seeds"] == [OLD_TEST_SEED]
    assert report["raw_primary"] == []
    assert report["raw_c1"] == []
    assert len(report["primary"]["claims"]) == 7
    assert all(claim["supported"] is None for claim in report["primary"]["claims"])
    assert "mean_per_seed_log10_ratio_by_claim" not in report["audit"]


def test_failed_campaign_preserves_rows_names_seed_and_arm_and_nulls_claims(
    monkeypatch, tmp_path: Path
) -> None:
    seeds = (OLD_TEST_SEED, 1001, 1002)
    monkeypatch.setattr(MODULE, "SEALED_SEEDS", seeds)
    monkeypatch.setattr(MODULE, "MIN_FACES", 2)
    monkeypatch.setattr(MODULE, "MIN_ELIGIBLE", 3)
    monkeypatch.setattr(MODULE, "_preflight", lambda *_args, **_kwargs: {"clean": True})

    def flaky_primary(_sample: SimpleNamespace, seed: int) -> dict:
        if seed == 1001:
            raise MODULE.DesignFailureError("tainted basis", arm="hard_cycle_ls")
        return {"seed": seed}

    monkeypatch.setattr(MODULE, "_evaluate_primary", flaky_primary)
    monkeypatch.setattr(MODULE, "_evaluate_c1", lambda _sample, seed: {"seed": seed})

    report = MODULE.run(
        tmp_path,
        tmp_path / "result.json",
        seal="seal.json",
        dataset_factory=_FakeDataset,
    )

    assert report["status"] == "design_failure"
    assert report["failure"]["seed"] == 1001
    assert report["failure"]["arm"] == "hard_cycle_ls"
    assert report["failure"]["phase"] == "campaign"
    assert report["failure"]["type"] == "DesignFailureError"
    assert "tainted basis" in report["failure"]["message"]
    assert [row["seed"] for row in report["raw_primary"]] == [OLD_TEST_SEED]
    assert [row["seed"] for row in report["raw_c1"]] == [OLD_TEST_SEED]
    assert len(report["primary"]["claims"]) == 7
    assert all(claim["supported"] is None for claim in report["primary"]["claims"])
    assert all(claim["estimate"] is None for claim in report["primary"]["claims"])
    assert {claim["id"] for claim in report["primary"]["claims"]} == set(
        MODULE.PRIMARY_CLAIM_IDS
    )
    assert report["audit"]["raw_primary_rows"] == 1
    assert "mean_per_seed_log10_ratio_by_claim" not in report["audit"]


def test_execution_failure_token_and_null_decisions_on_unexpected_fault(
    monkeypatch, tmp_path: Path
) -> None:
    seeds = (OLD_TEST_SEED, 1001)
    monkeypatch.setattr(MODULE, "SEALED_SEEDS", seeds)
    monkeypatch.setattr(MODULE, "MIN_FACES", 2)
    monkeypatch.setattr(MODULE, "MIN_ELIGIBLE", 2)
    monkeypatch.setattr(MODULE, "_preflight", lambda *_args, **_kwargs: {"clean": True})

    def broken_c1(_sample: SimpleNamespace, seed: int) -> dict:
        raise FloatingPointError("unexpected numeric fault")

    monkeypatch.setattr(
        MODULE, "_evaluate_primary", lambda _sample, seed: {"seed": seed}
    )
    monkeypatch.setattr(MODULE, "_evaluate_c1", broken_c1)

    report = MODULE.run(
        tmp_path,
        tmp_path / "result.json",
        seal="seal.json",
        dataset_factory=_FakeDataset,
    )

    assert report["status"] == "execution_failure"
    assert report["failure"]["seed"] == OLD_TEST_SEED
    assert report["failure"]["arm"] is None
    assert report["failure"]["type"] == "FloatingPointError"
    assert report["raw_primary"] == [{"seed": OLD_TEST_SEED}]
    assert report["raw_c1"] == []
    assert all(claim["supported"] is None for claim in report["primary"]["claims"])


def test_complete_run_audit_means_equal_raw_row_recomputation(
    monkeypatch, tmp_path: Path
) -> None:
    seeds = tuple(1000 + index for index in range(30))
    monkeypatch.setattr(MODULE, "SEALED_SEEDS", seeds)
    monkeypatch.setattr(MODULE, "MIN_FACES", 2)
    monkeypatch.setattr(MODULE, "MIN_ELIGIBLE", 30)
    monkeypatch.setattr(MODULE, "STEPS", 1)
    monkeypatch.setattr(MODULE, "N_TEST", 32)
    monkeypatch.setattr(MODULE, "_preflight", lambda *_args, **_kwargs: {"clean": True})

    report = MODULE.run(
        tmp_path,
        tmp_path / "result.json",
        seal="seal.json",
        dataset_factory=_FakeDataset,
    )

    assert report["status"] == "complete"
    audit = report["audit"]
    assert audit["declared_seeds"] == 30
    assert audit["eligible_seeds"] == 30
    assert audit["eligible_seed_ids"] == list(seeds)
    assert audit["raw_primary_rows"] == 30
    assert audit["raw_c1_rows"] == 30
    assert audit["c1_replicates_per_topology"] == MODULE.C1_REPLICATES
    assert audit["arm_names"] == list(MODULE.ARM_NAMES)

    for definition in MODULE._PRIMARY_CLAIM_DEFINITIONS:
        recomputed = statistics.fmean(
            math.log10(
                row["arms"][definition["numerator"]]["held_out_mse"]
                / row["arms"][definition["denominator"]]["held_out_mse"]
            )
            for row in report["raw_primary"]
        )
        assert audit["mean_per_seed_log10_ratio_by_claim"][
            definition["id"]
        ] == pytest.approx(recomputed, abs=1e-15)
    claims = {claim["id"]: claim for claim in report["primary"]["claims"]}
    for claim_id, mean in audit["mean_per_seed_log10_ratio_by_claim"].items():
        assert claims[claim_id]["estimate"] == pytest.approx(mean, abs=1e-15)

    ambient = statistics.fmean(
        math.log10(
            row["arms"]["ambient_adam"]["held_out_mse"]
            / row["arms"]["ambient_min_norm_ls"]["held_out_mse"]
        )
        for row in report["raw_primary"]
    )
    assert audit[
        "mean_per_seed_log10_ratio_ambient_adam_vs_min_norm_ls"
    ] == pytest.approx(ambient, abs=1e-15)

    # The completed campaign keeps the oracle out of every claim and log ratio.
    assert "generator_cycle_basis_oracle" not in json.dumps(report["primary"])
    assert "noninferiority" not in json.dumps(report["primary"]).lower()
    for row in report["raw_primary"]:
        oracle = row["arms"]["generator_cycle_basis_oracle"]
        assert "relative_error_to_mean_squared_test_target" in oracle
        assert not any("log" in key for key in oracle)
        gap = row["optimizer_descriptive"][
            "soft_adam_vs_closed_form_solution_gap_frobenius"
        ]
        assert math.isfinite(gap) and gap >= 0


def test_terminal_status_vocabulary_is_exactly_the_five_pinned_tokens() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    pinned = {
        "complete",
        "design_failure",
        "design_failure_insufficient_eligible",
        "execution_failure",
        "interrupted",
    }

    tokens = set(
        re.findall(
            r'"(complete|design_failure|design_failure_insufficient_eligible'
            r'|execution_failure|interrupted)"',
            source,
        )
    )
    assigned = set(re.findall(r'\["status"\]\s*=\s*"([a-z_]+)"', source))

    assert tokens == pinned
    assert assigned <= pinned
    # Every claim-decision token the CLI prints comes from the same pinned set.
    assert main_exit_logic_tokens(source) <= pinned


def main_exit_logic_tokens(source: str) -> set[str]:
    return set(re.findall(r'report\["status"\]\s*==\s*"([a-z_]+)"', source))
