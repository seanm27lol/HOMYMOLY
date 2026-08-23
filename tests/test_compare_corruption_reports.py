from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest

from scripts.compare_corruption_reports import compare_reports, main


def _synthetic_report(
    *, model_offset: float = 0.0, blocks: int = 4, kinds: int = 2
) -> dict:
    severities = [0.1, 0.2, 0.4]
    kind_names = ["edge_cochain_noise", "transport_rotation"][:kinds]
    per_batch = []
    for kind_number, kind in enumerate(kind_names):
        for severity_number, severity in enumerate(severities):
            for block_number in range(blocks):
                pattern = (3 * block_number + 2 * severity_number + kind_number) % 7
                topology = (
                    0.11 * (severity_number + 1)
                    + 0.027 * pattern
                    + 0.013 * block_number
                )
                damage = (
                    0.08 * (severity_number + 1)
                    + 0.019 * ((block_number + 2 * severity_number) % 5)
                    + model_offset * topology
                    + model_offset * 0.003 * (block_number % 3)
                )
                displacement = (
                    0.05 * (severity_number + 1)
                    + 0.007 * ((2 * block_number + severity_number) % 4)
                    + model_offset * 0.002
                )
                per_batch.append(
                    {
                        "kind": kind,
                        "severity": severity,
                        "block_id": f"{kind}:{block_number:04d}",
                        "block_number": block_number,
                        "batch_start": block_number * 16,
                        "block_seed": 100_000 * kind_number
                        + 1_000 * severity_number
                        + block_number,
                        "num_examples": 16,
                        "sigma_min": severity * 0.01,
                        "sigma_mean": severity * 0.51,
                        "sigma_max": severity * 0.99,
                        "damage_rate": damage,
                        "mean_ce_increase": damage * 1.3,
                        "topological_defect": topology + model_offset * 0.01,
                        "mean_embedding_displacement": displacement,
                        "mean_diagnostic": topology * 0.2,
                    }
                )
    return {
        "schema_version": 3,
        "checkpoint": "/tmp/model.pt",
        "checkpoint_sha256": "a" * 64,
        "config": "/tmp/config.yaml",
        "config_sha256": "b" * 64,
        "script_sha256": "c" * 64,
        "git": {"commit": "d" * 40, "status": ""},
        "command": ["python", "scripts/eval_corruption.py"],
        "execution": {"batch_size": 16, "max_batches": blocks},
        "topological_metric": {
            "name": "exact_srtd",
            "degree": 1,
            "max_dim": 2,
            "normalization": "full-matrix-quantile",
            "normalization_quantile": 0.9,
            "max_points": 24,
        },
        "severities": severities,
        "sampling": {
            "protocol": "sha256-block-and-sample-v1",
            "data_seed": 19,
            "experiment_seed": 23,
            "pairing_contract": "synthetic paired draws",
        },
        "analysis_protocol": {"method": "rank-residual-partial-spearman-v1"},
        "analysis": {},
        "per_batch": per_batch,
        "per_example": [],
    }


def _write_report(path: Path, report: dict) -> Path:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def test_exact_paired_analysis_is_deterministic_and_records_provenance(
    tmp_path: Path,
) -> None:
    baseline_path = _write_report(tmp_path / "baseline.json", _synthetic_report())
    candidate_path = _write_report(
        tmp_path / "candidate.json", _synthetic_report(model_offset=0.35)
    )

    first = compare_reports(
        baseline_path,
        [candidate_path],
        bootstrap_replicates=40,
        randomization_replicates=99,
        analysis_seed=8128,
        command=["synthetic", "comparison"],
    )
    second = compare_reports(
        baseline_path,
        [candidate_path],
        bootstrap_replicates=40,
        randomization_replicates=99,
        analysis_seed=8128,
        command=["synthetic", "comparison"],
    )

    assert first == second
    assert first["inputs"]["baseline"]["source_command"]
    assert len(first["inputs"]["baseline"]["sha256"]) == 64
    assert "not evaluate representation conversion" in first["claim_boundary"]
    result = first["comparisons"][0]["by_kind"]["edge_cochain_noise"]
    assert result["counts"] == {
        "paired_batch_observations": 12,
        "complete_blocks": 4,
        "severity_levels": 3,
    }
    assert math.isclose(
        result["candidate_minus_baseline"],
        result["candidate_adjusted_partial_spearman"]
        - result["baseline_adjusted_partial_spearman"],
    )
    assert len(result["paired_complete_block_bootstrap_95_ci"]) == 2
    randomization = result["paired_whole_block_model_label_randomization"]
    assert randomization["mode"] == "exact"
    assert randomization["assignments_evaluated"] == 16
    assert 0.0 <= randomization["pvalue_two_sided"] <= 1.0


def test_more_than_sixteen_blocks_uses_deterministic_monte_carlo(
    tmp_path: Path,
) -> None:
    baseline_path = _write_report(
        tmp_path / "baseline.json", _synthetic_report(blocks=17, kinds=1)
    )
    candidate_path = _write_report(
        tmp_path / "candidate.json",
        _synthetic_report(model_offset=0.2, blocks=17, kinds=1),
    )

    kwargs = {
        "bootstrap_replicates": 12,
        "randomization_replicates": 31,
        "analysis_seed": 99,
    }
    first = compare_reports(baseline_path, [candidate_path], **kwargs)
    second = compare_reports(baseline_path, [candidate_path], **kwargs)
    randomization = first["comparisons"][0]["by_kind"]["edge_cochain_noise"][
        "paired_whole_block_model_label_randomization"
    ]

    assert first == second
    assert randomization["mode"] == "deterministic-monte-carlo"
    assert randomization["assignments_evaluated"] == 31
    assert isinstance(randomization["seed"], int)


@pytest.mark.parametrize(
    "mismatch",
    [
        "schema",
        "sampling_seed",
        "severity_order",
        "missing_join_row",
        "block_seed",
        "num_examples",
        "sigma_summary",
    ],
)
def test_pairing_mismatches_are_rejected(tmp_path: Path, mismatch: str) -> None:
    baseline = _synthetic_report()
    candidate = copy.deepcopy(baseline)
    if mismatch == "schema":
        candidate["schema_version"] = 2
    elif mismatch == "sampling_seed":
        candidate["sampling"]["experiment_seed"] += 1
    elif mismatch == "severity_order":
        candidate["severities"] = list(reversed(candidate["severities"]))
    elif mismatch == "missing_join_row":
        candidate["per_batch"].pop()
    elif mismatch == "block_seed":
        candidate["per_batch"][0]["block_seed"] += 1
    elif mismatch == "num_examples":
        candidate["per_batch"][0]["num_examples"] += 1
    elif mismatch == "sigma_summary":
        candidate["per_batch"][0]["sigma_mean"] += 0.001

    baseline_path = _write_report(tmp_path / "baseline.json", baseline)
    candidate_path = _write_report(tmp_path / "candidate.json", candidate)
    with pytest.raises(ValueError):
        compare_reports(
            baseline_path,
            [candidate_path],
            bootstrap_replicates=4,
            randomization_replicates=4,
        )


def test_cli_writes_atomic_json_for_multiple_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_path = _write_report(tmp_path / "baseline.json", _synthetic_report())
    candidate_one = _write_report(
        tmp_path / "candidate-one.json", _synthetic_report(model_offset=0.1)
    )
    candidate_two = _write_report(
        tmp_path / "candidate-two.json", _synthetic_report(model_offset=0.2)
    )
    output = tmp_path / "nested" / "paired.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_corruption_reports.py",
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_one),
            str(candidate_two),
            "--output",
            str(output),
            "--bootstrap-replicates",
            "8",
            "--randomization-replicates",
            "8",
        ],
    )

    main()

    result = json.loads(output.read_text(encoding="utf-8"))
    assert len(result["comparisons"]) == 2
    assert result["command"][1:] == sys.argv
