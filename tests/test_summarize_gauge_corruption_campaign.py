from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "summarize_gauge_corruption_campaign.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_gauge_corruption_campaign", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

KINDS = ("edge_cochain_noise", "node_anchor_noise", "transport_rotation")
EVALUATOR_SHA256 = "e" * 64
COMPARATOR_SHA256 = "c" * 64
COMMIT = "8021292e97abfec91768f1b5437c883a42c29c60"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_side(
    project: Path, run_name: str, seed: int, statistics_by_kind: dict[str, float]
) -> dict[str, Any]:
    """Materialize one trained run's config, checkpoint, and final report."""

    config = project / "configs" / "gate3g" / f"{run_name}.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        f"experiment:\n  name: {run_name}\n  seed: {seed}\n"
        f"data:\n  seed: {seed}\n",
        encoding="utf-8",
    )

    run_dir = project / "artifacts" / "gate3g" / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "checkpoints" / "last.pt"
    checkpoint.write_bytes(f"weights-{run_name}".encode())

    report = run_dir / "corruption_report_final.json"
    report.write_text(
        json.dumps({"run": run_name, "adjusted": statistics_by_kind}, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "path": str(report),
        "sha256": _sha256(report),
        "schema_version": 3,
        "config": str(config),
        "config_sha256": _sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "script_sha256": EVALUATOR_SHA256,
        "git": {"commit": COMMIT, "status": ""},
        "checkpoint_load": {"strict": False, "unexpected_keys": []},
        "source_command": ["python", "scripts/eval_corruption.py", "--run", run_name],
    }


def _write_pair(
    project: Path, seed: int, differences: dict[str, float]
) -> Path:
    suffix = str(seed)[-2:]
    baseline_statistics = {kind: 0.05 for kind in KINDS}
    candidate_statistics = {
        kind: baseline_statistics[kind] + differences[kind] for kind in KINDS
    }
    baseline = _write_side(
        project, f"gauge-task-only-s{suffix}", seed, baseline_statistics
    )
    candidate = _write_side(
        project, f"gauge-plus-chain-s{suffix}", seed, candidate_statistics
    )

    by_kind = {
        kind: {
            "baseline_adjusted_partial_spearman": baseline_statistics[kind],
            "candidate_adjusted_partial_spearman": candidate_statistics[kind],
            "candidate_minus_baseline": differences[kind],
            "counts": {
                "complete_blocks": 13,
                "paired_batch_observations": 65,
                "severity_levels": 5,
            },
            "paired_whole_block_model_label_randomization": {
                "mode": "exact",
                "pvalue_two_sided": 0.5,
            },
        }
        for kind in KINDS
    }

    paired = Path(baseline["path"]).parent / "paired_comparison_final.json"
    paired.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "analysis_method": "paired-cross-model-corruption-v1",
                "claim_boundary": "Fixed-expert embedding diagnostic only.",
                "script_sha256": COMPARATOR_SHA256,
                "method": {"analysis_seed": 20260813},
                "inputs": {"baseline": baseline, "candidates": [candidate]},
                "validated_pairing_contract": {
                    "complete_blocks_required": True,
                    "join_key": ["kind", "severity", "block_id"],
                    "equal_fields": ["block_seed", "num_examples"],
                    "severities": [0.05, 0.1, 0.2, 0.4, 0.8],
                    "sampling_signature": {
                        "batch_size": 24,
                        "max_batches": 13,
                        "data_seed": seed,
                        "experiment_seed": seed,
                        "sampling_protocol": "sha256-block-and-sample-v1",
                        "sampling_metadata": {
                            "data_seed": seed,
                            "experiment_seed": seed,
                        },
                        "topological_metric": {"name": "exact_srtd", "degree": 1},
                    },
                },
                "comparisons": [{"candidate_number": 1, "by_kind": by_kind}],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return paired


def _campaign(project: Path, per_seed: list[dict[str, float]]) -> Path:
    for index, differences in enumerate(per_seed):
        _write_pair(project, 20260803 + index, differences)
    return project / "artifacts" / "gate3g"


def test_summary_aggregates_paired_differences_across_training_seeds(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    # Four seeds; transport_rotation differences average to exactly 0.10.
    per_seed = [
        {"edge_cochain_noise": 0.0, "node_anchor_noise": -0.02, "transport_rotation": 0.08},
        {"edge_cochain_noise": 0.0, "node_anchor_noise": -0.04, "transport_rotation": 0.12},
        {"edge_cochain_noise": 0.0, "node_anchor_noise": -0.06, "transport_rotation": 0.09},
        {"edge_cochain_noise": 0.0, "node_anchor_noise": -0.08, "transport_rotation": 0.11},
    ]
    root = _campaign(project, per_seed)

    report = MODULE.summarize(root, project_root=project, expected_pairs=4)

    assert report["validation"]["status"] == "passed"
    assert report["validation"]["validated_pairs"] == 4
    assert report["validation"]["distinct_training_seeds"] == 4
    assert report["validation"]["complete_blocks_per_kind"] == 13
    assert report["aggregation"]["student_t_degrees_of_freedom"] == 3
    assert report["aggregation"]["multiplicity_adjustment"] == "none"

    transport = report["by_kind"]["transport_rotation"]
    assert transport["seeds_in_order"] == [20260803, 20260804, 20260805, 20260806]
    assert transport["mean_difference"] == pytest.approx(0.10)
    assert transport["n_paired_seeds"] == 4
    # Every difference is positive, so the exact sign test hits its floor.
    assert transport["sensitivity_sign_test"]["pvalue_two_sided"] == pytest.approx(0.125)
    assert transport["interval_includes_zero"] is False

    # A kind that never moves has a degenerate interval pinned at zero.
    edge = report["by_kind"]["edge_cochain_noise"]
    assert edge["mean_difference"] == pytest.approx(0.0)
    assert edge["sensitivity_sign_test"]["ties_discarded"] == 4
    assert edge["sensitivity_sign_test"]["pvalue_two_sided"] is None

    assert report["decision"]["kinds_with_interval_excluding_zero"] == [
        "node_anchor_noise",
        "transport_rotation",
    ]
    assert "fixed-expert embedding diagnostic" in report["scope"].casefold()
    assert report["analysis_provenance"]["evaluator_sha256"] == EVALUATOR_SHA256
    assert report["analysis_provenance"]["comparator_sha256"] == COMPARATOR_SHA256
    assert report["analysis_provenance"]["git_commit"] == COMMIT


def test_exact_two_sided_sign_test_matches_binomial_enumeration() -> None:
    assert MODULE.exact_two_sided_sign_test([1.0] * 8)[
        "pvalue_two_sided"
    ] == pytest.approx(2 / 256)
    # 3 favorable and 5 unfavorable out of 8 untied seeds.
    assert MODULE.exact_two_sided_sign_test(
        [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    )["pvalue_two_sided"] == pytest.approx(2 * 93 / 256)
    # A balanced split saturates at one rather than exceeding it.
    assert MODULE.exact_two_sided_sign_test([1.0, 1.0, -1.0, -1.0])[
        "pvalue_two_sided"
    ] == 1.0


def test_summary_rejects_a_tampered_corruption_report(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = _campaign(project, [{kind: 0.01 for kind in KINDS} for _ in range(4)])
    tampered = (
        project
        / "artifacts"
        / "gate3g"
        / "gauge-plus-chain-s04"
        / "corruption_report_final.json"
    )
    tampered.write_text('{"run": "edited"}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="corruption report hash mismatch"):
        MODULE.summarize(root, project_root=project, expected_pairs=4)


def test_summary_rejects_a_tampered_checkpoint(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = _campaign(project, [{kind: 0.01 for kind in KINDS} for _ in range(4)])
    checkpoint = (
        project
        / "artifacts"
        / "gate3g"
        / "gauge-task-only-s03"
        / "checkpoints"
        / "last.pt"
    )
    checkpoint.write_bytes(b"replaced")

    with pytest.raises(RuntimeError, match="checkpoint hash mismatch"):
        MODULE.summarize(root, project_root=project, expected_pairs=4)


def test_summary_rejects_a_pair_that_is_not_seed_matched(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = _campaign(project, [{kind: 0.01 for kind in KINDS} for _ in range(4)])
    config = project / "configs" / "gate3g" / "gauge-plus-chain-s05.yaml"
    config.write_text(
        "experiment:\n  name: gauge-plus-chain-s05\n  seed: 19990101\n"
        "data:\n  seed: 19990101\n",
        encoding="utf-8",
    )
    paired = (
        project
        / "artifacts"
        / "gate3g"
        / "gauge-task-only-s05"
        / "paired_comparison_final.json"
    )
    document = json.loads(paired.read_text(encoding="utf-8"))
    document["inputs"]["candidates"][0]["config_sha256"] = _sha256(config)
    paired.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")

    with pytest.raises(RuntimeError, match="not seed-matched"):
        MODULE.summarize(root, project_root=project, expected_pairs=4)


def test_summary_rejects_a_report_generated_from_a_dirty_worktree(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    root = _campaign(project, [{kind: 0.01 for kind in KINDS} for _ in range(4)])
    paired = (
        project
        / "artifacts"
        / "gate3g"
        / "gauge-task-only-s03"
        / "paired_comparison_final.json"
    )
    document = json.loads(paired.read_text(encoding="utf-8"))
    document["inputs"]["baseline"]["git"]["status"] = " M src/homymoly/model.py"
    paired.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")

    with pytest.raises(RuntimeError, match="dirty worktree"):
        MODULE.summarize(root, project_root=project, expected_pairs=4)


def test_summary_rejects_an_unexpected_pair_count(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = _campaign(project, [{kind: 0.01 for kind in KINDS} for _ in range(3)])

    with pytest.raises(RuntimeError, match="expected 8 paired comparisons"):
        MODULE.summarize(root, project_root=project)


def test_summary_rejects_a_sampling_signature_that_drifts_from_the_seed(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    root = _campaign(project, [{kind: 0.01 for kind in KINDS} for _ in range(4)])
    paired = (
        project
        / "artifacts"
        / "gate3g"
        / "gauge-task-only-s06"
        / "paired_comparison_final.json"
    )
    document = json.loads(paired.read_text(encoding="utf-8"))
    document["validated_pairing_contract"]["sampling_signature"]["data_seed"] = 1
    paired.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")

    with pytest.raises(RuntimeError, match="sampling signature does not match"):
        MODULE.summarize(root, project_root=project, expected_pairs=4)


def test_summary_rejects_a_comparator_hash_that_is_not_shared(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = _campaign(project, [{kind: 0.01 for kind in KINDS} for _ in range(4)])
    paired = (
        project
        / "artifacts"
        / "gate3g"
        / "gauge-task-only-s04"
        / "paired_comparison_final.json"
    )
    document = json.loads(paired.read_text(encoding="utf-8"))
    document["script_sha256"] = "f" * 64
    paired.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")

    with pytest.raises(RuntimeError, match="comparator script is not shared"):
        MODULE.summarize(root, project_root=project, expected_pairs=4)
