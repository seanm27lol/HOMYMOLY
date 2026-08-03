from __future__ import annotations

import json

from homymoly.__main__ import main
from homymoly.config import load_config
from homymoly.stage1 import build_stage1_dataset, validate_foundation


def test_foundation_gate_integrates_balanced_data_and_exact_topology() -> None:
    report = validate_foundation(num_samples=6, seed=17, num_vertices=24)

    assert report["status"] == "passed"
    assert report["regime_counts"] == {"cell": 2, "graph": 2, "sheaf": 2}
    assert report["label_counts"] == {"0": 3, "1": 3}
    assert report["total_edges"] > 0
    assert report["total_candidate_faces"] >= report["total_active_faces"] > 0
    assert report["max_boundary_residual"] == 0.0
    assert report["max_chain_map_residual"] == 0.0
    assert report["max_cone_chain_residual"] == 0.0
    assert report["max_sheaf_operator_residual"] < 1e-12
    assert report["max_transport_orthogonality_residual"] < 1e-5
    assert report["oracles"] == {
        "triangle_graph_betti": [1, 1],
        "filled_triangle_betti": [1, 0, 0],
        "identity_cone_betti": [0, 0, 0],
        "inclusion_cone_betti": [0, 0, 1],
    }


def test_foundation_cli_prints_machine_readable_report(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(
        [
            "validate-foundation",
            "--config",
            "configs/stage1.yaml",
            "--samples",
            "6",
            "--vertices",
            "24",
        ]
    )

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "passed"
    assert report["num_samples"] == 6
    assert report["configured_dataset"]["num_samples"] == 6144
    assert sum(report["configured_dataset"]["split_sizes"].values()) == 6144


def test_runtime_config_builds_one_group_disjoint_dataset() -> None:
    config = load_config("configs/stage1.yaml")
    dataset, splits = build_stage1_dataset(config.data)

    split_sets = {name: set(indices) for name, indices in splits.items()}
    assert set.union(*split_sets.values()) == set(range(len(dataset)))
    assert split_sets["train"].isdisjoint(split_sets["validation"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["validation"].isdisjoint(split_sets["test"])

    group_sets = {
        name: {dataset.group_ids[index] for index in indices}
        for name, indices in splits.items()
    }
    assert group_sets["train"].isdisjoint(group_sets["validation"])
    assert group_sets["train"].isdisjoint(group_sets["test"])
    assert group_sets["validation"].isdisjoint(group_sets["test"])
