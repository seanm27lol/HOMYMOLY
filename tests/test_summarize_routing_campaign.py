from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_routing_campaign.py"
SPEC = importlib.util.spec_from_file_location("summarize_routing_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frozen_campaign_summary_and_decision(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    configs = []
    expected_hashes = {}
    margins = (0.08, 0.10, 0.12, 0.09, 0.11)
    for index, margin in enumerate(margins, start=1):
        name = f"frozen-s{index}"
        config = tmp_path / f"s{index}.yaml"
        config.write_text(f"artifacts:\n  run_name: {name}\n", encoding="utf-8")
        configs.append(config)
        expected_hashes[config.name] = hashlib.sha256(config.read_bytes()).hexdigest()
        run = artifacts / name
        run.mkdir(parents=True)
        fixed = 0.7
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "status": "gate-failed" if index == 5 else "completed",
                    "failed_gate": None,
                    "test": {
                        "hard_accuracy": fixed + margin,
                        "graph_expert_accuracy": fixed,
                        "cell_expert_accuracy": 0.65,
                        "sheaf_expert_accuracy": 0.66,
                        "dense_accuracy": 0.72,
                        "route_accuracy": 0.5,
                        "regime_route_mutual_information": 0.1,
                        "route_utilization_graph": 0.34,
                        "route_utilization_cell": 0.33,
                        "route_utilization_sheaf": 0.33,
                    },
                }
            ),
            encoding="utf-8",
        )
        (run / "environment.json").write_text(
            json.dumps(
                {
                    "git": {"commit": "abc123", "status": " M docs/note.md"},
                    "code_fingerprint": "source-sha256",
                    "torch_version": "2.13.0",
                    "cuda_version": "13.0",
                    "device": "cuda",
                    "device_name": "GB10",
                }
            ),
            encoding="utf-8",
        )

    report = MODULE.summarize(configs, artifacts, expected_hashes=expected_hashes)
    assert report["primary"]["mean_margin"] == pytest.approx(0.1)
    assert report["primary"]["decision"] == "supported"
    assert report["shared_git_revision"] == "abc123"
    assert report["shared_code_fingerprint"] == "source-sha256"
    assert report["rows"][0]["git_status_at_start"] == " M docs/note.md"
    assert report["shared_environment"]["device_name"] == "GB10"
    assert report["rows"][-1]["run_status"] == "gate-failed"
    assert len(report["rows"]) == 5


def test_summary_rejects_incomplete_run(tmp_path: Path) -> None:
    configs = []
    for index in range(5):
        config = tmp_path / f"s{index}.yaml"
        config.write_text(f"run_name: missing-{index}\n", encoding="utf-8")
        configs.append(config)
    with pytest.raises(FileNotFoundError):
        MODULE.summarize(
            configs,
            tmp_path / "artifacts",
            expected_hashes={
                config.name: hashlib.sha256(config.read_bytes()).hexdigest()
                for config in configs
            },
        )
