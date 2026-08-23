from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_compute_campaign.py"
SPEC = importlib.util.spec_from_file_location("summarize_compute_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

FINGERPRINT = "44408d7adf8467e594879b46e25a1cb7fd89a7e7a5d5f3446548bcbf3ed1096e"
COMMIT = "8021292e97abfec91768f1b5437c883a42c29c60"
TORCH = "2.13.0+cu130"
CUDA = "13.0"
DEVICE_NAME = "NVIDIA GB10"
IDENTIFIABLE_SEEDS = (20260821, 20260822)
IDENTIFIABLE_ABLATIONS = ("combined", "task_reconstruction")
ROUTING_SEEDS = (1, 2)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_identifiable(
    project: Path, seed: int, ablation: str, median: float
) -> Path:
    index = IDENTIFIABLE_SEEDS.index(seed) + 1
    run = project / "artifacts" / "identifiable-maps" / "campaign" / f"seed-{seed}" / ablation
    run.mkdir(parents=True, exist_ok=True)
    config = run / "effective_config.yaml"
    config.write_text(f"seed: {seed}\nablation: {ablation}\n", encoding="utf-8")
    checkpoint = run / "checkpoint.pt"
    checkpoint.write_bytes(f"weights-{seed}-{ablation}".encode())
    runner = project / "scripts" / "benchmark_identifiable_maps.py"
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("# identifiable runner\n", encoding="utf-8")

    return _write_json(
        project
        / "artifacts"
        / "identifiable-maps"
        / "benchmarks"
        / f"gb10-s{index}-{ablation}.json",
        {
            "ablation": ablation,
            "batch_size": 192,
            "best_epoch": 99,
            "chain_residual_max": 3.55e-15,
            "device": "cuda",
            "device_name": DEVICE_NAME,
            "experiment": "identifiable-map-checkpoint-inference-benchmark",
            "iterations": 100,
            "latency_ms": {
                "maximum": median + 1.0,
                "mean": median + 0.02,
                "median": median,
                "minimum": median - 0.01,
                "p10": median - 0.005,
                "p90": median + 0.03,
            },
            "map_tolerance": 1e-05,
            "materialization_checksum": 295.46,
            "measurement_scope": "model forward only",
            "parameters": 95448,
            "peak_cuda_memory_allocated_bytes": 35069440,
            "peak_cuda_memory_reserved_bytes": 35651584,
            "provenance": {
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": _sha256(checkpoint),
                },
                "code_fingerprint": "f" * 64,
                "command": ["python", "scripts/benchmark_identifiable_maps.py"],
                "config": {"path": str(config), "sha256": _sha256(config)},
                "cublas_workspace_config": ":4096:8",
                "cuda": CUDA,
                "deterministic_algorithms": True,
                "effective_ablation": ablation,
                "effective_seed": seed,
                "git_revision": COMMIT,
                "python": "3.12.3",
                "runner": {"path": str(runner), "sha256": _sha256(runner)},
                "torch": TORCH,
            },
            "schema_version": 1,
            "seed": seed,
            "status": "completed",
            "throughput_examples_per_second_at_median": 192.0 / (median / 1000.0),
            "warmup": 20,
        },
    )


def _write_routing(project: Path, index: int, routed: float, dense: float) -> Path:
    run_name = f"routing-confirmatory-v2-s{index}"
    run = project / "artifacts" / run_name / "checkpoints"
    run.mkdir(parents=True, exist_ok=True)
    checkpoint = run / "last.pt"
    checkpoint.write_bytes(f"weights-{run_name}".encode())
    config = project / "configs" / f"{run_name}.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(f"run: {run_name}\n", encoding="utf-8")

    def result(path: str, median: float, memory: int) -> dict[str, Any]:
        return {
            "batch_size": 64,
            "checksum": -1.0,
            "examples_per_second": 64.0 / (median / 1000.0),
            "iterations": 100,
            "latency_ms_mean": median + 0.5,
            "latency_ms_median": median,
            "latency_ms_p95": median + 4.0,
            "latency_ms_stdev": 1.5,
            "path": path,
            "peak_memory_bytes": memory,
        }

    return _write_json(
        project / "artifacts" / "benchmarks" / f"{run_name}-compute.json",
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "config": str(config),
            "config_sha256": _sha256(config),
            "dense_to_routed_speedup": dense / routed,
            "device": "cuda",
            "environment": {
                "command": ["python", "scripts/benchmark_compute.py"],
                "cuda": CUDA,
                "device_name": DEVICE_NAME,
                "git_revision": COMMIT,
                "platform": "Linux-6.17.0-1026-nvidia-aarch64-with-glibc2.39",
                "python": "3.12.3",
                "torch": TORCH,
            },
            "precision": "torch.bfloat16",
            "results": [
                result("routed", routed, 119415296),
                result("fixed_graph", routed / 2.0, 164400128),
                result("fixed_cell", routed / 1.8, 164400128),
                result("fixed_sheaf", routed / 1.7, 169857536),
                result("dense", dense, 169401344),
            ],
            "route_profile_on_benchmark_batch": {
                "cell": 0.375,
                "graph": 0.4375,
                "sheaf": 0.1875,
            },
            "routed_to_dense_latency_ratio": routed / dense,
            "status": "completed",
        },
    )


def _seal_receipt(project: Path, benchmarks: list[Path]) -> Path:
    outputs = [
        {
            "path": path.relative_to(project).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in benchmarks
    ]
    identifiable_steps = [
        f"benchmark-identifiable-s{index}-{ablation}"
        for index in range(1, len(IDENTIFIABLE_SEEDS) + 1)
        for ablation in IDENTIFIABLE_ABLATIONS
    ]
    routing_steps = [f"benchmark-routing-v2-s{index}" for index in ROUTING_SEEDS]
    fixed = ["summarize-identifiable-campaign", *identifiable_steps, *routing_steps]
    train = [
        f"train-step-{index}"
        for index in range(MODULE.EXPECTED_SCHEDULER_STEPS - len(fixed))
    ]
    return _write_json(
        project
        / "artifacts"
        / "scheduler"
        / MODULE.CAMPAIGN_ID
        / "runs"
        / FINGERPRINT
        / "campaign.complete.json",
        {
            "campaign_id": MODULE.CAMPAIGN_ID,
            "completed_at": "2026-08-23T03:01:26.739211+00:00",
            "completed_steps": [*train, *fixed],
            "launch_fingerprint": FINGERPRINT,
            "outputs": outputs,
            "retry_epoch": 0,
            "schema_version": 1,
            "status": "completed",
        },
    )


def _campaign(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    benchmarks: list[Path] = []
    # combined is slightly faster than task_reconstruction on every seed.
    for seed in IDENTIFIABLE_SEEDS:
        benchmarks.append(_write_identifiable(project, seed, "combined", 0.2750))
        benchmarks.append(
            _write_identifiable(project, seed, "task_reconstruction", 0.2760)
        )
    for index in ROUTING_SEEDS:
        benchmarks.append(_write_routing(project, index, routed=40.0, dense=60.0))
    receipt = _seal_receipt(project, benchmarks)
    return project, receipt


def _summarize(project: Path, receipt: Path) -> dict[str, Any]:
    return MODULE.summarize(
        project_root=project,
        identifiable_dir=project / "artifacts" / "identifiable-maps" / "benchmarks",
        routing_dir=project / "artifacts" / "benchmarks",
        receipt_path=receipt,
        expected_identifiable=len(IDENTIFIABLE_SEEDS) * len(IDENTIFIABLE_ABLATIONS),
        expected_routing=len(ROUTING_SEEDS),
    )


def test_summary_validates_and_aggregates_both_benchmark_families(
    tmp_path: Path,
) -> None:
    project, receipt = _campaign(tmp_path)

    report = _summarize(project, receipt)

    assert report["validation"]["status"] == "passed"
    assert report["validation"]["identifiable_benchmarks_validated"] == 4
    assert report["validation"]["routing_benchmarks_validated"] == 2
    assert report["validation"]["tail_statistics_not_pooled"] is True
    assert report["analysis_provenance"]["shared_git_revision"] == COMMIT
    assert report["analysis_provenance"]["shared_device_name"] == DEVICE_NAME
    assert report["analysis_provenance"]["sealed_receipt"]["launch_fingerprint"] == (
        FINGERPRINT
    )

    routing = report["routing"]
    assert routing["dense_to_routed_median_ratio"]["mean"] == pytest.approx(1.5)
    # The fastest fixed route is half the routed median in every fixture seed.
    assert routing["routed_to_fastest_fixed_median_ratio"]["mean"] == pytest.approx(2.0)
    assert set(routing["fastest_fixed_route_per_seed"].values()) == {"fixed_graph"}
    assert routing["routed_peak_memory_below_dense_in_every_seed"] is True

    identifiable = report["identifiable"]["by_ablation"]
    assert identifiable["combined"]["median_latency_ms"]["mean"] == pytest.approx(0.2750)
    assert identifiable["task_reconstruction"]["median_latency_ms"][
        "mean"
    ] == pytest.approx(0.2760)
    assert identifiable["combined"][
        "peak_cuda_memory_allocated_bytes_identical_across_seeds"
    ]
    contrast = report["identifiable"]["paired_contrast"]
    assert contrast["contrast"] == "combined minus task_reconstruction"
    assert contrast["mean"] == pytest.approx(-0.001)


def test_summary_keeps_the_two_tail_statistics_separate(tmp_path: Path) -> None:
    project, receipt = _campaign(tmp_path)

    report = _summarize(project, receipt)

    protocol = report["measurement_protocol"]
    assert protocol["identifiable"]["tail_statistic"] == "p90"
    assert protocol["routing"]["tail_statistic"] == "p95"
    assert "p90_latency_ms" in report["identifiable"]["by_ablation"]["combined"]
    assert "p95_latency_ms" in report["routing"]["runs"][0]
    # No pooled tail statistic exists anywhere in the emitted record.
    assert "p95" not in json.dumps(report["identifiable"]["by_ablation"])
    assert "p90" not in json.dumps(report["routing"]["runs"])


def test_summary_rejects_a_benchmark_missing_from_the_sealed_receipt(
    tmp_path: Path,
) -> None:
    project, receipt = _campaign(tmp_path)
    document = json.loads(receipt.read_text(encoding="utf-8"))
    document["outputs"] = [
        entry
        for entry in document["outputs"]
        if "routing-confirmatory-v2-s2" not in entry["path"]
    ]
    _write_json(receipt, document)

    with pytest.raises(RuntimeError, match="not listed in the sealed receipt"):
        _summarize(project, receipt)


def test_summary_rejects_a_benchmark_edited_after_sealing(tmp_path: Path) -> None:
    project, receipt = _campaign(tmp_path)
    edited = (
        project / "artifacts" / "identifiable-maps" / "benchmarks" / "gb10-s1-combined.json"
    )
    document = json.loads(edited.read_text(encoding="utf-8"))
    document["latency_ms"]["median"] = 0.0001
    _write_json(edited, document)

    with pytest.raises(RuntimeError, match="differs from the sealed receipt"):
        _summarize(project, receipt)


def test_summary_rejects_an_untrained_routing_benchmark(tmp_path: Path) -> None:
    project, receipt = _campaign(tmp_path)
    path = project / "artifacts" / "benchmarks" / "routing-confirmatory-v2-s1-compute.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["checkpoint"] = None
    _write_json(path, document)
    # Re-seal so the failure is attributable to the missing checkpoint alone.
    receipt_document = json.loads(receipt.read_text(encoding="utf-8"))
    for entry in receipt_document["outputs"]:
        if entry["path"].endswith("routing-confirmatory-v2-s1-compute.json"):
            entry["sha256"] = _sha256(path)
            entry["bytes"] = path.stat().st_size
    _write_json(receipt, receipt_document)

    with pytest.raises(RuntimeError, match="cannot be a trained measurement"):
        _summarize(project, receipt)


def test_summary_rejects_an_identifiable_benchmark_reporting_p95(
    tmp_path: Path,
) -> None:
    project, receipt = _campaign(tmp_path)
    path = (
        project / "artifacts" / "identifiable-maps" / "benchmarks" / "gb10-s2-combined.json"
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    document["latency_ms"]["p95"] = 0.31
    _write_json(path, document)
    receipt_document = json.loads(receipt.read_text(encoding="utf-8"))
    for entry in receipt_document["outputs"]:
        if entry["path"].endswith("gb10-s2-combined.json"):
            entry["sha256"] = _sha256(path)
            entry["bytes"] = path.stat().st_size
    _write_json(receipt, receipt_document)

    with pytest.raises(RuntimeError, match="unexpectedly reports p95"):
        _summarize(project, receipt)


def test_summary_rejects_a_receipt_that_is_not_sealed(tmp_path: Path) -> None:
    project, receipt = _campaign(tmp_path)
    document = json.loads(receipt.read_text(encoding="utf-8"))
    document["status"] = "running"
    _write_json(receipt, document)

    with pytest.raises(RuntimeError, match="not sealed as completed"):
        _summarize(project, receipt)


def test_summary_rejects_a_receipt_with_an_incomplete_step_list(
    tmp_path: Path,
) -> None:
    project, receipt = _campaign(tmp_path)
    document = json.loads(receipt.read_text(encoding="utf-8"))
    document["completed_steps"] = document["completed_steps"][:-1]
    _write_json(receipt, document)

    with pytest.raises(RuntimeError, match="completed steps"):
        _summarize(project, receipt)


def test_summary_rejects_a_checkpoint_that_changed_after_the_benchmark(
    tmp_path: Path,
) -> None:
    project, receipt = _campaign(tmp_path)
    checkpoint = (
        project
        / "artifacts"
        / "identifiable-maps"
        / "campaign"
        / f"seed-{IDENTIFIABLE_SEEDS[0]}"
        / "combined"
        / "checkpoint.pt"
    )
    checkpoint.write_bytes(b"replaced")

    with pytest.raises(RuntimeError, match="checkpoint hash mismatch"):
        _summarize(project, receipt)
