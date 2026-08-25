from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "export_publication_evidence.py"
PROJECT_ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("export_publication_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


COMMIT = "8021292e97abfec91768f1b5437c883a42c29c60"


def _corruption_report() -> dict[str, object]:
    return {
        "schema_version": 3,
        "checkpoint_sha256": "a" * 64,
        "git": {"commit": COMMIT, "status": ""},
        "analysis": {"transport_rotation": {"batch_observations": 2}},
        "per_batch": [
            {"block_id": "transport_rotation:0000", "damage_rate": 0.0},
            {"block_id": "transport_rotation:0001", "damage_rate": 0.5},
        ],
        "per_example": [{"sample_id": index} for index in range(64)],
    }


def _project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    (project / "artifacts" / "gate3g" / "run").mkdir(parents=True)
    (project / "results" / "summaries").mkdir(parents=True)
    (
        project / "artifacts" / "gate3g" / "run" / "corruption_report_final.json"
    ).write_text(
        json.dumps(_corruption_report(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (project / "artifacts" / "summary.json").write_text(
        json.dumps({"mean": 0.5, "shared_git_revision": COMMIT}, sort_keys=True),
        encoding="utf-8",
    )
    (project / "results" / "summaries" / "generated.json").write_text(
        json.dumps(
            {
                "validated_pairs": 8,
                "analysis_provenance": {"git_commit": COMMIT},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return project, project / "results"


def _specs() -> list[object]:
    return [
        MODULE.Spec(
            "results/summaries/generated.json",
            "summaries/generated.json",
            "compact-summary",
            "in-place",
            "Summary generated before export.",
        ),
        MODULE.Spec(
            "artifacts/summary.json",
            "summaries/copied.json",
            "endpoint-table",
            "copy",
            "Copied endpoint table.",
        ),
        MODULE.Spec(
            "artifacts/gate3g/run/corruption_report_final.json",
            "gate3g/run/corruption_report_final.compact.json",
            "corruption-report-derivative",
            "derive-corruption-report",
            "Compact corruption derivative.",
        ),
    ]


def test_compact_derivative_drops_per_example_and_keeps_the_analysis_unit() -> None:
    compact = MODULE.compact_corruption_report(_corruption_report())

    assert "per_example" not in compact
    assert len(compact["per_batch"]) == 2
    assert compact["analysis"] == {"transport_rotation": {"batch_observations": 2}}
    assert compact["checkpoint_sha256"] == "a" * 64
    assert compact["_derivative"]["dropped_keys"] == {"per_example": {"rows": 64}}
    assert compact["_derivative"]["retained_per_batch_rows"] == 2


def test_compact_derivative_refuses_a_report_without_batch_rows() -> None:
    document = _corruption_report()
    del document["per_batch"]

    with pytest.raises(ValueError, match="no per_batch rows"):
        MODULE.compact_corruption_report(document)


def test_export_writes_a_checksummed_manifest_for_every_allowlisted_file(
    tmp_path: Path,
) -> None:
    project, output = _project(tmp_path)

    manifest = MODULE.export(project_root=project, output_root=output, specs=_specs())

    assert manifest["schema_version"] == MODULE.SCHEMA_VERSION
    assert manifest["summary"]["files"] == 3
    assert manifest["summary"]["by_kind"] == {
        "compact-summary": 1,
        "corruption-report-derivative": 1,
        "endpoint-table": 1,
    }
    assert manifest["generating_command"][1].endswith("export_publication_evidence.py")
    # Provenance is read out of the evidence, never from the exporter's own HEAD.
    assert manifest["source_commits"] == {COMMIT: 3}

    entries = {entry["path"]: entry for entry in manifest["files"]}
    derivative = output / "gate3g" / "run" / "corruption_report_final.compact.json"
    assert "per_example" not in json.loads(derivative.read_text(encoding="utf-8"))
    recorded = entries["gate3g/run/corruption_report_final.compact.json"]
    assert recorded["sha256"] == hashlib.sha256(derivative.read_bytes()).hexdigest()
    assert recorded["bytes"] == derivative.stat().st_size
    assert recorded["derivation"] == "per-batch-lossless-v1"
    # The untruncated source stays pinned even though it is not exported.
    source = project / "artifacts" / "gate3g" / "run" / "corruption_report_final.json"
    assert recorded["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert recorded["source_bytes"] == source.stat().st_size

    # Files already sitting at their destination are hashed, never rewritten.
    assert entries["summaries/generated.json"]["source"] == (
        "results/summaries/generated.json"
    )


def test_export_is_byte_identical_when_the_evidence_is_unchanged(
    tmp_path: Path,
) -> None:
    project, output = _project(tmp_path)

    MODULE.export(project_root=project, output_root=output, specs=_specs())
    first = (output / MODULE.MANIFEST_NAME).read_bytes()
    MODULE.export(project_root=project, output_root=output, specs=_specs())
    second = (output / MODULE.MANIFEST_NAME).read_bytes()

    assert first == second


def test_export_refuses_a_checkpoint_even_when_a_specification_asks_for_it(
    tmp_path: Path,
) -> None:
    project, output = _project(tmp_path)
    (project / "artifacts" / "model.pt").write_bytes(b"weights")
    specs = [
        MODULE.Spec(
            "artifacts/model.pt",
            "checkpoints/model.pt",
            "benchmark-summary",
            "copy",
            "Should never be exportable.",
        )
    ]

    with pytest.raises(ValueError, match="excluded suffix"):
        MODULE.export(project_root=project, output_root=output, specs=specs)


def test_export_refuses_a_source_under_an_excluded_directory(tmp_path: Path) -> None:
    project, output = _project(tmp_path)
    (project / "artifacts" / "run" / "checkpoints").mkdir(parents=True)
    (project / "artifacts" / "run" / "checkpoints" / "note.json").write_text(
        "{}", encoding="utf-8"
    )
    specs = [
        MODULE.Spec(
            "artifacts/run/checkpoints/note.json",
            "gate3/note.json",
            "gate-decision",
            "copy",
            "Should never be exportable.",
        )
    ]

    with pytest.raises(ValueError, match="excluded directory"):
        MODULE.export(project_root=project, output_root=output, specs=specs)


def test_export_refuses_a_file_over_the_per_file_cap(tmp_path: Path) -> None:
    project, output = _project(tmp_path)

    with pytest.raises(ValueError, match="per-file cap"):
        MODULE.export(
            project_root=project,
            output_root=output,
            specs=_specs(),
            max_file_bytes=8,
        )


def test_export_fails_loudly_when_required_evidence_is_missing(tmp_path: Path) -> None:
    project, output = _project(tmp_path)
    (project / "artifacts" / "summary.json").unlink()

    with pytest.raises(FileNotFoundError, match="required evidence is missing"):
        MODULE.export(project_root=project, output_root=output, specs=_specs())


def test_export_refuses_evidence_with_no_recorded_generating_commit(
    tmp_path: Path,
) -> None:
    project, output = _project(tmp_path)
    (project / "artifacts" / "summary.json").write_text(
        json.dumps({"mean": 0.5}, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="no recorded generating commit"):
        MODULE.export(project_root=project, output_root=output, specs=_specs())


def test_export_records_more_than_one_generating_commit_when_campaigns_differ(
    tmp_path: Path,
) -> None:
    project, output = _project(tmp_path)
    earlier = "e69b07707950b6abe332366c51fe8c94254899f3"
    (project / "artifacts" / "summary.json").write_text(
        json.dumps({"mean": 0.5, "shared_git_revision": earlier}, sort_keys=True),
        encoding="utf-8",
    )

    manifest = MODULE.export(project_root=project, output_root=output, specs=_specs())

    assert manifest["source_commits"] == {earlier: 1, COMMIT: 2}


def test_verify_detects_a_tampered_bundle_file(tmp_path: Path) -> None:
    project, output = _project(tmp_path)
    MODULE.export(project_root=project, output_root=output, specs=_specs())
    (output / "summaries" / "copied.json").write_text('{"mean": 0.9}', encoding="utf-8")

    problems = MODULE.verify(project, output)

    assert any(
        "hash mismatch: summaries/copied.json" in problem for problem in problems
    )


def test_verify_detects_an_unlisted_file_and_a_changed_source(tmp_path: Path) -> None:
    project, output = _project(tmp_path)
    MODULE.export(project_root=project, output_root=output, specs=_specs())
    (output / "stray.json").write_text("{}", encoding="utf-8")
    (project / "artifacts" / "summary.json").write_text(
        json.dumps({"mean": 0.9}, sort_keys=True), encoding="utf-8"
    )

    problems = MODULE.verify(project, output)

    assert any("unlisted file in bundle: stray.json" in problem for problem in problems)
    assert any("source changed since export" in problem for problem in problems)


def test_the_real_allowlist_names_only_publishable_evidence() -> None:
    specs = MODULE.specifications()

    assert len(specs) == 50
    for spec in specs:
        assert Path(spec.source).suffix == ".json"
        assert Path(spec.destination).suffix == ".json"
        MODULE._reject_denied(Path(spec.source), "source")
        MODULE._reject_denied(Path(spec.destination), "destination")
    kinds = {spec.kind for spec in specs}
    assert kinds == {
        "benchmark-summary",
        "compact-summary",
        "corruption-report-derivative",
        "endpoint-table",
        "gate-decision",
    }
    conversion = {
        spec.destination: spec
        for spec in specs
        if spec.destination.startswith("campaigns/")
    }
    assert set(conversion) == {
        "campaigns/conversion-campaign-v1.json",
        "campaigns/conversion-campaign-v1-corrected.json",
    }
    assert (
        "Historical" in conversion["campaigns/conversion-campaign-v1.json"].description
    )
    assert (
        "Canonical corrected"
        in conversion["campaigns/conversion-campaign-v1-corrected.json"].description
    )
    # Every gauge seed contributes both trained runs and exactly one comparison.
    derivatives = [spec for spec in specs if spec.destination.startswith("gate3g/")]
    assert (
        sum(spec.kind == "corruption-report-derivative" for spec in derivatives) == 16
    )
    assert sum(spec.kind == "gate-decision" for spec in derivatives) == 8


def test_canonical_conversion_correction_proves_its_frozen_lineage() -> None:
    corrected = PROJECT_ROOT / MODULE.CORRECTED_CONVERSION_RESULT
    if not corrected.is_file():
        pytest.skip("corrected conversion campaign has not landed yet")

    document = json.loads(corrected.read_text(encoding="utf-8"))

    MODULE.validate_corrected_conversion_record(PROJECT_ROOT, document)

    document["correction"]["supersedes"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="historical result SHA-256"):
        MODULE.validate_corrected_conversion_record(PROJECT_ROOT, document)


def test_primary_interval_validator_uses_df28_bonferroni_quantile() -> None:
    historical = json.loads(
        (PROJECT_ROOT / MODULE.HISTORICAL_CONVERSION_RESULT).read_text(encoding="utf-8")
    )
    corrected = json.loads(json.dumps(historical))
    for term in ("exact", "cone", "rtd"):
        rows = corrected["primary"][term]["per_topology"]
        ratios = [
            math.log10(row["held_out_mse"] / row["baseline_held_out_mse"])
            for row in rows
        ]
        interval = MODULE._student_t_interval(ratios, MODULE.BONFERRONI_T_DF28)
        corrected["primary"][term]["interval_bonferroni_98_33"] = interval
        corrected["primary"][term]["improves_confirmatory"] = interval[1] < 0.0
        corrected["primary"][term]["harms_confirmatory"] = interval[0] > 0.0

    MODULE._validate_corrected_primary_intervals(corrected, historical)

    assert corrected["primary"]["exact"]["interval_bonferroni_98_33"] == (
        pytest.approx([-2.749161196, -1.511000534], abs=1e-9)
    )
    assert (
        corrected["primary"]["exact"]["interval_bonferroni_98_33"]
        != (historical["primary"]["exact"]["interval_bonferroni_98_33"])
    )
    corrected["primary"]["cone"]["interval_bonferroni_98_33"][0] -= 0.01
    with pytest.raises(ValueError, match="df=28"):
        MODULE._validate_corrected_primary_intervals(corrected, historical)


def test_c1_validator_recomputes_pearson_from_seed_keyed_raw_fits() -> None:
    weights = [float(index) for index in range(9)]
    rows = []
    correlations = []
    for seed in range(29):
        fits = [
            {
                "weight": weight,
                "held_out_mse": 10.0 ** ((index + 1) * (1.0 + seed / 1000.0)),
                "boundary_compatibility_defect_frobenius": 10.0 ** (index + 1),
            }
            for index, weight in enumerate(weights)
        ]
        correlation = MODULE._pearson(
            [
                math.log10(fit["boundary_compatibility_defect_frobenius"])
                for fit in fits
            ],
            [math.log10(fit["held_out_mse"]) for fit in fits],
        )
        correlations.append(correlation)
        rows.append({"seed": seed, "correlation": correlation, "fits": fits})
    interval = MODULE._student_t_interval(correlations, MODULE.T95_DF28)
    corrected = {
        "primary": {"exact": {"per_topology": [{"seed": seed} for seed in range(29)]}},
        "c1": {
            "weights_swept": weights,
            "n": 29,
            "mean_within_topology_correlation": sum(correlations) / 29,
            "interval_95": interval,
            "positive_topologies": 29,
            "supported": True,
            "supported_claim": (
                "positive within-seed regularization-path association only"
            ),
            "does_not_establish": [
                "independent predictive information",
                "off-path calibration",
                "causal effect",
            ],
            "per_topology_correlation": correlations,
            "per_topology": rows,
        },
    }
    historical = {"c1": {"supported": True}}

    MODULE._validate_corrected_c1(corrected, historical)

    corrected["c1"]["per_topology"][0]["correlation"] = 0.5
    with pytest.raises(ValueError, match="Pearson r"):
        MODULE._validate_corrected_c1(corrected, historical)


def test_h5_validator_recomputes_row_naive_and_topology_clustered_intervals() -> None:
    historical = json.loads(
        (PROJECT_ROOT / MODULE.HISTORICAL_CONVERSION_RESULT).read_text(encoding="utf-8")
    )
    corrected = json.loads(json.dumps(historical))
    routing = corrected["routing"]
    evaluation = [row for row in routing["trials"] if row["split"] == "evaluation"]
    values = []
    clustered: dict[int, list[float]] = {}
    selected_lower = 0
    for row in evaluation:
        routed = (
            row["cell_error"]
            if row["defect"] <= routing["threshold"]
            else row["graph_error"]
        )
        best = min(row["cell_error"], row["graph_error"])
        selected_lower += routed == best
        value = math.log10(routed / best)
        values.append(value)
        clustered.setdefault(row["seed"], []).append(value)
    cluster_means = [sum(clustered[seed]) / 2 for seed in sorted(clustered)]
    historical_interval = routing.pop("interval_95")
    routing.update(
        {
            "decision_informative": False,
            "historical_pseudoreplicated_interval_95": historical_interval,
            "topology_clustered_descriptive_interval_95": MODULE._student_t_interval(
                cluster_means, MODULE.T95_DF13, expected_n=14
            ),
            "topology_cluster_summaries": [
                {
                    "seed": seed,
                    "n_correlated_fits": 2,
                    "mean_log10_ratio": sum(clustered[seed]) / 2,
                    "endpoint_values": clustered[seed],
                }
                for seed in sorted(clustered)
            ],
            "supported": None,
            "decision": "withdrawn-non-informative",
            "evaluation_topology_clusters": 14,
            "selected_lower_error_rows": selected_lower,
        }
    )

    MODULE._validate_withdrawn_routing(corrected, historical)

    routing["trials"][1]["graph_error"] += 1e-14
    MODULE._validate_withdrawn_routing(corrected, historical)
    routing["trials"][1]["graph_error"] += 1e-6
    with pytest.raises(ValueError, match="routing raw trials changed"):
        MODULE._validate_withdrawn_routing(corrected, historical)
    routing["trials"][1]["graph_error"] -= 1e-6 + 1e-14

    assert routing["historical_pseudoreplicated_interval_95"] == pytest.approx(
        MODULE._student_t_interval(values, MODULE.T95_DF27, expected_n=28)
    )
    routing["topology_clustered_descriptive_interval_95"][1] += 0.1
    with pytest.raises(ValueError, match="topology-clustered"):
        MODULE._validate_withdrawn_routing(corrected, historical)


def test_design_validator_recomputes_canonical_scarce_probe_geometry() -> None:
    dimensions = [
        {
            "seed": seed,
            "edges": 16 if seed < 8 else 23,
            "faces": 11 if seed < 24 else 17,
        }
        for seed in range(29)
    ]
    document = {
        "design": {
            "training_pairs": 16,
            "held_out_pairs": 3072,
            "training_label_noise": {
                "distribution": "independent zero-mean Gaussian",
                "standard_deviation": 0.02,
            },
            "held_out_targets": "noiseless ground-truth linear responses",
            "primary_inference_unit": (
                "one eligible generator seed, jointly determining topology, training "
                "predictors and noise, and noiseless held-out predictors"
            ),
            "exchangeability_assumption": (
                "eligible seed-level joint replicates are exchangeable for the Student-t "
                "interval; the design does not separate topology heterogeneity from "
                "data/noise-realisation heterogeneity"
            ),
            "eligible_topology_dimensions": dimensions,
            "scarce_probe_geometry": {
                "training_pairs": 16,
                "median_edges_per_output_row": 23,
                "median_cycle_subspace_dimension": 11,
                "full_row_regressions_with_edges_gt_training_pairs": 21,
                "cycle_subspaces_with_faces_le_training_pairs": 24,
                "seeds_moving_from_edges_gt_n_to_faces_le_n": 16,
            },
            "objective_definitions": {
                "exact": {
                    "display_name": "boundary-compatibility penalty",
                    "formula": "mean((B1 @ W.T)^2)",
                    "frozen_key_is_historical_shorthand": True,
                    "is_exactness_of_a_sequence": False,
                    "structural_side_information": (
                        "B1 determines the target cycle subspace ker(B1); the penalty "
                        "does not directly use B2 or response labels, but the committed "
                        "deterministic generator algorithm can recover its noncanonical "
                        "B2 basis from the graph"
                    ),
                },
                "rtd": {
                    "is_representation_topology_divergence": False,
                    "target_alignment": (
                        "the generated truth discards cut-space directions while this "
                        "surrogate asks the lower-dimensional output to preserve the "
                        "full source distance geometry"
                    ),
                },
            },
        }
    }

    MODULE._validate_design_audit(document)

    document["design"]["scarce_probe_geometry"][
        "full_row_regressions_with_edges_gt_training_pairs"
    ] = 20
    with pytest.raises(ValueError, match="scarce-probe geometry"):
        MODULE._validate_design_audit(document)


def test_environment_validator_pins_lock_versions_cpu_and_clean_runner() -> None:
    actual = {
        **MODULE.EXPECTED_CAMPAIGN_ENVIRONMENT,
        "torch": "2.13.0+cu130",
    }
    document = {
        "provenance": {
            "git_status": "",
            "runner_sha256": MODULE._sha256(
                PROJECT_ROOT / "scripts" / "run_conversion_campaign.py"
            ),
            "dependencies": {
                key: actual[key] for key in ("torch", "networkx", "numpy")
            },
            "execution": {"tensor_device": "cpu", "cuda_visible_devices": ""},
            "environment": {
                "actual": actual,
                "expected": MODULE.EXPECTED_CAMPAIGN_ENVIRONMENT,
                "matches_expected": True,
                "lockfile": {
                    "path": MODULE.FROZEN_LOCKFILE_PATH,
                    "sha256": MODULE.FROZEN_LOCKFILE_SHA256,
                    "frozen_campaign_sha256": MODULE.FROZEN_LOCKFILE_SHA256,
                },
            },
        }
    }

    MODULE._validate_locked_environment(PROJECT_ROOT, document)

    document["provenance"]["environment"]["matches_expected"] = False
    with pytest.raises(ValueError, match="locked environment"):
        MODULE._validate_locked_environment(PROJECT_ROOT, document)
