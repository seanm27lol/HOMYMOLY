from __future__ import annotations

import importlib.util
import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_conversion_campaign.py"
SPEC = importlib.util.spec_from_file_location("run_conversion_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_correlation_is_conventional_pearson_not_sample_scaled() -> None:
    left = [float(value) for value in range(9)]
    right = [3.0 * value - 7.0 for value in left]

    # The schema-v1 estimator returned 8/9 here because it averaged products of
    # vectors divided by sample standard deviations. Conventional Pearson is 1.
    assert MODULE._correlation(left, right) == pytest.approx(1.0, abs=1e-15)


def test_correlation_matches_python_reference_for_non_affine_data() -> None:
    left = [-3.0, -1.0, 0.5, 2.0, 4.5, 8.0, 13.0, 21.0, 34.0]
    right = [5.0, 4.0, 7.0, 1.0, 9.0, 2.0, 11.0, 3.0, 15.0]

    assert MODULE._correlation(left, right) == pytest.approx(
        statistics.correlation(left, right), abs=1e-15
    )


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        ([1.0], [1.0], "at least two"),
        ([1.0, 2.0], [1.0], "equal length"),
        ([1.0, 1.0], [2.0, 3.0], "constant input"),
        ([1.0, math.inf], [2.0, 3.0], "finite"),
    ],
)
def test_correlation_rejects_undefined_inputs(
    left: list[float], right: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MODULE._correlation(left, right)


def test_frozen_generator_provenance_is_explicit_and_verified() -> None:
    provenance = MODULE._generator_provenance(PROJECT_ROOT)

    assert provenance == {
        "class": "homymoly.data.conversion.ConversionDataset",
        "path": "src/homymoly/data/conversion.py",
        "sha256": MODULE.FROZEN_GENERATOR_SHA256,
        "frozen_campaign_sha256": MODULE.FROZEN_GENERATOR_SHA256,
        "frozen_campaign_git_revision": MODULE.ORIGINAL_CAMPAIGN_GIT_REVISION,
        "matches_frozen_campaign": True,
    }


def test_generator_hash_mismatch_stops_before_campaign(tmp_path: Path) -> None:
    generator = tmp_path / MODULE.GENERATOR_SOURCE
    generator.parent.mkdir(parents=True)
    generator.write_text("# changed generator\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="conversion generator SHA-256"):
        MODULE._generator_provenance(tmp_path)


def test_correction_record_pins_the_historical_schema_v1_result() -> None:
    correction = MODULE._correction_record()

    assert MODULE.CORRECTED_RESULT_PATH == (
        "results/campaigns/conversion-campaign-v1-corrected.json"
    )
    assert correction["protocol_modified"] is False
    assert correction["data_or_fit_settings_modified"] is False
    assert correction["supersedes"] == {
        "path": "results/campaigns/conversion-campaign-v1.json",
        "schema_version": 1,
        "sha256": MODULE.SUPERSEDED_RESULT_SHA256,
        "runner_sha256": MODULE.SUPERSEDED_RUNNER_SHA256,
        "git_revision": MODULE.ORIGINAL_CAMPAIGN_GIT_REVISION,
    }
    assert MODULE._sha256(PROJECT_ROOT / MODULE.SUPERSEDED_RESULT_PATH) == (
        MODULE.SUPERSEDED_RESULT_SHA256
    )


def test_bonferroni_interval_uses_the_protocol_critical_value() -> None:
    """Regression: schema v1 accidentally used the wider two-sided 99% value."""

    historical = json.loads(
        (PROJECT_ROOT / MODULE.SUPERSEDED_RESULT_PATH).read_text(encoding="utf-8")
    )
    exact_rows = historical["primary"]["exact"]["per_topology"]
    paired = [
        math.log10(row["held_out_mse"] / row["baseline_held_out_mse"])
        for row in exact_rows
    ]

    assert MODULE._T_ADJ[28] == pytest.approx(2.546465223, abs=5e-10)
    assert MODULE._interval(paired, MODULE._T_ADJ) == pytest.approx(
        [-2.749161196, -1.511000534], abs=1e-9
    )
    assert historical["primary"]["exact"]["interval_bonferroni_98_33"] != (
        MODULE._interval(paired, MODULE._T_ADJ)
    )


def test_git_provenance_uses_the_requested_project_root(
    monkeypatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="abc123\n")

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)

    assert MODULE._git(tmp_path, "rev-parse", "HEAD") == "abc123"
    assert observed["command"] == ("git", "rev-parse", "HEAD")
    assert observed["cwd"] == tmp_path


def test_report_uses_schema_v2_and_embeds_generator_provenance(monkeypatch) -> None:
    seeds = tuple(range(10))

    class FakeDataset:
        def __init__(self, _count: int, *, seed: int, dtype: object) -> None:
            del dtype
            self.sample = SimpleNamespace(
                seed=seed,
                sample_id=f"fake-{seed}",
                num_vertices=8,
                num_edges=10,
                num_faces=3,
            )

        def __getitem__(self, index: int) -> SimpleNamespace:
            assert index == 0
            return self.sample

    def fake_fit(
        sample: SimpleNamespace, term: str | None, weight: float
    ) -> tuple[float, float]:
        seed_scale = 1.0 + sample.seed / 1000.0
        baseline = 1.0 + sample.seed / 100.0
        if term is None:
            return baseline, seed_scale
        term_scale = {"exact": 0.02, "cone": 0.03, "rtd": 0.04}[term]
        held_out = baseline * math.exp(term_scale * weight * seed_scale)
        violation = math.exp(-weight * seed_scale)
        return held_out, violation

    def fake_routing(
        sample: SimpleNamespace, weight: float
    ) -> tuple[float, float, float]:
        defect = 1.0 / (1.0 + weight) + sample.seed / 1000.0
        return defect, 1.0 + sample.seed / 100.0, 2.0 + sample.seed / 100.0

    monkeypatch.setattr(MODULE, "SEEDS", seeds)
    monkeypatch.setattr(MODULE, "MIN_ELIGIBLE", len(seeds))
    monkeypatch.setattr(MODULE, "ConversionDataset", FakeDataset)
    monkeypatch.setattr(MODULE, "_fit", fake_fit)
    monkeypatch.setattr(MODULE, "_routing_trial", fake_routing)

    report = MODULE.run(PROJECT_ROOT)

    assert report["schema_version"] == 2
    assert report["schema"] == {
        "name": "homymoly.conversion-campaign-result",
        "version": 2,
        "record_id": "conversion-campaign-v1-correction-1",
    }
    assert report["campaign"] == "conversion-campaign-v1"
    assert report["correction"]["id"] == "conversion-campaign-analysis-correction-1"
    assert {reason["id"] for reason in report["correction"]["reasons"]} == {
        "c1-pearson-normalisation",
        "bonferroni-critical-value",
    }
    assert report["correction"]["decision_changes"] == {
        "exact": False,
        "cone": False,
        "rtd": False,
        "c1": False,
    }
    assert report["protocol"]["document_hash_matches_frozen"] is True
    assert report["protocol"]["execution_matches_frozen_text"] is False
    assert report["protocol"]["implementation_deviations"][0]["id"] == (
        "compatibility-mean-normalisation"
    )
    assert report["provenance"]["generator"]["matches_frozen_campaign"] is True
    assert report["design"]["held_out_pairs"] == MODULE.N_HELD_OUT
    assert report["design"]["observation_noise_standard_deviation"] == MODULE.NOISE
    assert report["design"]["fit_scope"].startswith("one independently trained")
    assert report["design"]["free_parameters"]["median"] == 30
    assert report["c1"]["inference_role"].startswith("prespecified secondary")
    first_c1 = report["c1"]["per_topology"][0]
    assert first_c1["seed"] == seeds[0]
    assert [fit["weight"] for fit in first_c1["fits"]] == list(MODULE.C1_WEIGHTS)
    assert first_c1["correlation"] == pytest.approx(
        MODULE._correlation(
            [
                math.log10(fit["boundary_compatibility_defect_frobenius"])
                for fit in first_c1["fits"]
            ],
            [math.log10(fit["held_out_mse"]) for fit in first_c1["fits"]],
        )
    )
    assert (
        report["design"]["objective_definitions"]["cone"]["is_mapping_cone_homology"]
        is False
    )
    assert (
        report["design"]["objective_definitions"]["rtd"][
            "is_representation_topology_divergence"
        ]
        is False
    )
    assert report["routing"]["decision_informative"] is False
    for term in report["primary"].values():
        assert set(term["sensitivity_sign_test"]) == {
            "pvalue_two_sided",
            "negative",
            "positive",
            "ties_discarded",
        }
    assert all(
        value >= 0.0
        for value in (
            math.log10(
                (
                    row["cell_error"]
                    if row["defect"] <= report["routing"]["threshold"]
                    else row["graph_error"]
                )
                / min(row["cell_error"], row["graph_error"])
            )
            for row in report["routing"]["trials"]
            if row["split"] == "evaluation"
        )
    )
