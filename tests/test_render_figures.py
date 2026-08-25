from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from xml.etree import ElementTree

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "render_figures.py"
SPEC = importlib.util.spec_from_file_location("render_figures", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

RESULTS = Path(__file__).parents[1] / "results"


def _summary() -> dict[str, object]:
    return {
        "frozen_design": {
            "ablations": ["task_only", "cone_only", "combined", "rtd_only"],
            "chance_baselines": {"transformation_accuracy": 0.0833},
        },
        "by_ablation": {
            name: {
                "endpoints": {
                    "transformation_accuracy": {"mean": accuracy},
                    "map_mse": {"mean": mse},
                }
            }
            for name, accuracy, mse in (
                ("task_only", 1.0, 2.6e-17),
                ("cone_only", 0.0815, 0.109),
                ("combined", 1.0, 1.7e-16),
                ("rtd_only", 0.0833, 0.191),
            )
        },
    }


def test_recovery_figure_is_well_formed_and_reads_its_values_from_the_summary() -> None:
    markup = MODULE.figure_recovery(_summary())

    root = ElementTree.fromstring(markup)
    assert root.tag.endswith("svg")
    # Every declared ablation appears, and none is silently dropped.
    for name in ("task_only", "cone_only", "combined", "rtd_only"):
        assert MODULE.ABLATION_LABELS[name] in markup
    assert "cone proxy only" in markup
    assert "RTD-style surrogate only" in markup
    # The chance annotation is read from the summary, not hardcoded.
    assert "chance 0.0833" in markup
    # Structure-only controls are drawn in the second categorical slot.
    assert MODULE.SERIES[1] in markup


def test_recovery_figure_groups_controls_after_supervised_objectives() -> None:
    markup = MODULE.figure_recovery(_summary())

    positions = {
        name: markup.index(f">{MODULE.ABLATION_LABELS[name]}<")
        for name in ("task_only", "combined", "cone_only", "rtd_only")
    }
    assert positions["task_only"] < positions["combined"]
    assert positions["combined"] < positions["cone_only"]
    assert positions["cone_only"] < positions["rtd_only"]


def test_long_bars_label_inside_so_the_value_never_clips() -> None:
    inside = MODULE._value_label(0.0, 90.0, 100.0, 0.0, "1.000")
    outside = MODULE._value_label(0.0, 10.0, 100.0, 0.0, "0.08")

    assert 'text-anchor="end"' in inside[0]
    assert MODULE.SURFACE in inside[0]
    assert 'text-anchor="start"' in outside[0]


def test_every_generated_figure_parses_and_declares_an_accessible_label() -> None:
    if not (RESULTS / "MANIFEST.json").is_file():
        pytest.skip("tracked evidence bundle is not present")
    gate3 = json.loads(
        (RESULTS / "gate3" / "paired_comparison_final.json").read_text(encoding="utf-8")
    )
    gauge = json.loads(
        (RESULTS / "summaries" / "gauge-corruption-campaign.json").read_text(
            encoding="utf-8"
        )
    )
    compute = json.loads(
        (RESULTS / "summaries" / "compute-campaign.json").read_text(encoding="utf-8")
    )
    identifiable = json.loads(
        (RESULTS / "summaries" / "identifiable-campaign-summary.json").read_text(
            encoding="utf-8"
        )
    )

    for markup in (
        MODULE.figure_recovery(identifiable),
        MODULE.figure_contrasts(gate3, gauge),
        MODULE.figure_compute(compute),
    ):
        root = ElementTree.fromstring(markup)
        assert root.attrib.get("role") == "img"
        assert root.attrib.get("aria-label")
        assert root.find("{http://www.w3.org/2000/svg}title") is not None


def test_contrast_figure_plots_every_gate3_and_gauge_contrast() -> None:
    if not (RESULTS / "MANIFEST.json").is_file():
        pytest.skip("tracked evidence bundle is not present")
    gate3 = json.loads(
        (RESULTS / "gate3" / "paired_comparison_final.json").read_text(encoding="utf-8")
    )
    gauge = json.loads(
        (RESULTS / "summaries" / "gauge-corruption-campaign.json").read_text(
            encoding="utf-8"
        )
    )

    markup = MODULE.figure_contrasts(gate3, gauge)

    expected = sum(len(entry["by_kind"]) for entry in gate3["comparisons"])
    expected += len(gauge["by_kind"])
    assert markup.count("<circle") == expected


def test_conversion_campaign_names_proxy_objectives_as_surrogates() -> None:
    campaign = {
        "design": {"eligible_topologies": 2, "training_pairs": 16},
        "primary": {
            name: {
                "interval_bonferroni_98_33": interval,
                "interval_95": interval,
                "mean_log10_ratio": sum(interval) / 2,
                "improves_confirmatory": False,
                "harms_confirmatory": False,
            }
            for name, interval in (
                ("exact", [-0.3, 0.1]),
                ("cone", [-0.1, 0.2]),
                ("rtd", [-0.2, 0.3]),
            )
        },
        "routing": {
            "decision": "H5-WITHDRAWN-SENTINEL",
            "historical_pseudoreplicated_interval_95": [-999.0, 999.0],
            "topology_clustered_descriptive_interval_95": [-998.0, 998.0],
        },
    }

    markup = MODULE.figure_campaign(campaign)

    assert "singular-value cone surrogate" in markup
    assert "RTD-inspired distance surrogate" in markup
    assert "exp(-2·σ_min(W)); not mapping-cone homology" in markup
    assert "normalized pairwise-distance MSE" in markup
    assert "Boundary compatibility improves edge-to-cycle lifting" in markup
    assert "B₁Wᵀ = 0 (frozen key: exact)" in markup
    assert ">exact<" not in markup
    assert "no detected improvement" in markup
    assert ">inert<" not in markup
    assert (
        "eligible seed jointly fixes topology, predictors, and training noise" in markup
    )
    assert "same-family replication" in markup
    assert "Locked prospectively after outcome-informed weight selection" in markup
    assert "one execution deviation disclosed" in markup
    assert "preregistered" not in markup
    assert "one value per topology" not in markup
    assert "learned conversion" not in markup
    assert "H5-WITHDRAWN-SENTINEL" not in markup
    assert "-999" not in markup


def test_committed_figures_match_a_fresh_render(tmp_path: Path) -> None:
    """The tracked SVGs must be regenerable, so a stale figure cannot ship."""

    figures = Path(__file__).parents[1] / "docs" / "figures"
    corrected = RESULTS / "campaigns" / "conversion-campaign-v1-corrected.json"
    if (
        not (RESULTS / "MANIFEST.json").is_file()
        or not figures.is_dir()
        or not corrected.is_file()
    ):
        pytest.skip(
            "tracked evidence, corrected campaign, or figure directory is not present"
        )

    MODULE.main(["--results-root", str(RESULTS), "--output-dir", str(tmp_path)])

    for name in (
        "fig-recovery.svg",
        "fig-contrasts.svg",
        "fig-compute.svg",
        "fig-campaign.svg",
    ):
        assert (tmp_path / name).read_text(encoding="utf-8") == (
            figures / name
        ).read_text(encoding="utf-8"), (
            f"{name} is stale; re-run scripts/render_figures.py"
        )
