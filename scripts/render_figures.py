#!/usr/bin/env python3
"""Render the paper's data figures as SVG from the tracked evidence bundle.

Every value plotted here is read from ``results/`` rather than typed in, so a
figure cannot drift from the machine-readable record it illustrates. The output
is plain SVG: vector, dependency-free, diffable, and readable by the Markdown
renderer and by GitHub alike.

Figures are drawn on an explicit white surface because they are print artifacts
embedded in a PDF; they do not follow a viewer theme.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Categorical slots, assigned in fixed order, validated for light surface #ffffff:
# worst adjacent CVD dE 9.2 (deutan), worst adjacent normal-vision dE 27.6.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#7a7873"
GRID = "#e3e2df"
AXIS = "#b8b6b1"
SURFACE = "#ffffff"
FONT = "Arial, Helvetica, sans-serif"
MONO = "Consolas, 'DejaVu Sans Mono', monospace"


def _escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _text(
    x: float,
    y: float,
    content: str,
    *,
    size: float = 11.0,
    fill: str = INK,
    anchor: str = "start",
    weight: str = "normal",
    family: str = FONT,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" '
        f'font-size="{size:.1f}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}">{_escape(content)}</text>'
    )


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str,
    width: float = 1.0,
    dash: str | None = None,
) -> str:
    dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash_attribute}/>'
    )


def _rect(
    x: float, y: float, width: float, height: float, *, fill: str, radius: float = 0.0
) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(width, 0.0):.1f}" '
        f'height="{height:.1f}" rx="{radius}" ry="{radius}" fill="{fill}"/>'
    )


def _document(width: float, height: float, body: Iterable[str], title: str) -> str:
    parts = "\n  ".join(body)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
        f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'role="img" aria-label="{_escape(title)}">\n'
        f"  <title>{_escape(title)}</title>\n"
        f'  <rect width="{width:.0f}" height="{height:.0f}" fill="{SURFACE}"/>\n'
        f"  {parts}\n</svg>\n"
    )


def _legend(x: float, y: float, entries: list[tuple[str, str]]) -> list[str]:
    marks: list[str] = []
    cursor = x
    for colour, label in entries:
        marks.append(_rect(cursor, y - 7, 9, 9, fill=colour, radius=2))
        marks.append(_text(cursor + 14, y, label, size=10.5, fill=INK_SECONDARY))
        cursor += 20 + 6.6 * len(label)
    return marks


# --------------------------------------------------------------------------
# Figure: exact recovery by ablation
# --------------------------------------------------------------------------

NO_TASK_OR_RECONSTRUCTION = {"cone_only", "rtd_only"}
ABLATION_LABELS = {
    "task_only": "task only",
    "reconstruction_only": "reconstruction only",
    "task_reconstruction": "task + reconstruction",
    "task_reconstruction_cone": "task + recon + cone proxy",
    "task_reconstruction_rtd": "task + recon + RTD-style",
    "cone_only": "cone proxy only",
    "rtd_only": "RTD-style surrogate only",
    "combined": "task + recon + both proxies",
}


def _value_label(
    origin: float, span: float, panel_width: float, y: float, label: str
) -> list[str]:
    """Place a bar's value label outside the bar, or inside when it would clip."""

    if span > panel_width * 0.70:
        return [
            _text(
                origin + span - 6,
                y + 12,
                label,
                size=9.5,
                fill=SURFACE,
                anchor="end",
            )
        ]
    return [_text(origin + span + 6, y + 12, label, size=9.5, fill=INK_SECONDARY)]


def figure_recovery(summary: dict[str, Any]) -> str:
    declared = summary["frozen_design"]["ablations"]
    # Grouped for legibility: supervised objectives first, then the two
    # identifiability controls. Within each group the frozen registry order is
    # preserved, and every declared ablation is shown.
    order = [name for name in declared if name not in NO_TASK_OR_RECONSTRUCTION]
    order += [name for name in declared if name in NO_TASK_OR_RECONSTRUCTION]
    chance = float(
        summary["frozen_design"]["chance_baselines"]["transformation_accuracy"]
    )
    rows = [
        (
            name,
            float(
                summary["by_ablation"][name]["endpoints"]["transformation_accuracy"][
                    "mean"
                ]
            ),
            float(summary["by_ablation"][name]["endpoints"]["map_mse"]["mean"]),
        )
        for name in order
    ]

    width, height = 680.0, 342.0
    label_width = 198.0
    gap = 46.0
    panel_width = (width - label_width - gap - 46.0) / 2.0
    top = 80.0
    row_height = 26.0
    bar_height = 13.0

    left_x = label_width
    right_x = label_width + panel_width + gap

    body: list[str] = [
        _text(0, 20, "Recovery by training objective", size=13, weight="bold"),
        _text(
            0,
            36,
            "Mean over five seeds. Objectives without task/reconstruction sit at chance.",
            size=10.5,
            fill=INK_SECONDARY,
        ),
        _text(left_x, 54, "transformation accuracy", size=10.5, fill=INK_SECONDARY),
        _text(right_x, 54, "map MSE (log scale)", size=10.5, fill=INK_SECONDARY),
    ]

    # Left panel: accuracy on a linear 0..1 axis.
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left_x + fraction * panel_width
        body.append(_line(x, top, x, top + len(rows) * row_height, stroke=GRID))
        body.append(
            _text(
                x,
                top + len(rows) * row_height + 14,
                f"{fraction:g}",
                size=9.5,
                fill=INK_MUTED,
                anchor="middle",
            )
        )

    # Right panel: log10 map MSE.
    exponents = [-17, -13, -9, -5, -1]
    lo, hi = float(exponents[0]), float(exponents[-1])
    for exponent in exponents:
        x = right_x + (exponent - lo) / (hi - lo) * panel_width
        body.append(_line(x, top, x, top + len(rows) * row_height, stroke=GRID))
        body.append(
            _text(
                x,
                top + len(rows) * row_height + 14,
                f"1e{exponent}",
                size=9.5,
                fill=INK_MUTED,
                anchor="middle",
            )
        )

    for index, (name, accuracy, mse) in enumerate(rows):
        y = top + index * row_height
        colour = SERIES[1] if name in NO_TASK_OR_RECONSTRUCTION else SERIES[0]
        body.append(
            _text(
                label_width - 10,
                y + 11,
                ABLATION_LABELS.get(name, name),
                size=9.5,
                anchor="end",
            )
        )
        accuracy_span = accuracy * panel_width
        body.append(
            _rect(left_x, y + 2, accuracy_span, bar_height, fill=colour, radius=3)
        )
        body.extend(
            _value_label(
                left_x,
                accuracy_span,
                panel_width,
                y,
                f"{accuracy:.4f}".rstrip("0").rstrip(".") if accuracy < 1 else "1.000",
            )
        )
        exponent = math.log10(mse)
        span = max((exponent - lo) / (hi - lo), 0.0) * panel_width
        body.append(_rect(right_x, y + 2, span, bar_height, fill=colour, radius=3))
        body.extend(
            _value_label(
                right_x, span, panel_width, y, f"{mse:.1e}".replace("e-0", "e-")
            )
        )

    chance_x = left_x + chance * panel_width
    body.append(
        _line(
            chance_x,
            top - 14,
            chance_x,
            top + len(rows) * row_height,
            stroke=INK_MUTED,
            width=1.2,
            dash="4 3",
        )
    )
    body.append(
        _text(chance_x + 5, top - 17, "chance 0.0833", size=9.5, fill=INK_MUTED)
    )

    body.extend(
        _legend(
            0,
            height - 12,
            [
                (SERIES[0], "task or reconstruction supervision"),
                (SERIES[1], "no task/reconstruction term"),
            ],
        )
    )
    return _document(width, height, body, "Recovery by training objective")


# --------------------------------------------------------------------------
# Figure: corruption contrasts forest plot
# --------------------------------------------------------------------------


def figure_contrasts(gate3: dict[str, Any], gauge: dict[str, Any]) -> str:
    kind_label = {
        "edge_cochain_noise": "edge cochain",
        "node_anchor_noise": "node anchor",
        "transport_rotation": "transport rotation",
    }
    rows: list[tuple[str, str, float, float, float, int]] = []
    candidates = [
        Path(entry["path"]).parent.name for entry in gate3["inputs"]["candidates"]
    ]
    for comparison in gate3["comparisons"]:
        candidate = candidates[int(comparison["candidate_number"]) - 1]
        for kind, payload in sorted(comparison["by_kind"].items()):
            low, high = payload["paired_complete_block_bootstrap_95_ci"]
            rows.append(
                (
                    f"{candidate} / {kind_label.get(kind, kind)}",
                    "gate3",
                    float(payload["candidate_minus_baseline"]),
                    float(low),
                    float(high),
                    0,
                )
            )
    for kind in sorted(gauge["by_kind"]):
        payload = gauge["by_kind"][kind]
        low, high = payload["student_t_95_ci"]
        rows.append(
            (
                f"gauge / {kind_label.get(kind, kind)}",
                "gauge",
                float(payload["mean_difference"]),
                float(low),
                float(high),
                1,
            )
        )

    width = 680.0
    label_width = 210.0
    plot_width = width - label_width - 30.0
    top = 74.0
    row_height = 22.0
    height = top + len(rows) * row_height + 54.0

    lo = min(row[3] for row in rows)
    hi = max(row[4] for row in rows)
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad

    def scale(value: float) -> float:
        return label_width + (value - lo) / (hi - lo) * plot_width

    body: list[str] = [
        _text(
            0,
            20,
            "Every corruption contrast interval contains zero",
            size=13,
            weight="bold",
        ),
        _text(
            0,
            36,
            "Candidate minus baseline on the adjusted partial-Spearman endpoint, with 95% intervals.",
            size=10.5,
            fill=INK_SECONDARY,
        ),
        _text(
            0,
            50,
            "Gate-3 rows are conditional on a fixed checkpoint pair; gauge rows are across eight training seeds.",
            size=10.5,
            fill=INK_SECONDARY,
        ),
    ]

    for tick in (-0.4, -0.2, 0.0, 0.2, 0.4):
        if not lo <= tick <= hi:
            continue
        x = scale(tick)
        body.append(_line(x, top - 8, x, top + len(rows) * row_height, stroke=GRID))
        body.append(
            _text(
                x,
                top + len(rows) * row_height + 15,
                f"{tick:+g}" if tick else "0",
                size=9.5,
                fill=INK_MUTED,
                anchor="middle",
            )
        )

    zero_x = scale(0.0)
    body.append(
        _line(
            zero_x,
            top - 8,
            zero_x,
            top + len(rows) * row_height,
            stroke=INK_SECONDARY,
            width=1.4,
        )
    )

    for index, (label, family, estimate, low, high, slot) in enumerate(rows):
        y = top + index * row_height + row_height / 2.0
        colour = SERIES[slot]
        body.append(
            _text(label_width - 10, y + 3.5, label, size=9.5, anchor="end", family=MONO)
        )
        body.append(_line(scale(low), y, scale(high), y, stroke=colour, width=2))
        for endpoint in (low, high):
            body.append(
                _line(
                    scale(endpoint),
                    y - 3.5,
                    scale(endpoint),
                    y + 3.5,
                    stroke=colour,
                    width=2,
                )
            )
        body.append(
            f'<circle cx="{scale(estimate):.1f}" cy="{y:.1f}" r="4" fill="{colour}" '
            f'stroke="{SURFACE}" stroke-width="2"/>'
        )

    body.extend(
        _legend(
            0,
            height - 12,
            [
                (SERIES[0], "Gate-3 base (checkpoint-conditional)"),
                (SERIES[1], "gauge (eight training seeds)"),
            ],
        )
    )
    return _document(width, height, body, "Corruption contrast intervals")


# --------------------------------------------------------------------------
# Figure: trained routing compute
# --------------------------------------------------------------------------

PATH_ORDER = ("routed", "fixed_graph", "fixed_cell", "fixed_sheaf", "dense")
PATH_SLOT = {
    "routed": 0,
    "fixed_graph": 1,
    "fixed_cell": 1,
    "fixed_sheaf": 1,
    "dense": 2,
}


def figure_compute(compute: dict[str, Any]) -> str:
    runs = compute["routing"]["runs"]
    medians = {
        name: [float(run["median_latency_ms"][name]) for run in runs]
        for name in PATH_ORDER
    }
    p95s = {
        name: [float(run["p95_latency_ms"][name]) for run in runs]
        for name in PATH_ORDER
    }

    width, height = 680.0, 300.0
    label_width = 108.0
    plot_width = width - label_width - 96.0
    top = 74.0
    row_height = 30.0
    bar_height = 15.0
    hi = max(max(values) for values in p95s.values()) * 1.04

    body: list[str] = [
        _text(
            0, 20, "Trained routing inference latency on GB10", size=13, weight="bold"
        ),
        _text(
            0,
            36,
            "Median over 100 timed iterations, averaged across five seeds; whisker marks mean p95.",
            size=10.5,
            fill=INK_SECONDARY,
        ),
        _text(
            0,
            50,
            "Batch 64, bfloat16. Paths were timed in this order inside one process.",
            size=10.5,
            fill=INK_SECONDARY,
        ),
    ]

    for tick in range(0, int(hi) + 10, 10):
        x = label_width + tick / hi * plot_width
        body.append(
            _line(x, top - 6, x, top + len(PATH_ORDER) * row_height, stroke=GRID)
        )
        body.append(
            _text(
                x,
                top + len(PATH_ORDER) * row_height + 15,
                str(tick),
                size=9.5,
                fill=INK_MUTED,
                anchor="middle",
            )
        )
    body.append(
        _text(
            label_width + plot_width / 2,
            top + len(PATH_ORDER) * row_height + 32,
            "milliseconds",
            size=10,
            fill=INK_SECONDARY,
            anchor="middle",
        )
    )

    for index, name in enumerate(PATH_ORDER):
        y = top + index * row_height
        colour = SERIES[PATH_SLOT[name]]
        median = sum(medians[name]) / len(medians[name])
        p95 = sum(p95s[name]) / len(p95s[name])
        span = median / hi * plot_width
        body.append(
            _text(label_width - 10, y + 13, name, size=10, anchor="end", family=MONO)
        )
        body.append(_rect(label_width, y + 2, span, bar_height, fill=colour, radius=3))
        whisker = label_width + p95 / hi * plot_width
        body.append(
            _line(
                label_width + span,
                y + 9.5,
                whisker,
                y + 9.5,
                stroke=INK_MUTED,
                width=1.2,
            )
        )
        body.append(_line(whisker, y + 4, whisker, y + 15, stroke=INK_MUTED, width=1.2))
        body.append(
            _text(whisker + 8, y + 13, f"{median:.1f} ms", size=9.5, fill=INK_SECONDARY)
        )

    body.extend(
        _legend(
            0,
            height - 12,
            [
                (SERIES[0], "routed"),
                (SERIES[1], "single fixed route"),
                (SERIES[2], "dense three-expert"),
            ],
        )
    )
    return _document(width, height, body, "Trained routing inference latency")


# --------------------------------------------------------------------------
# Figure: the frozen edge-to-cycle lifting campaign
# --------------------------------------------------------------------------

TERM_LABEL = {
    "exact": (
        "boundary compatibility",
        "B₁Wᵀ = 0 (frozen key: exact)",
        0,
    ),
    "cone": (
        "singular-value cone surrogate",
        "exp(-2·σ_min(W)); not mapping-cone homology",
        1,
    ),
    "rtd": (
        "RTD-inspired distance surrogate",
        "normalized pairwise-distance MSE",
        2,
    ),
}


def figure_campaign(campaign: dict[str, Any]) -> str:
    # H5 routing inference was withdrawn after audit and is deliberately not a
    # figure input. This plot consumes only the three primary paired contrasts.
    primary = campaign["primary"]
    order = [name for name in ("exact", "cone", "rtd") if name in primary]
    rows = [(name, primary[name]) for name in order]

    width = 680.0
    label_width = 235.0
    # Reserve room on the right for the verdict and the printed interval.
    plot_width = width - label_width - 136.0
    top = 122.0
    row_height = 44.0
    height = top + len(rows) * row_height + 62.0

    lows = [row["interval_bonferroni_98_33"][0] for _, row in rows]
    highs = [row["interval_bonferroni_98_33"][1] for _, row in rows]
    lo, hi = min(lows + [0.0]), max(highs + [0.0])
    pad = (hi - lo) * 0.10
    lo, hi = lo - pad, hi + pad

    def scale(value: float) -> float:
        return label_width + (value - lo) / (hi - lo) * plot_width

    body: list[str] = [
        _text(
            0,
            20,
            "Boundary compatibility improves edge-to-cycle lifting",
            size=13,
            weight="bold",
        ),
        _text(
            0,
            37,
            "Paired log10 held-out-error ratio; negative favors the objective.",
            size=10.5,
            fill=INK_SECONDARY,
        ),
        _text(
            0,
            52,
            "One eligible seed jointly fixes topology, predictors, and training noise.",
            size=10.5,
            fill=INK_SECONDARY,
        ),
        _text(
            0,
            67,
            f"{campaign['design']['eligible_topologies']} eligible seeds, "
            f"{campaign['design']['training_pairs']} noisy training probes; "
            "same-family replication.",
            size=10.5,
            fill=INK_SECONDARY,
        ),
        _text(
            0,
            82,
            "Locked prospectively after outcome-informed weight selection; "
            "one execution deviation disclosed.",
            size=10,
            fill=INK_MUTED,
        ),
        _text(
            0,
            97,
            "Thick bar: Bonferroni 98.33% (governs the decision).  Thin bar: unadjusted 95%.",
            size=10,
            fill=INK_MUTED,
        ),
    ]

    for tick in (-3, -2, -1, 0, 1):
        if not lo <= tick <= hi:
            continue
        x = scale(tick)
        body.append(_line(x, top - 8, x, top + len(rows) * row_height, stroke=GRID))
        body.append(
            _text(
                x,
                top + len(rows) * row_height + 16,
                f"{tick:+d}" if tick else "0",
                size=9.5,
                fill=INK_MUTED,
                anchor="middle",
            )
        )
    zero = scale(0.0)
    body.append(
        _line(
            zero,
            top - 8,
            zero,
            top + len(rows) * row_height,
            stroke=INK_SECONDARY,
            width=1.4,
        )
    )
    body.append(
        _text(
            label_width + plot_width / 2,
            top + len(rows) * row_height + 34,
            "log10 ratio of held-out error",
            size=10,
            fill=INK_SECONDARY,
            anchor="middle",
        )
    )

    for index, (name, row) in enumerate(rows):
        y = top + index * row_height + row_height / 2.0 - 4
        label, gloss, slot = TERM_LABEL[name]
        colour = SERIES[slot]
        verdict = (
            "improves"
            if row["improves_confirmatory"]
            else ("harms" if row["harms_confirmatory"] else "no detected improvement")
        )
        body.append(
            _text(label_width - 12, y + 1, label, size=11, anchor="end", family=MONO)
        )
        body.append(
            _text(label_width - 12, y + 15, gloss, size=9, anchor="end", fill=INK_MUTED)
        )
        adjusted = row["interval_bonferroni_98_33"]
        plain = row["interval_95"]
        body.append(
            _line(scale(plain[0]), y, scale(plain[1]), y, stroke=colour, width=2)
        )
        body.append(
            _line(scale(adjusted[0]), y, scale(adjusted[1]), y, stroke=colour, width=6)
        )
        centre = scale(row["mean_log10_ratio"])
        body.append(
            f'<circle cx="{centre:.1f}" cy="{y:.1f}" r="4.5" fill="{SURFACE}" '
            f'stroke="{colour}" stroke-width="2.5"/>'
        )
        # The exact interval dominates the axis, so cone and rtd compress to a
        # few pixels. Print the adjusted interval beside each row so the decision
        # is legible even where the bar is not.
        anchor_x = scale(adjusted[1]) + 8
        body.append(
            _text(anchor_x, y + 1, verdict, size=10, fill=colour, weight="bold")
        )
        body.append(
            _text(
                anchor_x,
                y + 14,
                f"[{adjusted[0]:+.3f}, {adjusted[1]:+.3f}]",
                size=9,
                fill=INK_MUTED,
                family=MONO,
            )
        )

    body.extend(
        _legend(
            0,
            height - 14,
            [
                (SERIES[0], "improves"),
                (SERIES[1], "harms"),
                (SERIES[2], "no detected improvement"),
            ],
        )
    )
    return _document(width, height, body, "Edge-to-cycle lifting primary contrasts")


# --------------------------------------------------------------------------
# Figure: untouched-seed replication of the seven frozen claims
# --------------------------------------------------------------------------

FUTILITY_CLAIM_ID = "h7-rtd-bounded-benefit-futility"
GOVERNING_BOUND = {
    "less": ("one_sided_upper_bound", "upper bound"),
    "greater": ("one_sided_lower_bound", "lower bound"),
}


def figure_replication(campaign: dict[str, Any]) -> str:
    claims = campaign["primary"]["claims"]
    rows: list[tuple[str, str, float, float, str, bool]] = []
    for claim in claims:
        key, bound_name = GOVERNING_BOUND[claim["direction"]]
        rows.append(
            (
                claim["id"],
                f"{claim['numerator_arm']} / {claim['reference_arm']}",
                float(claim["mean_log10_ratio"]),
                float(claim[key]),
                bound_name,
                bool(claim["supported"]),
            )
        )
    futility = next(claim for claim in claims if claim["id"] == FUTILITY_CLAIM_ID)
    margin = float(futility["threshold"])
    descriptive_mean = float(
        campaign["descriptive"]["ambient_adam_vs_min_norm_ls"]["mean_log10_ratio"]
    )
    eligible = int(campaign["eligibility"]["eligible"])

    width = 680.0
    label_width = 272.0
    plot_width = width - label_width - 8.0
    top = 100.0
    row_height = 42.0
    divider_gap = 12.0
    descriptive_height = 34.0
    plot_bottom = top + len(rows) * row_height + divider_gap + descriptive_height
    height = plot_bottom + 60.0

    extremes = [value for row in rows for value in (row[2], row[3])]
    lo = min(extremes + [0.0, margin, descriptive_mean])
    hi = max(extremes + [0.0, margin, descriptive_mean])
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad

    def scale(value: float) -> float:
        return label_width + (value - lo) / (hi - lo) * plot_width

    body: list[str] = [
        _text(
            0,
            20,
            f"Untouched-seed replication ({eligible} eligible generator seeds)",
            size=13,
            weight="bold",
        ),
        _text(
            0,
            36,
            "Mean paired log10 held-out-MSE ratio per frozen claim; "
            "negative favors the numerator arm.",
            size=10.5,
            fill=INK_SECONDARY,
        ),
        _text(
            0,
            51,
            "Point: seed mean. Whisker ends at the governing one-sided "
            "Bonferroni bound in the claim's direction.",
            size=10.5,
            fill=INK_SECONDARY,
        ),
        _text(
            0,
            66,
            "Seven-claim confirmatory family at familywise alpha 0.05; "
            "same-generator-family replication, not real-data validation.",
            size=10,
            fill=INK_MUTED,
        ),
        _text(
            0,
            81,
            "H7 is a bounded-benefit/futility statement only; it cannot "
            "establish equality or absence of every benefit.",
            size=10,
            fill=INK_MUTED,
        ),
    ]

    for tick in (-3, -2, -1, 0):
        if not lo <= tick <= hi:
            continue
        x = scale(tick)
        body.append(_line(x, top - 8, x, plot_bottom, stroke=GRID))
        body.append(
            _text(
                x,
                plot_bottom + 16,
                f"{tick:+d}" if tick else "0",
                size=9.5,
                fill=INK_MUTED,
                anchor="middle",
            )
        )
    zero = scale(0.0)
    body.append(
        _line(zero, top - 8, zero, plot_bottom, stroke=INK_SECONDARY, width=1.4)
    )
    body.append(
        _text(
            label_width + plot_width / 2,
            plot_bottom + 34,
            "log10 ratio of held-out MSE (numerator / reference)",
            size=10,
            fill=INK_SECONDARY,
            anchor="middle",
        )
    )

    for index, (claim_id, pair, estimate, bound, bound_name, supported) in enumerate(
        rows
    ):
        y = top + index * row_height + row_height / 2.0
        colour = SERIES[0] if supported else SERIES[1]
        verdict = "supported" if supported else "not supported"
        body.append(
            _text(
                label_width - 12, y - 8, claim_id, size=9.5, anchor="end", family=MONO
            )
        )
        body.append(
            _text(
                label_width - 12,
                y + 4,
                pair,
                size=8.5,
                anchor="end",
                fill=INK_MUTED,
                family=MONO,
            )
        )
        body.append(
            _text(
                label_width - 12,
                y + 16,
                f"{verdict} · {bound_name} {bound:+.4f}",
                size=9,
                anchor="end",
                fill=colour,
            )
        )
        left, right = sorted((estimate, bound))
        body.append(_line(scale(left), y, scale(right), y, stroke=colour, width=2))
        body.append(
            _line(scale(bound), y - 4.5, scale(bound), y + 4.5, stroke=colour, width=2)
        )
        body.append(
            f'<circle cx="{scale(estimate):.1f}" cy="{y:.1f}" r="4.5" '
            f'fill="{colour if supported else SURFACE}" '
            f'stroke="{SURFACE if supported else colour}" stroke-width="2.5"/>'
        )
        if claim_id == FUTILITY_CLAIM_ID:
            margin_x = scale(margin)
            body.append(
                _line(
                    margin_x,
                    y - 7,
                    margin_x,
                    y + 7,
                    stroke=INK_MUTED,
                    width=1.2,
                    dash="2 2",
                )
            )
            body.append(
                _text(
                    margin_x - 6,
                    y + 3,
                    f"futility margin {margin}",
                    size=8.5,
                    fill=INK_MUTED,
                    anchor="end",
                )
            )

    divider_y = top + len(rows) * row_height + divider_gap / 2.0
    descriptive_y = divider_y + divider_gap / 2.0 + descriptive_height / 2.0
    body.append(
        _line(label_width, divider_y, width - 8, divider_y, stroke=AXIS, dash="3 3")
    )
    body.append(
        _text(
            label_width - 12,
            descriptive_y - 8,
            "ambient_adam / ambient_min_norm_ls",
            size=9,
            anchor="end",
            fill=INK_SECONDARY,
            family=MONO,
        )
    )
    body.append(
        _text(
            label_width - 12,
            descriptive_y + 3,
            "descriptive optimizer diagnostic",
            size=8.5,
            anchor="end",
            fill=INK_MUTED,
        )
    )
    body.append(
        _text(
            label_width - 12,
            descriptive_y + 14,
            f"mean {descriptive_mean:+.4f} · no confirmatory decision",
            size=8.5,
            anchor="end",
            fill=INK_MUTED,
        )
    )
    marker = scale(descriptive_mean)
    body.append(_rect(marker - 3.5, descriptive_y - 3.5, 7, 7, fill=INK_MUTED))

    body.extend(
        _legend(
            0,
            height - 12,
            [
                (SERIES[0], "supported (filled point)"),
                (SERIES[1], "not supported (open point)"),
                (INK_MUTED, "descriptive (muted square)"),
            ],
        )
    )
    return _document(
        width,
        height,
        body,
        f"Untouched-seed replication ({eligible} eligible generator seeds)",
    )


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=project_root / "results")
    parser.add_argument(
        "--output-dir", type=Path, default=project_root / "docs" / "figures"
    )
    args = parser.parse_args(argv)

    results = args.results_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    def load(relative: str) -> dict[str, Any]:
        return json.loads((results / relative).read_text(encoding="utf-8"))

    figures = {
        "fig-recovery.svg": figure_recovery(
            load("summaries/identifiable-campaign-summary.json")
        ),
        "fig-contrasts.svg": figure_contrasts(
            load("gate3/paired_comparison_final.json"),
            load("summaries/gauge-corruption-campaign.json"),
        ),
        "fig-compute.svg": figure_compute(load("summaries/compute-campaign.json")),
        "fig-campaign.svg": figure_campaign(
            load("campaigns/conversion-campaign-v1-corrected.json")
        ),
        "fig-replication.svg": figure_replication(
            load("campaigns/lifting-replication-v2.json")
        ),
    }
    for name, markup in figures.items():
        (output / name).write_text(markup, encoding="utf-8")
    print(
        json.dumps({"figures": sorted(figures), "output": str(output)}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
