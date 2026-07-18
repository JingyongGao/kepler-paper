from __future__ import annotations

import itertools
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Sequence

import reportlab
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import (
    Circle,
    Drawing,
    Line,
    Path as ShapePath,
    Polygon,
    Rect,
    String,
)
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from common import (
    FIGURES,
    ROOT,
    V35_AUDIT,
    V35_RESULTS,
    V36_AUDIT,
    V36_RESULTS,
    atomic_write_json,
    load_json,
    require_frozen_source_hashes,
    write_csv,
)


WIDTH = 720
HEIGHT = 432
REPORTLAB_FONTS = Path(reportlab.__file__).resolve().parent / "fonts"
FONT = "KeplerVera"
FONT_BOLD = "KeplerVera-Bold"
pdfmetrics.registerFont(TTFont(FONT, str(REPORTLAB_FONTS / "Vera.ttf")))
pdfmetrics.registerFont(TTFont(FONT_BOLD, str(REPORTLAB_FONTS / "VeraBd.ttf")))
INK = colors.HexColor("#20252B")
MUTED = colors.HexColor("#65717E")
GRID = colors.HexColor("#D9DEE3")
PAPER = colors.white
BLUE = colors.HexColor("#0072B2")
ORANGE = colors.HexColor("#E69F00")
GREEN = colors.HexColor("#009E73")
PURPLE = colors.HexColor("#7A5195")
RED = colors.HexColor("#D55E00")
SKY = colors.HexColor("#56B4E9")
CONDITION_COLORS = {
    "C0": MUTED,
    "C1": BLUE,
    "C2": GREEN,
    "C3": ORANGE,
    "R0": MUTED,
    "R1": BLUE,
    "R2": RED,
    "geometry_concentrated": PURPLE,
    "geometry_diffuse": ORANGE,
}


def text(
    drawing: Drawing,
    x: float,
    y: float,
    value: Any,
    *,
    size: float = 8,
    color: colors.Color = INK,
    anchor: str = "start",
    bold: bool = False,
    angle: float = 0,
) -> None:
    drawing.add(
        String(
            x,
            y,
            str(value),
            fontName=FONT_BOLD if bold else FONT,
            fontSize=size,
            fillColor=color,
            textAnchor=anchor,
            angle=angle,
        )
    )


def title(drawing: Drawing, panel_title: str, status: str) -> None:
    text(drawing, 28, HEIGHT - 28, panel_title, size=13, bold=True)
    badge_color = PURPLE if status == "Exploratory" else GREEN
    drawing.add(
        Rect(
            WIDTH - 111,
            HEIGHT - 37,
            83,
            18,
            rx=4,
            ry=4,
            fillColor=colors.Color(
                badge_color.red,
                badge_color.green,
                badge_color.blue,
                alpha=0.14,
            ),
            strokeColor=badge_color,
            strokeWidth=0.6,
        )
    )
    text(
        drawing,
        WIDTH - 69.5,
        HEIGHT - 32,
        status.upper(),
        size=7,
        color=badge_color,
        anchor="middle",
        bold=True,
    )


def panel_label(drawing: Drawing, x: float, y: float, label: str, name: str) -> None:
    text(drawing, x, y, label, size=10, bold=True)
    text(drawing, x + 17, y, name, size=9, bold=True)


def arrow(
    drawing: Drawing,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: colors.Color = MUTED,
) -> None:
    drawing.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.2))
    angle = math.atan2(y2 - y1, x2 - x1)
    length = 7
    spread = 0.45
    points = [
        x2,
        y2,
        x2 - length * math.cos(angle - spread),
        y2 - length * math.sin(angle - spread),
        x2 - length * math.cos(angle + spread),
        y2 - length * math.sin(angle + spread),
    ]
    drawing.add(Polygon(points, fillColor=color, strokeColor=color))


def box(
    drawing: Drawing,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    sublabel: str = "",
    color: colors.Color = BLUE,
) -> None:
    drawing.add(
        Rect(
            x,
            y,
            w,
            h,
            rx=5,
            ry=5,
            fillColor=colors.Color(color.red, color.green, color.blue, alpha=0.10),
            strokeColor=color,
            strokeWidth=1.0,
        )
    )
    text(drawing, x + w / 2, y + h / 2 + 4, label, anchor="middle", size=9, bold=True)
    if sublabel:
        text(
            drawing,
            x + w / 2,
            y + h / 2 - 9,
            sublabel,
            anchor="middle",
            size=7,
            color=MUTED,
        )


def axes(
    drawing: Drawing,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    xticks: Sequence[float],
    yticks: Sequence[float],
    xlabel: str,
    ylabel: str,
    xformat: Callable[[float], str] = lambda value: f"{value:g}",
    yformat: Callable[[float], str] = lambda value: f"{value:g}",
) -> tuple[Callable[[float], float], Callable[[float], float]]:
    def mx(value: float) -> float:
        return x + (value - xmin) / (xmax - xmin) * w

    def my(value: float) -> float:
        return y + (value - ymin) / (ymax - ymin) * h

    for value in yticks:
        py = my(value)
        drawing.add(Line(x, py, x + w, py, strokeColor=GRID, strokeWidth=0.5))
        text(drawing, x - 7, py - 2.5, yformat(value), size=7, color=MUTED, anchor="end")
    for value in xticks:
        px = mx(value)
        drawing.add(Line(px, y, px, y - 3, strokeColor=INK, strokeWidth=0.7))
        text(drawing, px, y - 13, xformat(value), size=7, color=MUTED, anchor="middle")
    drawing.add(Line(x, y, x + w, y, strokeColor=INK, strokeWidth=0.8))
    drawing.add(Line(x, y, x, y + h, strokeColor=INK, strokeWidth=0.8))
    text(drawing, x + w / 2, y - 29, xlabel, size=8, anchor="middle")
    text(drawing, x, y + h + 8, ylabel, size=7, color=MUTED)
    return mx, my


def line_path(
    drawing: Drawing,
    points: Sequence[tuple[float, float]],
    *,
    color: colors.Color,
    width: float = 1.0,
    dash: Sequence[float] | None = None,
) -> None:
    if not points:
        return
    path = ShapePath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    path.fillColor = None
    path.strokeColor = color
    path.strokeWidth = width
    path.strokeDashArray = dash
    drawing.add(path)


def marker(
    drawing: Drawing,
    x: float,
    y: float,
    color: colors.Color,
    *,
    radius: float = 3,
    open_marker: bool = False,
) -> None:
    drawing.add(
        Circle(
            x,
            y,
            radius,
            fillColor=PAPER if open_marker else color,
            strokeColor=color,
            strokeWidth=1.0,
        )
    )


def export(drawing: Drawing, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / f"{stem}.pdf"
    svg = FIGURES / f"{stem}.svg"
    png = FIGURES / f"{stem}.png"
    drawing.initialFontName = FONT
    renderPDF.drawToFile(
        drawing,
        str(pdf),
        invariant=1,
        initialFontName=FONT,
    )
    renderSVG.drawToFile(drawing, str(svg))
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        raise RuntimeError("pdftoppm is required to render deterministic PNG files")
    subprocess.run(
        [
            pdftoppm,
            "-png",
            "-singlefile",
            "-r",
            "180",
            str(pdf),
            str(png.with_suffix("")),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def checkpoint(cell: dict[str, Any], exposure: int, phase: str = "standard") -> dict:
    matches = [
        item
        for item in cell["checkpoint_metrics"]
        if int(item["exposed_pairs"]) == exposure and item["phase"] == phase
    ]
    if len(matches) != 1:
        raise ValueError((cell["run_id"], exposure, phase, len(matches)))
    return matches[0]


def blend(
    start: colors.Color,
    end: colors.Color,
    amount: float,
) -> colors.Color:
    weight = min(1.0, max(0.0, amount))
    return colors.Color(
        start.red + (end.red - start.red) * weight,
        start.green + (end.green - start.green) * weight,
        start.blue + (end.blue - start.blue) * weight,
    )


def binned_pair_density_ratio(
    condition: dict[str, Any],
    *,
    p: int,
    bins: int = 14,
) -> dict[str, Any]:
    coefficient_1, coefficient_2 = (
        int(value) for value in condition["coefficients"]
    )
    cue_coefficient = int(condition["cue_coefficient_c"])
    selected_residues = {int(value) for value in condition["B_m"]}
    m = int(condition["m"])
    lam = float(condition["lambda"])
    mass = [[0.0 for _ in range(bins)] for _ in range(bins)]
    counts = [[0 for _ in range(bins)] for _ in range(bins)]
    minimum = math.inf
    maximum = -math.inf
    for x1 in range(p):
        row = min(bins - 1, x1 * bins // p)
        for x2 in range(p):
            column = min(bins - 1, x2 * bins // p)
            z = (coefficient_1 * x1 + coefficient_2 * x2) % p
            selected = ((z - cue_coefficient * x1) % p) in selected_residues
            probability = (1.0 - lam) / (p * p)
            if selected:
                probability += lam / (m * p)
            mass[row][column] += probability
            counts[row][column] += 1
            minimum = min(minimum, probability)
            maximum = max(maximum, probability)
    tolerance = 5e-18
    if abs(minimum - float(condition["minimum_pair_probability"])) > tolerance:
        raise AssertionError("minimum pair probability does not match frozen audit")
    if abs(maximum - float(condition["maximum_pair_probability"])) > tolerance:
        raise AssertionError("maximum pair probability does not match frozen audit")
    ratios = [
        [
            mass[row][column] / (counts[row][column] / (p * p))
            for column in range(bins)
        ]
        for row in range(bins)
    ]
    return {
        "bins": bins,
        "density_ratio_to_uniform": ratios,
        "minimum_pair_probability": minimum,
        "maximum_pair_probability": maximum,
    }


def density_heatmap(
    drawing: Drawing,
    x: float,
    y: float,
    size: float,
    ratios: Sequence[Sequence[float]],
    label: str,
    *,
    deviation_scale: float,
) -> None:
    bins = len(ratios)
    cell = size / bins
    scale = deviation_scale if deviation_scale > 0 else 1.0
    for row_index, row in enumerate(ratios):
        for column_index, value in enumerate(row):
            deviation = (value - 1.0) / scale
            color = (
                blend(GRID, BLUE, 0.82 * deviation)
                if deviation >= 0
                else blend(GRID, ORANGE, 0.82 * -deviation)
            )
            drawing.add(
                Rect(
                    x + column_index * cell,
                    y + (bins - 1 - row_index) * cell,
                    cell + 0.15,
                    cell + 0.15,
                    fillColor=color,
                    strokeColor=None,
                )
            )
    drawing.add(
        Rect(
            x,
            y,
            size,
            size,
            fillColor=None,
            strokeColor=INK,
            strokeWidth=0.6,
        )
    )
    text(drawing, x + size / 2, y + size + 11, label, size=7.5, anchor="middle", bold=True)
    text(drawing, x + size / 2, y - 13, "x2", size=7, anchor="middle", color=MUTED)
    text(drawing, x - 10, y + size / 2, "x1", size=7, anchor="middle", color=MUTED, angle=90)


def figure_01(v35_audit: dict, v36_audit: dict) -> None:
    comparison_conditions = {
        item["condition_id"]: item
        for item in v36_audit["conditions"]
        if item["rule_id"] == "rule_0"
        and item["permutation_id"] == "permutation_0"
        and item["condition_id"] in {"R0", "R2"}
    }
    density_comparison = {
        condition: binned_pair_density_ratio(item, p=int(v36_audit["p"]))
        for condition, item in comparison_conditions.items()
    }
    conditions = {
        item["condition_id"]: item["target_singleton_cue_index"]
        for item in v36_audit["conditions"]
        if item["rule_id"] == "rule_0" and item["permutation_id"] == "permutation_0"
    }
    source = {
        "figure": 1,
        "status": "task definition plus exploratory and confirmatory cue families",
        "p": v36_audit["p"],
        "eta": v36_audit["eta"],
        "v3_5_coefficients": v35_audit["coefficients"],
        "v3_5_cue_coefficient": v35_audit["cue_coefficient_c"],
        "v3_6_rules": v36_audit["rules"],
        "v3_6_confirmatory_targets": conditions,
        "m": v36_audit["m"],
        "pair_distribution_comparison": {
            condition: {
                "distribution_id": comparison_conditions[condition][
                    "distribution_id"
                ],
                "coefficients": comparison_conditions[condition]["coefficients"],
                "cue_coefficient_c": comparison_conditions[condition][
                    "cue_coefficient_c"
                ],
                "B_m": comparison_conditions[condition]["B_m"],
                "lambda": comparison_conditions[condition]["lambda"],
                "target_singleton_cue_index": comparison_conditions[condition][
                    "target_singleton_cue_index"
                ],
                **density_comparison[condition],
            }
            for condition in ("R0", "R2")
        },
        "data_source": "artifacts/confirmatory_distribution_audit.json",
    }
    atomic_write_json(FIGURES / "fig01_task_cue_schematic.source.json", source)
    d = Drawing(WIDTH, HEIGHT)
    title(d, "Pure-synergy task and transient full-support cue family", "Confirmatory")
    panel_label(d, 28, 372, "A", "Parent-to-center-to-target graph")
    box(d, 36, 287, 63, 42, "X1", "parent", BLUE)
    box(d, 36, 225, 63, 42, "X2", "parent", ORANGE)
    box(
        d,
        137,
        246,
        119,
        52,
        "Z = a1 X1 + a2 X2",
        f"mod p, p={source['p']}",
        GREEN,
    )
    box(
        d,
        287,
        246,
        92,
        52,
        "Y = Z + epsilon",
        f"eta={source['eta']}",
        PURPLE,
    )
    arrow(d, 99, 308, 137, 282, BLUE)
    arrow(d, 99, 246, 137, 264, ORANGE)
    arrow(d, 256, 272, 287, 272, GREEN)

    panel_label(d, 28, 190, "B", "Pure-synergy identities at R0")
    text(d, 37, 157, "I(Y; X1) = I(Y; X2) = 0", size=8.5, bold=True)
    text(d, 37, 137, "I(Y; X1,X2) = log(p) - H(nu_eta)", size=8.5)
    text(d, 37, 117, "X1, X2, Z, and Y are marginally uniform", size=8)
    text(d, 37, 97, "All p^2 ordered parent pairs have positive support", size=8)
    text(d, 37, 72, "Cue conditions preserve marginals, not parent independence.", size=7.5, color=MUTED)

    panel_label(d, 410, 372, "C", "Pair-distribution geometry")
    comparison_scale = max(
        abs(value - 1.0)
        for row in density_comparison["R2"]["density_ratio_to_uniform"]
        for value in row
    )
    density_heatmap(
        d,
        424,
        225,
        104,
        density_comparison["R0"]["density_ratio_to_uniform"],
        "R0: lambda = 0",
        deviation_scale=comparison_scale,
    )
    density_heatmap(
        d,
        571,
        225,
        104,
        density_comparison["R2"]["density_ratio_to_uniform"],
        f"R2: C_single = {conditions['R2']:g}",
        deviation_scale=comparison_scale,
    )
    text(d, 550, 199, "binned pair-density ratio to uniform", size=7, anchor="middle", color=MUTED)
    drawing_legend_y = 183
    for legend_x, legend_color, legend_label in (
        (470, ORANGE, "< 1"),
        (550, GRID, "= 1"),
        (630, BLUE, "> 1"),
    ):
        d.add(Rect(legend_x - 17, drawing_legend_y - 2, 12, 8, fillColor=legend_color, strokeColor=None))
        text(d, legend_x, drawing_legend_y, legend_label, size=7, color=MUTED)

    panel_label(d, 410, 151, "D", "Cue window and return to full support")
    x0, x1 = 424, 681
    y = 99
    d.add(Line(x0, y, x1, y, strokeColor=INK, strokeWidth=1.0))
    cue_end = x0 + (x1 - x0) / 2
    d.add(
        Rect(
            x0,
            y - 18,
            cue_end - x0,
            36,
            fillColor=colors.Color(BLUE.red, BLUE.green, BLUE.blue, alpha=0.12),
            strokeColor=None,
        )
    )
    d.add(
        Rect(
            cue_end,
            y - 18,
            x1 - cue_end,
            36,
            fillColor=colors.Color(MUTED.red, MUTED.green, MUTED.blue, alpha=0.10),
            strokeColor=None,
        )
    )
    for xpos, label in ((x0, "0"), (cue_end, "5M"), (x1, "10M")):
        d.add(Line(xpos, y - 22, xpos, y + 22, strokeColor=INK, strokeWidth=0.8))
        text(d, xpos, y - 34, label, anchor="middle", size=8)
    text(d, (x0 + cue_end) / 2, y + 27, "R1/R2 cue-bearing", anchor="middle", size=8, bold=True)
    text(d, (cue_end + x1) / 2, y + 27, "uniform full support", anchor="middle", size=8, bold=True)
    text(d, 424, 48, "Only the sampler switches at 5M; training continues to 10M.", size=7.5, color=MUTED)
    export(d, "fig01_task_cue_schematic")


def figure_02(v35: dict) -> None:
    conditions = ("C0", "C1", "C2", "C3")
    rows: list[dict[str, Any]] = []
    for cell in v35["cells"]:
        if cell["condition_id"] not in conditions:
            continue
        for item in cell["checkpoint_metrics"]:
            if item["phase"] == "post_switch":
                continue
            rows.append(
                {
                    "condition": cell["condition_id"],
                    "seed": cell["seed"],
                    "exposure": item["exposed_pairs"],
                    "phase": item["phase"],
                    "R": item["R"],
                    "R_ci_low": item["R_bootstrap_95_ci"][0],
                    "R_ci_high": item["R_bootstrap_95_ci"][1],
                    "onset_ordinal": cell["transition"]["onset_ordinal"],
                    "onset_exposure": cell["transition"]["onset_exposure"],
                    "establishment_exposure": cell["transition"][
                        "establishment_exposure"
                    ],
                }
            )
    write_csv(FIGURES / "fig02_v35_dose_trajectories.source.csv", rows)
    d = Drawing(WIDTH, HEIGHT)
    title(d, "v3.5 cue-dose trajectories and frozen onset outcomes", "Exploratory")
    panel_label(d, 38, 372, "A", "Exact full-support R")
    mx, my = axes(
        d,
        64,
        92,
        430,
        250,
        xmin=0,
        xmax=10,
        ymin=0,
        ymax=1.08,
        xticks=(0, 2, 4, 5, 6, 8, 10),
        yticks=(0, 0.25, 0.5, 0.75, 1.0),
        xlabel="Pair exposures (millions)",
        ylabel="Normalized exact KL, R",
    )
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], int(row["seed"]))].append(row)
    for (condition, _seed), points in grouped.items():
        points.sort(key=lambda item: item["exposure"])
        color = CONDITION_COLORS[condition]
        faint = colors.Color(color.red, color.green, color.blue, alpha=0.22)
        line_path(
            d,
            [(mx(item["exposure"] / 1e6), my(item["R"])) for item in points],
            color=faint,
            width=0.7,
        )
    for condition in conditions:
        by_exposure: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            if row["condition"] == condition:
                by_exposure[int(row["exposure"])].append(row)
        summary = []
        for exposure, items in sorted(by_exposure.items()):
            summary.append(
                (
                    exposure / 1e6,
                    mean(item["R"] for item in items),
                    mean(item["R_ci_low"] for item in items),
                    mean(item["R_ci_high"] for item in items),
                )
            )
        lower = [(mx(x), my(max(0, low))) for x, _value, low, _high in summary]
        upper = [(mx(x), my(min(1.08, high))) for x, _value, _low, high in summary]
        polygon = [coord for point in lower for coord in point] + [
            coord for point in reversed(upper) for coord in point
        ]
        color = CONDITION_COLORS[condition]
        d.add(
            Polygon(
                polygon,
                fillColor=colors.Color(color.red, color.green, color.blue, alpha=0.10),
                strokeColor=None,
            )
        )
        line_path(
            d,
            [(mx(x), my(value)) for x, value, _low, _high in summary],
            color=color,
            width=2.0,
        )
        for x, value, _low, _high in summary:
            marker(d, mx(x), my(value), color, radius=2.2)
    d.add(Line(mx(5), 92, mx(5), 342, strokeColor=MUTED, strokeWidth=0.9, strokeDashArray=[3, 2]))
    text(d, mx(5) + 4, 329, "cue removed", size=7, color=MUTED)
    text(d, 70, 43, "Thin lines: seeds; band: mean registered row-bootstrap CI endpoints", size=7, color=MUTED)

    panel_label(d, 526, 372, "B", "Onset ordinal")
    bx, by = axes(
        d,
        540,
        92,
        145,
        250,
        xmin=-0.5,
        xmax=3.5,
        ymin=1,
        ymax=8.2,
        xticks=(0, 1, 2, 3),
        yticks=(1, 2, 3, 4, 5, 6, 7, 8),
        xlabel="Cue dose",
        ylabel="Onset ordinal (lower = earlier)",
        xformat=lambda value: conditions[int(value)],
    )
    for index, condition in enumerate(conditions):
        ordinals = [
            int(cell["transition"]["onset_ordinal"])
            for cell in v35["cells"]
            if cell["condition_id"] == condition
        ]
        color = CONDITION_COLORS[condition]
        for seed, ordinal in enumerate(ordinals):
            marker(d, bx(index + (seed - 2) * 0.045), by(ordinal), color, radius=3)
        d.add(
            Line(
                bx(index - 0.22),
                by(mean(ordinals)),
                bx(index + 0.22),
                by(mean(ordinals)),
                strokeColor=INK,
                strokeWidth=2,
            )
        )
    text(d, 548, 43, "Ordinal 8 = no verified onset", size=7, color=MUTED)
    export(d, "fig02_v35_dose_trajectories")


def figure_03(v36: dict) -> None:
    ordinals = v36["primary_confirmatory_test"]["ordered_onset_test"][
        "onset_ordinals_by_block"
    ]
    metadata = {
        item["block_id"]: item
        for item in v36["primary_confirmatory_test"]["block_metadata"]
    }
    rows = []
    for block, values in ordinals.items():
        for condition in ("R0", "R1", "R2"):
            rows.append(
                {
                    "block_id": block,
                    "rule_id": metadata[block]["rule_id"],
                    "permutation_id": metadata[block]["permutation_id"],
                    "model_seed_index": metadata[block]["model_seed_index"],
                    "condition": condition,
                    "onset_ordinal": values[condition],
                }
            )
    write_csv(FIGURES / "fig03_v36_paired_onset.source.csv", rows)
    d = Drawing(WIDTH, HEIGHT)
    title(d, "v3.6 paired-block representation onset", "Confirmatory")
    panel_label(d, 42, 370, "A", "Eight confirmatory paired blocks")
    mx, my = axes(
        d,
        95,
        82,
        500,
        270,
        xmin=-0.25,
        xmax=2.25,
        ymin=1,
        ymax=8.25,
        xticks=(0, 1, 2),
        yticks=(1, 2, 3, 4, 5, 6, 7, 8),
        xlabel="Registered cue dose",
        ylabel="Onset ordinal (lower = earlier)",
        xformat=lambda value: ("R0", "R1", "R2")[int(value)],
    )
    for block, values in ordinals.items():
        item = metadata[block]
        color = BLUE if item["rule_id"] == "rule_0" else RED
        dash = None if item["permutation_id"] == "permutation_0" else [4, 2]
        points = [(mx(index), my(values[condition])) for index, condition in enumerate(("R0", "R1", "R2"))]
        line_path(d, points, color=colors.Color(color.red, color.green, color.blue, alpha=0.65), width=1.2, dash=dash)
        for index, point in enumerate(points):
            marker(
                d,
                point[0],
                point[1],
                color,
                radius=3.2,
                open_marker=item["model_seed_index"] == 1,
            )
    d.add(Line(95, my(8), 595, my(8), strokeColor=MUTED, strokeWidth=0.8, strokeDashArray=[2, 2]))
    text(d, 602, my(8) - 2, "terminal", size=7, color=MUTED)
    text(d, 113, 23, "Blue: rule 0; orange-red: rule 1", size=8, color=MUTED)
    text(d, 330, 23, "Solid/dashed: permutation 0/1", size=8, color=MUTED)
    text(d, 548, 23, "Filled/open: init 0/1", size=8, color=MUTED)
    primary = v36["primary_confirmatory_test"]["ordered_onset_test"]
    text(
        d,
        610,
        319,
        f"p = {primary['exact_one_sided_p']:.3g}",
        size=9,
        bold=True,
        color=GREEN,
    )
    text(d, 610, 301, f"{primary['upper_tail_count']} / {primary['exact_permutation_count']:,}", size=7, color=MUTED)
    export(d, "fig03_v36_paired_onset")


def exact_null(v36: dict) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    test = v36["primary_confirmatory_test"]["ordered_onset_test"]
    ranks2 = [[int(round(value * 2)) for value in ranks] for ranks in test["block_ranks"]]
    weights = (1, 2, 3)
    distribution = Counter({0: 1})
    for ranks in ranks2:
        block = Counter(
            sum(weight * rank for weight, rank in zip(weights, perm, strict=True))
            for perm in itertools.permutations(ranks)
        )
        combined: Counter[int] = Counter()
        for total, count in distribution.items():
            for contribution, block_count in block.items():
                combined[total + contribution] += count * block_count
        distribution = combined
    observed2 = int(test["observed_statistic_doubled_integer"])
    total = sum(distribution.values())
    upper = sum(count for statistic, count in distribution.items() if statistic >= observed2)
    rows = [
        {
            "statistic": statistic / 2,
            "statistic_doubled_integer": statistic,
            "assignment_count": count,
            "probability": count / total,
            "upper_tail_at_or_above_observed": statistic >= observed2,
        }
        for statistic, count in sorted(distribution.items())
    ]
    metadata = {
        "exact_assignment_count": total,
        "observed_statistic": observed2 / 2,
        "upper_tail_count": upper,
        "exact_one_sided_p": upper / total,
        "v3_5_combined": False,
    }
    if total != test["exact_permutation_count"] or upper != test["upper_tail_count"]:
        raise AssertionError("exact null reconstruction does not match frozen result")
    return rows, metadata


def figure_04(v36: dict) -> None:
    rows, metadata = exact_null(v36)
    write_csv(FIGURES / "fig04_v36_exact_null.source.csv", rows)
    atomic_write_json(FIGURES / "fig04_v36_exact_null.source.json", metadata)
    d = Drawing(WIDTH, HEIGHT)
    title(d, "Exact blocked Page-style permutation null", "Confirmatory")
    panel_label(d, 44, 370, "A", "All within-block assignments")
    xmin = min(float(row["statistic"]) for row in rows)
    xmax = max(float(row["statistic"]) for row in rows)
    ymax = max(math.log10(int(row["assignment_count"]) + 1) for row in rows)
    mx, my = axes(
        d,
        78,
        84,
        560,
        270,
        xmin=xmin - 0.5,
        xmax=xmax + 0.5,
        ymin=0,
        ymax=math.ceil(ymax),
        xticks=tuple(range(int(math.ceil(xmin / 5) * 5), int(xmax) + 1, 5)),
        yticks=tuple(range(0, int(math.ceil(ymax)) + 1)),
        xlabel="Page-style statistic",
        ylabel="log10(exact assignment count + 1)",
    )
    bar_width = max(1.0, 0.76 * (mx(xmin + 1) - mx(xmin)))
    for row in rows:
        height = math.log10(int(row["assignment_count"]) + 1)
        color = RED if row["upper_tail_at_or_above_observed"] else BLUE
        d.add(
            Rect(
                mx(float(row["statistic"])) - bar_width / 2,
                my(0),
                bar_width,
                my(height) - my(0),
                fillColor=colors.Color(color.red, color.green, color.blue, alpha=0.72),
                strokeColor=None,
            )
        )
    observed = float(metadata["observed_statistic"])
    d.add(Line(mx(observed), 84, mx(observed), 354, strokeColor=RED, strokeWidth=1.6))
    text(d, mx(observed) - 4, 337, f"observed = {observed:g}", size=8, color=RED, anchor="end", bold=True)
    text(
        d,
        82,
        25,
        f"Exact upper tail: {metadata['upper_tail_count']} / {metadata['exact_assignment_count']:,}; "
        f"one-sided p = {metadata['exact_one_sided_p']:.8g}",
        size=8,
        color=MUTED,
    )
    export(d, "fig04_v36_exact_null")


def figure_05(v36: dict) -> None:
    rows = []
    selected = [
        cell
        for cell in v36["cells"]
        if cell["condition_id"] in ("R1", "R2")
        and cell["transition"]["threshold_state_at_5M"]
    ]
    for cell in selected:
        for exposure, phase in ((5_000_000, "pre_switch"), (6_000_000, "standard"), (10_000_000, "standard")):
            item = checkpoint(cell, exposure, phase)
            rows.append(
                {
                    "run_id": cell["run_id"],
                    "block_id": cell["block_id"],
                    "condition": cell["condition_id"],
                    "rule_id": cell["rule_id"],
                    "exposure": exposure,
                    "phase": phase,
                    "R": item["R"],
                    "R_ci_low": item["R_bootstrap_95_ci"][0],
                    "R_ci_high": item["R_bootstrap_95_ci"][1],
                    "threshold_positive": (
                        cell["transition"][
                            {
                                5_000_000: "threshold_state_at_5M",
                                6_000_000: "threshold_state_at_6M",
                                10_000_000: "threshold_state_at_10M",
                            }[exposure]
                        ]
                    ),
                }
            )
    write_csv(FIGURES / "fig05_v36_persistence.source.csv", rows)
    d = Drawing(WIDTH, HEIGHT)
    title(d, "Predictive-state persistence after cue removal", "Confirmatory")
    panel_label(d, 44, 370, "A", "Cells positive at the 5M pre-switch checkpoint")
    mx, my = axes(
        d,
        90,
        82,
        540,
        270,
        xmin=4.6,
        xmax=10.4,
        ymin=-3.2,
        ymax=0.1,
        xticks=(5, 6, 10),
        yticks=(-3, -2, -1, 0),
        xlabel="Pair exposures (millions)",
        ylabel="Full-support R (log10 scale)",
        yformat=lambda value: f"{10 ** value:g}",
    )
    by_run: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_run[row["run_id"]].append(row)
    for run_id, points in by_run.items():
        points.sort(key=lambda item: item["exposure"])
        condition = points[0]["condition"]
        color = CONDITION_COLORS[condition]
        coords = [
            (mx(item["exposure"] / 1e6), my(math.log10(max(float(item["R"]), 1e-4))))
            for item in points
        ]
        line_path(
            d,
            coords,
            color=colors.Color(color.red, color.green, color.blue, alpha=0.55),
            width=1.0,
        )
        for item, (px, py) in zip(points, coords, strict=True):
            low = my(math.log10(max(float(item["R_ci_low"]), 1e-4)))
            high = my(math.log10(max(float(item["R_ci_high"]), 1e-4)))
            d.add(Line(px, low, px, high, strokeColor=color, strokeWidth=0.7))
            marker(d, px, py, color, radius=2.6, open_marker=condition == "R1")
    threshold_y = my(math.log10(0.9))
    d.add(Line(90, threshold_y, 630, threshold_y, strokeColor=MUTED, strokeWidth=0.8, strokeDashArray=[3, 2]))
    text(d, 636, threshold_y - 2, "R=0.9", size=7, color=MUTED)
    text(d, mx(5), 365, "cue -> uniform", size=8, color=MUTED, anchor="middle")
    text(d, 105, 23, "Open: R1 (5 cells); filled: R2 (8 cells); whiskers: registered row-bootstrap 95% CI", size=8, color=MUTED)
    export(d, "fig05_v36_persistence")


def figure_06(v36: dict) -> None:
    rows = []
    for cell in v36["cells"]:
        if cell["condition_id"] not in ("R1", "R2"):
            continue
        rows.append(
            {
                "run_id": cell["run_id"],
                "condition": cell["condition_id"],
                "rule_id": cell["rule_id"],
                "permutation_id": cell["permutation_id"],
                "model_seed_index": cell["model_seed_index"],
                "onset_ordinal": cell["transition"]["onset_ordinal"],
                "verified_onset_by_10M": not cell["transition"]["onset_terminal_coded"],
                "R_5M_pre": cell["R_5M_pre"],
                "R_10M": cell["R_10M"],
            }
        )
    write_csv(FIGURES / "fig06_v36_rule_heterogeneity.source.csv", rows)
    d = Drawing(WIDTH, HEIGHT)
    title(d, "Rule-specific mid-dose heterogeneity and high-dose robustness", "Confirmatory")
    for panel_index, condition in enumerate(("R1", "R2")):
        x0 = 72 + panel_index * 330
        panel_label(
            d,
            x0 - 25,
            370,
            "A" if panel_index == 0 else "B",
            f"{condition} onset by coefficient rule",
        )
        mx, my = axes(
            d,
            x0,
            85,
            235,
            255,
            xmin=-0.45,
            xmax=1.45,
            ymin=1,
            ymax=8.2,
            xticks=(0, 1),
            yticks=(1, 2, 3, 4, 5, 6, 7, 8),
            xlabel="Coefficient rule",
            ylabel="Onset ordinal (lower = earlier)",
            xformat=lambda value: ("rule 0", "rule 1")[int(value)],
        )
        selected = [row for row in rows if row["condition"] == condition]
        for rule_index, rule in enumerate(("rule_0", "rule_1")):
            items = [row for row in selected if row["rule_id"] == rule]
            for item in items:
                jitter = (
                    (-0.09 if item["permutation_id"] == "permutation_0" else 0.09)
                    + (-0.025 if item["model_seed_index"] == 0 else 0.025)
                )
                marker(
                    d,
                    mx(rule_index + jitter),
                    my(float(item["onset_ordinal"])),
                    CONDITION_COLORS[condition],
                    radius=3.5,
                    open_marker=item["model_seed_index"] == 1,
                )
            average = mean(float(item["onset_ordinal"]) for item in items)
            d.add(
                Line(
                    mx(rule_index - 0.22),
                    my(average),
                    mx(rule_index + 0.22),
                    my(average),
                    strokeColor=INK,
                    strokeWidth=2.0,
                )
            )
            verified = sum(bool(item["verified_onset_by_10M"]) for item in items)
            text(
                d,
                mx(rule_index),
                28,
                f"{verified}/{len(items)} verified",
                anchor="middle",
                size=7,
                color=MUTED,
            )
    export(d, "fig06_v36_rule_heterogeneity")


def figure_07(v35: dict) -> None:
    condition_order = ("geometry_concentrated", "C2", "geometry_diffuse")
    labels = ("m=1", "m=64", "m=128")
    rows = []
    for cell in v35["cells"]:
        if cell["condition_id"] not in condition_order:
            continue
        rows.append(
            {
                "run_id": cell["run_id"],
                "condition": cell["condition_id"],
                "m": cell["m"],
                "seed": cell["seed"],
                "target_C_single": cell["target_singleton_cue_index"],
                "onset_ordinal": cell["transition"]["onset_ordinal"],
                "onset_exposure": cell["transition"]["onset_exposure"],
                "R_5M_pre": cell["R_5M_pre"],
                "R_10M": cell["R_10M"],
            }
        )
    write_csv(FIGURES / "fig07_v35_geometry_comparison.source.csv", rows)
    d = Drawing(WIDTH, HEIGHT)
    title(d, "Matched-C_single cue geometry is not interchangeable", "Exploratory")
    panel_label(d, 40, 370, "A", "Frozen onset ordinal")
    mx, my = axes(
        d,
        80,
        85,
        255,
        255,
        xmin=-0.3,
        xmax=2.3,
        ymin=1,
        ymax=7.2,
        xticks=(0, 1, 2),
        yticks=(1, 2, 3, 4, 5, 6, 7),
        xlabel="Cue support geometry",
        ylabel="Onset ordinal (lower = earlier)",
        xformat=lambda value: labels[int(value)],
    )
    for seed in range(5):
        items = {
            row["condition"]: row
            for row in rows
            if int(row["seed"]) == seed
        }
        points = [
            (mx(index), my(float(items[condition]["onset_ordinal"])))
            for index, condition in enumerate(condition_order)
        ]
        line_path(d, points, color=colors.Color(BLUE.red, BLUE.green, BLUE.blue, alpha=0.45), width=1.0)
        for index, point in enumerate(points):
            marker(d, point[0], point[1], CONDITION_COLORS[condition_order[index]], radius=3)

    panel_label(d, 385, 370, "B", "Exact R at cue removal")
    rx, ry = axes(
        d,
        425,
        85,
        235,
        255,
        xmin=-0.3,
        xmax=2.3,
        ymin=0,
        ymax=0.21,
        xticks=(0, 1, 2),
        yticks=(0, 0.05, 0.10, 0.15, 0.20),
        xlabel="Cue support geometry",
        ylabel="R at 5M pre-switch",
        xformat=lambda value: labels[int(value)],
    )
    for seed in range(5):
        items = {
            row["condition"]: row
            for row in rows
            if int(row["seed"]) == seed
        }
        points = [
            (rx(index), ry(float(items[condition]["R_5M_pre"])))
            for index, condition in enumerate(condition_order)
        ]
        line_path(d, points, color=colors.Color(BLUE.red, BLUE.green, BLUE.blue, alpha=0.40), width=0.9)
        for index, point in enumerate(points):
            marker(d, point[0], point[1], CONDITION_COLORS[condition_order[index]], radius=3)
    text(d, 94, 23, "All three conditions target the same registered C_single = 0.065092", size=8, color=MUTED)
    export(d, "fig07_v35_geometry_comparison")


def main() -> None:
    require_frozen_source_hashes()
    v35_audit = load_json(V35_AUDIT)
    v36_audit = load_json(V36_AUDIT)
    v35 = load_json(V35_RESULTS)
    v36 = load_json(V36_RESULTS)
    figure_01(v35_audit, v36_audit)
    figure_02(v35)
    figure_03(v36)
    figure_04(v36)
    figure_05(v36)
    figure_06(v36)
    figure_07(v35)
    print(f"rendered 7 figures to {FIGURES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
