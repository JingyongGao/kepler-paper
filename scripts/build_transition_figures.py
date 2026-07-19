from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper/scripts"))
import build_figures as bf  # noqa: E402


bf.FIGURES = ROOT / "figures"


def load(name: str):
    return json.loads((ROOT / "source-data" / name).read_text(encoding="utf-8"))


def legend(drawing, entries, x, y):
    for index, (label, color, dash) in enumerate(entries):
        offset = index * 105
        drawing.add(bf.Line(x + offset, y, x + offset + 18, y, strokeColor=color, strokeWidth=1.8, strokeDashArray=dash))
        bf.text(drawing, x + offset + 24, y - 2.5, label, size=7, color=bf.MUTED)


def figure_1() -> None:
    source = load("figure1_representation_emergence.json")
    exposures = [value / 1e6 for value in source["exposures"]]
    drawing = bf.Drawing(bf.WIDTH, bf.HEIGHT)
    bf.title(drawing, "Representation emergence across frozen trajectories", "Exploratory")
    colors = {"R0": bf.MUTED, "R2": bf.RED}

    panels = [
        (48, "A", "Full-support R", -3.0, 0.1, (-3, -2, -1, 0)),
        (278, "B", "Layer-6 probe accuracy", 0.0, 1.02, (0, 0.5, 1.0)),
        (508, "C", "Layer-6 Fourier R2", -0.05, 0.5, (0, 0.25, 0.5)),
    ]
    for x0, label, name, ymin, ymax, yticks in panels:
        bf.panel_label(drawing, x0 - 22, 367, label, name)
        mx, my = bf.axes(
            drawing,
            x0,
            88,
            175,
            250,
            xmin=1.7,
            xmax=10.3,
            ymin=ymin,
            ymax=ymax,
            xticks=(2, 4, 6, 8, 10),
            yticks=yticks,
            xlabel="Pair exposures (millions)",
            ylabel="log10 R" if label == "A" else ("accuracy" if label == "B" else "variance-normalized R2"),
            yformat=(lambda value: f"{10 ** value:g}") if label == "A" else (lambda value: f"{value:g}"),
        )
        for condition in ("R0", "R2"):
            aggregate = source["conditions"][condition]
            if label == "A":
                values = [math.log10(max(value, 1e-3)) for value in aggregate["R_mean"]]
                bf.line_path(drawing, [(mx(x), my(y)) for x, y in zip(exposures, values)], color=colors[condition], width=1.8)
            elif label == "B":
                linear = aggregate["layers"]["6"]["linear_accuracy_mean"]
                mlp = aggregate["layers"]["6"]["mlp_accuracy_mean"]
                bf.line_path(drawing, [(mx(x), my(y)) for x, y in zip(exposures, linear)], color=colors[condition], width=1.8)
                bf.line_path(drawing, [(mx(x), my(y)) for x, y in zip(exposures, mlp)], color=colors[condition], width=1.2, dash=(4, 2))
            else:
                values = aggregate["layers"]["6"]["fourier_r2_mean"]
                bf.line_path(drawing, [(mx(x), my(y)) for x, y in zip(exposures, values)], color=colors[condition], width=1.8)
    legend(drawing, [("R0", bf.MUTED, None), ("R2", bf.RED, None), ("MLP (panel B)", bf.INK, (4, 2))], 205, 45)
    bf.text(drawing, 48, 21, "Means across eight paired confirmatory blocks; Fourier is continuous because no registered R2 threshold exists.", size=7, color=bf.MUTED)
    bf.export(drawing, "figure1_representation_emergence")


def figure_2() -> None:
    source = load("figure2_spectral_dynamics.json")
    by_condition = defaultdict(list)
    for point in source["checkpoints"]:
        by_condition[point["condition"]].append(point)
    for points in by_condition.values():
        points.sort(key=lambda item: item["exposure"])
    drawing = bf.Drawing(bf.WIDTH, bf.HEIGHT)
    bf.title(drawing, "Spectral signatures at representative transition checkpoints", "Exploratory")
    panels = [
        (88, "A", "Maximum Hessian eigenvalue", "lambda_max"),
        (412, "B", "Effective spectral dimension", "effective_spectral_dimension_top10"),
    ]
    colors = {"R0": bf.MUTED, "R2": bf.RED}
    for x0, label, name, key in panels:
        values = [float(point[key]) for point in source["checkpoints"]]
        low = min(values)
        high = max(values)
        margin = max((high - low) * 0.15, abs(high) * 0.05, 0.05)
        bf.panel_label(drawing, x0 - 30, 367, label, name)
        mx, my = bf.axes(
            drawing,
            x0,
            88,
            235,
            250,
            xmin=2.5,
            xmax=10.4,
            ymin=low - margin,
            ymax=high + margin,
            xticks=(3, 4, 5, 6, 8, 10),
            yticks=(low, (low + high) / 2, high),
            xlabel="Pair exposures (millions)",
            ylabel="Ritz estimate" if label == "A" else "d_eff (top 10)",
            yformat=lambda value: f"{value:.3g}",
        )
        for condition in ("R0", "R2"):
            points = by_condition[condition]
            coords = [(mx(point["exposure"] / 1e6), my(float(point[key]))) for point in points]
            bf.line_path(drawing, coords, color=colors[condition], width=1.8)
            for point, (x, y) in zip(points, coords):
                bf.marker(drawing, x, y, colors[condition], radius=3.2, open_marker=point["role"] == "pre_transition")
                bf.text(drawing, x, y + 9, point["role"].replace("_transition", ""), size=6, color=bf.MUTED, anchor="middle")
    legend(drawing, [("R0", bf.MUTED, None), ("R2", bf.RED, None)], 265, 45)
    bf.text(drawing, 88, 21, "CPU exact HVPs on the same fixed evaluation batches; associations only, not causal mechanism claims.", size=7, color=bf.MUTED)
    bf.export(drawing, "figure2_spectral_dynamics")


def figure_3() -> None:
    source = load("figure3_cue_removal_timing.json")
    drawing = bf.Drawing(bf.WIDTH, bf.HEIGHT)
    bf.title(drawing, "Cue-removal timing: observed boundaries and unavailable continuations", "Exploratory")

    immediate = {3: [], 4: [], 5: []}
    continuation = defaultdict(list)
    for run in source["runs"]:
        immediate[3].append(float(run["switches"]["3000000"]["immediate_full_support_R"]))
        immediate[4].append(float(run["switches"]["4000000"]["immediate_full_support_R"]))
        immediate[5].append(float(run["switches"]["5000000"]["R_pre"]))
        for point in run["switches"]["5000000"]["continuation"]:
            continuation[int(point["exposed_pairs"])].append(float(point["R"]))

    bf.panel_label(drawing, 37, 367, "A", "Immediate frozen R2 state at switch boundary")
    mx, my = bf.axes(
        drawing, 72, 88, 245, 250, xmin=2.6, xmax=5.4, ymin=-2.6, ymax=0.1,
        xticks=(3, 4, 5), yticks=(-2, -1, 0), xlabel="Removal time (millions)", ylabel="Full-support R (log10)",
        yformat=lambda value: f"{10 ** value:g}",
    )
    for exposure in (3, 4, 5):
        values = [math.log10(max(value, 1e-3)) for value in immediate[exposure]]
        x = mx(exposure)
        drawing.add(bf.Line(x, my(min(values)), x, my(max(values)), strokeColor=bf.RED, strokeWidth=1.2))
        bf.marker(drawing, x, my(sum(values) / len(values)), bf.RED, radius=4)

    bf.panel_label(drawing, 365, 367, "B", "Continuation availability and observed 5M persistence")
    mx2, my2 = bf.axes(
        drawing, 400, 88, 245, 250, xmin=2.6, xmax=10.4, ymin=-2.6, ymax=0.1,
        xticks=(3, 4, 5, 6, 7.5, 10), yticks=(-2, -1, 0), xlabel="Pair exposures (millions)", ylabel="Mean full-support R (log10)",
        yformat=lambda value: f"{10 ** value:g}",
    )
    for exposure in (3, 4):
        x = mx2(exposure)
        drawing.add(bf.Line(x, my2(-2.45), x, my2(-2.1), strokeColor=bf.PURPLE, strokeWidth=5))
    bf.text(drawing, mx2(4.7), my2(-2.38), "3M/4M continuation unavailable", size=6, color=bf.PURPLE, anchor="middle")
    observed_points = [(5.0, sum(immediate[5]) / len(immediate[5]))]
    for exposure in (6_000_000, 7_500_000, 10_000_000):
        observed_points.append((exposure / 1e6, sum(continuation[exposure]) / len(continuation[exposure])))
    coords = [(mx2(x), my2(math.log10(max(y, 1e-3)))) for x, y in observed_points]
    bf.line_path(drawing, coords, color=bf.RED, width=1.8)
    for x, y in coords:
        bf.marker(drawing, x, y, bf.RED, radius=3.2)
    bf.text(drawing, 72, 45, "Points/ranges: observed frozen states across 8 R2 blocks", size=7, color=bf.MUTED)
    bf.text(drawing, 400, 45, "Purple: unavailable continuation; red: observed 5M-switch continuation", size=7, color=bf.MUTED)
    bf.export(drawing, "figure3_cue_removal_timing")


def main() -> None:
    figure_1()
    figure_2()
    figure_3()


if __name__ == "__main__":
    main()
