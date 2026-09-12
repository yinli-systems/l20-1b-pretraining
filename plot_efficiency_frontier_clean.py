#!/usr/bin/env python3
"""A deliberately minimal, editorial-style view of the frozen frontier."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "reports/final-efficiency-20260912/frontier-summary.json"
OUT = ROOT / "reports/plots"

INK = "#172A3A"
MUTED = "#6D7D89"
LIGHT = "#D2D9DE"
GRID = "#E5E9EC"
OURS = "#006D77"
TINY = "#8775A3"
FRONTIER = "#E66F66"
GOLD = "#C99700"


def load() -> tuple[list[dict], list[str]]:
    payload = json.loads(SOURCE.read_text())
    if payload.get("status") != "declared_subset_complete" or len(payload["points"]) != 38:
        raise ValueError("Expected frozen complete 38-point summary")
    return payload["points"], payload["apparent_candidate_frontier"]


def x(point: dict) -> float:
    return point["compute"] / 1e20


def token_b(point: dict) -> float:
    return point["tokens"] / 1e9


def token_label(value: float) -> str:
    return f"{value / 1000:g}T" if value >= 1000 else f"{value:g}B"


def halo(text) -> None:
    text.set_path_effects([pe.withStroke(linewidth=3.2, foreground="white")])


def label(ax, point: dict, text: str, offset: tuple[float, float], *, color=MUTED,
          size=8.3, weight="normal", align="left") -> None:
    annotation = ax.annotate(
        text, (x(point), point["score"]), xytext=offset, textcoords="offset points",
        color=color, fontsize=size, fontweight=weight, ha=align, va="center", zorder=8,
    )
    halo(annotation)


def main() -> None:
    points, frontier_ids = load()
    lookup = {point["id"]: point for point in points}
    ours = lookup["ours_20b"]

    tiny_ids = [
        "tinyllama_early_10b", "tinyllama_early_21b", "tinyllama_early_31b",
        "tinyllama_early_63b", "tinyllama_early_105b", "tinyllama_503b",
        "tinyllama_1t", "tinyllama_1_5t", "tinyllama_2t", "tinyllama_2_5t",
        "tinyllama_3t",
    ]
    tiny = [lookup[i] for i in tiny_ids]
    frontier = sorted([lookup[i] for i in frontier_ids], key=x)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    })
    fig, ax = plt.subplots(figsize=(16, 9.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Every frozen point remains present, but non-story points recede.
    ax.scatter([x(p) for p in points], [p["score"] for p in points], s=36,
               color=LIGHT, edgecolor="white", linewidth=0.5, alpha=0.58, zorder=2)

    # TinyLlama is the one comparison trajectory that directly supports the headline.
    ax.plot([x(p) for p in tiny], [p["score"] for p in tiny], color=TINY,
            linewidth=1.65, alpha=0.72, marker="o", markersize=5.0,
            markeredgecolor="white", markeredgewidth=0.65, zorder=4)

    # Frozen point-estimate candidate frontier.
    ax.plot([x(p) for p in frontier], [p["score"] for p in frontier], color=FRONTIER,
            linewidth=1.55, alpha=0.78, linestyle=(0, (2, 3)), zorder=5)
    ax.scatter([x(p) for p in frontier], [p["score"] for p in frontier], s=128,
               facecolor="white", edgecolor=FRONTIER, linewidth=1.45, zorder=6)

    # Teacher/synthetic reference is visually distinct but intentionally quiet.
    phi = lookup["phi_1_5"]
    ax.scatter([x(phi)], [phi["score"]], s=105, marker="P", color=GOLD,
               edgecolor="white", linewidth=0.8, zorder=6)
    label(ax, phi, "Phi-1.5 · teacher/synthetic reference", (10, 0),
          color="#967500", size=8.7, weight="bold")

    # Ours gets a single high-contrast mark and one annotation card.
    ax.hlines(ours["score"], xmin=0.55, xmax=x(ours), color=OURS, linewidth=1.05,
              linestyle=(0, (2, 3)), alpha=0.32, zorder=3)
    ax.vlines(x(ours), ymin=39, ymax=ours["score"], color=OURS, linewidth=1.05,
              linestyle=(0, (2, 3)), alpha=0.32, zorder=3)
    ax.scatter([x(ours)], [ours["score"]], s=1050, color=OURS, alpha=0.10,
               linewidth=0, zorder=7)
    ax.scatter([x(ours)], [ours["score"]], s=470, facecolor="white", edgecolor=OURS,
               linewidth=1.35, zorder=8)
    ax.scatter([x(ours)], [ours["score"]], s=300, marker="*", color=OURS,
               edgecolor="white", linewidth=1.15, zorder=9)
    ax.annotate(
        "OURS  ·  1.10B parameters  ·  20B tokens\n"
        "51.26%  ·  +1.02 pp vs TinyLlama 1T\n"
        "paired 95% CI  [+0.16, +1.89]",
        xy=(x(ours), ours["score"]), xycoords="data",
        xytext=(0.035, 0.80), textcoords="axes fraction",
        fontsize=10.2, fontweight="bold", color="white",
        arrowprops={"arrowstyle": "-", "color": OURS, "lw": 1.0, "alpha": 0.72},
        bbox={"boxstyle": "round,pad=0.55,rounding_size=0.14", "fc": OURS,
              "ec": OURS, "lw": 1.0}, zorder=10,
    )
    label(ax, ours, "OUR MODEL", (0, -30), color=OURS, size=9.3,
          weight="bold", align="center")

    # Label only selected TinyLlama checkpoints: early anchor, transition, and late curve.
    selected_tiny = {
        "tinyllama_early_10b": ("10B", (-2, -16)),
        "tinyllama_early_105b": ("105B", (0, 11)),
        "tinyllama_503b": ("503B", (0, 11)),
        "tinyllama_1t": ("1T", (0, 11)),
        "tinyllama_1_5t": ("1.5T", (0, 11)),
        "tinyllama_2t": ("2T", (0, 11)),
        "tinyllama_2_5t": ("2.5T", (-4, 12)),
        "tinyllama_3t": ("3T", (8, -14)),
    }
    for pid, (text, offset) in selected_tiny.items():
        label(ax, lookup[pid], text, offset, color=TINY, size=8, align="center")
    text = ax.annotate("TinyLlama trajectory", (x(tiny[5]), tiny[5]["score"]),
                       xytext=(12, -20), textcoords="offset points", color=TINY,
                       fontsize=9, fontweight="bold")
    halo(text)

    # Frontier labels are concise; sequential QC7/FW3 checkpoints use token-only tags.
    frontier_labels = [
        ("tinyllama_early_10b", "TinyLlama · 10B", (9, 11)),
        ("dd_dclm_baseline_qc_7p_fw3_10000", "QC7/FW3 · 14B", (-6, 13)),
        ("dd_dclm_baseline_qc_7p_fw3_12500", "18B", (-4, 13)),
        ("dd_dclm_baseline_qc_7p_fw3_15000", "22B", (0, 13)),
        ("dd_dclm_baseline_qc_7p_fw3_20000", "29B", (2, 14)),
        ("weborganizer_domain_mix", "WebOrganizer mix · 29B", (9, 11)),
        ("dd_dclm_baseline_qc_20p_42500", "QC20 · 61B", (10, 12)),
    ]
    for pid, text, offset in frontier_labels:
        label(ax, lookup[pid], text, offset, color=FRONTIER, size=7.9,
              weight="bold" if len(text) > 3 else "normal")

    # Standalone baselines: direct labels remove the need for a legend.
    standalone = [
        ("pythia_1b", "Pythia-1B", (8, -13)),
        ("opt_1_3b", "OPT-1.3B", (8, 10)),
        ("falcon_rw_1b", "Falcon-RW-1B", (8, 10)),
    ]
    for pid, text, offset in standalone:
        label(ax, lookup[pid], text, offset, color=MUTED, size=8.2)

    # One unobtrusive inline key replaces a boxed multi-item legend.
    ax.text(0.745, 0.065, "●  TinyLlama trajectory", transform=ax.transAxes,
            color=TINY, fontsize=8.5, fontweight="bold")
    ax.text(0.745, 0.038, "○  evaluated natural-corpus frontier", transform=ax.transAxes,
            color=FRONTIER, fontsize=8.5, fontweight="bold")
    ax.text(0.745, 0.011, "●  other frozen checkpoints", transform=ax.transAxes,
            color="#AAB4BB", fontsize=8.5)

    ax.set_xscale("log")
    ax.set_xlim(0.55, 250)
    ax.set_ylim(39, 67.2)
    ticks = [0.6, 1, 2, 5, 10, 20, 50, 100, 200]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.minorticks_off()
    ax.set_yticks([40, 45, 50, 55, 60, 65])
    ax.grid(axis="y", color=GRID, linewidth=0.85)
    ax.grid(axis="x", color=GRID, linewidth=0.65, alpha=0.72)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=7)

    ax.set_ylabel("7-task macro accuracy (%)", fontsize=10.5,
                  fontweight="bold", labelpad=12)

    fig.suptitle("20B tokens reach TinyLlama’s 1–2T score range", x=0.055, ha="left",
                 fontsize=23, fontweight="bold", color=INK)
    fig.text(0.056, 0.924,
             "Matched protocol · 38 frozen checkpoints · x: 6ND compute proxy (×10²⁰ FLOPs, log) · y: 7-task macro accuracy",
             color=MUTED, fontsize=10.5)
    fig.text(0.056, 0.018,
             "The red line is a point-estimate frontier only within the evaluated natural-corpus candidates; it is not a global-SOTA claim. "
             "Compute excludes data construction and experiment-search cost.",
             color=MUTED, fontsize=8.3)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "efficiency-frontier-clean-20260912.png"
    svg = OUT / "efficiency-frontier-clean-20260912.svg"
    pdf = OUT / "efficiency-frontier-clean-20260912.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(json.dumps({"points": len(points), "png": str(png), "svg": str(svg), "pdf": str(pdf)}))


if __name__ == "__main__":
    main()
