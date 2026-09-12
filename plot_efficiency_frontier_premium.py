#!/usr/bin/env python3
"""Create a publication-grade single-panel efficiency frontier figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "reports/final-efficiency-20260912/frontier-summary.json"
OUT = ROOT / "reports/plots"
OURS_PARAMS = 1_100_048_384

INK = "#152536"
MUTED = "#607383"
GRID = "#CED7DE"
RED = "#F04444"
TEAL = "#007F86"
PURPLE = "#7D56A3"
GOLD = "#D9A514"
COLORS = {
    "dclm": "#3975B7",
    "qc20": "#F28E3B",
    "fineweb": "#2A9D68",
    "qc7fw3": "#D84D62",
    "web": "#8B5E4B",
    "legacy": "#7E8B98",
}

FAMILIES = {
    "DCLM": ("dd_dclm_baseline_", "dclm", "o"),
    "DCLM + QC 20%": ("dd_dclm_baseline_qc_20p_", "qc20", "s"),
    "FineWeb-Edu": ("dd_fineweb_edu_", "fineweb", "^"),
    "QC 7% + FW-Edu 3%": ("dd_dclm_baseline_qc_7p_fw3_", "qc7fw3", "D"),
}


def load() -> tuple[list[dict], list[str]]:
    payload = json.loads(SOURCE.read_text())
    if payload.get("status") != "declared_subset_complete" or len(payload["points"]) != 38:
        raise ValueError("Expected frozen 38-point result set")
    return payload["points"], payload["apparent_candidate_frontier"]


def cx(point: dict) -> float:
    """Approximate compute in units of 1e20 FLOPs."""
    return point["compute"] / 1e20


def tok(point: dict) -> float:
    return point["tokens"] / 1e9


def token_label(value: float) -> str:
    return f"{value / 1000:g}T" if value >= 1000 else f"{value:g}B"


def outlined(text) -> None:
    text.set_path_effects([pe.withStroke(linewidth=3.2, foreground="#FAFBFC")])


def annotate(ax, point: dict, label: str, offset: tuple[float, float], *, color=MUTED,
             size=8.2, weight="normal", arrow=False, box=False, zorder=8) -> None:
    props = {"arrowstyle": "-", "color": color, "lw": 0.7, "alpha": 0.65} if arrow else None
    bbox = ({"boxstyle": "round,pad=0.38,rounding_size=0.15", "fc": "white",
             "ec": color, "lw": 0.9, "alpha": 0.96} if box else None)
    text = ax.annotate(
        label, (cx(point), point["score"]), xytext=offset, textcoords="offset points",
        color=color, fontsize=size, fontweight=weight, arrowprops=props, bbox=bbox,
        zorder=zorder,
    )
    if not box:
        outlined(text)


def main() -> None:
    points, frontier_ids = load()
    lookup = {p["id"]: p for p in points}
    ours = lookup["ours_20b"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": "#415565",
        "ytick.color": "#415565",
    })
    fig, ax = plt.subplots(figsize=(18, 10.4), constrained_layout=True)
    fig.patch.set_facecolor("#F7F9FA")
    ax.set_facecolor("#FBFCFD")

    # Quiet background guide bands improve reading without dominating the marks.
    ax.axhspan(50, 55, color="#DDEBF0", alpha=0.28, zorder=0)
    ax.axhspan(55, 60, color="#DDEFE8", alpha=0.23, zorder=0)

    # Connected data-mixture trajectories.
    for label, (prefix, color_key, marker) in FAMILIES.items():
        family = sorted([p for p in points if p["id"].startswith(prefix)], key=tok)
        ax.plot(
            [cx(p) for p in family], [p["score"] for p in family],
            color=COLORS[color_key], lw=1.8, alpha=0.88, marker=marker,
            markersize=6.7, markeredgecolor="white", markeredgewidth=0.65,
            zorder=4, label=label,
        )

    # TinyLlama trajectory is one continuous series from 10B through 3T.
    tiny_ids = [
        "tinyllama_early_10b", "tinyllama_early_21b", "tinyllama_early_31b",
        "tinyllama_early_63b", "tinyllama_early_105b", "tinyllama_503b",
        "tinyllama_1t", "tinyllama_1_5t", "tinyllama_2t", "tinyllama_2_5t",
        "tinyllama_3t",
    ]
    tiny = [lookup[i] for i in tiny_ids]
    ax.plot(
        [cx(p) for p in tiny], [p["score"] for p in tiny], color=PURPLE,
        lw=2.2, marker="o", markersize=6, markeredgecolor="white",
        markeredgewidth=0.7, zorder=4, label="TinyLlama trajectory",
    )

    # Standalone baselines and teacher/synthetic reference.
    legacy_ids = ["pythia_1b", "opt_1_3b", "falcon_rw_1b"]
    legacy = [lookup[i] for i in legacy_ids]
    ax.scatter([cx(p) for p in legacy], [p["score"] for p in legacy], s=72,
               color=COLORS["legacy"], edgecolor="white", linewidth=0.8, zorder=5)
    phi = lookup["phi_1_5"]
    ax.scatter([cx(phi)], [phi["score"]], s=130, marker="P", color=GOLD,
               edgecolor="white", linewidth=0.9, zorder=6)
    web = [lookup["weborganizer_dclm"], lookup["weborganizer_domain_mix"]]
    ax.scatter([cx(p) for p in web], [p["score"] for p in web], s=105, marker="X",
               color=COLORS["web"], edgecolor="white", linewidth=0.8, zorder=6)

    # Point-estimate frontier, exactly as frozen in the evaluation summary.
    frontier = sorted([lookup[i] for i in frontier_ids], key=cx)
    ax.plot([cx(p) for p in frontier], [p["score"] for p in frontier],
            color=RED, lw=2.0, linestyle=(0, (2, 3)), zorder=3)
    ax.scatter([cx(p) for p in frontier], [p["score"] for p in frontier],
               s=160, facecolor="none", edgecolor=RED, linewidth=1.8, zorder=7)

    # Ours: halo + star + compact statistical callout.
    ax.scatter([cx(ours)], [ours["score"]], s=440, marker="o", color=TEAL,
               alpha=0.12, linewidth=0, zorder=8)
    ax.scatter([cx(ours)], [ours["score"]], s=245, marker="*", color=TEAL,
               edgecolor="white", linewidth=1.0, zorder=9)
    ax.annotate(
        "OURS  ·  1.10B params  ·  20.0B tokens\n"
        "51.26%  ·  vs TinyLlama 1T: +1.02 pp\n"
        "paired 95% CI  [+0.16, +1.89]",
        xy=(cx(ours), ours["score"]), xycoords="data",
        xytext=(0.032, 0.765), textcoords="axes fraction",
        color=TEAL, fontsize=9.2, fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": TEAL, "lw": 1.0, "alpha": 0.72},
        bbox={"boxstyle": "round,pad=0.42,rounding_size=0.15", "fc": "white",
              "ec": TEAL, "lw": 1.0, "alpha": 0.97},
        zorder=12,
    )

    # Short checkpoint labels keep trajectories legible without repeating model names.
    tiny_offsets = {
        "tinyllama_early_10b": (-8, -15), "tinyllama_early_21b": (-7, -15),
        "tinyllama_early_31b": (-5, 8), "tinyllama_early_63b": (-6, 8),
        "tinyllama_early_105b": (-7, 8), "tinyllama_503b": (-7, 8),
        "tinyllama_1t": (-3, 8), "tinyllama_1_5t": (-2, 8),
        "tinyllama_2t": (-1, 8), "tinyllama_2_5t": (-14, 9),
        "tinyllama_3t": (5, -15),
    }
    for p in tiny:
        annotate(ax, p, token_label(tok(p)), tiny_offsets[p["id"]], color=PURPLE, size=7.8)

    # Direct end labels for the four mixture curves; positions are intentionally staggered.
    endpoint_specs = [
        ("dd_dclm_baseline_42500", "DCLM · 61B", (14, -15), "dclm"),
        ("dd_fineweb_edu_42500", "FineWeb-Edu · 61B", (14, 5), "fineweb"),
        ("dd_dclm_baseline_qc_7p_fw3_42500", "QC7/FW3 · 61B", (14, -14), "qc7fw3"),
        ("dd_dclm_baseline_qc_20p_42500", "QC20 · 61B", (14, 8), "qc20"),
    ]
    for pid, label, offset, color_key in endpoint_specs:
        annotate(ax, lookup[pid], label, offset, color=COLORS[color_key], size=8.2,
                 weight="bold")

    # Full labels only for standalone/key references.
    labels = [
        ("weborganizer_dclm", "WebOrganizer · DCLM", (9, -17), COLORS["web"]),
        ("weborganizer_domain_mix", "WebOrganizer · domain mix", (9, 9), COLORS["web"]),
        ("pythia_1b", "Pythia-1B", (7, -15), COLORS["legacy"]),
        ("opt_1_3b", "OPT-1.3B", (7, 8), COLORS["legacy"]),
        ("falcon_rw_1b", "Falcon-RW-1B", (7, 8), COLORS["legacy"]),
        ("phi_1_5", "Phi-1.5  ·  teacher/synthetic reference", (9, 9), "#9A7500"),
    ]
    for pid, label, offset, color in labels:
        annotate(ax, lookup[pid], label, offset, color=color, size=8.3,
                 weight="bold" if pid in {"phi_1_5", "weborganizer_domain_mix"} else "normal")

    ax.set_xscale("log")
    ax.set_xlim(0.52, 250)
    ax.set_ylim(38.8, 67.2)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 4, 5, 6, 7, 8, 9)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.grid(which="major", color=GRID, linewidth=0.95, alpha=0.82)
    ax.grid(which="minor", axis="x", color=GRID, linewidth=0.6, alpha=0.42)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#91A0AC")
    ax.set_xlabel("Approximate training compute  ·  6ND proxy (×10²⁰ FLOPs, log scale)",
                  fontsize=11.5, fontweight="bold", labelpad=12)
    ax.set_ylabel("Matched-protocol 7-task macro accuracy (%)", fontsize=11.5,
                  fontweight="bold", labelpad=10)

    # Equivalent-token top axis makes the compute scale intuitive while remaining honest.
    to_tokens = lambda x: x * 1e20 / (6 * OURS_PARAMS) / 1e9
    from_tokens = lambda t: t * 1e9 * 6 * OURS_PARAMS / 1e20
    top = ax.secondary_xaxis("top", functions=(to_tokens, from_tokens))
    top.set_xscale("log")
    top.set_xlabel("Equivalent training tokens for a 1.10B-parameter model", color=MUTED,
                   fontsize=9.3, labelpad=9)
    top.xaxis.set_major_formatter(FuncFormatter(lambda value, _: token_label(value)))
    top.tick_params(colors=MUTED, labelsize=8)
    top.spines["top"].set_color("#AAB6BF")

    legend_handles = [
        Line2D([0], [0], color=PURPLE, marker="o", lw=2, label="TinyLlama"),
        Line2D([0], [0], color=COLORS["dclm"], marker="o", lw=2, label="DCLM"),
        Line2D([0], [0], color=COLORS["qc20"], marker="s", lw=2, label="DCLM + QC 20%"),
        Line2D([0], [0], color=COLORS["fineweb"], marker="^", lw=2, label="FineWeb-Edu"),
        Line2D([0], [0], color=COLORS["qc7fw3"], marker="D", lw=2, label="QC7/FW3"),
        Line2D([0], [0], color=COLORS["web"], marker="X", lw=0, label="WebOrganizer"),
        Line2D([0], [0], color=COLORS["legacy"], marker="o", lw=0, label="Other baselines"),
        Line2D([0], [0], color=GOLD, marker="P", lw=0, label="Teacher/synthetic ref."),
        Line2D([0], [0], color=RED, lw=2, linestyle=(0, (2, 3)), label="Point-estimate frontier"),
    ]
    legend = ax.legend(handles=legend_handles, title="MODEL FAMILY", loc="lower right",
                       ncol=3, fontsize=8.1, title_fontsize=8.3, frameon=True,
                       borderpad=0.9, columnspacing=1.3, handlelength=2.1)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#D4DCE2")
    legend.get_frame().set_alpha(0.95)

    fig.suptitle("Training Compute Efficiency at ~1B Parameters", x=0.045, ha="left",
                 fontsize=24, fontweight="bold", color=INK)
    fig.text(0.046, 0.928,
             "Matched protocol  ·  38 frozen checkpoints  ·  trajectory labels show training tokens",
             color=MUTED, fontsize=11)
    fig.text(0.046, 0.021,
             "Red dashed line: point-estimate frontier within evaluated natural-corpus candidates — not a global-SOTA claim.  "
             "Compute excludes data construction and experiment-search cost.  Paired uncertainty is reported separately.",
             color=MUTED, fontsize=8.5)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "efficiency-frontier-premium-20260912.png"
    svg = OUT / "efficiency-frontier-premium-20260912.svg"
    pdf = OUT / "efficiency-frontier-premium-20260912.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(json.dumps({"points": len(points), "png": str(png), "svg": str(svg), "pdf": str(pdf)}))


if __name__ == "__main__":
    main()
