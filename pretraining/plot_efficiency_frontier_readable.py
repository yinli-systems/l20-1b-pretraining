#!/usr/bin/env python3
"""Render a readable, multi-panel view of the frozen efficiency frontier data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "reports/final-efficiency-20260912/frontier-summary.json"
OUT = ROOT / "reports/plots"

COLORS = {
    "ours": "#087F8C",
    "tinyllama": "#7A5195",
    "dclm": "#3B6FB6",
    "qc20": "#E07A2D",
    "fineweb": "#2A9D63",
    "qc7fw3": "#D1495B",
    "weborganizer": "#8A5A44",
    "other": "#7B8794",
    "phi": "#D39B00",
}

FAMILIES = {
    "DCLM": ("dd_dclm_baseline_", "dclm", "o"),
    "DCLM + QC 20%": ("dd_dclm_baseline_qc_20p_", "qc20", "s"),
    "FineWeb-Edu": ("dd_fineweb_edu_", "fineweb", "^"),
    "QC 7% + FW-Edu 3%": ("dd_dclm_baseline_qc_7p_fw3_", "qc7fw3", "D"),
}


def load_points() -> list[dict]:
    payload = json.loads(SOURCE.read_text())
    if payload.get("status") != "declared_subset_complete" or len(payload["points"]) != 38:
        raise ValueError("Expected the frozen, complete 38-point frontier summary")
    return payload["points"]


def by_id(points: list[dict]) -> dict[str, dict]:
    return {point["id"]: point for point in points}


def token_b(point: dict) -> float:
    return point["tokens"] / 1e9


def compute_e20(point: dict) -> float:
    return point["compute"] / 1e20


def style_axes(ax) -> None:
    ax.grid(True, color="#DDE3E8", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#344554")


def plot_family(ax, points: list[dict], label: str, prefix: str, color_key: str, marker: str,
                *, x_kind: str = "tokens", max_tokens: float | None = None) -> None:
    family = [p for p in points if p["id"].startswith(prefix)]
    if max_tokens is not None:
        family = [p for p in family if token_b(p) <= max_tokens]
    family.sort(key=token_b)
    x = [token_b(p) if x_kind == "tokens" else compute_e20(p) for p in family]
    ax.plot(x, [p["score"] for p in family], color=COLORS[color_key], marker=marker,
            markersize=6.5, linewidth=2, label=label)


def plot_ours(ax, ours: dict, *, x_kind: str = "tokens", annotate: bool = True) -> None:
    x = token_b(ours) if x_kind == "tokens" else compute_e20(ours)
    ax.scatter([x], [ours["score"]], s=220, marker="*", color=COLORS["ours"],
               edgecolor="white", linewidth=1.2, zorder=10, label="Ours 20B")
    if annotate:
        ax.annotate(f'Ours\n{ours["score"]:.2f}%', (x, ours["score"]), xytext=(8, 10),
                    textcoords="offset points", color=COLORS["ours"], weight="bold",
                    fontsize=9)


def tinyllama_points(points: list[dict]) -> list[dict]:
    ids = [
        "tinyllama_early_10b", "tinyllama_early_21b", "tinyllama_early_31b",
        "tinyllama_early_63b", "tinyllama_early_105b", "tinyllama_503b",
        "tinyllama_1t", "tinyllama_1_5t", "tinyllama_2t", "tinyllama_2_5t",
        "tinyllama_3t",
    ]
    lookup = by_id(points)
    return [lookup[i] for i in ids]


def main() -> None:
    points = load_points()
    lookup = by_id(points)
    ours = lookup["ours_20b"]
    tiny = tinyllama_points(points)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#233442",
        "text.color": "#233442",
    })
    fig, axes = plt.subplots(2, 2, figsize=(19, 13), constrained_layout=True)

    # A: global context with only selective labels.
    ax = axes[0, 0]
    natural = [p for p in points if p["category"] == "natural_corpus_base"]
    ax.scatter([compute_e20(p) for p in natural], [p["score"] for p in natural],
               s=35, color="#B6C0C9", alpha=0.65, label="Other evaluated points")
    ax.plot([compute_e20(p) for p in tiny], [p["score"] for p in tiny],
            color=COLORS["tinyllama"], marker="o", markersize=4.5, linewidth=1.8,
            label="TinyLlama trajectory")
    phi = lookup["phi_1_5"]
    ax.scatter([compute_e20(phi)], [phi["score"]], s=85, marker="P", color=COLORS["phi"],
               label="Phi-1.5 (teacher/synthetic ref.)", zorder=6)
    plot_ours(ax, ours, x_kind="compute")
    for pid, label, offset in [
        ("weborganizer_domain_mix", "WebOrganizer mix", (7, 7)),
        ("tinyllama_2_5t", "TinyLlama 2.5T", (-86, 10)),
        ("falcon_rw_1b", "Falcon-RW-1B", (7, 7)),
        ("phi_1_5", "Phi-1.5", (7, 7)),
    ]:
        p = lookup[pid]
        ax.annotate(label, (compute_e20(p), p["score"]), xytext=offset,
                    textcoords="offset points", fontsize=8.5)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.set_xlabel("Approx. training compute, 6ND (×10²⁰ FLOPs; log scale)")
    ax.set_ylabel("7-task macro accuracy (%)")
    ax.set_title("A. Global context — labels intentionally selective")
    ax.legend(loc="lower right", frameon=True, fontsize=8.5)
    style_axes(ax)

    # B: close-up around the user's 20B-token compute regime.
    ax = axes[0, 1]
    for label, (prefix, color_key, marker) in FAMILIES.items():
        plot_family(ax, points, label, prefix, color_key, marker, max_tokens=32)
    early = [p for p in tiny if token_b(p) <= 32]
    ax.plot([token_b(p) for p in early], [p["score"] for p in early],
            color=COLORS["tinyllama"], marker="o", linewidth=2, label="TinyLlama early")
    web = [lookup["weborganizer_dclm"], lookup["weborganizer_domain_mix"]]
    ax.scatter([token_b(p) for p in web], [p["score"] for p in web], s=75, marker="X",
               color=COLORS["weborganizer"], label="WebOrganizer", zorder=6)
    plot_ours(ax, ours)
    ax.axvspan(18, 22, color=COLORS["ours"], alpha=0.055)
    ax.set_xlim(8, 33)
    ax.set_ylim(39, 56.5)
    ax.set_xlabel("Training tokens (billions)")
    ax.set_ylabel("7-task macro accuracy (%)")
    ax.set_title("B. 10–32B-token close-up — the crowded region expanded")
    ax.legend(loc="lower right", ncol=2, frameon=True, fontsize=8.2)
    style_axes(ax)

    # C: data-mixture scaling curves, separated by lines instead of labels.
    ax = axes[1, 0]
    for label, (prefix, color_key, marker) in FAMILIES.items():
        plot_family(ax, points, label, prefix, color_key, marker)
    ax.scatter([token_b(p) for p in web], [p["score"] for p in web], s=80, marker="X",
               color=COLORS["weborganizer"], label="WebOrganizer @ 28.8B", zorder=6)
    plot_ours(ax, ours)
    ax.set_xlim(11, 65)
    ax.set_ylim(48, 58)
    ax.set_xlabel("Training tokens (billions)")
    ax.set_ylabel("7-task macro accuracy (%)")
    ax.set_title("C. Data-mixture scaling — four trajectories no longer overlap")
    ax.legend(loc="lower right", ncol=2, frameon=True, fontsize=8.5)
    style_axes(ax)

    # D: TinyLlama's complete trajectory on a dedicated token axis.
    ax = axes[1, 1]
    ax.plot([token_b(p) for p in tiny], [p["score"] for p in tiny],
            color=COLORS["tinyllama"], marker="o", markersize=6, linewidth=2.2,
            label="TinyLlama checkpoints")
    plot_ours(ax, ours)
    for p in tiny:
        tok = token_b(p)
        label = f"{tok/1000:g}T" if tok >= 1000 else f"{tok:g}B"
        dy = 8 if p["id"] not in {"tinyllama_2_5t", "tinyllama_3t"} else -15
        ax.annotate(label, (tok, p["score"]), xytext=(0, dy), textcoords="offset points",
                    ha="center", fontsize=8, color=COLORS["tinyllama"])
    ax.axhline(ours["score"], color=COLORS["ours"], linewidth=1.1, alpha=0.55,
               linestyle="--")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v/1000:g}T" if v >= 1000 else f"{v:g}B"
    ))
    ax.set_ylim(39, 56)
    ax.set_xlabel("Training tokens (log scale)")
    ax.set_ylabel("7-task macro accuracy (%)")
    ax.set_title("D. TinyLlama trajectory — each checkpoint is visible")
    ax.legend(loc="lower right", frameon=True, fontsize=8.5)
    style_axes(ax)

    fig.suptitle("1.1B Pretraining Efficiency — Readable Multi-Scale View", fontsize=19,
                 fontweight="bold")
    fig.text(
        0.5, -0.012,
        "Frozen 38-point matched-protocol subset. Point estimates only; paired 95% CIs remain in the result table. "
        "Compute excludes data construction and experiment-search cost; this is not a global-SOTA claim.",
        ha="center", va="bottom", fontsize=9, color="#536574",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "efficiency-frontier-readable-20260912.png"
    svg = OUT / "efficiency-frontier-readable-20260912.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    print(json.dumps({"source": str(SOURCE), "points": len(points), "png": str(png), "svg": str(svg)}))


if __name__ == "__main__":
    main()
