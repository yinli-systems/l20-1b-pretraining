#!/usr/bin/env python3
"""Extract complete TensorBoard loss metrics and render the final English figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_TRAINING_CSV = ROOT / "reports/metrics/training-loss-100-step.csv"
DEFAULT_VALIDATION_CSV = ROOT / "reports/metrics/validation-loss.csv"
DEFAULT_OUTPUT_PREFIX = ROOT / "reports/figures/loss-curve-final-en"
DEFAULT_RECEIPT = ROOT / "reports/receipts/loss-curve-final-en.json"
WINDOW_STEPS = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def extract_event_metrics(
    event_file: Path,
    training_csv: Path,
    validation_csv: Path,
) -> dict[str, object]:
    import numpy as np
    from tensorboard.backend.event_processing import event_accumulator

    accumulator = event_accumulator.EventAccumulator(
        str(event_file), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))
    required = {"loss", "step", "tokens", "val_loss", "val_ppl"}
    missing = sorted(required - scalar_tags)
    if missing:
        raise ValueError(f"missing TensorBoard scalar tags: {missing}")

    loss_events = accumulator.Scalars("loss")
    step_events = accumulator.Scalars("step")
    token_events = accumulator.Scalars("tokens")
    if not (len(loss_events) == len(step_events) == len(token_events)):
        raise ValueError("training scalar series have different lengths")

    points: list[tuple[int, int, float]] = []
    event_step_to_optimizer_step: dict[int, int] = {}
    for loss_event, step_event, token_event in zip(
        loss_events, step_events, token_events
    ):
        if not (loss_event.step == step_event.step == token_event.step):
            raise ValueError("training scalar event steps are not aligned")
        optimizer_step = int(round(step_event.value))
        tokens = int(round(token_event.value))
        points.append((optimizer_step, tokens, float(loss_event.value)))
        event_step_to_optimizer_step[step_event.step] = optimizer_step

    optimizer_steps = [point[0] for point in points]
    expected_steps = list(range(1, len(points) + 1))
    if optimizer_steps != expected_steps:
        raise ValueError("optimizer steps are not contiguous from one")
    if any(right[1] <= left[1] for left, right in zip(points, points[1:])):
        raise ValueError("training tokens are not strictly increasing")

    training_rows: list[dict[str, object]] = []
    for start in range(0, len(points), WINDOW_STEPS):
        window = points[start : start + WINDOW_STEPS]
        losses = np.asarray([point[2] for point in window], dtype=np.float64)
        tokens = np.asarray([point[1] for point in window], dtype=np.float64)
        training_rows.append(
            {
                "step_start": window[0][0],
                "step_end": window[-1][0],
                "tokens_b": f"{tokens.mean() / 1e9:.6f}",
                "loss_mean": f"{losses.mean():.6f}",
                "loss_p10": f"{np.percentile(losses, 10):.6f}",
                "loss_p90": f"{np.percentile(losses, 90):.6f}",
            }
        )

    val_loss_events = accumulator.Scalars("val_loss")
    val_ppl_events = accumulator.Scalars("val_ppl")
    if len(val_loss_events) != len(val_ppl_events):
        raise ValueError("validation loss and perplexity series have different lengths")

    final_optimizer_step = points[-1][0]
    final_tokens = points[-1][1]
    tokens_per_step = final_tokens // final_optimizer_step
    if tokens_per_step * final_optimizer_step != final_tokens:
        raise ValueError("final token count is not step-aligned")

    validation_rows: list[dict[str, object]] = []
    for index, (loss_event, ppl_event) in enumerate(
        zip(val_loss_events, val_ppl_events)
    ):
        if loss_event.step != ppl_event.step:
            raise ValueError("validation loss and perplexity events are not aligned")
        optimizer_step = event_step_to_optimizer_step.get(loss_event.step)
        if optimizer_step is None and index == len(val_loss_events) - 1:
            optimizer_step = final_optimizer_step
        if optimizer_step is None:
            raise ValueError(f"cannot map validation event step {loss_event.step}")
        loss = float(loss_event.value)
        ppl = float(ppl_event.value)
        if not math.isclose(math.exp(loss), ppl, rel_tol=2e-6):
            raise ValueError("validation perplexity does not match exp(loss)")
        validation_rows.append(
            {
                "step": optimizer_step,
                "tokens_b": f"{optimizer_step * tokens_per_step / 1e9:.6f}",
                "loss": f"{loss:.9f}",
                "ppl": f"{ppl:.9f}",
            }
        )

    expected_validation_steps = list(range(500, final_optimizer_step, 500)) + [
        final_optimizer_step
    ]
    actual_validation_steps = [int(row["step"]) for row in validation_rows]
    if actual_validation_steps != expected_validation_steps:
        raise ValueError("validation steps do not match the frozen 500-step schedule")

    write_csv(
        training_csv,
        ["step_start", "step_end", "tokens_b", "loss_mean", "loss_p10", "loss_p90"],
        training_rows,
    )
    write_csv(validation_csv, ["step", "tokens_b", "loss", "ppl"], validation_rows)

    return {
        "event_file_sha256": sha256_file(event_file),
        "event_file_bytes": event_file.stat().st_size,
        "scalar_counts": {
            "loss": len(loss_events),
            "tokens": len(token_events),
            "val_loss": len(val_loss_events),
            "val_ppl": len(val_ppl_events),
        },
        "optimizer_steps": final_optimizer_step,
        "prediction_tokens": final_tokens,
        "tokens_per_step": tokens_per_step,
        "training_windows": len(training_rows),
        "validation_points": len(validation_rows),
    }


def read_numeric_csv(path: Path) -> dict[str, list[float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return {key: [float(row[key]) for row in rows] for key in rows[0]}


def render_figure(training_csv: Path, validation_csv: Path, output_prefix: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    training = read_numeric_csv(training_csv)
    validation = read_numeric_csv(validation_csv)
    train_tokens = np.asarray(training["tokens_b"])
    train_mean = np.asarray(training["loss_mean"])
    train_p10 = np.asarray(training["loss_p10"])
    train_p90 = np.asarray(training["loss_p90"])
    val_tokens = np.asarray(validation["tokens_b"])
    val_loss = np.asarray(validation["loss"])
    val_ppl = np.asarray(validation["ppl"])

    blue = "#2563EB"
    blue_fill = "#93C5FD"
    red = "#F0445E"
    violet = "#7C3AED"
    ink = "#101828"
    muted = "#667085"
    grid = "#D0D5DD"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11.5,
            "axes.edgecolor": "#98A2B3",
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.labelcolor": ink,
            "xtick.color": "#344054",
            "ytick.color": "#344054",
            "svg.hashsalt": "l20-1b-loss-final-en-v1",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), dpi=200)
    fig.patch.set_facecolor("#F8FAFC")
    for ax in axes:
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, color=grid, alpha=0.55, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_handles = [
        Patch(facecolor=blue_fill, edgecolor=blue, alpha=0.30, label="100-step P10–P90"),
        Line2D([0], [0], color=blue, linewidth=2.4, label="100-step mean training loss"),
        Line2D(
            [0],
            [0],
            marker="o",
            color=red,
            markerfacecolor=red,
            markeredgecolor="white",
            linewidth=1.2,
            markersize=7,
            label="Held-out validation loss",
        ),
    ]

    for ax in axes[:1]:
        ax.fill_between(train_tokens, train_p10, train_p90, color=blue_fill, alpha=0.30)
        ax.plot(train_tokens, train_mean, color=blue, linewidth=2.35)
        ax.plot(val_tokens, val_loss, color=red, alpha=0.38, linewidth=1.2)
        ax.scatter(
            val_tokens,
            val_loss,
            s=46,
            color=red,
            edgecolor="white",
            linewidth=1.2,
            zorder=5,
        )
        ax.set_xlabel("Cumulative training tokens (billions)")
        ax.set_ylabel("Cross-entropy loss")
        ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=9.5)

    axes[0].set_title("Full training convergence", loc="left", color=ink, pad=12)
    axes[0].set_xlim(0, 20.25)
    axes[0].set_ylim(2.25, 9.05)
    axes[0].set_xticks(np.arange(0, 21, 2))

    late_mask = train_tokens >= 7.0
    axes[1].fill_between(
        train_tokens[late_mask], train_p10[late_mask], train_p90[late_mask], color=blue_fill, alpha=0.30
    )
    axes[1].plot(train_tokens[late_mask], train_mean[late_mask], color=blue, linewidth=2.35)
    val_late_mask = val_tokens >= 7.0
    axes[1].plot(val_tokens[val_late_mask], val_loss[val_late_mask], color=red, alpha=0.38, linewidth=1.2)
    axes[1].scatter(
        val_tokens[val_late_mask],
        val_loss[val_late_mask],
        s=46,
        color=red,
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
    )
    axes[1].set_xlabel("Cumulative training tokens (billions)")
    axes[1].set_ylabel("Cross-entropy loss")
    axes[1].set_title("Late-stage training detail", loc="left", color=ink, pad=12)
    axes[1].set_xlim(7.0, 20.25)
    axes[1].set_ylim(2.36, 2.69)
    axes[1].set_xticks(np.arange(8, 21, 2))
    axes[1].axvline(val_tokens[-1], color=violet, linestyle=(0, (4, 3)), linewidth=1.35, alpha=0.8)
    axes[1].text(
        val_tokens[-1] - 0.18,
        2.372,
        "20.00B · complete",
        color=violet,
        fontsize=9.5,
        fontweight="bold",
        ha="right",
        va="bottom",
    )
    axes[1].annotate(
        f"Final held-out result\nLoss {val_loss[-1]:.4f} · PPL {val_ppl[-1]:.4f}",
        xy=(val_tokens[-1], val_loss[-1]),
        xytext=(15.25, 2.505),
        color=red,
        fontsize=10.2,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": red, "linewidth": 1.4},
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#FDA4AF", "alpha": 0.96},
    )
    axes[1].legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=9.5)

    fig.suptitle(
        "L20-1B-20B-Base — From-Scratch Pretraining",
        x=0.5,
        y=0.965,
        fontsize=23,
        fontweight="bold",
        color=ink,
    )
    fig.text(
        0.5,
        0.918,
        "1.10B parameters · 19.9997B prediction tokens · one NVIDIA L20",
        ha="center",
        va="center",
        fontsize=12.5,
        color=muted,
    )
    fig.text(
        0.5,
        0.035,
        "Training: non-overlapping 100-step windows (mean and P10–P90).  Validation: fixed held-out set every 500 steps plus final step.  All curves are measured; no extrapolation.",
        ha="center",
        va="center",
        fontsize=9.5,
        color=muted,
    )
    fig.subplots_adjust(left=0.062, right=0.986, bottom=0.13, top=0.83, wspace=0.15)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    fig.savefig(
        png_path,
        dpi=200,
        facecolor=fig.get_facecolor(),
        metadata={
            "Software": "Matplotlib",
            "Description": "Complete English loss curve for L20-1B-20B-Base",
        },
    )
    fig.savefig(
        svg_path,
        facecolor=fig.get_facecolor(),
        metadata={
            "Creator": "Matplotlib",
            "Date": None,
            "Description": "Complete English loss curve for L20-1B-20B-Base",
        },
    )
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-file", type=Path)
    parser.add_argument("--training-csv", type=Path, default=DEFAULT_TRAINING_CSV)
    parser.add_argument("--validation-csv", type=Path, default=DEFAULT_VALIDATION_CSV)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()

    source: dict[str, object] | None = None
    if args.event_file is not None:
        source = extract_event_metrics(args.event_file, args.training_csv, args.validation_csv)

    png_path, svg_path = render_figure(
        args.training_csv, args.validation_csv, args.output_prefix
    )
    validation = read_numeric_csv(args.validation_csv)
    source_event = source
    if source_event is None and args.receipt.exists():
        previous = json.loads(args.receipt.read_text(encoding="utf-8"))
        source_event = previous.get("source_event")
    exact_prediction_tokens = (
        int(source_event["prediction_tokens"])
        if source_event is not None
        else int(round(validation["tokens_b"][-1] * 1e9))
    )
    result = {
        "schema_version": 1,
        "language": "en",
        "measurement_boundary": "all plotted curves are measured; no extrapolation",
        "aggregation": {
            "training": "non-overlapping 100-step mean with 10th and 90th percentiles",
            "validation": "fixed held-out evaluation every 500 steps plus the final step",
        },
        "final": {
            "step": int(validation["step"][-1]),
            "prediction_tokens": exact_prediction_tokens,
            "validation_loss": validation["loss"][-1],
            "validation_perplexity": validation["ppl"][-1],
        },
        "files": {
            "training_csv": {"path": str(args.training_csv.relative_to(ROOT)), "sha256": sha256_file(args.training_csv)},
            "validation_csv": {"path": str(args.validation_csv.relative_to(ROOT)), "sha256": sha256_file(args.validation_csv)},
            "png": {"path": str(png_path.relative_to(ROOT)), "sha256": sha256_file(png_path)},
            "svg": {"path": str(svg_path.relative_to(ROOT)), "sha256": sha256_file(svg_path)},
        },
    }
    if source_event is not None:
        result["source_event"] = source_event
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["final"], sort_keys=True))


if __name__ == "__main__":
    main()
