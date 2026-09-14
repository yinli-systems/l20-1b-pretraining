#!/usr/bin/env python3
"""Export hash-bound, windowed full-run telemetry from one TensorBoard event file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


EXPECTED_EVENT_SHA256 = "295a8a4cfa71709450b559029cf13830781dc7fa9d505e93d6f1ed227a333407"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite {path}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def export(
    event_path: Path,
    output_path: Path,
    *,
    optimizer_steps: int,
    tokens_per_step: int,
    window_size: int,
    peak_flops_per_second: float,
    expected_event_sha256: str,
) -> dict[str, Any]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise RuntimeError("tensorboard is required for source-event export") from exc

    if not event_path.is_file() or event_path.is_symlink():
        raise ValueError("event path must be a regular file")
    event_hash = sha256_file(event_path)
    if event_hash != expected_event_sha256:
        raise ValueError(
            f"event SHA-256 mismatch: actual={event_hash}, expected={expected_event_sha256}"
        )
    if "tfevents" not in event_path.name:
        raise ValueError("TensorBoard EventAccumulator requires a filename containing 'tfevents'")

    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    required = {"step", "tokens", "val_loss", "val_ppl", "device/flops_per_sec", "device/items_per_sec"}
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"event file is missing scalar tags: {missing}")

    step_events = accumulator.Scalars("step")
    if len(step_events) != optimizer_steps:
        raise ValueError(f"expected {optimizer_steps} optimizer steps, found {len(step_events)}")
    step_map = {event.step: int(round(event.value)) for event in step_events}
    if sorted(step_map.values()) != list(range(1, optimizer_steps + 1)):
        raise ValueError("optimizer step scalar is not exactly contiguous")

    token_events = accumulator.Scalars("tokens")
    token_map = {event.step: int(round(event.value)) for event in token_events}
    if set(token_map) != set(step_map):
        raise ValueError("token and optimizer-step scalars are not event-aligned")
    expected_final_tokens = optimizer_steps * tokens_per_step
    if token_map[max(token_map)] != expected_final_tokens:
        raise ValueError(
            f"final token count mismatch: actual={token_map[max(token_map)]}, "
            f"expected={expected_final_tokens}"
        )

    def keyed(tag: str) -> dict[int, tuple[float, float]]:
        result: dict[int, tuple[float, float]] = {}
        for event in accumulator.Scalars(tag):
            if event.step not in step_map:
                raise ValueError(f"{tag} has no matching optimizer step")
            result[step_map[event.step]] = (float(event.value), float(event.wall_time))
        return result

    flops = keyed("device/flops_per_sec")
    throughput = keyed("device/items_per_sec")
    if set(flops) != set(throughput):
        raise ValueError("FLOP/s and token/s samples are not step-aligned")
    covered = sorted(flops)
    if not covered or any(flops[step][0] <= 0 or throughput[step][0] <= 0 for step in covered):
        raise ValueError("telemetry contains no usable positive samples")

    header = [
        "window",
        "declared_step_start",
        "declared_step_end",
        "covered_step_start",
        "covered_step_end",
        "sample_count",
        "first_wall_unix",
        "last_wall_unix",
        "sum_model_flops_per_second",
        "sum_reciprocal_model_flops_per_second",
        "mean_model_flops_per_second",
        "harmonic_model_flops_per_second",
        "sum_tokens_per_second",
        "harmonic_tokens_per_second",
        "covered_logger_seconds",
    ]
    rows: list[list[Any]] = []
    for declared_start in range(1, optimizer_steps + 1, window_size):
        declared_end = min(optimizer_steps, declared_start + window_size - 1)
        selected = [step for step in covered if declared_start <= step <= declared_end]
        if not selected:
            raise ValueError(f"telemetry window {declared_start}-{declared_end} has no samples")
        flops_values = [flops[step][0] for step in selected]
        token_rates = [throughput[step][0] for step in selected]
        reciprocal_flops = sum(1.0 / value for value in flops_values)
        logger_seconds = sum(tokens_per_step / value for value in token_rates)
        rows.append(
            [
                len(rows) + 1,
                declared_start,
                declared_end,
                selected[0],
                selected[-1],
                len(selected),
                f"{flops[selected[0]][1]:.6f}",
                f"{flops[selected[-1]][1]:.6f}",
                f"{sum(flops_values):.6f}",
                f"{reciprocal_flops:.18g}",
                f"{sum(flops_values) / len(selected):.6f}",
                f"{len(selected) / reciprocal_flops:.6f}",
                f"{sum(token_rates):.6f}",
                f"{len(selected) / sum(1.0 / value for value in token_rates):.9f}",
                f"{logger_seconds:.9f}",
            ]
        )
    _atomic_csv(output_path, header, rows)

    flops_values = [flops[step][0] for step in covered]
    reciprocal_flops = sum(1.0 / value for value in flops_values)
    logger_seconds = sum(tokens_per_step / throughput[step][0] for step in covered)
    return {
        "event": {
            "path": str(event_path),
            "bytes": event_path.stat().st_size,
            "sha256": event_hash,
        },
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "coverage": {
            "optimizer_steps": optimizer_steps,
            "samples": len(covered),
            "fraction": len(covered) / optimizer_steps,
            "uncovered_steps": sorted(set(range(1, optimizer_steps + 1)) - set(covered)),
        },
        "mfu": {
            "peak_flops_per_second": peak_flops_per_second,
            "step_weighted": (sum(flops_values) / len(covered)) / peak_flops_per_second,
            "time_weighted": (len(covered) / reciprocal_flops) / peak_flops_per_second,
        },
        "throughput": {
            "covered_logger_seconds": logger_seconds,
            "aggregate_tokens_per_second": tokens_per_step * len(covered) / logger_seconds,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimizer-steps", type=int, default=19148)
    parser.add_argument("--tokens-per-step", type=int, default=1044480)
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--peak-flops-per-second", type=float, default=132e12)
    parser.add_argument("--expected-event-sha256", default=EXPECTED_EVENT_SHA256)
    args = parser.parse_args()
    result = export(
        args.event,
        args.output,
        optimizer_steps=args.optimizer_steps,
        tokens_per_step=args.tokens_per_step,
        window_size=args.window_size,
        peak_flops_per_second=args.peak_flops_per_second,
        expected_event_sha256=args.expected_event_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
