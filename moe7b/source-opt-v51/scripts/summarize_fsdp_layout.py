"""Summarize a bounded 7B FSDP2 layout screen with fail-closed gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--warmup-records", type=int, default=2)
    parser.add_argument("--baseline-throughput", type=float, required=True)
    parser.add_argument("--baseline-causal-mfu", type=float, required=True)
    parser.add_argument("--minimum-speedup", type=float, default=1.10)
    parser.add_argument("--minimum-headroom-gib", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    manifest_path = arguments.run / "run-manifest.json"
    status_path = arguments.run / "training-status.json"
    metrics_path = arguments.run / "metrics.jsonl"
    manifest = json.loads(manifest_path.read_text())
    status = json.loads(status_path.read_text())
    records = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    records = [row for row in records if "tokens_per_second" in row]
    measured = records[arguments.warmup_records :]
    if not measured:
        raise ValueError("no steady-state records remain after warmup")

    finite_keys = (
        "loss",
        "cross_entropy",
        "gradient_norm",
        "tokens_per_second",
        "causal_useful_matmul_mfu",
    )
    all_finite = all(
        all(isinstance(row[key], (int, float)) and math.isfinite(row[key]) for key in finite_keys)
        for row in measured
    )
    throughput = statistics.median(row["tokens_per_second"] for row in measured)
    causal_mfu = statistics.median(
        row["causal_useful_matmul_mfu"] for row in measured
    )
    peak_reserved = max(
        row["peak_memory_reserved_bytes_rank_max"] for row in records
    )
    device_bytes = min(row["memory_bytes"] for row in manifest["runtime"]["devices"])
    headroom = device_bytes - peak_reserved
    speedup = throughput / arguments.baseline_throughput
    causal_mfu_ratio = causal_mfu / arguments.baseline_causal_mfu
    gates = {
        "complete": status.get("status") == "COMPLETED",
        "all_finite": all_finite,
        "eight_verified_devices": len(manifest["runtime"]["devices"]) == 8,
        "constant_tokens_per_optimizer_step": manifest["tokens_per_step"] == 131072,
        "memory_headroom": headroom >= arguments.minimum_headroom_gib * 1024**3,
        "minimum_speedup": speedup >= arguments.minimum_speedup,
    }
    result = {
        "schema": "cvcr-moe-7b-fsdp2-layout-screen-v1",
        "claim_boundary": (
            "Bounded systems evidence only. A passing layout still requires a matched "
            "all-tensor update-parity gate before it can replace the active training layout."
        ),
        "run": str(arguments.run),
        "run_manifest_sha256": sha256(manifest_path),
        "metrics_sha256": sha256(metrics_path),
        "training_status_sha256": sha256(status_path),
        "microbatch": manifest["microbatch"],
        "accumulation": manifest["accumulation"],
        "tokens_per_step": manifest["tokens_per_step"],
        "records_after_warmup": len(measured),
        "median_tokens_per_second": throughput,
        "median_causal_useful_matmul_mfu": causal_mfu,
        "baseline_tokens_per_second": arguments.baseline_throughput,
        "baseline_causal_useful_matmul_mfu": arguments.baseline_causal_mfu,
        "throughput_ratio": speedup,
        "causal_mfu_ratio": causal_mfu_ratio,
        "peak_reserved_bytes": peak_reserved,
        "minimum_device_memory_bytes": device_bytes,
        "headroom_bytes": headroom,
        "last_cross_entropy": measured[-1]["cross_entropy"],
        "gates": gates,
        "promotion": (
            "CANDIDATE_FOR_UPDATE_PARITY" if all(gates.values()) else "REJECT"
        ),
    }
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
