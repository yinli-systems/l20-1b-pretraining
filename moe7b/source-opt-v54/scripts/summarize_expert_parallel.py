"""Apply fail-closed MFU and memory gates to the 8-way expert-parallel screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--warmup-records", type=int, default=2)
    parser.add_argument("--minimum-mfu", type=float, default=0.50)
    parser.add_argument("--minimum-headroom-gib", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    manifest_path = arguments.run / "run-manifest.json"
    metrics_path = arguments.run / "metrics.jsonl"
    status_path = arguments.run / "training-status.json"
    manifest = json.loads(manifest_path.read_text())
    status = json.loads(status_path.read_text())
    records = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    measured = records[arguments.warmup_records :]
    if not measured:
        raise ValueError("no steady-state expert-parallel records remain")
    finite_keys = (
        "loss",
        "cross_entropy",
        "gradient_norm",
        "tokens_per_second",
        "causal_useful_matmul_mfu",
    )
    all_finite = all(
        all(math.isfinite(float(row[key])) for key in finite_keys) for row in measured
    )
    throughput = statistics.median(row["tokens_per_second"] for row in measured)
    mfu = statistics.median(row["causal_useful_matmul_mfu"] for row in measured)
    peak_reserved = max(row["peak_memory_reserved_bytes_rank_max"] for row in records)
    device_bytes = min(row["memory_bytes"] for row in manifest["runtime"]["devices"])
    headroom = device_bytes - peak_reserved
    gates = {
        "complete": status.get("status") == "COMPLETED",
        "all_finite": all_finite,
        "eight_verified_devices": len(manifest["runtime"]["devices"]) == 8,
        "expert_parallel_degree_eight": manifest["expert_parallel_degree"] == 8,
        "two_experts_per_rank": manifest["experts_per_rank"] == 2,
        "constant_tokens_per_optimizer_step": manifest["tokens_per_step"] == 131072,
        "memory_headroom": headroom >= arguments.minimum_headroom_gib * 1024**3,
        "minimum_causal_useful_mfu": mfu >= arguments.minimum_mfu,
    }
    result = {
        "schema": "cvcr-moe-7b-expert-parallel-screen-summary-v1",
        "claim_boundary": (
            "A pass qualifies this pure-BF16 layout only for update-parity, optimizer-state, "
            "checkpoint, and sustained-run validation. It cannot start formal training yet."
        ),
        "run_manifest_sha256": sha256(manifest_path),
        "metrics_sha256": sha256(metrics_path),
        "training_status_sha256": sha256(status_path),
        "records_after_warmup": len(measured),
        "median_tokens_per_second": throughput,
        "median_causal_useful_matmul_mfu": mfu,
        "minimum_causal_useful_mfu": arguments.minimum_mfu,
        "peak_reserved_bytes": peak_reserved,
        "minimum_device_memory_bytes": device_bytes,
        "headroom_bytes": headroom,
        "last_cross_entropy": measured[-1]["cross_entropy"],
        "gates": gates,
        "promotion": (
            "CANDIDATE_FOR_NUMERICAL_QUALIFICATION"
            if all(gates.values())
            else "REJECT"
        ),
    }
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
