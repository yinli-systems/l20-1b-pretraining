#!/usr/bin/env python3
"""Build an auditable two-pass training manifest for a GPU-time-matched CE arm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic


def duplicate_train_rows(rows: list[dict[str, Any]], passes: int) -> list[dict[str, Any]]:
    if passes < 1:
        raise ValueError("passes must be positive")
    output: list[dict[str, Any]] = []
    train = [row for row in rows if row["split"] == "train"]
    development = [row for row in rows if row["split"] != "train"]
    for pass_index in range(passes):
        suffix = f"::compute-pass-{pass_index + 1}"
        for source in train:
            row = dict(source)
            row["source_scene_family_id"] = source["scene_family_id"]
            row["source_scene_pair_id"] = source["scene_pair_id"]
            row["compute_pass_index"] = pass_index + 1
            row["scene_family_id"] = source["scene_family_id"] + suffix
            row["scene_pair_id"] = source["scene_pair_id"] + suffix
            row["statistical_cluster_id"] = row["scene_pair_id"]
            output.append(row)
    output.extend(dict(row) for row in development)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("time-matched output or receipt already exists")
    rows = [json.loads(line) for line in args.source.read_text().splitlines() if line]
    output = duplicate_train_rows(rows, args.passes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with temporary.open("w") as handle:
        for row in output:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(args.output)
    counts = {
        split: sum(row["split"] == split for row in output)
        for split in sorted({row["split"] for row in output})
    }
    train_rows = [row for row in output if row["split"] == "train"]
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete",
        "passes": args.passes,
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "rows_by_split": counts,
        "train_scene_families": len(train_rows),
        "train_scene_pairs": len({row["scene_pair_id"] for row in train_rows}),
        "source_train_scene_pairs": len({row["source_scene_pair_id"] for row in train_rows}),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "completed_at": utc_now(),
        "claim_boundary": "Training rows are exact semantic repeats with identity suffixes for a compute-matched CE control; no new data diversity is claimed.",
    }
    write_json_atomic(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
