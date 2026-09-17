#!/usr/bin/env python3
"""Create an exact-prediction-token stage manifest from the completed MoE7B pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite stage manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def artifact_inventory(build_root: Path, config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for component in config["components"]:
        directory = build_root / "packed" / component["component_id"]
        receipts = sorted(directory.glob("*.receipt.json"))
        if not receipts:
            raise ValueError(f"component has no completed receipts: {component['component_id']}")
        for receipt_path in receipts:
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("status") != "PACKED_CANDIDATE_NOT_ADMITTED":
                raise ValueError(f"unexpected receipt status: {receipt_path}")
            train = receipt["artifacts"]["train"]
            path = Path(train["path"])
            if path.stat().st_size != train["bytes"] or sha256_file(path) != train["sha256"]:
                raise ValueError(f"packed artifact identity mismatch: {path}")
            blocks = int(receipt["train_blocks"])
            if path.stat().st_size != blocks * int(config["block_tokens"]) * 2:
                raise ValueError(f"packed artifact size/block mismatch: {path}")
            result[component["aggregate_source"]].append(
                {
                    "component_id": component["component_id"],
                    "path": str(path),
                    "sha256": train["sha256"],
                    "blocks": blocks,
                    "prediction_tokens": blocks * int(config["prediction_tokens_per_block"]),
                    "receipt_path": str(receipt_path),
                    "receipt_sha256": sha256_file(receipt_path),
                    "input": receipt["input"],
                }
            )
    return dict(result)


def allocate(
    files: list[dict[str, Any]], cursor: int, count: int
) -> tuple[list[dict[str, Any]], int]:
    remaining = count
    segments = []
    logical_start = 0
    for record in files:
        logical_end = logical_start + int(record["prediction_tokens"])
        if cursor >= logical_end:
            logical_start = logical_end
            continue
        in_file_start = max(0, cursor - logical_start)
        available = logical_end - max(cursor, logical_start)
        take = min(remaining, available)
        if take:
            segments.append(
                {
                    "component_id": record["component_id"],
                    "path": record["path"],
                    "sha256": record["sha256"],
                    "prediction_start": in_file_start,
                    "prediction_count": take,
                    "first_block": in_file_start // 2048,
                    "first_block_prediction_offset": in_file_start % 2048,
                }
            )
            cursor += take
            remaining -= take
            if remaining == 0:
                return segments, cursor
        logical_start = logical_end
    raise ValueError(f"source capacity exhausted with {remaining} prediction tokens unallocated")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--build-contract", type=Path, default=ROOT / "data" / "moe7b_150b_build_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.build_contract.read_text())
    plan_path = ROOT / config["mixture_plan"]
    plan = json.loads(plan_path.read_text())
    progress_path = args.build_root / "progress.json"
    progress = json.loads(progress_path.read_text())
    if not progress.get("all_targets_reached"):
        raise ValueError("cannot finalize before every component reaches its target")
    inventory = artifact_inventory(args.build_root, config)
    cursors = defaultdict(int)
    stages = []
    total = 0
    for stage in plan["stages"]:
        sources = []
        stage_total = 0
        for source_id, count in stage["source_quotas"].items():
            segments, cursors[source_id] = allocate(inventory[source_id], cursors[source_id], int(count))
            sources.append(
                {
                    "source_id": source_id,
                    "prediction_tokens": int(count),
                    "segments": segments,
                }
            )
            stage_total += int(count)
        if stage_total != int(stage["target_tokens"]):
            raise ValueError(f"stage {stage['stage']} quotas do not sum to target")
        stages.append(
            {
                "stage": stage["stage"],
                "purpose": stage["purpose"],
                "prediction_tokens": stage_total,
                "sources": sources,
            }
        )
        total += stage_total
    if total != int(plan["budget"]["target_exposed_tokens"]):
        raise ValueError("allocated stage total does not equal frozen budget")
    manifest = {
        "schema": "moe7b-exact-stage-mixture-manifest-v1",
        "status": "STAGE_MANIFEST_COMPLETE_NOT_ADMITTED",
        "build_id": config["build_id"],
        "build_contract": {"path": str(args.build_contract), "sha256": sha256_file(args.build_contract)},
        "mixture_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "build_progress": {"path": str(progress_path), "sha256": sha256_file(progress_path)},
        "dtype": "uint16-little-endian-raw",
        "block_tokens": int(config["block_tokens"]),
        "prediction_tokens_per_block": int(config["prediction_tokens_per_block"]),
        "exact_prediction_tokens": total,
        "stages": stages,
        "source_artifacts": inventory,
        "admission": {
            "training_admitted": False,
            "open_gates": [
                "final_license_and_attribution_approval",
                "semantic_paraphrase_dedup_audit",
                "complete_public_eval_variant_coverage_including_RULER",
                "clean_room_contamination_audit",
                "language_domain_and_human_content_audits",
                "train_validation_test_isolation",
            ],
        },
        "claim_boundary": "This manifest allocates exactly 150B next-token targets without replay. It remains fail-closed for training until every listed admission gate is independently evidenced.",
    }
    atomic_json(args.output, manifest)
    print(json.dumps({"status": manifest["status"], "exact_prediction_tokens": total, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
