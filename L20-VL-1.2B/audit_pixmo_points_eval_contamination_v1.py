#!/usr/bin/env python3
"""Audit exact image overlap between PixMo-Points-Eval and project manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import pyarrow.parquet as pq


STATUS = "authorized_pixmo_points_eval_exact_project_overlap_audit_only_v1"
IMAGE_HASH_KEYS = {
    "image_sha256",
    "base_image_sha256",
    "edited_image_sha256",
    "invariant_image_sha256",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def extract_image_hashes(value: Any) -> set[str]:
    output: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in IMAGE_HASH_KEYS and isinstance(child, str) and SHA256_RE.fullmatch(child):
                output.add(child)
            else:
                output.update(extract_image_hashes(child))
    elif isinstance(value, list):
        for child in value:
            output.update(extract_image_hashes(child))
    return output


def audit_manifest(path: Path, evaluation_hashes: set[str]) -> dict[str, Any]:
    hashes: set[str] = set()
    rows = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows += 1
        hashes.update(extract_image_hashes(json.loads(line)))
    overlap = sorted(hashes & evaluation_hashes)
    return {
        "rows": rows,
        "unique_image_hashes": len(hashes),
        "overlap_count": len(overlap),
        "overlap_sha256": overlap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    if not args.protocol.exists():
        raise FileNotFoundError(args.protocol)

    from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic

    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != STATUS:
        raise RuntimeError("exact project-overlap audit is not authorized")
    for field in ("image_download_authorized", "training_authorized", "model_evaluation_authorized"):
        if protocol.get(field) is not False:
            raise RuntimeError(f"{field} must remain false")
    root = Path(__file__).resolve().parent
    for path, key in (
        (Path(__file__), "auditor_sha256"),
        (root / "test_audit_pixmo_points_eval_contamination_v1.py", "test_sha256"),
    ):
        if sha256_file(path) != protocol["source_code"][key]:
            raise RuntimeError(f"source hash mismatch: {path.name}")
    receipt_path = Path(protocol["output_receipt"])
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    for name, item in protocol["prerequisites"].items():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"prerequisite hash mismatch: {name}")

    parquet = Path(protocol["prerequisites"]["evaluation_parquet"]["path"])
    evaluation_hashes = set(pq.read_table(parquet, columns=["image_sha256"])["image_sha256"].to_pylist())
    if len(evaluation_hashes) != int(protocol["expected_unique_evaluation_images"]):
        raise RuntimeError("evaluation image count changed")
    sources = []
    seen_paths: set[str] = set()
    for item in protocol["project_manifests"]:
        path = Path(item["path"])
        if str(path) in seen_paths:
            raise RuntimeError(f"duplicate manifest path: {path}")
        seen_paths.add(str(path))
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"manifest hash mismatch: {path}")
        result = audit_manifest(path, evaluation_hashes)
        sources.append({**item, **result})
    blocking = [item for item in sources if item["evidence_class"] in {"model_training", "model_development"} and item["overlap_count"]]
    any_overlap = [item for item in sources if item["overlap_count"]]
    if blocking:
        decision = "blocked_due_to_known_project_train_or_development_overlap"
        next_step = "Do not use overlapping rows for an independent project-level evaluation."
    else:
        decision = "pass_zero_known_project_train_or_development_exact_overlap"
        next_step = "Freeze image acquisition and metric protocols; retain the upstream vision-pretraining contamination caveat."
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_pixmo_points_eval_exact_project_overlap_audit",
        "completed_at": utc_now(),
        "decision": decision,
        "protocol": {"path": str(args.protocol), "sha256": sha256_file(args.protocol)},
        "auditor_sha256": sha256_file(Path(__file__)),
        "evaluation_unique_image_sha256": len(evaluation_hashes),
        "manifests": sources,
        "blocking_manifest_count": len(blocking),
        "manifests_with_any_overlap": len(any_overlap),
        "known_project_exact_overlap_sha256": sorted({value for item in sources for value in item["overlap_sha256"]}),
        "image_download_started": False,
        "model_under_test_accessed": False,
        "training_started": False,
        "model_evaluation_started": False,
        "upstream_vision_pretraining_overlap_auditable": False,
        "next_allowed_step": next_step,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(receipt_path, receipt)
    print(json.dumps({
        "status": receipt["status"],
        "decision": decision,
        "manifests": len(sources),
        "manifests_with_any_overlap": len(any_overlap),
        "blocking_manifest_count": len(blocking),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
