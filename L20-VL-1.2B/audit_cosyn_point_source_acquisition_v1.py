#!/usr/bin/env python3
"""Verify two pinned CoSyn-point parquet shards after controlled transfer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent
STATUS = "authorized_cosyn_point_bounded_acquisition_audit_v1"
REQUIRED_COLUMNS = {"id", "image", "questions", "answer_points", "names"}


def validate_source(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    if actual_bytes != int(spec["expected_bytes"]):
        raise RuntimeError(f"byte-size mismatch: {path.name}")
    parquet = pq.ParquetFile(path)
    columns = set(parquet.schema_arrow.names)
    if columns != REQUIRED_COLUMNS:
        raise RuntimeError(f"unexpected schema: {sorted(columns)}")
    table = pq.read_table(path, columns=["id"])
    ids = table.column("id").to_pylist()
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate ids within {path.name}")
    return {
        "path": str(path),
        "bytes": actual_bytes,
        "sha256": sha256_file(path),
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "columns": sorted(columns),
        "unique_ids": len(ids),
        "ids": set(ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != STATUS or protocol.get("training_authorized") is not False:
        raise SystemExit("bounded acquisition-audit protocol required")
    for path, key in (
        (Path(__file__), "auditor_sha256"),
        (ROOT / "test_audit_cosyn_point_source_acquisition_v1.py", "test_sha256"),
    ):
        if sha256_file(path) != protocol["source_code"][key]:
            raise SystemExit(f"source hash mismatch: {path.name}")
    output = Path(protocol["output"]["receipt"])
    if output.exists():
        raise FileExistsError(output)
    records = {}
    for name, spec in protocol["sources"].items():
        record = validate_source(Path(spec["destination"]), spec)
        ids = record.pop("ids")
        records[name] = record
        records[name]["_ids_for_overlap"] = ids
    names = list(records)
    overlap = records[names[0]]["_ids_for_overlap"] & records[names[1]]["_ids_for_overlap"]
    if overlap:
        raise RuntimeError(f"cross-shard id overlap: {len(overlap)}")
    for record in records.values():
        record.pop("_ids_for_overlap")
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_cosyn_point_bounded_acquisition_audit_v1",
        "completed_at": utc_now(),
        "protocol": {"path": str(args.protocol), "sha256": sha256_file(args.protocol)},
        "dataset_repository": protocol["dataset_repository"],
        "sources": records,
        "cross_shard_id_overlap": 0,
        "official_validation_accessed": False,
        "training_started": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
