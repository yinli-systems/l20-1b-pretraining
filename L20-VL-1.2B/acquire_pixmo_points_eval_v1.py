#!/usr/bin/env python3
"""Acquire pinned PixMo-Points-Eval metadata without downloading images."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests


STATUS = "authorized_pixmo_points_eval_metadata_acquisition_only_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def profile_parquet(path: Path) -> dict[str, Any]:
    table = pq.read_table(path, columns=["image_url", "image_sha256", "label", "points"])
    expected_fields = ["image_url", "image_sha256", "label", "points"]
    if table.column_names != expected_fields:
        raise RuntimeError(f"unexpected projected schema: {table.column_names}")
    rows = table.to_pylist()
    point_counts, xs, ys = [], [], []
    image_urls, image_hashes = set(), set()
    for index, row in enumerate(rows):
        if not str(row["image_url"]).startswith(("http://", "https://")):
            raise RuntimeError(f"invalid image URL at row {index}")
        if not SHA256_RE.fullmatch(str(row["image_sha256"])):
            raise RuntimeError(f"invalid image SHA256 at row {index}")
        if not str(row["label"]).strip():
            raise RuntimeError(f"empty label at row {index}")
        points = row["points"] or []
        for point in points:
            x, y = float(point["x"]), float(point["y"])
            if not (0.0 <= x <= 100.0 and 0.0 <= y <= 100.0):
                raise RuntimeError(f"point outside stored 0-100 coordinate space at row {index}")
            xs.append(x)
            ys.append(y)
        point_counts.append(len(points))
        image_urls.add(row["image_url"])
        image_hashes.add(row["image_sha256"])

    mask_table = pq.read_table(path, columns=["masks"])
    mask_counts = pc.list_value_length(mask_table["masks"]).to_pylist()
    return {
        "rows": len(rows),
        "unique_image_urls": len(image_urls),
        "unique_image_sha256": len(image_hashes),
        "total_points": sum(point_counts),
        "rows_with_zero_points": sum(value == 0 for value in point_counts),
        "rows_with_multiple_points": sum(value > 1 for value in point_counts),
        "minimum_points_per_row": min(point_counts),
        "maximum_points_per_row": max(point_counts),
        "total_mask_instances": sum(mask_counts),
        "minimum_masks_per_row": min(mask_counts),
        "maximum_masks_per_row": max(mask_counts),
        "rows_where_point_and_mask_counts_differ": sum(a != b for a, b in zip(point_counts, mask_counts)),
        "stored_point_x_range": [min(xs), max(xs)],
        "stored_point_y_range": [min(ys), max(ys)],
    }


def validate_protocol(path: Path) -> dict[str, Any]:
    from train_stage_a_full_token import sha256_file

    protocol = json.loads(path.read_text())
    if protocol.get("status") != STATUS:
        raise RuntimeError("PixMo points eval metadata acquisition is not authorized")
    for field in ("image_download_authorized", "training_authorized", "model_evaluation_authorized"):
        if protocol.get(field) is not False:
            raise RuntimeError(f"{field} must remain false")
    root = Path(__file__).resolve().parent
    for file_path, key in (
        (Path(__file__), "acquirer_sha256"),
        (root / "test_acquire_pixmo_points_eval_v1.py", "test_sha256"),
    ):
        if sha256_file(file_path) != protocol["source_code"][key]:
            raise RuntimeError(f"source hash mismatch: {file_path.name}")
    return protocol


def verify_source_file(path: Path, source: dict[str, Any]) -> None:
    from train_stage_a_full_token import sha256_file

    if path.stat().st_size != int(source["bytes"]):
        raise RuntimeError("source byte count mismatch")
    if sha256_file(path) != source["sha256"]:
        raise RuntimeError("source SHA256 mismatch")


def download(protocol: dict[str, Any], use_existing_verified_source: bool = False) -> Path:
    from train_stage_a_full_token import sha256_file

    output = Path(protocol["output"]["parquet"])
    if output.exists():
        if not use_existing_verified_source:
            raise FileExistsError(output)
        verify_source_file(output, protocol["source"])
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with requests.get(protocol["source"]["url"], stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with temporary.open("xb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
        verify_source_file(temporary, protocol["source"])
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--use-existing-verified-source", action="store_true")
    args = parser.parse_args()
    protocol = validate_protocol(args.protocol)
    receipt_path = Path(protocol["output"]["receipt"])
    if receipt_path.exists():
        raise FileExistsError(receipt_path)

    from train_stage_a_full_token import sha256_file, utc_now, write_json_atomic

    parquet = download(protocol, use_existing_verified_source=args.use_existing_verified_source)
    profile = profile_parquet(parquet)
    expected = protocol["expected_profile"]
    for key, value in expected.items():
        if profile.get(key) != value:
            raise RuntimeError(f"profile mismatch for {key}: {profile.get(key)} != {value}")
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_verified_pixmo_points_eval_metadata_acquisition_only",
        "completed_at": utc_now(),
        "dataset": protocol["dataset"],
        "source": protocol["source"],
        "protocol": {"path": str(args.protocol), "sha256": sha256_file(args.protocol)},
        "parquet": {"path": str(parquet), "bytes": parquet.stat().st_size, "sha256": sha256_file(parquet)},
        "profile": profile,
        "images_downloaded": 0,
        "model_under_test_accessed": False,
        "training_started": False,
        "model_evaluation_started": False,
        "next_allowed_step": "Freeze image acquisition, integrity, contamination, and human-audit protocol before downloading benchmark images.",
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
