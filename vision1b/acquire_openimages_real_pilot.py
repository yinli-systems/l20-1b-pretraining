#!/usr/bin/env python3
"""Acquire the hash-pinned, visually audited Open Images real-data pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path, expected_bytes: int | None = None) -> None:
    if path.exists() and (expected_bytes is None or path.stat().st_size == expected_bytes):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "vision1b-research-pilot/1"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary_path.open("wb") as output:
            while block := response.read(8 * 1024 * 1024):
                output.write(block)
        if expected_bytes is not None and temporary_path.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"size mismatch for {url}: {temporary_path.stat().st_size} != {expected_bytes}"
            )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_json(path: Path, expected_sha256: str) -> dict:
    if sha256(path) != expected_sha256:
        raise RuntimeError(f"source receipt hash mismatch: {path}")
    return json.loads(path.read_text())


def selected_ids(protocol: dict, repo_root: Path) -> list[str]:
    source = protocol["source"]
    generation = load_json(
        (repo_root / "vision1b" / source["selection_source_path"]).resolve(),
        source["selection_source_sha256"],
    )
    audit = load_json(
        (repo_root / "vision1b" / source["visual_audit_path"]).resolve(),
        source["visual_audit_sha256"],
    )
    if audit.get("decision") != "pass_for_bounded_localization_router_training_only":
        raise RuntimeError("the frozen visual audit did not pass")
    ids: list[str] = []
    for sheet in generation["audit_sheets"]:
        if sheet["split"] != "train":
            continue
        for family_id in sheet["family_ids"]:
            prefix = "oi-er-train-"
            if not family_id.startswith(prefix):
                raise RuntimeError(f"unexpected family id: {family_id}")
            ids.append(family_id.removeprefix(prefix))
    expected = int(source["expected_unique_images"])
    if len(ids) != expected or len(set(ids)) != expected:
        raise RuntimeError(f"expected {expected} unique train image IDs, found {len(set(ids))}")
    return sorted(ids)


def metadata_rows(path: Path, ids: list[str], required_license: str) -> dict[str, dict[str, str]]:
    wanted = set(ids)
    found: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id = row.get("ImageID")
            if image_id not in wanted:
                continue
            if row.get("Subset") != "train":
                raise RuntimeError(f"non-train row selected: {image_id}")
            if row.get("License") != required_license:
                raise RuntimeError(f"license mismatch for {image_id}: {row.get('License')}")
            found[image_id] = row
    missing = sorted(wanted - found.keys())
    if missing:
        raise RuntimeError(f"metadata rows missing: {missing}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    protocol = json.loads(arguments.protocol.read_text())
    if protocol.get("status") != "authorized_bounded_real_image_self_distillation_pilot":
        raise RuntimeError("authorized pilot protocol required")
    repo_root = arguments.protocol.resolve().parent.parent
    ids = selected_ids(protocol, repo_root)
    output = arguments.output_root
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / "sources" / "train-images-boxable-with-rotation.csv"
    source = protocol["source"]
    download(source["image_metadata_url"], metadata, int(source["image_metadata_expected_bytes"]))
    rows = metadata_rows(metadata, ids, source["required_image_license"])
    manifest_rows = []
    for image_id in ids:
        image_path = output / "images" / f"{image_id}.jpg"
        download(source["image_url_template"].format(image_id=image_id), image_path)
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
            mode = image.mode
            image.convert("RGB").load()
        minimum = int(protocol["data_gates"]["minimum_width_and_height"])
        if min(width, height) < minimum:
            raise RuntimeError(f"image too small: {image_id} {width}x{height}")
        row = rows[image_id]
        manifest_rows.append(
            {
                "image_id": image_id,
                "split": "train",
                "image_path": str(image_path.resolve()),
                "image_bytes": image_path.stat().st_size,
                "image_sha256": sha256(image_path),
                "width": width,
                "height": height,
                "mode": mode,
                "license": row["License"],
                "original_url": row.get("OriginalURL", ""),
                "original_landing_url": row.get("OriginalLandingURL", ""),
                "author": row.get("Author", ""),
                "author_profile_url": row.get("AuthorProfileURL", ""),
                "title": row.get("Title", ""),
            }
        )
    manifest = output / "manifest.jsonl"
    temporary = manifest.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows))
    os.replace(temporary, manifest)
    receipt = {
        "status": "PASS",
        "protocol_sha256": sha256(arguments.protocol),
        "metadata": {"path": str(metadata), "bytes": metadata.stat().st_size, "sha256": sha256(metadata)},
        "manifest": {"path": str(manifest), "rows": len(manifest_rows), "sha256": sha256(manifest)},
        "all_images_decode": True,
        "all_images_minimum_224": True,
        "all_licenses_exact_match": True,
        "claim_boundary": protocol["claim_boundary"],
    }
    receipt_path = output / "acquisition-receipt.json"
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, receipt_path)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
