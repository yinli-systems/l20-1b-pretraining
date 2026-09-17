#!/usr/bin/env python3
"""Acquire a deterministic, attribution-preserving Open Images 25k pilot."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import heapq
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.request

from PIL import Image, ImageDraw, ImageOps


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def select_candidates(metadata: Path, protocol: dict) -> list[dict[str, str]]:
    source = protocol["source"]
    cap = int(source["candidate_images"])
    seed = int(source["selection_seed"])
    required_license = source["required_image_license"]
    heap: list[tuple[int, str, dict[str, str]]] = []
    with metadata.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id = row.get("ImageID", "")
            if (
                row.get("Subset") != "train"
                or row.get("License") != required_license
                or not image_id
                or not row.get("OriginalLandingURL")
                or not row.get("Author")
            ):
                continue
            key = int.from_bytes(hashlib.sha256(f"{seed}|{image_id}".encode()).digest(), "big")
            entry = (-key, image_id, row)
            if len(heap) < cap:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
    if len(heap) != cap:
        raise RuntimeError(f"insufficient metadata candidates: {len(heap)} != {cap}")
    return [entry[2] for entry in sorted(heap, key=lambda item: (-item[0], item[1]))]


def download_one(row: dict[str, str], protocol: dict, image_root: Path) -> tuple[dict | None, str | None]:
    image_id = row["ImageID"]
    path = image_root / image_id[:2] / f"{image_id}.jpg"
    maximum_bytes = int(protocol["data_gates"]["maximum_single_image_bytes"])
    minimum = int(protocol["data_gates"]["minimum_width_and_height"])
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = protocol["source"]["image_url_template"].format(image_id=image_id)
        error = None
        for attempt in range(3):
            temporary_path = None
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "vision1b-research-pilot/1"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    length = int(response.headers.get("Content-Length", 0))
                    if length and length > maximum_bytes:
                        return None, "declared_too_large"
                    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
                        temporary_path = Path(temporary.name)
                        total = 0
                        while block := response.read(1024 * 1024):
                            total += len(block)
                            if total > maximum_bytes:
                                raise RuntimeError("image exceeds maximum bytes")
                            temporary.write(block)
                os.replace(temporary_path, path)
                break
            except Exception as exception:
                error = exception
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
                if attempt == 2:
                    return None, f"download:{type(error).__name__}"
                time.sleep(0.5 * (attempt + 1))
    try:
        if path.stat().st_size > maximum_bytes:
            return None, "too_large"
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
            image.convert("RGB").load()
        if min(width, height) < minimum:
            return None, "too_small"
        digest = sha256(path)
        return {
            "image_id": image_id,
            "split": "train",
            "image_path": str(path.resolve()),
            "image_bytes": path.stat().st_size,
            "image_sha256": digest,
            "width": width,
            "height": height,
            "mode": mode,
            "license": row["License"],
            "original_url": row.get("OriginalURL", ""),
            "original_landing_url": row["OriginalLandingURL"],
            "author": row["Author"],
            "author_profile_url": row.get("AuthorProfileURL", ""),
            "title": row.get("Title", ""),
        }, None
    except Exception as exception:
        path.unlink(missing_ok=True)
        return None, f"decode:{type(exception).__name__}"


def make_audit_sheets(rows: list[dict], output: Path, seed: int) -> list[dict]:
    chosen = sorted(rows, key=lambda row: hashlib.sha256(f"{seed}|audit|{row['image_id']}".encode()).digest())[:64]
    records = []
    for sheet_index in range(4):
        sheet_rows = chosen[sheet_index * 16:(sheet_index + 1) * 16]
        canvas = Image.new("RGB", (1024, 1024), "white")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(sheet_rows):
            with Image.open(row["image_path"]) as handle:
                image = ImageOps.fit(handle.convert("RGB"), (250, 220), method=Image.Resampling.LANCZOS)
            column, line = index % 4, index // 4
            left, top = column * 256, line * 256
            canvas.paste(image, (left + 3, top + 3))
            draw.text((left + 5, top + 226), row["image_id"], fill="black")
        path = output / "audit" / f"openimages-25k-audit-{sheet_index:02d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, quality=92)
        records.append({"path": str(path), "sha256": sha256(path), "image_ids": [row["image_id"] for row in sheet_rows]})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    arguments = parser.parse_args()
    protocol = json.loads(arguments.protocol.read_text())
    if protocol.get("status") != "authorized_bounded_real_image_self_distillation_pilot":
        raise RuntimeError("authorized protocol required")
    source = protocol["source"]
    if arguments.metadata.stat().st_size != source["image_metadata_expected_bytes"] or sha256(arguments.metadata) != source["image_metadata_sha256"]:
        raise RuntimeError("metadata identity mismatch")
    output = arguments.output_root
    output.mkdir(parents=True, exist_ok=True)
    candidates = select_candidates(arguments.metadata, protocol)
    target = int(source["expected_unique_images"])
    accepted: list[dict] = []
    failures: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        futures = [executor.submit(download_one, row, protocol, output / "images") for row in candidates]
        for completed, future in enumerate(as_completed(futures), start=1):
            row, reason = future.result()
            if row is not None:
                accepted.append(row)
            else:
                failures[reason or "unknown"] = failures.get(reason or "unknown", 0) + 1
            if completed % 500 == 0:
                atomic_json(output / "progress.json", {"completed_candidates": completed, "accepted": len(accepted), "failures": failures})
    accepted.sort(key=lambda row: hashlib.sha256(f"{source['selection_seed']}|{row['image_id']}".encode()).digest())
    unique = []
    seen_hashes = set()
    for row in accepted:
        if row["image_sha256"] in seen_hashes:
            failures["exact_duplicate"] = failures.get("exact_duplicate", 0) + 1
            continue
        seen_hashes.add(row["image_sha256"])
        unique.append(row)
        if len(unique) == target:
            break
    if len(unique) != target:
        raise RuntimeError(f"only {len(unique)} unique passing images for target {target}")
    manifest = output / "manifest.jsonl"
    temporary = manifest.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in unique))
    os.replace(temporary, manifest)
    audit_sheets = make_audit_sheets(unique, output, int(source["selection_seed"]))
    receipt = {
        "status": "PENDING_FIXED_VISUAL_AUDIT",
        "protocol_sha256": sha256(arguments.protocol),
        "metadata": {"path": str(arguments.metadata), "bytes": arguments.metadata.stat().st_size, "sha256": sha256(arguments.metadata)},
        "candidates": len(candidates),
        "passing_before_exact_dedupe": len(accepted),
        "failures": failures,
        "manifest": {"path": str(manifest), "rows": len(unique), "sha256": sha256(manifest)},
        "audit_sheets": audit_sheets,
        "claim_boundary": protocol["claim_boundary"],
    }
    atomic_json(output / "acquisition-receipt.json", receipt)
    atomic_json(output / "progress.json", {"completed_candidates": len(candidates), "accepted": len(unique), "status": receipt["status"]})
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
