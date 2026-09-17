#!/usr/bin/env python3
"""Fail-closed, shard-parallel admission for the official Open Images archives."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import heapq
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import tarfile
from typing import Iterable, Iterator

from PIL import Image, ImageDraw, ImageOps


BLOCK = 16 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(BLOCK):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    os.replace(temporary, path)
    return count, sha256(path)


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    if protocol.get("status") != "ADMISSION_PLAN_NOT_ADMITTED":
        raise RuntimeError("expected a fail-closed admission protocol")
    perceptual = protocol["perceptual_deduplication"]
    if int(perceptual["bands"]) * int(perceptual["bits_per_band"]) != 128:
        raise RuntimeError("perceptual band contract must cover 128 bits")
    if int(perceptual["maximum_hamming_distance"]) >= int(perceptual["bands"]):
        raise RuntimeError("LSH completeness requires distance < band count")
    return protocol


def ensure_free_space(path: Path, protocol: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    required = int(protocol.get("storage", {}).get("minimum_free_bytes", 0))
    if shutil.disk_usage(path).free < required:
        raise RuntimeError(f"free-space floor reached: {path}")


def combined_dhash_128(image: Image.Image) -> str:
    resized = image.convert("L").resize((9, 9), Image.Resampling.LANCZOS)
    pixels = list(resized.getdata())
    horizontal = 0
    vertical = 0
    for y in range(8):
        for x in range(8):
            horizontal = (horizontal << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
            vertical = (vertical << 1) | (pixels[y * 9 + x] > pixels[(y + 1) * 9 + x])
    return f"{horizontal:016x}{vertical:016x}"


def pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    prefix = f"RGB|{rgb.width}|{rgb.height}|".encode()
    return sha256_bytes(prefix + rgb.tobytes())


def image_id_from_member(name: str) -> str | None:
    path = Path(name)
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        return None
    image_id = path.stem
    if len(image_id) != 16 or any(character not in "0123456789abcdefABCDEF" for character in image_id):
        return None
    return image_id.lower()


def build_metadata(arguments: argparse.Namespace) -> None:
    protocol = load_protocol(arguments.protocol)
    ensure_free_space(arguments.output.parent, protocol)
    contract = protocol["metadata"]
    metadata = arguments.metadata
    if metadata.stat().st_size != int(contract["expected_bytes"]):
        raise RuntimeError("official metadata byte length mismatch")
    if sha256(metadata) != contract["sha256"]:
        raise RuntimeError("official metadata SHA-256 mismatch")
    output = arguments.output
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    connection.execute(
        "CREATE TABLE metadata (image_id TEXT PRIMARY KEY, original_url TEXT NOT NULL, "
        "landing_url TEXT NOT NULL, license TEXT NOT NULL, author TEXT NOT NULL, "
        "author_profile_url TEXT NOT NULL, title TEXT NOT NULL) WITHOUT ROWID"
    )
    counts = Counter()
    required = list(contract["required_fields"])
    with metadata.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(field not in reader.fieldnames for field in required):
            raise RuntimeError("official metadata schema mismatch")
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        for row in reader:
            counts["rows"] += 1
            if row["Subset"] != contract["required_split"]:
                counts["wrong_split"] += 1
                continue
            if row["License"] != contract["required_license"]:
                counts["wrong_license"] += 1
                continue
            if not row["ImageID"] or not row["OriginalLandingURL"] or not row["Author"]:
                counts["missing_attribution"] += 1
                continue
            batch.append(
                (
                    row["ImageID"].lower(),
                    row["OriginalURL"],
                    row["OriginalLandingURL"],
                    row["License"],
                    row["Author"],
                    row["AuthorProfileURL"],
                    row["Title"],
                )
            )
            if len(batch) == 10000:
                connection.executemany("INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                counts["accepted"] += len(batch)
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
            counts["accepted"] += len(batch)
    connection.commit()
    connection.execute("PRAGMA optimize")
    connection.close()
    os.replace(temporary, output)
    receipt = {
        "counts": dict(counts),
        "database": {"bytes": output.stat().st_size, "path": str(output), "sha256": sha256(output)},
        "metadata": {"bytes": metadata.stat().st_size, "path": str(metadata), "sha256": sha256(metadata)},
        "protocol_sha256": sha256(arguments.protocol),
        "status": "OFFICIAL_METADATA_JOIN_READY_NOT_ADMITTED",
    }
    atomic_json(arguments.receipt, receipt)


def verified_metadata_database(database: Path, receipt_path: Path, protocol_path: Path) -> sqlite3.Connection:
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "OFFICIAL_METADATA_JOIN_READY_NOT_ADMITTED":
        raise RuntimeError("metadata join receipt has not passed")
    if receipt["protocol_sha256"] != sha256(protocol_path):
        raise RuntimeError("metadata join protocol drift")
    if receipt["database"]["sha256"] != sha256(database):
        raise RuntimeError("metadata database drift")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def acquisition_archive(receipt_path: Path, archive: Path, protocol: dict) -> dict:
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "DOWNLOADED_NOT_ADMITTED":
        raise RuntimeError("full acquisition receipt is required before indexing")
    if receipt.get("protocol_sha256") != protocol["acquisition"]["protocol_sha256"]:
        raise RuntimeError("acquisition protocol identity mismatch")
    rows = [row for row in receipt.get("archives", []) if row.get("filename") == archive.name]
    if len(rows) != 1:
        raise RuntimeError("archive is absent or duplicated in acquisition receipt")
    row = rows[0]
    if archive.stat().st_size != int(row["bytes"]) or sha256(archive) != row["sha256"]:
        raise RuntimeError("archive identity mismatch")
    return row


def index_shard(arguments: argparse.Namespace) -> None:
    protocol = load_protocol(arguments.protocol)
    ensure_free_space(arguments.output_manifest.parent, protocol)
    archive_identity = acquisition_archive(arguments.acquisition_receipt, arguments.archive, protocol)
    metadata = verified_metadata_database(arguments.metadata_database, arguments.metadata_receipt, arguments.protocol)
    maximum = int(protocol["image_gates"]["maximum_member_bytes"])
    minimum_width = int(protocol["image_gates"]["minimum_width"])
    minimum_height = int(protocol["image_gates"]["minimum_height"])
    accepted: list[dict] = []
    rejected = Counter()
    seen_ids: set[str] = set()
    with tarfile.open(arguments.archive, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            image_id = image_id_from_member(member.name)
            if image_id is None:
                rejected["non_image_member"] += 1
                continue
            if image_id in seen_ids:
                raise RuntimeError(f"duplicate image ID inside archive: {image_id}")
            seen_ids.add(image_id)
            row = metadata.execute("SELECT * FROM metadata WHERE image_id = ?", (image_id,)).fetchone()
            if row is None:
                rejected["metadata_or_rights_gate"] += 1
                continue
            if member.size > maximum:
                rejected["too_large"] += 1
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                rejected["missing_member_stream"] += 1
                continue
            payload = extracted.read(maximum + 1)
            if len(payload) != member.size or len(payload) > maximum:
                rejected["member_size_mismatch"] += 1
                continue
            try:
                with Image.open(io.BytesIO(payload)) as candidate:
                    candidate.verify()
                with Image.open(io.BytesIO(payload)) as candidate:
                    width, height = candidate.size
                    mode = candidate.mode
                    rgb = candidate.convert("RGB")
                    rgb.load()
                if width < minimum_width or height < minimum_height:
                    rejected["too_small"] += 1
                    continue
                accepted.append(
                    {
                        "archive_bytes": int(archive_identity["bytes"]),
                        "archive_filename": arguments.archive.name,
                        "archive_sha256": archive_identity["sha256"],
                        "author": row["author"],
                        "author_profile_url": row["author_profile_url"],
                        "dhash128": combined_dhash_128(rgb),
                        "file_bytes": len(payload),
                        "file_sha256": sha256_bytes(payload),
                        "height": height,
                        "image_id": image_id,
                        "license": row["license"],
                        "member": member.name,
                        "mode": mode,
                        "original_landing_url": row["landing_url"],
                        "original_url": row["original_url"],
                        "pixel_sha256": pixel_sha256(rgb),
                        "split": "train",
                        "title": row["title"],
                        "width": width,
                    }
                )
            except Exception as error:
                rejected[f"decode:{type(error).__name__}"] += 1
    metadata.close()
    accepted.sort(key=lambda row: row["image_id"])
    rows, manifest_hash = atomic_jsonl(arguments.output_manifest, accepted)
    receipt = {
        "accepted_rows": rows,
        "acquisition_receipt_sha256": sha256(arguments.acquisition_receipt),
        "archive": archive_identity,
        "manifest": {"path": str(arguments.output_manifest), "rows": rows, "sha256": manifest_hash},
        "metadata_receipt_sha256": sha256(arguments.metadata_receipt),
        "protocol_sha256": sha256(arguments.protocol),
        "rejected": dict(rejected),
        "status": "SHARD_INDEXED_NOT_ADMITTED",
    }
    atomic_json(arguments.output_receipt, receipt)


def jsonl(path: Path) -> Iterator[dict]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def verified_reduction(receipt_path: Path, protocol_path: Path) -> tuple[dict, Path]:
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "GLOBAL_DEDUP_COMPLETE_PENDING_AUDITS_NOT_ADMITTED":
        raise RuntimeError("global reduction receipt has not passed")
    if receipt.get("protocol_sha256") != sha256(protocol_path):
        raise RuntimeError("global reduction protocol drift")
    manifest = Path(receipt["manifest"]["path"])
    if sha256(manifest) != receipt["manifest"]["sha256"]:
        raise RuntimeError("admission candidate manifest drift")
    if int(receipt["manifest"]["rows"]) != int(receipt["accepted_rows"]):
        raise RuntimeError("admission candidate row-count drift")
    return receipt, manifest


def bands(value: str, count: int, width: int) -> Iterator[tuple[int, int]]:
    number = int(value, 16)
    mask = (1 << width) - 1
    for index in range(count):
        shift = (count - index - 1) * width
        yield index, (number >> shift) & mask


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def verified_downstream_denylist(manifest: Path, receipt_path: Path, protocol: dict) -> list[dict]:
    receipt = json.loads(receipt_path.read_text())
    required = protocol["reduction"]["required_downstream_denylist_status"]
    if receipt.get("status") != required:
        raise RuntimeError("downstream denylist is not frozen")
    if receipt.get("manifest_sha256") != sha256(manifest):
        raise RuntimeError("downstream denylist identity mismatch")
    rows = list(jsonl(manifest))
    if int(receipt.get("rows", -1)) != len(rows):
        raise RuntimeError("downstream denylist row-count mismatch")
    for row in rows:
        if not row.get("file_sha256") or not row.get("pixel_sha256") or len(row.get("dhash128", "")) != 32:
            raise RuntimeError("downstream denylist row is incomplete")
    return rows


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while block := handle.read(BLOCK):
            digest.update(block)
    return digest.hexdigest()


def idx_images(path: Path) -> Iterator[bytes]:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, columns = struct.unpack(">IIII", handle.read(16))
        if magic != 2051 or rows != 28 or columns != 28:
            raise RuntimeError(f"unexpected IDX image schema: {path}")
        size = rows * columns
        for _ in range(count):
            payload = handle.read(size)
            if len(payload) != size:
                raise RuntimeError(f"truncated IDX image file: {path}")
            yield payload
        if handle.read(1):
            raise RuntimeError(f"trailing IDX image bytes: {path}")


def build_mnist_denylist(arguments: argparse.Namespace) -> None:
    frozen = json.loads(arguments.frozen_eval_protocol.read_text())
    if frozen.get("status") != "PREREGISTERED_FROZEN_FEATURE_DIAGNOSTIC":
        raise RuntimeError("frozen evaluation protocol is required")
    rows: list[dict] = []
    source_files = []
    for dataset_name, dataset in sorted(frozen["datasets"].items()):
        for filename, expected_md5 in sorted(dataset["files"].items()):
            path = arguments.data_root / dataset_name.replace("_", "-") / filename
            if md5(path) != expected_md5:
                raise RuntimeError(f"downstream source identity mismatch: {path}")
            source_files.append({"md5": expected_md5, "path": str(path)})
            if "images-idx3" not in filename:
                continue
            split = "test" if filename.startswith("t10k") else "train"
            for index, payload in enumerate(idx_images(path)):
                image = Image.frombytes("L", (28, 28), payload).convert("RGB")
                identifier = f"{dataset_name}:{split}:{index:05d}"
                rows.append(
                    {
                        "dhash128": combined_dhash_128(image),
                        "file_sha256": sha256_bytes(identifier.encode() + b"|" + payload),
                        "image_id": identifier,
                        "pixel_sha256": pixel_sha256(image),
                        "source": identifier,
                    }
                )
    rows.sort(key=lambda row: row["image_id"])
    count, manifest_hash = atomic_jsonl(arguments.output_manifest, rows)
    receipt = {
        "frozen_eval_protocol_sha256": sha256(arguments.frozen_eval_protocol),
        "manifest_sha256": manifest_hash,
        "rows": count,
        "source_files": source_files,
        "status": "FROZEN_DOWNSTREAM_DENYLIST",
    }
    atomic_json(arguments.output_receipt, receipt)


def nearby(connection: sqlite3.Connection, table: str, value: str, band_count: int, band_width: int, limit: int) -> str | None:
    checked: set[str] = set()
    for band_index, band_value in bands(value, band_count, band_width):
        for image_id, candidate in connection.execute(
            f"SELECT image_id, dhash FROM {table} WHERE band_index = ? AND band_value = ?",
            (band_index, band_value),
        ):
            if image_id in checked:
                continue
            checked.add(image_id)
            if hamming(value, candidate) <= limit:
                return image_id
    return None


def reduce_indexes(arguments: argparse.Namespace) -> None:
    protocol = load_protocol(arguments.protocol)
    ensure_free_space(arguments.output_root, protocol)
    protocol_hash = sha256(arguments.protocol)
    receipts = [json.loads(path.read_text()) for path in arguments.index_receipts]
    if len(receipts) != int(protocol["acquisition"]["expected_archive_count"]):
        raise RuntimeError("all frozen archive indexes are required")
    manifests: list[Path] = []
    archives: dict[str, dict] = {}
    acquisition_receipts: set[str] = set()
    metadata_receipts: set[str] = set()
    for path, receipt in zip(arguments.index_receipts, receipts):
        if receipt.get("status") != "SHARD_INDEXED_NOT_ADMITTED" or receipt.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"invalid shard index receipt: {path}")
        archive = receipt["archive"]["filename"]
        if archive in archives:
            raise RuntimeError("duplicate indexed archive")
        archives[archive] = receipt["archive"]
        acquisition_receipts.add(receipt["acquisition_receipt_sha256"])
        metadata_receipts.add(receipt["metadata_receipt_sha256"])
        manifest = Path(receipt["manifest"]["path"])
        if sha256(manifest) != receipt["manifest"]["sha256"]:
            raise RuntimeError("shard index manifest drift")
        manifests.append(manifest)
    if len(acquisition_receipts) != 1:
        raise RuntimeError("shard indexes do not share one acquisition receipt")
    if len(metadata_receipts) != 1:
        raise RuntimeError("shard indexes do not share one metadata receipt")
    downstream = verified_downstream_denylist(arguments.downstream_manifest, arguments.downstream_receipt, protocol)
    perceptual = protocol["perceptual_deduplication"]
    band_count = int(perceptual["bands"])
    band_width = int(perceptual["bits_per_band"])
    distance = int(perceptual["maximum_hamming_distance"])
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    database = arguments.output_root / "dedup.sqlite"
    temporary_database = database.with_suffix(".sqlite.tmp")
    temporary_database.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary_database)
    connection.executescript(
        "CREATE TABLE exact (image_id TEXT PRIMARY KEY, file_sha TEXT UNIQUE, pixel_sha TEXT UNIQUE, dhash TEXT NOT NULL);"
        "CREATE TABLE accepted_bands (band_index INTEGER, band_value INTEGER, image_id TEXT, dhash TEXT);"
        "CREATE INDEX accepted_band_lookup ON accepted_bands(band_index, band_value);"
        "CREATE TABLE downstream_exact (image_id TEXT PRIMARY KEY, file_sha TEXT, pixel_sha TEXT, dhash TEXT);"
        "CREATE INDEX downstream_file ON downstream_exact(file_sha);"
        "CREATE INDEX downstream_pixel ON downstream_exact(pixel_sha);"
        "CREATE TABLE downstream_bands (band_index INTEGER, band_value INTEGER, image_id TEXT, dhash TEXT);"
        "CREATE INDEX downstream_band_lookup ON downstream_bands(band_index, band_value);"
    )
    for downstream_index, row in enumerate(downstream):
        image_id = str(row.get("image_id") or row.get("source") or f"downstream-{downstream_index:09d}")
        connection.execute(
            "INSERT INTO downstream_exact VALUES (?, ?, ?, ?)",
            (image_id, row["file_sha256"], row["pixel_sha256"], row["dhash128"]),
        )
        connection.executemany(
            "INSERT INTO downstream_bands VALUES (?, ?, ?, ?)",
            [(index, value, image_id, row["dhash128"]) for index, value in bands(row["dhash128"], band_count, band_width)],
        )
    output_manifest = arguments.output_root / "admission-candidates.jsonl"
    temporary_manifest = output_manifest.with_suffix(".jsonl.tmp")
    streams = [iter(jsonl(path)) for path in manifests]
    merged = heapq.merge(*streams, key=lambda row: row["image_id"])
    rejected = Counter()
    accepted = 0
    previous_id = None
    with temporary_manifest.open("w") as output:
        for row in merged:
            image_id = row["image_id"]
            if image_id == previous_id:
                raise RuntimeError(f"image ID appears in multiple archives: {image_id}")
            previous_id = image_id
            if connection.execute(
                "SELECT 1 FROM downstream_exact WHERE file_sha = ? OR pixel_sha = ? LIMIT 1",
                (row["file_sha256"], row["pixel_sha256"]),
            ).fetchone():
                rejected["downstream_exact_overlap"] += 1
                continue
            if nearby(connection, "downstream_bands", row["dhash128"], band_count, band_width, distance):
                rejected["downstream_perceptual_overlap"] += 1
                continue
            if connection.execute(
                "SELECT 1 FROM exact WHERE file_sha = ? OR pixel_sha = ? LIMIT 1",
                (row["file_sha256"], row["pixel_sha256"]),
            ).fetchone():
                rejected["global_exact_duplicate"] += 1
                continue
            if nearby(connection, "accepted_bands", row["dhash128"], band_count, band_width, distance):
                rejected["global_perceptual_duplicate"] += 1
                continue
            connection.execute(
                "INSERT INTO exact VALUES (?, ?, ?, ?)",
                (image_id, row["file_sha256"], row["pixel_sha256"], row["dhash128"]),
            )
            connection.executemany(
                "INSERT INTO accepted_bands VALUES (?, ?, ?, ?)",
                [(index, value, image_id, row["dhash128"]) for index, value in bands(row["dhash128"], band_count, band_width)],
            )
            output.write(json.dumps(row, sort_keys=True) + "\n")
            accepted += 1
            if accepted % 10000 == 0:
                connection.commit()
    connection.commit()
    connection.execute("PRAGMA optimize")
    connection.close()
    os.replace(temporary_manifest, output_manifest)
    os.replace(temporary_database, database)
    receipt = {
        "accepted_rows": accepted,
        "acquisition_receipt_sha256": next(iter(acquisition_receipts)),
        "archives": [archives[name] for name in sorted(archives)],
        "database_sha256": sha256(database),
        "downstream_denylist_sha256": sha256(arguments.downstream_manifest),
        "downstream_receipt_sha256": sha256(arguments.downstream_receipt),
        "index_receipt_sha256": [sha256(path) for path in arguments.index_receipts],
        "manifest": {"path": str(output_manifest), "rows": accepted, "sha256": sha256(output_manifest)},
        "metadata_receipt_sha256": next(iter(metadata_receipts)),
        "protocol_sha256": protocol_hash,
        "rejected": dict(rejected),
        "status": "GLOBAL_DEDUP_COMPLETE_PENDING_AUDITS_NOT_ADMITTED",
    }
    atomic_json(arguments.output_root / "reduction-receipt.json", receipt)


def selected_audit_rows(manifest: Path, count: int, seed: int) -> list[dict]:
    heap: list[tuple[int, str, dict]] = []
    for row in jsonl(manifest):
        key = int.from_bytes(hashlib.sha256(f"{seed}|audit|{row['image_id']}".encode()).digest(), "big")
        entry = (-key, row["image_id"], row)
        if len(heap) < count:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    if len(heap) != count:
        raise RuntimeError("insufficient admitted candidates for frozen audit")
    return [entry[2] for entry in sorted(heap, key=lambda item: (-item[0], item[1]))]


def render_audit(arguments: argparse.Namespace) -> None:
    protocol = load_protocol(arguments.protocol)
    ensure_free_space(arguments.output_root, protocol)
    reduction, manifest = verified_reduction(arguments.reduction_receipt, arguments.protocol)
    contract = protocol["audit"]
    rows = selected_audit_rows(manifest, int(contract["images"]), int(contract["seed"]))
    by_archive: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_archive.setdefault(row["archive_filename"], {})[row["member"]] = row
    images: dict[str, Image.Image] = {}
    for filename, wanted in by_archive.items():
        archive_path = arguments.archive_root / filename
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                if member.name not in wanted:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    raise RuntimeError("audit member stream missing")
                payload = handle.read()
                if sha256_bytes(payload) != wanted[member.name]["file_sha256"]:
                    raise RuntimeError("audit image identity drift")
                with Image.open(io.BytesIO(payload)) as candidate:
                    images[wanted[member.name]["image_id"]] = candidate.convert("RGB")
    if len(images) != len(rows):
        raise RuntimeError("not all frozen audit images were found")
    columns = int(contract["sheet_columns"])
    lines = int(contract["sheet_rows"])
    per_sheet = columns * lines
    sheet_receipts = []
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    for sheet_index in range((len(rows) + per_sheet - 1) // per_sheet):
        subset = rows[sheet_index * per_sheet:(sheet_index + 1) * per_sheet]
        canvas = Image.new("RGB", (columns * 256, lines * 256), "white")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(subset):
            thumbnail = ImageOps.fit(images[row["image_id"]], (250, 220), method=Image.Resampling.LANCZOS)
            column, line = index % columns, index // columns
            left, top = column * 256, line * 256
            canvas.paste(thumbnail, (left + 3, top + 3))
            draw.text((left + 5, top + 226), row["image_id"], fill="black")
        path = arguments.output_root / f"audit-{sheet_index:03d}.jpg"
        canvas.save(path, quality=92)
        sheet_receipts.append({"image_ids": [row["image_id"] for row in subset], "path": str(path), "sha256": sha256(path)})
    index = arguments.output_root / "audit-index.json"
    receipt = {
        "candidate_manifest_sha256": reduction["manifest"]["sha256"],
        "image_ids": [row["image_id"] for row in rows],
        "protocol_sha256": sha256(arguments.protocol),
        "sheets": sheet_receipts,
        "status": "PENDING_HUMAN_CONTENT_SAFETY_AUDIT",
    }
    atomic_json(index, receipt)


def pack_shard(arguments: argparse.Namespace) -> None:
    protocol = load_protocol(arguments.protocol)
    ensure_free_space(arguments.output_tar.parent, protocol)
    reduction, manifest = verified_reduction(arguments.reduction_receipt, arguments.protocol)
    archive_rows = [row for row in reduction["archives"] if row["filename"] == arguments.archive.name]
    if len(archive_rows) != 1:
        raise RuntimeError("source archive is not bound to the global reduction")
    archive_identity = archive_rows[0]
    if arguments.archive.stat().st_size != int(archive_identity["bytes"]):
        raise RuntimeError("source archive byte length drift")
    selected = {row["member"]: row for row in jsonl(manifest) if row["archive_filename"] == arguments.archive.name}
    temporary = arguments.output_tar.with_suffix(arguments.output_tar.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with tarfile.open(arguments.archive, "r:gz") as source, tarfile.open(temporary, "w") as output:
        for member in source:
            row = selected.get(member.name)
            if row is None:
                continue
            handle = source.extractfile(member)
            if handle is None:
                raise RuntimeError("packed member stream missing")
            payload = handle.read()
            if sha256_bytes(payload) != row["file_sha256"]:
                raise RuntimeError("packed member identity drift")
            target = tarfile.TarInfo(f"images/{row['image_id']}.jpg")
            target.size = len(payload)
            target.mode = int(protocol["packing"]["member_mode"])
            target.mtime = int(protocol["packing"]["member_mtime"])
            target.uid = target.gid = 0
            target.uname = target.gname = ""
            output.addfile(target, io.BytesIO(payload))
            written += 1
    if written != len(selected):
        raise RuntimeError("not all selected members were packed")
    os.replace(temporary, arguments.output_tar)
    receipt = {
        "candidate_manifest_sha256": reduction["manifest"]["sha256"],
        "output": {"bytes": arguments.output_tar.stat().st_size, "path": str(arguments.output_tar), "sha256": sha256(arguments.output_tar)},
        "protocol_sha256": sha256(arguments.protocol),
        "rows": written,
        "source_archive": arguments.archive.name,
        "source_archive_bytes": int(archive_identity["bytes"]),
        "source_archive_sha256": archive_identity["sha256"],
        "status": "PACKED_SHARD_READY_NOT_ADMITTED",
    }
    atomic_json(arguments.output_receipt, receipt)


def finalize(arguments: argparse.Namespace) -> None:
    protocol = load_protocol(arguments.protocol)
    reduction, manifest = verified_reduction(arguments.reduction_receipt, arguments.protocol)
    manifest_hash = reduction["manifest"]["sha256"]
    protocol_hash = sha256(arguments.protocol)
    audit_index = json.loads(arguments.audit_index.read_text())
    audit_decision = json.loads(arguments.audit_decision.read_text())
    safety = json.loads(arguments.safety_receipt.read_text())
    if audit_index.get("status") != "PENDING_HUMAN_CONTENT_SAFETY_AUDIT":
        raise RuntimeError("frozen audit index missing")
    if audit_index.get("candidate_manifest_sha256") != manifest_hash:
        raise RuntimeError("audit index is not bound to candidate manifest")
    if audit_index.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("audit index protocol drift")
    if audit_decision.get("status") != protocol["required_final_receipts"]["human_audit_status"]:
        raise RuntimeError("human audit has not passed")
    if audit_decision.get("audit_index_sha256") != sha256(arguments.audit_index):
        raise RuntimeError("human audit decision is not bound to audit index")
    if safety.get("status") != protocol["required_final_receipts"]["safety_audit_status"]:
        raise RuntimeError("safety audit has not passed")
    if safety.get("candidate_manifest_sha256") != manifest_hash:
        raise RuntimeError("safety audit is not bound to candidate manifest")
    if int(safety.get("scanned_rows", -1)) != int(reduction["accepted_rows"]):
        raise RuntimeError("safety audit does not cover every candidate")
    if not safety.get("policy_sha256"):
        raise RuntimeError("safety policy identity missing")
    pack_receipts = [json.loads(path.read_text()) for path in arguments.pack_receipts]
    if len(pack_receipts) != int(protocol["acquisition"]["expected_archive_count"]):
        raise RuntimeError("all deterministic packed shards are required")
    if any(receipt.get("status") != "PACKED_SHARD_READY_NOT_ADMITTED" for receipt in pack_receipts):
        raise RuntimeError("a packed shard has not passed")
    if any(receipt.get("protocol_sha256") != protocol_hash for receipt in pack_receipts):
        raise RuntimeError("a packed shard protocol has drifted")
    if any(receipt.get("candidate_manifest_sha256") != manifest_hash for receipt in pack_receipts):
        raise RuntimeError("packed shard candidate identity mismatch")
    archives = sorted(receipt["source_archive"] for receipt in pack_receipts)
    expected_archives = sorted(row["filename"] for row in reduction["archives"])
    if archives != expected_archives:
        raise RuntimeError("packed source archive set mismatch")
    if sum(int(receipt["rows"]) for receipt in pack_receipts) != int(reduction["accepted_rows"]):
        raise RuntimeError("packed row count does not match admitted candidates")
    for receipt in pack_receipts:
        output = Path(receipt["output"]["path"])
        if output.stat().st_size != int(receipt["output"]["bytes"]):
            raise RuntimeError("packed shard byte length drift")
        if sha256(output) != receipt["output"]["sha256"]:
            raise RuntimeError("packed shard SHA-256 drift")
    final = {
        "audit_decision_sha256": sha256(arguments.audit_decision),
        "candidate_manifest": {"path": str(manifest), "rows": reduction["accepted_rows"], "sha256": manifest_hash},
        "claim_boundary": protocol["claim_boundary"],
        "packed_shards": [receipt["output"] for receipt in sorted(pack_receipts, key=lambda row: row["source_archive"])],
        "protocol_sha256": protocol_hash,
        "safety_receipt_sha256": sha256(arguments.safety_receipt),
        "status": "TRAINING_ADMITTED",
        "training_admitted_count": int(reduction["accepted_rows"]),
    }
    atomic_json(arguments.output_receipt, final)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    metadata = commands.add_parser("build-metadata")
    metadata.add_argument("--protocol", type=Path, required=True)
    metadata.add_argument("--metadata", type=Path, required=True)
    metadata.add_argument("--output", type=Path, required=True)
    metadata.add_argument("--receipt", type=Path, required=True)
    metadata.set_defaults(function=build_metadata)
    index = commands.add_parser("index-shard")
    index.add_argument("--protocol", type=Path, required=True)
    index.add_argument("--acquisition-receipt", type=Path, required=True)
    index.add_argument("--archive", type=Path, required=True)
    index.add_argument("--metadata-database", type=Path, required=True)
    index.add_argument("--metadata-receipt", type=Path, required=True)
    index.add_argument("--output-manifest", type=Path, required=True)
    index.add_argument("--output-receipt", type=Path, required=True)
    index.set_defaults(function=index_shard)
    reduce = commands.add_parser("reduce")
    reduce.add_argument("--protocol", type=Path, required=True)
    reduce.add_argument("--index-receipts", type=Path, nargs="+", required=True)
    reduce.add_argument("--downstream-manifest", type=Path, required=True)
    reduce.add_argument("--downstream-receipt", type=Path, required=True)
    reduce.add_argument("--output-root", type=Path, required=True)
    reduce.set_defaults(function=reduce_indexes)
    denylist = commands.add_parser("build-mnist-denylist")
    denylist.add_argument("--frozen-eval-protocol", type=Path, required=True)
    denylist.add_argument("--data-root", type=Path, required=True)
    denylist.add_argument("--output-manifest", type=Path, required=True)
    denylist.add_argument("--output-receipt", type=Path, required=True)
    denylist.set_defaults(function=build_mnist_denylist)
    audit = commands.add_parser("render-audit")
    audit.add_argument("--protocol", type=Path, required=True)
    audit.add_argument("--reduction-receipt", type=Path, required=True)
    audit.add_argument("--archive-root", type=Path, required=True)
    audit.add_argument("--output-root", type=Path, required=True)
    audit.set_defaults(function=render_audit)
    pack = commands.add_parser("pack-shard")
    pack.add_argument("--protocol", type=Path, required=True)
    pack.add_argument("--reduction-receipt", type=Path, required=True)
    pack.add_argument("--archive", type=Path, required=True)
    pack.add_argument("--output-tar", type=Path, required=True)
    pack.add_argument("--output-receipt", type=Path, required=True)
    pack.set_defaults(function=pack_shard)
    final = commands.add_parser("finalize")
    final.add_argument("--protocol", type=Path, required=True)
    final.add_argument("--reduction-receipt", type=Path, required=True)
    final.add_argument("--audit-index", type=Path, required=True)
    final.add_argument("--audit-decision", type=Path, required=True)
    final.add_argument("--safety-receipt", type=Path, required=True)
    final.add_argument("--pack-receipts", type=Path, nargs="+", required=True)
    final.add_argument("--output-receipt", type=Path, required=True)
    final.set_defaults(function=finalize)
    return root


def main() -> None:
    arguments = parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
