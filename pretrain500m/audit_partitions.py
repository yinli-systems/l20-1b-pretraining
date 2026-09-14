#!/usr/bin/env python3
"""Audit independently packed partitions and emit exact cross-part exclusions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=16 * 1024**2)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--expected-parts", type=int, default=14)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.scratch_dir or args.output_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    seen_path = work_dir / "seen.sqlite"
    if seen_path.exists():
        seen_path.unlink()
    connection = sqlite3.connect(seen_path)
    if args.scratch_dir:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-8388608")
    else:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE seen(hash BLOB PRIMARY KEY, first_part INTEGER NOT NULL) WITHOUT ROWID"
    )
    records = []
    try:
        for index in range(args.expected_parts):
            label = f"part-{index:02d}"
            root = args.parts_root / label
            manifest_path = root / "web" / "manifest.json"
            database_path = root / "dedup.sqlite"
            if not manifest_path.is_file() or not database_path.is_file():
                raise RuntimeError(f"missing completed partition: {label}")
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("partition_label") != label:
                raise RuntimeError(f"partition label mismatch: {label}")
            if manifest.get("block_size") != 2049:
                raise RuntimeError(f"block size mismatch: {label}")
            if not manifest.get("source_exhausted"):
                raise RuntimeError(f"partition did not scan its complete source file: {label}")
            check = sqlite3.connect(database_path)
            try:
                integrity = check.execute("PRAGMA integrity_check").fetchone()[0]
                document_count = check.execute("SELECT count(*) FROM documents").fetchone()[0]
            finally:
                check.close()
            if integrity != "ok":
                raise RuntimeError(f"database integrity failed for {label}: {integrity}")
            connection.execute("ATTACH DATABASE ? AS candidate", (str(database_path),))
            overlaps = connection.execute(
                "SELECT count(*) FROM candidate.documents AS d JOIN seen AS s ON d.hash=s.hash"
            ).fetchone()[0]
            exclusion_path = work_dir / f"exclude-{label}.sqlite"
            if exclusion_path.exists():
                exclusion_path.unlink()
            exclusion = sqlite3.connect(exclusion_path)
            exclusion.execute("CREATE TABLE documents(hash BLOB PRIMARY KEY) WITHOUT ROWID")
            if overlaps:
                rows = connection.execute(
                    "SELECT d.hash FROM candidate.documents AS d JOIN seen AS s ON d.hash=s.hash"
                )
                exclusion.executemany("INSERT INTO documents(hash) VALUES (?)", rows)
            exclusion.commit()
            exclusion.close()
            connection.execute(
                "INSERT OR IGNORE INTO seen(hash,first_part) SELECT hash,? FROM candidate.documents",
                (index,),
            )
            connection.commit()
            connection.execute("DETACH DATABASE candidate")
            record = {
                "part": label,
                "manifest_sha256": digest(manifest_path),
                "dedup_db_sha256": digest(database_path),
                "documents": document_count,
                "cross_part_duplicates": overlaps,
                "exclusion_db": str(args.output_dir / exclusion_path.name),
                "exclusion_db_sha256": digest(exclusion_path),
            }
            records.append(record)
            print(json.dumps({"progress": record}, sort_keys=True), flush=True)
    finally:
        connection.close()
    if args.scratch_dir:
        for record in records:
            destination = Path(record["exclusion_db"])
            atomic_copy(work_dir / destination.name, destination)
    total = sum(record["cross_part_duplicates"] for record in records)
    receipt = {
        "status": "PASS_ZERO_CROSS_PART_DUPLICATES" if total == 0 else "RERUN_WITH_EXCLUSIONS_REQUIRED",
        "parts_root": str(args.parts_root),
        "expected_parts": args.expected_parts,
        "documents": sum(record["documents"] for record in records),
        "cross_part_duplicates": total,
        "parts": records,
    }
    atomic_json(args.output_dir / "cross-part-dedup.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
