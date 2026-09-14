#!/usr/bin/env python3
"""Hardlink clean partitions and plan only the cross-partition reruns needed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def link_tree(source: Path, destination: Path) -> tuple[int, int]:
    files = 0
    bytes_linked = 0
    for root, directories, names in os.walk(source):
        relative = Path(root).relative_to(source)
        target_root = destination / relative
        target_root.mkdir(parents=True, exist_ok=True)
        for directory in directories:
            (target_root / directory).mkdir(exist_ok=True)
        for name in names:
            original = Path(root) / name
            target = target_root / name
            os.link(original, target)
            files += 1
            bytes_linked += original.stat().st_size
    return files, bytes_linked


def array_spec(indices: list[int]) -> str:
    if not indices:
        return ""
    groups: list[str] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        groups.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = index
    groups.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(groups)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--audit-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.audit_receipt.read_text())
    if receipt.get("status") != "RERUN_WITH_EXCLUSIONS_REQUIRED":
        raise RuntimeError("audit does not require a dedup rerun")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    rerun: list[int] = []
    linked: list[dict[str, object]] = []
    for index, record in enumerate(receipt["parts"]):
        label = f"part-{index:02d}"
        if record["part"] != label:
            raise RuntimeError(f"unexpected part order at {index}: {record['part']}")
        source = args.parts_root / label
        manifest = source / "web" / "manifest.json"
        database = source / "dedup.sqlite"
        if digest(manifest) != record["manifest_sha256"] or digest(database) != record["dedup_db_sha256"]:
            raise RuntimeError(f"audit input changed after receipt: {label}")
        if int(record["cross_part_duplicates"]) > 0:
            rerun.append(index)
        else:
            files, bytes_linked = link_tree(source, args.output_root / label)
            linked.append({"part": label, "files": files, "logical_bytes": bytes_linked})

    plan = {
        "status": "RERUN_REQUIRED" if rerun else "NO_RERUN_TASKS",
        "audit_receipt": str(args.audit_receipt),
        "audit_receipt_sha256": digest(args.audit_receipt),
        "source_parts_root": str(args.parts_root),
        "output_parts_root": str(args.output_root),
        "hardlinked_zero_overlap_parts": linked,
        "rerun_indices": rerun,
        "slurm_array_spec": array_spec(rerun),
    }
    temporary = args.plan.with_suffix(args.plan.suffix + ".tmp")
    temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.plan)
    print(json.dumps(plan, sort_keys=True))


if __name__ == "__main__":
    main()
