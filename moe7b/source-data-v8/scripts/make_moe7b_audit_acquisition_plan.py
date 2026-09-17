#!/usr/bin/env python3
"""Select a bounded, deterministic real-data audit set from frozen source trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SPLIT_PARTS = {"test", "validation", "valid", "dev", "eval", "evaluation"}


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite acquisition plan: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def is_training_path(path: str) -> bool:
    return not any(part.casefold() in FORBIDDEN_SPLIT_PARTS for part in PurePosixPath(path).parts)


def validate_file_record(record: dict) -> None:
    path = record.get("path")
    if not isinstance(path, str) or not path or path.startswith("/") or ".." in PurePosixPath(path).parts:
        raise ValueError(f"unsafe repository path: {path!r}")
    if not isinstance(record.get("size"), int) or record["size"] <= 0:
        raise ValueError(f"invalid size for {path}")
    digest = record.get("lfs_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"missing or invalid LFS SHA-256 for {path}")


def smallest(records: list[dict], prefix: str) -> dict:
    candidates = [record for record in records if record["path"].startswith(prefix) and is_training_path(record["path"])]
    if not candidates:
        raise ValueError(f"no eligible training shard for prefix={prefix!r}")
    return min(candidates, key=lambda record: (record["size"], record["path"]))


def selection_prefixes(source_id: str) -> list[str]:
    policies = {
        "fineweb_edu": ["sample/100BT/"],
        "dclm_baseline": ["global-shard_"],
        "fineweb2_multilingual": [
            "data/cmn_Hani/train/",
            "data/spa_Latn/train/",
            "data/fra_Latn/train/",
            "data/deu_Latn/train/",
            "data/arb_Arab/train/",
        ],
        "stack_v3_permissive": ["data/"],
        "finemath": ["finemath-4plus/", "finemath-3plus/"],
        "olmoe_science_reference": ["data/pes2o/", "data/wiki/"],
    }
    try:
        return policies[source_id]
    except KeyError as error:
        raise ValueError(f"no audit selection policy for {source_id}") from error


def select_records(source_id: str, records: list[dict], stack_shards: int) -> list[dict]:
    if source_id != "stack_v3_permissive":
        return [smallest(records, prefix) for prefix in selection_prefixes(source_id)]
    candidates = sorted(
        (record for record in records if record["path"].startswith("data/") and is_training_path(record["path"])),
        key=lambda record: record["path"],
    )
    if stack_shards < 1 or stack_shards > len(candidates):
        raise ValueError(f"invalid Stack v3 shard count: {stack_shards}")
    if stack_shards == 1:
        return [min(candidates, key=lambda record: (record["size"], record["path"]))]
    indices = [round(index * (len(candidates) - 1) / (stack_shards - 1)) for index in range(stack_shards)]
    if len(set(indices)) != stack_shards:
        raise ValueError("Stack v3 stratification produced duplicate indices")
    return [candidates[index] for index in indices]


def build_plan(plan_path: Path, metadata_dir: Path, stack_shards: int = 8) -> dict:
    mixture = json.loads(plan_path.read_text(encoding="utf-8"))
    if mixture.get("status") != "FROZEN_CANDIDATE_PLAN_NOT_ADMITTED":
        raise ValueError("mixture plan is not in the expected fail-closed state")
    planned_sources = []
    identities: set[tuple[str, str]] = set()
    for source in mixture["sources"]:
        source_id = source["source_id"]
        metadata_path = metadata_dir / f"{source_id}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "METADATA_SNAPSHOT_ONLY_NOT_ADMITTED":
            raise ValueError(f"unexpected metadata status for {source_id}")
        for key in ("source_id", "dataset_id", "subset"):
            if metadata.get(key) != source.get(key):
                raise ValueError(f"{source_id}: {key} mismatch")
        if metadata.get("requested_revision") != source["revision"] or metadata.get("resolved_revision") != source["revision"]:
            raise ValueError(f"{source_id}: immutable revision mismatch")
        records = metadata.get("files")
        if not isinstance(records, list) or not records:
            raise ValueError(f"{source_id}: empty file inventory")
        for record in records:
            validate_file_record(record)
        selected = select_records(source_id, records, stack_shards)
        planned_files = []
        for record in sorted(selected, key=lambda item: item["path"]):
            identity = (source["dataset_id"], record["path"])
            if identity in identities:
                raise ValueError(f"duplicate selected repository path: {identity}")
            identities.add(identity)
            planned_files.append({"path": record["path"], "size": record["size"], "lfs_sha256": record["lfs_sha256"]})
        planned_sources.append(
            {
                "source_id": source_id,
                "dataset_id": source["dataset_id"],
                "subset": source["subset"],
                "revision": source["revision"],
                "category": source["category"],
                "metadata_manifest": {"file": metadata_path.name, "sha256": sha256_file(metadata_path)},
                "files": planned_files,
            }
        )
    body = {
        "schema": "moe7b-content-audit-acquisition-plan-v1",
        "status": "PLANNED_NOT_DOWNLOADED_NOT_ADMITTED",
        "mixture_plan": {"file": plan_path.name, "sha256": sha256_file(plan_path)},
        "selection_policy": {
            "default": "smallest_lfs_backed_training_shard_per_required_component_then_lexical_path",
            "fineweb2": "one shard for each frozen language",
            "finemath": "one 4plus shard and one 3plus shard",
            "olmoe_science_reference": "one peS2o shard and one wiki shard",
            "stack_v3": f"{stack_shards} path-stratified physical shards spanning the frozen inventory for license-yield variance auditing",
            "forbidden_split_path_parts": sorted(FORBIDDEN_SPLIT_PARTS),
        },
        "sources": planned_sources,
        "file_count": sum(len(source["files"]) for source in planned_sources),
        "total_bytes": sum(record["size"] for source in planned_sources for record in source["files"]),
        "claim_boundary": "This plan selects real bytes for bounded schema, license, privacy, and content auditing. It admits no documents or training tokens and is not a training manifest.",
    }
    body["plan_sha256"] = hashlib.sha256(canonical_bytes(body)).hexdigest()
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "data" / "moe7b_150b_mixture_v1.json")
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stack-shards", type=int, default=8)
    args = parser.parse_args()
    plan = build_plan(args.plan, args.metadata_dir, stack_shards=args.stack_shards)
    atomic_json(args.output, plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
