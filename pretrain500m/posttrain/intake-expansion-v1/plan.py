"""Build the frozen unique-data expansion plan for F2/F3 confirmation.

The plan derives required unique prediction tokens from the exact short-screen
manifests. Confirmation uses four times the screen budget and every newly
admitted source remains capped at two epochs, so the minimum unique corpus is
twice the largest per-source screen quota. A 35% acquisition reserve covers
quality, family, decontamination, and packing losses; it is not pre-admission.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCREEN_TOKENS_PER_BLOCK = 2048
SAFETY_FACTOR = 1.35
EXPECTED = {
    "F2-screen-manifest.json": "a6120a304a16d5b569787cc542b5917cb8bf13ba49e97e2083cbf0ff3235591c",
    "F3-screen-manifest.json": "4cf466475ef56cff66fcd90e6077532f7657de0e5e127126dce91e30aa72b6cb",
    "history.json": "3124a2dccb869231487bb4f6d4aada05b1c34dc4b5c8b7e9f63a860038833097",
    "parquet-metadata.json": "32e70dd190cbc50bde2217f32184bd035ae7bcfbda6588ef06d88c7e847e8b47",
    "finemath-shard00-parquet-metadata.json": "17095dc27ae33d2b5c3f8e029dcbab0adb530a4292f2e0313aefc31e486ce6ce",
    "finemath-shard00-metadata.json": "59a388356fb0ee66616b6c21d4eb4be485efabf29a5c2932779d55e177b14b17",
    "source-shards.json": "dcac0813b1e2e2a7e718c790a884234e3687052dca7da552914b61060e7b7cd5",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def load(name: str):
    path = HERE / name
    expected = EXPECTED.get(name)
    if expected and sha256(path) != expected:
        raise ValueError(f"frozen input changed: {name}")
    return json.loads(path.read_text())


def manifest_unique_tokens(manifest: dict, source_id: str) -> int:
    for source in manifest["sources"]:
        if source["id"] == source_id:
            return sum(int(shard["blocks"]) * SCREEN_TOKENS_PER_BLOCK for shard in source["shards"])
    return 0


def is_admitted_history(receipt: dict, source_id: str) -> bool:
    path = receipt["path"]
    return (
        "diverse-intake-v1/" in path
        or "diverse-intake-v2/" in path
        or (source_id == "finemath4" and "diverse-intake-v4-finemath/" in path)
    )


def limits(source_id: str, groups: int) -> dict:
    # Caps are intentionally generous relative to observed receipts but finite.
    is_code = source_id.startswith("code_")
    return {
        "max_rows": groups * 1000,
        "max_text_bytes": (2 if source_id == "pdf_en" else 1) * 1024**3,
        "max_parquet_network_bytes": (2 if source_id in {"pdf_en", "dclm"} else 1) * 1024**3,
        "max_code_attempts": groups * 1000 if is_code else 0,
        "max_code_payload_bytes": 2 * 1024**3 if is_code else 0,
    }


def build_plan() -> dict:
    f2, f3 = load("F2-screen-manifest.json"), load("F3-screen-manifest.json")
    history = load("history.json")["receipts"]
    metadata = {x["id"]: x for x in load("parquet-metadata.json")["segments"]}
    shard00 = load("finemath-shard00-parquet-metadata.json")
    source_shards = {x["id"]: x for x in load("source-shards.json")["segments"]}
    ids = sorted((set(f2["source_block_quotas"]) | set(f3["source_block_quotas"])) - {"fineweb_edu"})
    sources = []
    segments = []
    for source_id in ids:
        screen_quota = max(
            int(f2["source_block_quotas"].get(source_id, 0)),
            int(f3["source_block_quotas"].get(source_id, 0)),
        ) * SCREEN_TOKENS_PER_BLOCK
        unique_minimum = screen_quota * 2
        current_unique = max(manifest_unique_tokens(f2, source_id), manifest_unique_tokens(f3, source_id))
        admitted = [r for r in history if r["source_id"] == source_id and is_admitted_history(r, source_id)]
        pending = [r for r in history if r["source_id"] == source_id and "diverse-intake-v3/" in r["path"]]
        admitted_groups = sum(len(r["groups"]) for r in admitted)
        if admitted_groups <= 0 or current_unique <= 0:
            raise ValueError(f"no measured admitted yield for {source_id}")
        yield_per_group = current_unique / admitted_groups
        deficit = max(0, unique_minimum - current_unique)
        gross_groups = math.ceil(deficit * SAFETY_FACTOR / yield_per_group)
        pending_groups = sum(len(r["groups"]) for r in pending)
        new_groups = max(0, gross_groups - pending_groups)
        sources.append({
            "source_id": source_id,
            "screen_quota_prediction_tokens": screen_quota,
            "confirmation_quota_prediction_tokens": screen_quota * 4,
            "unique_prediction_tokens_required_at_two_epoch_cap": unique_minimum,
            "current_admitted_unique_prediction_tokens": current_unique,
            "measured_admitted_groups": admitted_groups,
            "measured_prediction_tokens_per_group": yield_per_group,
            "unique_prediction_token_deficit": deficit,
            "acquisition_safety_factor": SAFETY_FACTOR,
            "gross_new_groups_including_reserve": gross_groups,
            "ready_unadmitted_groups": pending_groups,
            "groups_to_acquire": new_groups,
        })
        base = source_shards[source_id]
        relevant = [r for r in history if r["source_id"] == source_id]
        excluded = sorted({g for r in relevant for g in r["groups"]})

        def add_segment(segment_id: str, path: str, size: int, digest: str, take: int, row_groups: int, refs: list[dict]):
            if take <= 0:
                return
            if take > row_groups - len(excluded if path == base["path"] else []):
                raise ValueError(f"insufficient unused groups for {segment_id}")
            segment = {
                "segment_id": segment_id,
                "logical_source_id": source_id,
                "repo_id": base["repo_id"],
                "revision": base["revision"],
                "path": path,
                "bytes": size,
                "lfs_sha256": digest,
                "declared_dataset_license": base["declared_dataset_license"],
                "history": [{"path": r["path"], "sha256": r["sha256"]} for r in refs],
                "exclude_row_groups": excluded if path == base["path"] else [],
                "additional_row_groups": take,
                "parquet_row_groups": row_groups,
                "training_admitted": False,
            }
            segment.update(limits(source_id, take))
            segments.append(segment)

        if source_id == "finemath4":
            remaining = metadata[source_id]["row_groups"] - len(excluded)
            take_old = min(new_groups, remaining)
            add_segment("finemath4_shard53_tail", base["path"], base["bytes"], base["sha256"], take_old,
                        metadata[source_id]["row_groups"], relevant)
            take_new = new_groups - take_old
            fine_meta = load("finemath-shard00-metadata.json")["entry"]
            add_segment("finemath4_shard00", shard00["path"], shard00["bytes"], fine_meta["lfs"]["oid"],
                        take_new, shard00["row_groups"], [])
        else:
            add_segment(source_id + "_extra", base["path"], base["bytes"], base["sha256"], new_groups,
                        metadata[source_id]["row_groups"], relevant)
    fine_metadata = load("finemath-shard00-metadata.json")
    return {
        "schema_version": 1,
        "status": "FROZEN_UNIQUE_DATA_EXPANSION_PLAN_NOT_ADMITTED",
        "created_utc": fine_metadata["retrieved_utc"],
        "purpose": "two-seed 2.147B-token F2/F3 confirmation from the same base checkpoint",
        "rule": "unique minimum = 2 * largest exact short-screen source quota; acquire measured deficit with 35% pre-admission reserve",
        "screen_manifest_sha256": {"F2": EXPECTED["F2-screen-manifest.json"], "F3": EXPECTED["F3-screen-manifest.json"]},
        "source_repeat_cap": 2.0,
        "training_admitted": False,
        "sources": sources,
        "segments": segments,
    }


def main() -> None:
    output = HERE / "plan.json"
    output.write_text(json.dumps(build_plan(), indent=2, sort_keys=True) + "\n")
    print(sha256(output), output)


if __name__ == "__main__":
    main()
