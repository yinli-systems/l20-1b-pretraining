#!/usr/bin/env python3
"""Create deterministic, pair-atomic folds for the audited posture pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    temporary.replace(path)
    return sha256_file(path)


def validate_pairs(pairs: list[dict[str, Any]], expected_pairs: int) -> None:
    if len(pairs) != expected_pairs:
        raise ValueError(f"expected {expected_pairs} pairs, found {len(pairs)}")
    pair_ids: set[str] = set()
    image_ids: set[str] = set()
    image_hashes: set[str] = set()
    for pair in pairs:
        pair_id = pair["pair_id"]
        if pair_id in pair_ids:
            raise ValueError(f"duplicate pair id: {pair_id}")
        pair_ids.add(pair_id)
        if pair.get("candidate_answers") != ["sitting", "standing"]:
            raise ValueError(f"unexpected candidates for {pair_id}")
        images = (pair["image_a"], pair["image_b"])
        if {image["answer"] for image in images} != {"sitting", "standing"}:
            raise ValueError(f"pair is not posture-balanced: {pair_id}")
        for image in images:
            if image["class_name"] != pair["class_name"]:
                raise ValueError(f"class mismatch in pair {pair_id}")
            if image.get("human_audit", {}).get("decision") != "accept":
                raise ValueError(f"non-accepted image in pair {pair_id}")
            image_id = image["image_id"]
            image_hash = image["image_sha256"]
            if image_id in image_ids or image_hash in image_hashes:
                raise ValueError(f"image leakage or duplicate in pair {pair_id}")
            image_ids.add(image_id)
            image_hashes.add(image_hash)


def assign_folds(
    pairs: list[dict[str, Any]], fold_count: int, seed: int
) -> list[dict[str, Any]]:
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[pair["class_name"]].append(pair)

    fold_loads = [0] * fold_count
    assignments: dict[str, int] = {}
    # Largest strata first lets the remainder allocation keep total fold sizes balanced.
    for class_name, class_pairs in sorted(
        grouped.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        ranked_pairs = sorted(
            class_pairs,
            key=lambda pair: stable_hash(seed, "pair-rank", class_name, pair["pair_id"]),
        )
        base, remainder = divmod(len(ranked_pairs), fold_count)
        extra_folds = sorted(
            range(fold_count),
            key=lambda fold: (
                fold_loads[fold],
                stable_hash(seed, "extra-fold", class_name, fold),
            ),
        )[:remainder]
        capacities = {
            fold: base + int(fold in extra_folds) for fold in range(fold_count)
        }
        slots = [
            fold
            for fold in sorted(
                range(fold_count),
                key=lambda value: stable_hash(seed, "slot-order", class_name, value),
            )
            for _ in range(capacities[fold])
        ]
        if len(slots) != len(ranked_pairs):
            raise RuntimeError("fold slot construction failed")
        for pair, fold in zip(ranked_pairs, slots):
            assignments[pair["pair_id"]] = fold
            fold_loads[fold] += 1

    output = [
        {
            **pair,
            "crossfit": {
                "fold": assignments[pair["pair_id"]],
                "unit": "pair_id",
                "pair_atomic": True,
            },
        }
        for pair in pairs
    ]
    return sorted(output, key=lambda row: (row["crossfit"]["fold"], row["pair_id"]))


def validate_assignment(rows: list[dict[str, Any]], fold_count: int) -> None:
    fold_counts = Counter(row["crossfit"]["fold"] for row in rows)
    if set(fold_counts) != set(range(fold_count)):
        raise RuntimeError("one or more folds are empty")
    if max(fold_counts.values()) - min(fold_counts.values()) > 1:
        raise RuntimeError("total fold sizes differ by more than one pair")
    class_folds = Counter(
        (row["class_name"], row["crossfit"]["fold"]) for row in rows
    )
    for class_name in {row["class_name"] for row in rows}:
        counts = [class_folds[(class_name, fold)] for fold in range(fold_count)]
        if max(counts) - min(counts) > 1:
            raise RuntimeError(f"class fold imbalance for {class_name}")


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("status") != "authorized_posture_crossfit_split_only":
        raise RuntimeError("posture crossfit split construction is not authorized")
    if protocol.get("training_authorized") is not False:
        raise RuntimeError("split protocol must not authorize parameter updates")
    source = protocol["source_code"]
    if sha256_file(Path(__file__)) != source["splitter_sha256"]:
        raise RuntimeError("splitter source hash mismatch")
    test_path = ROOT / "test_split_openimages_posture_crossfit_v1.py"
    if sha256_file(test_path) != source["test_sha256"]:
        raise RuntimeError("splitter test source hash mismatch")
    for label, record in protocol["inputs"].items():
        record_path = resolve_path(record["path"])
        if not record_path.is_file() or sha256_file(record_path) != record["sha256"]:
            raise RuntimeError(f"crossfit input mismatch: {label}")
    finalization = json.loads(
        resolve_path(protocol["inputs"]["finalization_receipt"]["path"]).read_text()
    )
    if finalization.get("status") != "complete_human_accepted_balanced_pairs":
        raise RuntimeError("finalized posture pairs are not complete")
    if finalization.get("training_authorized") is not False:
        raise RuntimeError("finalization receipt unexpectedly authorizes training")
    return protocol


def build(protocol_path: Path) -> dict[str, Any]:
    protocol = validate_protocol(protocol_path)
    pair_path = resolve_path(protocol["inputs"]["pair_manifest"]["path"])
    pairs = load_jsonl(pair_path)
    validate_pairs(pairs, int(protocol["data"]["expected_pairs"]))
    rows = assign_folds(
        pairs,
        fold_count=int(protocol["crossfit"]["fold_count"]),
        seed=int(protocol["crossfit"]["seed"]),
    )
    fold_count = int(protocol["crossfit"]["fold_count"])
    validate_assignment(rows, fold_count)

    output_path = resolve_path(protocol["outputs"]["fold_manifest"])
    output_hash = write_jsonl_atomic(output_path, rows)
    fold_counts = Counter(row["crossfit"]["fold"] for row in rows)
    class_fold_counts = Counter(
        (row["class_name"], row["crossfit"]["fold"]) for row in rows
    )
    receipt = {
        "schema_version": "2026-09-14-v1",
        "status": "complete_posture_crossfit_split_only",
        "training_authorized": False,
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "input_pair_manifest": {"path": str(pair_path), "sha256": sha256_file(pair_path)},
        "fold_manifest": {
            "path": str(output_path),
            "sha256": output_hash,
            "pairs": len(rows),
            "images": 2 * len(rows),
        },
        "fold_counts": [
            {"fold": fold, "pairs": fold_counts[fold]}
            for fold in range(fold_count)
        ],
        "class_fold_counts": [
            {
                "class_name": class_name,
                "fold": fold,
                "pairs": class_fold_counts[(class_name, fold)],
            }
            for class_name in sorted({row["class_name"] for row in rows})
            for fold in range(fold_count)
        ],
        "leakage_audit": {
            "pair_atomic": True,
            "unique_pair_ids": len({row["pair_id"] for row in rows}),
            "unique_image_ids": len(
                {
                    image["image_id"]
                    for row in rows
                    for image in (row["image_a"], row["image_b"])
                }
            ),
            "unique_image_sha256": len(
                {
                    image["image_sha256"]
                    for row in rows
                    for image in (row["image_a"], row["image_b"])
                }
            ),
            "human_rejected_images_used": 0,
        },
        "statistical_plan": protocol["statistical_plan"],
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(resolve_path(protocol["outputs"]["receipt"]), receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.protocol.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
