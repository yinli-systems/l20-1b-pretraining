#!/usr/bin/env python3
"""Extend the frozen exclusion union with identity-bound four-tranche audits."""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


REASONS = {
    "supplemental": "verified_supplemental_exact_span_exclusion",
    "legacy": "legacy_13word_hash_candidate_conservative_exclusion",
    "old_source": "old_source_normalized_overlap_conservative_exclusion",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def add(mapping: dict[str, set[str]], text_hash: str, reason: str) -> None:
    if len(text_hash) != 64 or any(c not in "0123456789abcdef" for c in text_hash):
        raise ValueError("invalid text SHA256")
    mapping.setdefault(text_hash, set()).add(reason)


def build(prior_path: Path, supplemental_path: Path, legacy_path: Path,
          old_source_path: Path) -> dict:
    prior = load_json(prior_path)
    supplemental = load_json(supplemental_path)
    legacy = load_json(legacy_path)
    old_source = load_json(old_source_path)
    if prior["status"] != "POLICY_EXCLUSIONS_BOUND_NOT_APPLIED_TO_PACK":
        raise ValueError("prior union is not bound")
    if supplemental["status"] != "SUPPLEMENTAL_EXACT_SPAN_AUDIT_COMPLETE_ADMISSION_PENDING":
        raise ValueError("supplemental audit incomplete")
    if legacy["status"] != "LEGACY_HASH_SCAN_COMPLETE_NOT_ADMITTED":
        raise ValueError("legacy audit incomplete")
    if old_source["status"] != "OLD_SOURCE_NORMALIZED_OVERLAP_COMPLETE_NOT_ADMITTED":
        raise ValueError("old-source audit incomplete")

    exclusions = {key: set(value) for key, value in prior["exclude_text_sha256_reasons"].items()}
    bindings = {str(prior_path): sha256(prior_path)}
    counts = Counter()

    supplemental_candidates = supplemental_path.parent / "candidate-exclusions.jsonl"
    if sha256(supplemental_candidates) != supplemental["candidates_sha256"]:
        raise ValueError("supplemental candidate identity mismatch")
    bindings[str(supplemental_path)] = sha256(supplemental_path)
    bindings[str(supplemental_candidates)] = sha256(supplemental_candidates)
    for row in read_jsonl(supplemental_candidates):
        add(exclusions, row["text_sha256"], REASONS["supplemental"])
        counts["supplemental_candidate_rows"] += 1

    bindings[str(legacy_path)] = sha256(legacy_path)
    for record in legacy["files"]:
        path = Path(record["candidates"])
        if sha256(path) != record["candidates_sha256"]:
            raise ValueError("legacy candidate identity mismatch")
        bindings[str(path)] = record["candidates_sha256"]
        for row in read_jsonl(path):
            add(exclusions, row["text_sha256"], REASONS["legacy"])
            counts["legacy_candidate_rows"] += 1
    if counts["legacy_candidate_rows"] != legacy["candidate_rows"]:
        raise ValueError("legacy candidate count mismatch")

    bindings[str(old_source_path)] = sha256(old_source_path)
    old_records = old_source["files"]
    if isinstance(old_records, dict):
        old_records = old_records.values()
    for record in old_records:
        path = Path(record["matches"])
        if sha256(path) != record["matches_sha256"]:
            raise ValueError("old-source match identity mismatch")
        bindings[str(path)] = record["matches_sha256"]
        for row in read_jsonl(path):
            for document in row["new_documents"]:
                add(exclusions, document["text_sha256"], REASONS["old_source"])
                counts["old_source_new_document_references"] += 1

    frozen = {key: sorted(value) for key, value in sorted(exclusions.items())}
    reason_counts = Counter(reason for values in frozen.values() for reason in values)
    return {
        "schema": "p529m-exclusion-union-v2",
        "status": "POLICY_EXCLUSIONS_BOUND_NOT_APPLIED_TO_PACK",
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "prior_union_sha256": sha256(prior_path),
        "input_bindings": dict(sorted(bindings.items())),
        "input_counts": dict(sorted(counts.items())),
        "reason_unique_text_counts": dict(sorted(reason_counts.items())),
        "unique_excluded_text_hashes": len(frozen),
        "exclude_text_sha256_reasons": frozen,
        "training_admitted": False,
        "raw_files_modified": False,
        "family_expansion_completed": False,
        "packing_has_consumed_plan": False,
        "scope": (
            "Conservative union of the frozen three-tranche exclusions and all "
            "identity-bound four-tranche supplemental, legacy-hash, and old-source candidates."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--supplemental-report", type=Path, required=True)
    parser.add_argument("--legacy-report", type=Path, required=True)
    parser.add_argument("--old-source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.prior, args.supplemental_report, args.legacy_report,
                   args.old_source_report)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "unique_excluded_text_hashes", "input_counts")}, sort_keys=True))


if __name__ == "__main__":
    main()
