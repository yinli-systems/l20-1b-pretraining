#!/usr/bin/env python3
"""Select and pack a FineMath top-up without changing frozen reserved splits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import unicodedata
import zlib


TOKENS = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]|[^\W_]+|[^\w\s]", re.UNICODE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def repeated_ngram_coverage(text: str, width: int = 10) -> float:
    tokens = TOKENS.findall(unicodedata.normalize("NFC", text).casefold())
    if len(tokens) < width:
        return 0.0
    grams = [tuple(tokens[i:i + width]) for i in range(len(tokens) - width + 1)]
    counts = Counter(grams)
    covered = bytearray(len(tokens))
    for offset, gram in enumerate(grams):
        if counts[gram] > 1:
            covered[offset:offset + width] = b"\x01" * width
    return sum(covered) / len(tokens)


def quality_reasons(text: str, text_hash: str, policy: dict) -> tuple[list[str], float, float]:
    coverage = repeated_ngram_coverage(text)
    blob = text.encode("utf-8")
    compression = len(zlib.compress(blob, 9)) / max(1, len(blob))
    reasons = []
    extreme = policy["extreme_repetition_rule"]
    if (coverage >= extreme["repeated_10gram_coverage_at_least"] and
            compression <= extreme["compression_ratio_at_most"]):
        reasons.append("extreme_repetition_and_compression")
    if text_hash in policy["manual_rejects"]:
        reasons.append("manual_" + policy["manual_rejects"][text_hash])
    return sorted(reasons), coverage, compression


def select_candidates(prior: dict[tuple[str, int, int], str], expanded_rows: list[dict],
                      source: str = "finemath4", tranche: int = 3):
    families = defaultdict(list)
    for row in expanded_rows:
        families[row["family_id"]].append(row)
    selected, rejected = [], Counter()
    for family_id, members in families.items():
        new = [row for row in members if row["source_id"] == source and row["tranche"] == tranche]
        if not new:
            continue
        old_members = [row for row in members if (row["source_id"], row["tranche"], row["row"]) in prior]
        family_reasons = {reason for row in members for reason in row["family_exclusion_reasons"]}
        if old_members:
            rejected["observed_preexisting_family_bridge_documents"] += len(new)
            rejected["observed_preexisting_family_bridge_families"] += 1
            continue
        if family_reasons:
            rejected["quarantined_or_policy_excluded_documents"] += len(new)
            rejected["quarantined_or_policy_excluded_families"] += 1
            continue
        eligible = [row for row in new if row["selected_eligible_representative"] and
                    row["passes_filters_and_bound_exclusions"]]
        selected.extend(eligible)
        rejected["unselected_duplicate_or_ineligible_documents"] += len(new) - len(eligible)
    return sorted(selected, key=lambda row: row["row"]), dict(sorted(rejected.items()))


def load_bound_report(path: Path, expected: str, status: str) -> dict:
    if sha256(path) != expected:
        raise ValueError("report identity mismatch")
    report = json.loads(path.read_text())
    if report["status"] != status:
        raise ValueError("input stage incomplete")
    return report


def assignment(report_path: Path, report: dict) -> Path:
    path = report_path.parent / "row-assignments.jsonl.gz"
    if sha256(path) != report["output_files"][str(path)]:
        raise ValueError("assignment identity mismatch")
    return path


def run(args) -> dict:
    prior_report = load_bound_report(args.prior_family_report, args.expected_prior_family_sha256,
        "FAMILY_CLOSURE_AND_RESERVED_ASSIGNMENTS_COMPLETE_NOT_ADMITTED")
    expanded_report = load_bound_report(args.expanded_family_report, args.expected_expanded_family_sha256,
        "FAMILY_CLOSURE_AND_RESERVED_ASSIGNMENTS_COMPLETE_NOT_ADMITTED")
    raw_report = load_bound_report(args.raw_report, args.expected_raw_sha256,
        "COMBINED_RAW_INTAKE_MEASURED_NOT_ADMITTED")
    decisions = load_bound_report(args.quality_decisions, args.expected_quality_decisions_sha256,
        "QUALITY_REVIEW_EXCLUSIONS_BOUND_NOT_APPLIED")
    if sha256(args.tokenizer) != args.expected_tokenizer_sha256:
        raise ValueError("tokenizer identity mismatch")

    prior_assignment = assignment(args.prior_family_report, prior_report)
    expanded_assignment = assignment(args.expanded_family_report, expanded_report)
    prior = {}
    for row in rows(prior_assignment):
        prior[row["source_id"], row["tranche"], row["row"]] = row["text_sha256"]
    expanded_rows = list(rows(expanded_assignment))
    for row in expanded_rows:
        key = row["source_id"], row["tranche"], row["row"]
        if key in prior and prior[key] != row["text_sha256"]:
            raise ValueError("preexisting row identity changed in expanded family graph")
    if len(prior) != sum((row["source_id"], row["tranche"], row["row"]) in prior
                         for row in expanded_rows):
        raise ValueError("expanded family graph does not preserve every prior row")
    candidates, family_rejections = select_candidates(prior, expanded_rows)
    candidate_by_row = {row["row"]: row for row in candidates}

    raw_root = Path(raw_report["tranches"][3]["path"])
    raw_path = raw_root / "finemath4.jsonl.gz"
    expected_raw_file = expanded_report["input_bindings"].get(str(raw_path))
    if not expected_raw_file or sha256(raw_path) != expected_raw_file:
        raise ValueError("top-up raw file identity mismatch")

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    eos = tokenizer.token_to_id("<|endoftext|>")
    if eos != 50279:
        raise ValueError("tokenizer EOS differs from frozen model")
    kept, rejected_quality = [], Counter()
    rejected_quality_documents = 0
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle):
            bound = candidate_by_row.get(row_number)
            if bound is None:
                continue
            raw = json.loads(line)
            text = raw.get("text") or raw.get("content") or ""
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            if text_hash != bound["text_sha256"]:
                raise ValueError("selected raw text identity changed")
            reasons, coverage, compression = quality_reasons(text, text_hash, decisions["policy"])
            if reasons:
                rejected_quality.update(reasons)
                rejected_quality_documents += 1
                continue
            token_ids = tokenizer.encode(text, add_special_tokens=False).ids + [eos]
            if len(token_ids) != bound["encoded_tokens_including_one_eos"]:
                raise ValueError("bound token count changed")
            kept.append((bound, token_ids, coverage, compression))
    if len(kept) + rejected_quality_documents != len(candidates):
        raise ValueError("top-up quality review coverage incomplete")
    total_tokens = sum(len(token_ids) for _, token_ids, _, _ in kept)
    array_tokens = total_tokens // 2049 * 2049
    blocks = array_tokens // 2049
    if blocks < args.minimum_blocks:
        raise ValueError(f"top-up has {blocks} blocks; minimum is {args.minimum_blocks}")
    required_free = 18 * 1024**3 + array_tokens * 2 + 64 * 1024**2
    if shutil.disk_usage(args.output.parent).free < required_free:
        raise ValueError("aggregate top-up packing headroom insufficient")
    if args.output.exists():
        raise ValueError("output already exists")
    args.output.mkdir()

    import numpy as np
    array_path = args.output / "finemath4.topup.blocks.npy"
    array = np.lib.format.open_memmap(array_path, mode="w+", dtype=np.uint16, shape=(array_tokens,))
    index_path = args.output / "finemath4.topup.documents.jsonl.gz"
    cursor = written = 0
    tail = []
    with gzip.open(index_path, "xt", encoding="utf-8", compresslevel=1) as index:
        for bound, token_ids, coverage, compression in kept:
            start = cursor
            count = min(len(token_ids), max(0, array_tokens - written))
            if count:
                array[written:written + count] = token_ids[:count]
                written += count
            tail.extend(token_ids[count:])
            cursor += len(token_ids)
            index.write(json.dumps({
                "source_id": "finemath4", "tranche": 3, "row": bound["row"],
                "text_sha256": bound["text_sha256"], "family_id": bound["family_id"],
                "diagnostic_partition_overridden_to_train": bound["partition"],
                "start": start, "end": cursor, "first_document_target_offset": start + 1,
                "repeated_10gram_coverage": coverage, "compression_ratio": compression,
            }) + "\n")
    if written != array_tokens or cursor != total_tokens or len(tail) != total_tokens - array_tokens or len(tail) >= 2049:
        raise ValueError("packed top-up accounting mismatch")
    array.flush()
    tail_path = args.output / "finemath4.topup.tail.json"
    tail_path.write_text(json.dumps({"token_ids": tail, "virtual_offset": array_tokens,
                                     "training_admitted": False}, indent=2) + "\n")
    bindings = {
        str(args.prior_family_report): args.expected_prior_family_sha256,
        str(prior_assignment): prior_report["output_files"][str(prior_assignment)],
        str(args.expanded_family_report): args.expected_expanded_family_sha256,
        str(expanded_assignment): expanded_report["output_files"][str(expanded_assignment)],
        str(args.raw_report): args.expected_raw_sha256,
        str(raw_path): expected_raw_file,
        str(args.quality_decisions): args.expected_quality_decisions_sha256,
        str(args.tokenizer): args.expected_tokenizer_sha256,
    }
    for path, expected in bindings.items():
        if sha256(Path(path)) != expected:
            raise ValueError("bound input changed during top-up packing")
    report = {
        "schema": "p529m-finemath-topup-v1",
        "status": "FINEMATH_TOPUP_PACK_COMPLETE_NOT_ADMITTED",
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input_bindings": dict(sorted(bindings.items())),
        "family_rejections": family_rejections,
        "selected_new_only_representatives_before_quality": len(candidates),
        "quality_rejected_documents": rejected_quality_documents,
        "quality_rejection_counts": dict(sorted(rejected_quality.items())),
        "documents": len(kept), "encoded_tokens": total_tokens,
        "array_tokens": array_tokens, "tail_tokens": len(tail),
        "blocks": blocks, "prediction_tokens": blocks * 2048,
        "output": {"path": str(array_path), "sha256": sha256(array_path), "dtype": "uint16"},
        "tail": {"path": str(tail_path), "sha256": sha256(tail_path)},
        "document_index": {"path": str(index_path), "sha256": sha256(index_path)},
        "reserved_split_policy": "preserve all prior assignments; accept only expanded-graph new-only families as train",
        "minimum_blocks_required": args.minimum_blocks,
        "training_admitted": False, "training_launched": False,
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("prior-family-report", "expanded-family-report", "raw-report",
                 "quality-decisions", "tokenizer", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-prior-family-sha256", "expected-expanded-family-sha256",
                 "expected-raw-sha256", "expected-quality-decisions-sha256",
                 "expected-tokenizer-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--minimum-blocks", type=int, default=624)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({key: report[key] for key in ("status", "documents", "blocks",
                                                    "prediction_tokens")}, sort_keys=True))


if __name__ == "__main__":
    main()
