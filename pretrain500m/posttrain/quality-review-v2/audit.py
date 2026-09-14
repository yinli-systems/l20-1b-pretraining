"""Audit selected prose for repeated/template and injected-web content.

This emits review candidates only.  It never changes eligibility or grants
training admission.
"""
from collections import Counter, defaultdict
import argparse
import datetime
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import re
import time
import unicodedata
import zlib


TOKENS = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]|[^\W_]+|[^\w\s]", re.UNICODE)
SPANISH_ADULT = re.compile(
    r"\b(?:porno|porn[oó]graf\w*|puta\w*|prostitut\w*|escort\w*|scort\w*|follar\w*|webcam)\b",
    re.IGNORECASE,
)
POLICY = {
    "schema": "p529m-selected-quality-review-v2",
    "seed": 20260914,
    "ngram_tokens": 10,
    "candidate_repeated_ngram_coverage": 0.20,
    "candidate_low_compression_ratio": 0.20,
    "low_compression_minimum_characters": 1000,
    "spanish_adult_document_hits": 8,
    "spanish_adult_tail_distinct_terms": 3,
    "japanese_dating_phrase_hits": 3,
    "training_admitted": False,
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def normalized_tokens(text):
    return TOKENS.findall(unicodedata.normalize("NFC", text).casefold())


def repeated_ngram_coverage(tokens, width=10):
    if len(tokens) < width:
        return 0.0, 0
    grams = [tuple(tokens[i : i + width]) for i in range(len(tokens) - width + 1)]
    counts = Counter(grams)
    covered = bytearray(len(tokens))
    maximum = max(counts.values(), default=0)
    for i, gram in enumerate(grams):
        if counts[gram] > 1:
            covered[i : i + width] = b"\x01" * width
    return sum(covered) / len(tokens), maximum


def compression_ratio(text):
    blob = text.encode("utf-8")
    return len(zlib.compress(blob, 9)) / max(1, len(blob))


def candidate_reasons(source, text, coverage, compression):
    reasons = []
    if not source.startswith("code_"):
        if coverage > POLICY["candidate_repeated_ngram_coverage"]:
            reasons.append("repeated_10gram_coverage")
        if len(text) >= POLICY["low_compression_minimum_characters"] and compression < POLICY["candidate_low_compression_ratio"]:
            reasons.append("low_compression_ratio")
    if source == "multilingual_spa_Latn":
        hits = SPANISH_ADULT.findall(text)
        tail_hits = {v.casefold() for v in SPANISH_ADULT.findall(text[len(text) * 2 // 3 :])}
        if len(hits) >= POLICY["spanish_adult_document_hits"] and len(tail_hits) >= POLICY["spanish_adult_tail_distinct_terms"]:
            reasons.append("adult_keyword_stuffing")
        if len(text) < 2000 and "Wikia no es accesible" in text and "bloqueo de anuncios" in text:
            reasons.append("dominant_platform_boilerplate")
    if source == "multilingual_jpn_Jpan":
        if text.count("出会い掲示板") >= POLICY["japanese_dating_phrase_hits"]:
            reasons.append("repeated_dating_keyword")
        ui = sum(phrase in text for phrase in ("引用をストックしました", "引用するにはまずログイン", "引用をストックできません", "限定公開記事"))
        if len(text) < 1000 and ui >= 2:
            reasons.append("dominant_platform_boilerplate")
    if source == "multilingual_cmn_Hani":
        if "本文关键词" in text and "本文来源" in text:
            reasons.append("seo_footer")
    return sorted(set(reasons))


def inspect_file(item):
    source, tranche, path, expected_hash, selected = item
    if sha(path) != expected_hash:
        raise ValueError(f"raw input changed: {path}")
    candidates = []
    metrics = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            bound = selected.get(row_index)
            if bound is None:
                continue
            row = json.loads(line)
            text = row.get("text") or row.get("content") or ""
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            if text_hash != bound["text_sha256"]:
                raise ValueError("selected text identity changed")
            tokens = normalized_tokens(text)
            coverage, maximum = repeated_ngram_coverage(tokens)
            compression = compression_ratio(text)
            reasons = candidate_reasons(source, text, coverage, compression)
            metric = {
                "source_id": source,
                "partition": bound["partition"],
                "tranche": tranche,
                "row": row_index,
                "text_sha256": text_hash,
                "family_id": bound["family_id"],
                "characters": len(text),
                "tokens": len(tokens),
                "repeated_10gram_coverage": coverage,
                "maximum_10gram_occurrences": maximum,
                "compression_ratio": compression,
                "candidate_reasons": reasons,
            }
            metrics.append(metric)
            if reasons:
                metric = dict(metric)
                metric.update(start=text[:500], middle=text[max(0, len(text)//2-250):len(text)//2+250], end=text[-500:])
                candidates.append(metric)
    if sha(path) != expected_hash:
        raise ValueError(f"raw input changed during review: {path}")
    return metrics, candidates


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[round((len(values) - 1) * q)]


def run(args):
    started = time.monotonic()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be 1..4")
    if sha(args.family_report) != args.expected_family_sha256:
        raise ValueError("family report identity mismatch")
    family = json.loads(args.family_report.read_text())
    if family["status"] != "FAMILY_CLOSURE_AND_RESERVED_ASSIGNMENTS_COMPLETE_NOT_ADMITTED":
        raise ValueError("family stage incomplete")
    assignment = args.family_report.parent / "row-assignments.jsonl.gz"
    if sha(assignment) != family["output_files"][str(assignment)]:
        raise ValueError("family assignment identity mismatch")
    raw_path = next(Path(p) for p in family["input_bindings"] if p.endswith("/diverse-audit-v2-three-tranche/report.json"))
    if sha(raw_path) != family["input_bindings"][str(raw_path)]:
        raise ValueError("raw report identity mismatch")
    roots = [Path(t["path"]) for t in json.loads(raw_path.read_text())["tranches"]]
    selected = defaultdict(dict)
    with gzip.open(assignment, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            allowed = ({'development','confirmation'} if args.partition == 'reserved'
                       else {args.partition})
            if row["partition"] in allowed and row["selected_eligible_representative"]:
                selected[row["source_id"], row["tranche"]][row["row"]] = row
    work = []
    for (source, tranche), rows in selected.items():
        path = roots[tranche] / f"{source}.jsonl.gz"
        work.append((source, tranche, path, family["input_bindings"][str(path)], rows))
    all_metrics, candidates = [], []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for metrics, found in pool.imap_unordered(inspect_file, work):
            all_metrics.extend(metrics)
            candidates.extend(found)
    if len(all_metrics) != sum(len(v) for v in selected.values()):
        raise ValueError("selected review coverage incomplete")
    by_source = {}
    for source in sorted({m["source_id"] for m in all_metrics}):
        rows = [m for m in all_metrics if m["source_id"] == source]
        by_source[source] = {
            "documents": len(rows),
            "candidate_documents": sum(bool(m["candidate_reasons"]) for m in rows),
            "repeated_10gram_coverage_p50": percentile([m["repeated_10gram_coverage"] for m in rows], .5),
            "repeated_10gram_coverage_p95": percentile([m["repeated_10gram_coverage"] for m in rows], .95),
            "repeated_10gram_coverage_p99": percentile([m["repeated_10gram_coverage"] for m in rows], .99),
            "compression_ratio_p01": percentile([m["compression_ratio"] for m in rows], .01),
            "compression_ratio_p05": percentile([m["compression_ratio"] for m in rows], .05),
            "reason_counts": dict(Counter(r for m in rows for r in m["candidate_reasons"])),
        }
    final = {
        "status": "SELECTED_QUALITY_REVIEW_CANDIDATES_COMPLETE_NOT_ADMITTED",
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "policy": POLICY,
        "family_report_sha256": args.expected_family_sha256,
        "partition_scope": args.partition,
        "selected_documents_reviewed": len(all_metrics),
        "candidate_documents": len(candidates),
        "sources": by_source,
        "candidates": sorted(candidates, key=lambda x: (x["source_id"], x["text_sha256"])),
        "training_admitted": False,
        "limitations": [
            "Candidate heuristics require review and are not automatic exclusion decisions.",
            "The review covers selected train documents only; held-out partitions are unchanged.",
            "Structural heuristics do not certify factual correctness or educational value.",
        ],
    }
    blob = (json.dumps(final, ensure_ascii=False, indent=2) + "\n").encode()
    if args.output.exists():
        raise ValueError("output already exists")
    args.output.write_bytes(blob)
    print(json.dumps({"status": final["status"], "reviewed": len(all_metrics), "candidates": len(candidates), "elapsed_seconds": final["elapsed_seconds"], "sha256": hashlib.sha256(blob).hexdigest()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-report", type=Path, required=True)
    parser.add_argument("--expected-family-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--partition", choices=('train','development','confirmation','reserved'), default='train')
    run(parser.parse_args())
