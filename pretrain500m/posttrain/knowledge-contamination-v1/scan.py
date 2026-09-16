#!/usr/bin/env python3
"""Conservative exact and lexical-near SciQ scan over current raw intake."""

import argparse
from collections import Counter, defaultdict
import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import unicodedata

import ahocorasick

from audit_raw_intake_v2 import preflight, require


ROWS_SHA256 = "55cc70950fdd8892d6c7c2cbd458d2c3c6fc5f88548b909799d147b68e7c16da"
POLICY = {
    "schema": "p529m-sciq-exact-lexical-near-contamination-v1",
    "normalization": "NFKC, casefold, collapse Unicode whitespace",
    "word_tokenization": "Unicode alphanumeric runs excluding underscore",
    "exact": "normalized question as a contiguous substring",
    "question_option_shingle_tokens": 5,
    "question_option_minimum_coverage": 0.5,
    "question_option_minimum_distinct_matches": 2,
    "support_shingle_tokens": 13,
    "support_minimum_coverage": 0.15,
    "support_minimum_distinct_matches": 8,
    "scope": "three raw intake tranches; superset of selected F2 documents",
    "action": "exclude affected reference rows from development and confirmation pools",
}


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", normalize(text), flags=re.UNICODE)


def shingles(tokens: list[str], size: int) -> set[tuple[str, ...]]:
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index:index + size])
            for index in range(len(tokens) - size + 1)}


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    if sha(args.rows) != ROWS_SHA256:
        raise ValueError("SciQ canonical-row digest mismatch")
    references = [json.loads(line) for line in args.rows.read_text().splitlines()]
    require(len(references) == 1000 and
            [row["ordinal"] for row in references] == list(range(1000)),
            "invalid SciQ reference rows")
    files, bindings, tranches = preflight(args.input)
    args.output.mkdir(exist_ok=False)

    exact = ahocorasick.Automaton()
    q_index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    s_index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    q_totals = []
    s_totals = []
    for row in references:
        ordinal = row["ordinal"]
        question = normalize(row["question"])
        if question in exact:
            existing = list(exact.get(question)); existing.append(ordinal)
            exact.add_word(question, tuple(existing))
        else:
            exact.add_word(question, (ordinal,))
        question_options = " ".join([
            row["question"], row["distractor1"], row["distractor2"],
            row["distractor3"], row["correct_answer"],
        ])
        q = shingles(words(question_options), POLICY["question_option_shingle_tokens"])
        s = shingles(words(row["support"]), POLICY["support_shingle_tokens"])
        q_totals.append(len(q)); s_totals.append(len(s))
        for key in q: q_index[key].append(ordinal)
        for key in s: s_index[key].append(ordinal)
    exact.make_automaton()

    contaminated = set()
    counts = Counter()
    per_source = {}
    with (args.output / "candidate-reference-hits.jsonl").open("x") as writer:
        for file in files:
            source_key = f"{file['tranche']}:{file['source_id']}"
            current = Counter()
            with gzip.open(file["path"], "rt", encoding="utf-8") as stream:
                for document_ordinal, line in enumerate(stream):
                    row = json.loads(line)
                    text = row.get("text") or row.get("content")
                    require(isinstance(text, str), "invalid raw text")
                    normalized = normalize(text)
                    exact_hits = set()
                    for _, owners in exact.iter(normalized): exact_hits.update(owners)
                    tokens = words(text)
                    q_counts = Counter(); s_counts = Counter()
                    for key in shingles(tokens, POLICY["question_option_shingle_tokens"]):
                        for owner in q_index.get(key, ()): q_counts[owner] += 1
                    for key in shingles(tokens, POLICY["support_shingle_tokens"]):
                        for owner in s_index.get(key, ()): s_counts[owner] += 1
                    lexical_q = {owner for owner, matched in q_counts.items()
                                 if matched >= max(POLICY["question_option_minimum_distinct_matches"],
                                    math.ceil(POLICY["question_option_minimum_coverage"] * q_totals[owner]))}
                    lexical_s = {owner for owner, matched in s_counts.items()
                                 if matched >= max(POLICY["support_minimum_distinct_matches"],
                                    math.ceil(POLICY["support_minimum_coverage"] * s_totals[owner]))}
                    owners = exact_hits | lexical_q | lexical_s
                    if owners:
                        contaminated.update(owners)
                        current["documents_with_candidate_hits"] += 1
                        for owner in sorted(owners):
                            writer.write(json.dumps({
                                "reference_ordinal": owner,
                                "source_id": file["source_id"],
                                "tranche": file["tranche"],
                                "document_ordinal": document_ordinal,
                                "document_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                "exact_question": owner in exact_hits,
                                "question_option_shingles_matched": q_counts[owner],
                                "question_option_shingles_total": q_totals[owner],
                                "support_shingles_matched": s_counts[owner],
                                "support_shingles_total": s_totals[owner],
                            }, sort_keys=True) + "\n")
                            current["reference_hit_records"] += 1
                    current["rows_scanned"] += 1
                    current["text_characters_scanned"] += len(text)
                    if current["rows_scanned"] % 1000 == 0:
                        atomic_json(args.output / "progress.json", {
                            "status": "RUNNING", "source": source_key,
                            "rows_in_source": current["rows_scanned"],
                            "completed_files": len(per_source),
                            "unique_references_hit": len(contaminated),
                            "elapsed_seconds": time.time() - started,
                        })
            require(current["rows_scanned"] == file["receipt"]["rows_written"],
                    "raw row count mismatch")
            per_source[source_key] = dict(current); counts.update(current)

    for path, expected in bindings.items(): require(sha(path) == expected, "raw source changed")
    clean = sorted(set(range(1000)) - contaminated)
    require(len(clean) >= 800, "fewer than 800 clean SciQ references remain")
    result = {
        "schema": POLICY["schema"], "status": "PASS_CLEAN_REFERENCE_POOL_READY",
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "training_admitted": False, "policy": POLICY,
        "policy_sha256": hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest(),
        "rows_sha256": sha(args.rows), "tranches": tranches,
        "input_bindings": {str(path): value for path, value in bindings.items()},
        "counts": dict(counts), "sources": per_source,
        "contaminated_reference_ordinals": sorted(contaminated),
        "clean_reference_ordinals": clean,
        "clean_reference_count": len(clean),
        "candidate_hits_sha256": sha(args.output / "candidate-reference-hits.jsonl"),
        "elapsed_seconds": time.time() - started,
        "limits": [
            "This detects exact questions and fixed lexical shingle overlap, not semantic paraphrases or model-ancestor contamination.",
            "The three scanned raw tranches are a conservative superset of documents selected for F2.",
            "A hit excludes the affected reference row from both proxy roles; it does not prove intentional benchmark training.",
        ],
    }
    atomic_json(args.output / "report.json", result)
    print(json.dumps({key: result[key] for key in
                      ("status", "counts", "clean_reference_count", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
