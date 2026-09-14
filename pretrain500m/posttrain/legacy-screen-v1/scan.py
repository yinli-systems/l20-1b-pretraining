"""Parallel legacy benchmark hash scan and old-normalization index; no admission."""
import argparse
from collections import Counter
import datetime
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import time
import unicodedata

WORD_RE = re.compile(r'[a-z0-9]+')
HASHES = frozenset()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def legacy_normalized_hash(text):
    # Match the original FineWeb packer exactly: lower(), not casefold().
    norm = ' '.join(unicodedata.normalize('NFKC', text).lower().split())
    return hashlib.sha256(norm.encode()).hexdigest()


def first_legacy_match(text, hashes):
    words = WORD_RE.findall(text.lower())
    for i in range(len(words) - 12):
        h = hashlib.blake2b(' '.join(words[i:i+13]).encode(), digest_size=8).digest()
        if h in hashes:
            return {'word_offset': i, 'ngram_blake2b_64': h.hex()}
    return None


def atomic_json(path, value):
    tmp = path.with_suffix('.next')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def scan_file(item):
    source, tranche, path, expected_rows, output = item
    start = time.monotonic()
    index = Path(output) / f'{tranche}-{source}.index.jsonl.gz'
    hits = Path(output) / f'{tranche}-{source}.candidates.jsonl'
    progress = Path(output) / f'{tranche}-{source}.progress.json'
    rows = candidates = 0
    with gzip.open(path, 'rt', encoding='utf-8') as inp, \
            gzip.open(index, 'xt', encoding='utf-8', compresslevel=1) as idx, \
            hits.open('x') as out:
        for row_number, line in enumerate(inp):
            row = json.loads(line)
            text = row.get('text') or row.get('content')
            if not isinstance(text, str) or not text.strip():
                raise ValueError('missing text')
            raw_hash = hashlib.sha256(text.encode()).hexdigest()
            match = first_legacy_match(text, HASHES)
            record = {'source_id': source, 'tranche': tranche, 'row': row_number,
                      'text_sha256': raw_hash, 'legacy_normalized_sha256': legacy_normalized_hash(text)}
            idx.write(json.dumps(record) + '\n')
            if match is not None:
                out.write(json.dumps(dict(record, match=match, provenance=row['_provenance'])) + '\n')
                candidates += 1
            rows += 1
            if rows % 1000 == 0:
                atomic_json(progress, dict(status='RUNNING', rows=rows, candidates=candidates,
                                          elapsed_seconds=time.monotonic()-start))
    if rows != expected_rows:
        raise ValueError('row count changed during legacy scan')
    result = dict(source_id=source, tranche=tranche, rows=rows, candidate_rows=candidates,
                  index=str(index), index_sha256=sha(index), candidates=str(hits),
                  candidates_sha256=sha(hits), elapsed_seconds=time.monotonic()-start)
    atomic_json(progress, dict(result, status='COMPLETED'))
    return result


def run(args):
    global HASHES
    started = time.monotonic()
    if not 1 <= args.workers <= 4:
        raise ValueError('worker count must be 1..4')
    if sha(args.database) != args.expected_database_sha256:
        raise ValueError('legacy database identity mismatch')
    if shutil.disk_usage(args.output.parent).free < args.min_free_gib * 1024**3:
        raise ValueError('disk headroom below acquisition floor')
    # Reuse the frozen receipt/physical-group preflight; its sources are separately bound.
    sys.path.insert(0, str(args.audit_source))
    from audit_raw_intake_v2 import preflight
    files, bindings, tranches = preflight(args.input)
    con = sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        provenance = con.execute('SELECT repo, config, revision, split, rows FROM provenance').fetchall()
        if len(provenance) != 20:
            raise ValueError('unexpected legacy reference coverage')
        HASHES = frozenset(row[0] for row in con.execute('SELECT hash FROM ngrams'))
        if not HASHES or any(not isinstance(h, bytes) or len(h) != 8 for h in HASHES):
            raise ValueError('invalid legacy ngram hash set')
    finally:
        con.close()
    args.output.mkdir(exist_ok=False)
    atomic_json(args.output/'launch.json', dict(status='SCANNING', workers=args.workers,
                database_sha256=args.expected_database_sha256, reference_hashes=len(HASHES),
                expected_rows=sum(t['rows'] for t in tranches), training_admitted=False))
    work = [(f['source_id'], f['tranche'], str(f['path']), f['receipt']['rows_written'], str(args.output))
            for f in files]
    # Larger files start first; output identities and matching do not depend on scheduling.
    work.sort(key=lambda x: -Path(x[2]).stat().st_size)
    # Linux fork shares the immutable hash table rather than reloading it per source.
    with mp.get_context('fork').Pool(args.workers) as pool:
        results = list(pool.imap_unordered(scan_file, work, chunksize=1))
    for path, digest in bindings.items():
        if sha(path) != digest:
            raise ValueError('bound input changed during legacy scan')
    if sha(args.database) != args.expected_database_sha256:
        raise ValueError('legacy database changed during scan')
    for root in args.input:
        if (root/'writer.lock').exists():
            raise ValueError('new input writer appeared')
        for pattern in ['*.receipt.json', '*.jsonl.gz']:
            if set(root.glob(pattern)) != {p for p in bindings if p.parent == root and p.match(pattern)}:
                raise ValueError('input file set changed during scan')
    total = sum(x['rows'] for x in results)
    if total != sum(t['rows'] for t in tranches):
        raise ValueError('total row mismatch')
    report = dict(status='LEGACY_HASH_SCAN_COMPLETE_NOT_ADMITTED',
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), rows=total,
        candidate_rows=sum(x['candidate_rows'] for x in results), reference_hashes=len(HASHES),
        database_sha256=args.expected_database_sha256, provenance=provenance,
        input_bindings={str(k): v for k, v in bindings.items()},
        files=sorted(results, key=lambda x: (x['tranche'], x['source_id'])),
        elapsed_seconds=time.monotonic()-started, workers=args.workers, training_admitted=False,
        scope='Original ASCII [a-z0-9]+ lowercased 13-word BLAKE2b-64 membership; first hit per row.',
        limitations=['Hash matches are candidate exclusions; original reference spans are not available in this database.',
                     'Short, Unicode, translated, paraphrased and renamed-code contamination are not cleared by this scan.',
                     'No source text was deleted, no family expansion or packing applied, and no admission receipt issued.',
                     'The old-normalization index enables a subsequent old-corpus comparison; that comparison is not done here.'])
    atomic_json(args.output/'report.json', report)
    print(json.dumps({k: report[k] for k in ['status','rows','candidate_rows','elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', action='append', type=Path, required=True)
    p.add_argument('--database', type=Path, required=True)
    p.add_argument('--expected-database-sha256', required=True)
    p.add_argument('--audit-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--min-free-gib', type=float, default=18)
    run(p.parse_args())
