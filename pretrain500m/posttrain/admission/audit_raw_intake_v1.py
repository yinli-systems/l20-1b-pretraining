"""Measure actual first-tranche token/family deficits without admitting data."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import urlsplit
from tokenizers import Tokenizer


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.next')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def leaf_id(sid):
    mapping = {'pdf_en': 'finepdfs_edu', 'cosmopedia2': 'synthetic',
               'finemath4': 'HuggingFaceTB/finemath:finemath-4plus',
               'infiwebmath4': 'HuggingFaceTB/finemath:infiwebmath-4plus',
               'code_python': 'Python', 'code_javascript': 'JavaScript',
               'code_typescript': 'TypeScript', 'code_cpp': 'Cpp', 'code_java': 'Java'}
    if sid.startswith('multilingual_'):
        return sid.removeprefix('multilingual_')
    return mapping.get(sid, sid)


def family(row, sid):
    if sid.startswith('code_'):
        name = str(row.get('repo_name', '')).strip().strip('/').lower()
        return ('repository:' + name, 'repository_aliases_unresolved') if name else (None, 'missing_repository')
    if sid == 'cosmopedia2':
        seed = row.get('seed_data')
        if isinstance(seed, dict):
            identity = seed.get('url') or seed.get('id')
            if identity:
                return 'seed:' + str(identity), 'parent_candidate_unverified'
        return None, 'seed_source_label_is_not_parent_document_identity'
    url = row.get('url')
    try:
        hostname = urlsplit(str(url)).hostname
    except ValueError:
        hostname = None
    if hostname:
        return 'host:' + hostname.lower().removeprefix('www.'), 'host_only_registrable_domain_and_aliases_unresolved'
    return None, 'missing_url_host'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--expected-tokenizer-sha256', required=True)
    parser.add_argument('--quotas', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert sha(args.tokenizer) == args.expected_tokenizer_sha256, 'tokenizer hash mismatch'
    intake = json.loads((args.input / 'intake-summary.json').read_text())
    assert intake['status'] == 'RAW_INTAKE_FINISHED_NOT_ADMITTED'
    args.output.mkdir(exist_ok=False)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    quotas = json.loads(args.quotas.read_text())
    required = {}
    for groups in quotas['screen'].values():
        for group in groups.values():
            for leaf, value in group['leaves'].items():
                required[leaf] = max(required.get(leaf, 0), (value['prediction_tokens'] + 1) // 2)
    exact_seen = {}
    candidates_seen = {}
    global_duplicates = Counter()
    candidate_duplicates = Counter()
    summaries = {}
    started = time.time()
    with (args.output / 'row-index.jsonl').open('x') as index:
        for source in sorted(args.input.glob('*.jsonl.gz')):
            sid = source.name.removesuffix('.jsonl.gz')
            receipt = json.loads((args.input / (sid + '.receipt.json')).read_text())
            assert receipt['status'] == 'RAW_SEGMENT_READY'
            source_hash = sha(source)
            assert source_hash == receipt['output_sha256'], sid + ' output checksum mismatch'
            seen_local = set()
            families = set()
            family_status = Counter()
            counts = Counter()
            lengths = Counter()
            with gzip.open(source, 'rt', encoding='utf-8') as handle:
                for number, line in enumerate(handle):
                    row = json.loads(line)
                    text = row.get('text')
                    assert isinstance(text, str) and text.strip(), (sid, number, 'missing text')
                    assert row['_provenance']['source_id'] == sid
                    raw_hash = hashlib.sha256(text.encode()).hexdigest()
                    if row.get('content_sha256'):
                        assert row['content_sha256'] == raw_hash, (sid, number, 'code text hash mismatch')
                    # Only byte-identical text is deduplicated for this upper bound.
                    # Compatibility/case/whitespace candidates are counted for review.
                    normalized = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip().casefold()
                    candidate_hash = hashlib.sha256(normalized.encode()).hexdigest()
                    tokens = len(tokenizer.encode(text, add_special_tokens=False).ids) + 1
                    counts['rows'] += 1
                    counts['encoded_tokens_including_one_eos'] += tokens
                    if raw_hash not in seen_local:
                        seen_local.add(raw_hash)
                        counts['exact_unique_encoded_tokens_including_one_eos'] += tokens
                    else:
                        counts['within_source_byte_identical_duplicates'] += 1
                    previous = exact_seen.setdefault(raw_hash, sid)
                    if previous != sid:
                        global_duplicates['|'.join(sorted((previous, sid)))] += 1
                    previous_candidate = candidates_seen.setdefault(candidate_hash, (sid, raw_hash))
                    if previous_candidate[1] != raw_hash:
                        candidate_duplicates['|'.join(sorted((previous_candidate[0], sid)))] += 1
                    key, status = family(row, sid)
                    family_status[status] += 1
                    if key:
                        families.add(key)
                    else:
                        counts['missing_family_rows'] += 1
                    lengths['lt128' if tokens < 128 else '128_to_2048' if tokens <= 2048 else '2049_to_8192' if tokens <= 8192 else '8193_to_32768' if tokens <= 32768 else 'gt32768'] += 1
                    index.write(json.dumps({'source_id': sid, 'row': number, 'text_sha256': raw_hash,
                                            'normalized_candidate_sha256': candidate_hash,
                                            'encoded_tokens_including_one_eos': tokens,
                                            'family_candidate': key, 'family_status': status,
                                            'provenance': row['_provenance']}, ensure_ascii=False) + '\n')
                    if counts['rows'] % 1000 == 0:
                        atomic_json(args.output / 'progress.json', {'status': 'RUNNING', 'source': sid,
                                    'source_rows': counts['rows'], 'completed_sources': len(summaries),
                                    'elapsed_seconds': time.time() - started})
            assert counts['rows'] == receipt['rows_written'], sid + ' receipt row count mismatch'
            assert sha(source) == source_hash, sid + ' changed while auditing'
            upper = counts['exact_unique_encoded_tokens_including_one_eos'] // 2049 * 2048
            minimum = required[leaf_id(sid)]
            summaries[sid] = dict(counts, byte_identical_unique_documents=len(seen_local),
                                  provisional_family_candidates=len(families), family_status=dict(family_status),
                                  token_length_histogram=dict(lengths), raw_file_sha256=source_hash,
                                  packed_prediction_tokens_upper_bound_before_filtering=upper,
                                  min_unique_prediction_tokens_for_largest_single_screen_at_two_epochs=minimum,
                                  shortage_lower_bound_before_validation_and_quality_filters=max(0, minimum - upper))
            atomic_json(args.output / 'source-counts.json', summaries)
    assert sum(s['rows'] for s in summaries.values()) == intake['rows']
    report = {'status': 'RAW_INTAKE_MEASURED_NOT_ADMITTED', 'training_admitted': False,
              'tokenizer_sha256': args.expected_tokenizer_sha256, 'quota_plan_sha256': sha(args.quotas),
              'sources': summaries, 'rows': intake['rows'], 'elapsed_seconds': time.time() - started,
              'cross_source_byte_identical_overlap_rows': dict(global_duplicates),
              'normalized_nonidentical_candidate_overlap_rows': dict(candidate_duplicates),
              'row_index_sha256': sha(args.output / 'row-index.jsonl'),
              'limits': ['Token counts include one EOS per document and are upper bounds before filtering and held-out reserves.',
                         'Host and repository candidates are not certified independent families; registrable domains, aliases and synthetic parents remain unresolved.',
                         'No near-duplicate removal, old-FineWeb overlap, benchmark decontamination or quality admission has been performed.',
                         'Normalized nonidentical candidates are not automatically discarded, especially for mathematical notation and code.',
                         'Two-epoch shortage is a lower bound for one screen run only, not cumulative screening/confirmation sufficiency.']}
    atomic_json(args.output / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('sources', 'limits')}), flush=True)


if __name__ == '__main__':
    main()
