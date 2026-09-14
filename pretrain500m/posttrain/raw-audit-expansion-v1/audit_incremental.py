"""Extend a bound historical raw audit by scanning only appended tranches.

Historical files and their tokenizer index are re-hashed and their row records are
re-keyed to the generalized file ordinal.  Raw text in the frozen prefix is not
read or tokenized again.  Exact ownership is append-stable: a historical owner is
never displaced by a later tranche.
"""
import argparse
from collections import Counter, defaultdict
import datetime
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import re
import time
import unicodedata

from tokenizers import Tokenizer

from audit_raw_intake_v1 import atomic_json, family, sha
from audit_raw_intake_v2 import bound_json, preflight, require


COUNT_FIELDS = (
    'rows', 'encoded_tokens_including_one_eos',
    'source_unique_encoded_tokens_including_one_eos',
    'globally_unique_documents_assigned',
    'globally_unique_encoded_tokens_including_one_eos_assigned',
    'within_source_byte_identical_duplicates_across_all_tranches',
    'missing_family_rows',
)

WORKER_TOKENIZER = None


def initialize_worker(tokenizer_path):
    global WORKER_TOKENIZER
    WORKER_TOKENIZER = Tokenizer.from_file(str(tokenizer_path))
    WORKER_TOKENIZER.no_truncation()
    WORKER_TOKENIZER.no_padding()


def scan_new_file(item):
    """Validate and tokenize one appended raw file into a deterministic staging index."""
    f, staging = item
    sid, tranche = f['source_id'], f['tranche']
    stage_path = Path(staging) / f"{tranche}-{f['file_id']}.jsonl"
    progress_path = Path(staging) / f"{tranche}-{f['file_id']}.progress.json"
    physical_seen = set()
    nrows = 0
    with gzip.open(f['path'], 'rt', encoding='utf-8') as handle, stage_path.open('x') as writer:
        for row_number, line in enumerate(handle):
            row = json.loads(line)
            text = row.get('text') or row.get('content')
            require(isinstance(text, str) and bool(text.strip()), 'missing text')
            p = row['_provenance']
            require(p['source_id'] == sid and
                    (p['repo_id'], p['revision'], p['shard']) == f['identity'],
                    'row provenance/source mismatch')
            group = f['groups'].get(p['row_group'])
            require(group is not None, 'row references an unrecorded group')
            offset = p['row_in_group']
            require(type(offset) is int and 0 <= offset < group['rows_read'],
                    'row outside group bounds')
            require(p['physical_row'] == group['physical_row_offset'] + offset,
                    'physical row offset mismatch')
            physical_key = (p['row_group'], offset)
            require(physical_key not in physical_seen, 'repeated physical row')
            physical_seen.add(physical_key)
            raw_hash = hashlib.sha256(text.encode()).hexdigest()
            if sid.startswith('code_'):
                require(row.get('content_sha256') == raw_hash, 'code text hash mismatch')
            elif row.get('content_sha256'):
                require(row['content_sha256'] == raw_hash, 'text hash mismatch')
            normalized = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip().casefold()
            candidate_hash = hashlib.sha256(normalized.encode()).hexdigest()
            tokens = len(WORKER_TOKENIZER.encode(text, add_special_tokens=False).ids) + 1
            family_key, status = family(row, sid)
            writer.write(json.dumps(dict(source_id=sid, tranche=tranche, row=row_number,
                        text_sha256=raw_hash, normalized_candidate_sha256=candidate_hash,
                        encoded_tokens_including_one_eos=tokens, family_candidate=family_key,
                        family_status=status, provenance=p), ensure_ascii=False) + '\n')
            nrows += 1
            if nrows % 1000 == 0:
                atomic_json(progress_path, dict(status='RUNNING_TOKENIZE', source=sid,
                            tranche=tranche, rows=nrows))
    require(nrows == f['receipt']['rows_written'], 'source row count mismatch: ' + sid)
    require(sha(f['path']) == f['receipt']['output_sha256'], 'raw file changed during worker scan')
    atomic_json(progress_path, dict(status='TOKENIZED', source=sid, tranche=tranche, rows=nrows))
    return dict(source_id=sid, file_id=f['file_id'], tranche=tranche,
                root_index=f['root_index'], path=str(f['path']), rows=nrows,
                raw_file_sha256=f['receipt']['output_sha256'], stage_path=str(stage_path),
                stage_sha256=sha(stage_path))


def map_historical_files(files, prior_files, prefix, bindings):
    """Map an already-generalized prior file ordinal through its bound absolute path."""
    current_by_path={str(Path(f['path']).resolve()):f for f in files if f['root_index']<prefix}
    if len(current_by_path)!=sum(f['root_index']<prefix for f in files):
        raise ValueError('duplicate historical physical path')
    mapped={}
    for old_file in prior_files:
        key=(old_file['source_id'],old_file['tranche'])
        require(key not in mapped,'duplicate historical file ordinal')
        path=str(Path(old_file['path']).resolve())
        require(path in current_by_path,'historical file missing')
        current=current_by_path[path]
        require(old_file['source_id']==current['source_id'] and
                old_file['rows']==current['receipt']['rows_written'] and
                old_file['raw_file_sha256']==bindings[current['path']],
                'historical raw identity changed')
        mapped[key]=current
    require(len(mapped)==len(current_by_path),'historical file count changed')
    return mapped


def audit(inputs, prior_report_path, tokenizer_path, expected_tokenizer_sha256,
          plan_path, output, workers=4):
    started = time.time()
    require(type(workers) is int and 1 <= workers <= 8, 'workers must be 1..8')
    require(sha(tokenizer_path) == expected_tokenizer_sha256, 'tokenizer hash mismatch')
    files, bindings, tranches = preflight(inputs)
    bindings[tokenizer_path] = expected_tokenizer_sha256
    prior = bound_json(prior_report_path, bindings)
    require(prior['status'] == 'COMBINED_RAW_INTAKE_MEASURED_NOT_ADMITTED',
            'historical audit incomplete')
    require(prior['tokenizer_sha256'] == expected_tokenizer_sha256,
            'historical tokenizer differs')
    prefix = len(prior['tranches'])
    require(0 < prefix < len(tranches), 'no appended tranche')
    for old, current in zip(prior['tranches'], tranches[:prefix]):
        require(Path(old['path']).resolve() == Path(current['path']).resolve(),
                'historical tranche order/path changed')
        require(old['rows'] == current['rows'] and old['sources'] == current['sources'] and
                old['summary_sha256'] == current['summary_sha256'],
                'historical tranche metadata changed')
    for path_string, digest in prior['input_bindings'].items():
        path = Path(path_string)
        require(path.exists() and sha(path) == digest, 'historical binding changed: ' + path_string)

    prior_index = prior_report_path.parent / 'row-index.jsonl'
    require(sha(prior_index) == prior['row_index_sha256'], 'historical row index changed')
    plan = bound_json(plan_path, bindings)
    require(plan.get('status') == 'FROZEN_UNIQUE_DATA_EXPANSION_PLAN_NOT_ADMITTED',
            'wrong expansion plan')
    require(plan['source_repeat_cap'] == 2.0, 'unexpected source repeat cap')
    required = {x['source_id']: x['unique_prediction_tokens_required_at_two_epoch_cap']
                for x in plan['sources']}

    # Map prior generalized ordinals by immutable physical path. This also supports
    # more than one file for the same source inside a historical input root.
    old_key_to_file=map_historical_files(files,prior['files'],prefix,bindings)

    output.mkdir(exist_ok=False)
    staging = output / 'staging'
    staging.mkdir()
    exact_seen, normalized_seen = {}, {}
    source_seen, source_families = defaultdict(set), defaultdict(set)
    source_counts, family_statuses, lengths = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    overlap = Counter(prior.get('byte_identical_overlap_rows', {}))
    candidates = Counter(prior.get('normalized_nonidentical_candidates', {}))
    for sid, summary in prior['sources'].items():
        source_counts[sid].update({k: summary.get(k, 0) for k in COUNT_FIELDS})
        family_statuses[sid].update(summary.get('family_status', {}))
        lengths[sid].update(summary.get('token_length_histogram', {}))

    new_files = sorted((x for x in files if x['root_index'] >= prefix),
                       key=lambda x: (x['source_id'], x['tranche']))
    require(all(f['source_id'] in required for f in new_files),
            'an appended source is absent from expansion plan')
    tasks = [(f, str(staging)) for f in new_files]
    index_path = output / 'row-index.jsonl'
    with mp.get_context('fork').Pool(workers, initializer=initialize_worker,
                                     initargs=(str(tokenizer_path),)) as pool:
        pending = pool.map_async(scan_new_file, tasks, chunksize=1)
        with index_path.open('x') as index:
            historical_rows = 0
            with prior_index.open() as handle:
                for line in handle:
                    row = json.loads(line)
                    key = (row['source_id'], row['tranche'])
                    require(key in old_key_to_file, 'historical index references unknown file')
                    row['tranche'] = old_key_to_file[key]['tranche']
                    owner = row['exact_dedup_owner']
                    owner_key = (owner['source_id'], owner['tranche'])
                    require(owner_key in old_key_to_file, 'historical owner references unknown file')
                    owner = dict(owner)
                    owner['tranche'] = old_key_to_file[owner_key]['tranche']
                    row['exact_dedup_owner'] = owner
                    raw_hash = row['text_sha256']
                    source_seen[row['source_id']].add(raw_hash)
                    if row.get('family_candidate'):
                        source_families[row['source_id']].add(row['family_candidate'])
                    exact_seen.setdefault(raw_hash, owner)
                    normalized_seen.setdefault(row['normalized_candidate_sha256'], raw_hash)
                    index.write(json.dumps(row, ensure_ascii=False) + '\n')
                    historical_rows += 1
            require(historical_rows == prior['rows'], 'historical index row count changed')
            new_file_counts = pending.get()
            new_file_counts.sort(key=lambda x: (x['source_id'], x['tranche']))
            appended_rows = 0
            for result in new_file_counts:
                stage_path = Path(result['stage_path'])
                require(sha(stage_path) == result['stage_sha256'], 'staging index changed')
                with stage_path.open() as stage:
                    for line in stage:
                        row = json.loads(line)
                        sid, tranche = row['source_id'], row['tranche']
                        counts = source_counts[sid]
                        raw_hash = row['text_sha256']
                        tokens = row['encoded_tokens_including_one_eos']
                        counts['rows'] += 1
                        counts['encoded_tokens_including_one_eos'] += tokens
                        if raw_hash not in source_seen[sid]:
                            source_seen[sid].add(raw_hash)
                            counts['source_unique_encoded_tokens_including_one_eos'] += tokens
                        else:
                            counts['within_source_byte_identical_duplicates_across_all_tranches'] += 1
                        owner = exact_seen.get(raw_hash)
                        if owner is None:
                            owner = dict(source_id=sid, tranche=tranche, row=row['row'])
                            exact_seen[raw_hash] = owner
                            counts['globally_unique_documents_assigned'] += 1
                            counts['globally_unique_encoded_tokens_including_one_eos_assigned'] += tokens
                        else:
                            scope = 'cross_source' if owner['source_id'] != sid else (
                                'cross_tranche_same_source' if owner['tranche'] != tranche else 'within_file')
                            overlap[scope] += 1
                        previous = normalized_seen.setdefault(row['normalized_candidate_sha256'], raw_hash)
                        if previous != raw_hash:
                            candidates['normalized_nonidentical_rows'] += 1
                        family_statuses[sid][row['family_status']] += 1
                        if row.get('family_candidate'):
                            source_families[sid].add(row['family_candidate'])
                        else:
                            counts['missing_family_rows'] += 1
                        bucket = ('lt128' if tokens < 128 else '128_to_2048' if tokens <= 2048 else
                                  '2049_to_8192' if tokens <= 8192 else
                                  '8193_to_32768' if tokens <= 32768 else 'gt32768')
                        lengths[sid][bucket] += 1
                        row['exact_dedup_owner'] = owner
                        index.write(json.dumps(row, ensure_ascii=False) + '\n')
                        appended_rows += 1
                        if appended_rows % 10000 == 0:
                            atomic_json(output / 'progress.json', dict(status='MERGING_INCREMENTAL',
                                appended_rows=appended_rows, total_appended_rows=sum(x['rows'] for x in new_file_counts),
                                elapsed_seconds=time.time() - started))
    for result in new_file_counts:
        Path(result['stage_path']).unlink()
        progress = staging / (Path(result['stage_path']).stem + '.progress.json')
        if progress.exists():
            progress.unlink()
    staging.rmdir()

    for path, digest in bindings.items():
        require(sha(path) == digest, 'input changed during audit: ' + str(path))
    for tranche in tranches:
        root = Path(tranche['path'])
        require(not (root / 'writer.lock').exists(), 'new intake writer appeared during audit')
        for pattern in ['*.receipt.json', '*.jsonl.gz']:
            actual = set(root.glob(pattern))
            expected = {p for p in bindings if p.parent == root and p.match(pattern)}
            require(actual == expected, 'intake file set changed during audit')

    summaries = {}
    for sid, counts in sorted(source_counts.items()):
        source_upper = counts['source_unique_encoded_tokens_including_one_eos'] // 2049 * 2048
        assigned_upper = counts['globally_unique_encoded_tokens_including_one_eos_assigned'] // 2049 * 2048
        minimum = required.get(sid, 0)
        summaries[sid] = dict(counts, byte_identical_unique_documents=len(source_seen[sid]),
            provisional_family_candidates=len(source_families[sid]),
            family_status=dict(family_statuses[sid]), token_length_histogram=dict(lengths[sid]),
            packed_prediction_tokens_upper_bound_before_filtering=source_upper,
            globally_deduplicated_assigned_prediction_tokens_upper_bound=assigned_upper,
            min_unique_prediction_tokens_for_confirmation_at_two_epochs=minimum,
            shortage_lower_bound_before_validation_and_quality_filters=max(0, minimum-assigned_upper))
    total_rows = sum(x['rows'] for x in tranches)
    require(total_rows == sum(x['rows'] for x in summaries.values()), 'combined row count mismatch')
    file_counts = []
    for f in files:
        file_counts.append(dict(source_id=f['source_id'], file_id=f['file_id'], tranche=f['tranche'],
                                root_index=f['root_index'], path=str(f['path']),
                                rows=f['receipt']['rows_written'], raw_file_sha256=bindings[f['path']]))
    report = dict(status='COMBINED_RAW_INTAKE_MEASURED_NOT_ADMITTED', training_admitted=False,
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        tokenizer_sha256=expected_tokenizer_sha256, expansion_plan_sha256=bindings[plan_path],
        incremental_from_report=str(prior_report_path), incremental_from_report_sha256=bindings[prior_report_path],
        tokenizer_workers=workers,
        historical_ownership_policy='append_stable_prior_owner_then_new_source_and_file_order',
        physical_group_disjointness='PASS', input_hashes_stable='PASS', tranches=tranches,
        input_bindings={str(p): d for p, d in bindings.items()}, files=file_counts, sources=summaries,
        rows=total_rows, globally_byte_identical_unique_documents=len(exact_seen),
        byte_identical_overlap_rows=dict(overlap), normalized_nonidentical_candidates=dict(candidates),
        row_index_sha256=sha(index_path), elapsed_seconds=time.time()-started,
        limits=['Historical raw text was not re-tokenized; its frozen report, row index, receipts and raw files were re-hashed.',
                'Exact ownership is append-stable; later tranches cannot displace a historical owner.',
                'Raw upper bounds include one EOS per document, before filtering and held-out reserves.',
                'Families are provisional until the bound family-closure stage.',
                'Two-epoch shortages use the frozen 2.147B-token F2/F3 confirmation requirements.'])
    atomic_json(output / 'source-counts.json', summaries)
    atomic_json(output / 'report.json', report)
    atomic_json(output / 'progress.json', dict(status=report['status'], rows=total_rows,
                                             elapsed_seconds=report['elapsed_seconds']))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='append', type=Path, required=True)
    parser.add_argument('--prior-report', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--expected-tokenizer-sha256', required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    report = audit(args.input, args.prior_report, args.tokenizer,
                   args.expected_tokenizer_sha256, args.plan, args.output, args.workers)
    print(json.dumps({k: report[k] for k in ['status', 'rows',
          'globally_byte_identical_unique_documents', 'byte_identical_overlap_rows',
          'elapsed_seconds']}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
