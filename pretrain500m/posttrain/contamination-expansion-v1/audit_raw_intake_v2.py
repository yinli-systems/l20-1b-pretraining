"""Audit completed raw tranches together; never issue training admission."""
import argparse
from collections import Counter, defaultdict
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata

from tokenizers import Tokenizer
from audit_raw_intake_v1 import atomic_json, family, leaf_id, sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound_json(path, bindings):
    data = path.read_bytes()
    bindings[path] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def preflight(inputs):
    """Bind old and generalized intake receipts and reject physical reuse."""
    roots = [p.resolve(strict=True) for p in inputs]
    require(bool(roots), 'no intake directories')
    require(len(roots) == len(set(roots)), 'duplicate intake directory')
    files, bindings, tranches = [], {}, []
    groups_seen = set()
    prior_by_source, receipt_by_path = {}, {}
    for ordinal, root in enumerate(roots):
        require(not (root / 'writer.lock').exists(), 'intake still has a writer: ' + str(root))
        summary_path = root / 'intake-summary.json'
        summary = bound_json(summary_path, bindings)
        require(summary['status'] == 'RAW_INTAKE_FINISHED_NOT_ADMITTED', 'intake is unfinished')
        require(summary['sources_blocked'] == 0, 'intake has blocked sources')
        receipts = sorted(root.glob('*.receipt.json'))
        require(bool(receipts), 'intake has no sources')
        require(len(receipts) == summary['sources_ready'], 'summary source count mismatch')
        require({p.name.removesuffix('.receipt.json') for p in receipts} ==
                {p.name.removesuffix('.jsonl.gz') for p in root.glob('*.jsonl.gz')},
                'raw/receipt file set mismatch')
        require(not list(root.glob('*.part')), 'partial raw output exists')
        tranche_rows = 0
        for receipt_path in receipts:
            file_id = receipt_path.name.removesuffix('.receipt.json')
            receipt = bound_json(receipt_path, bindings)
            receipt_hash = bindings[receipt_path]
            generalized = 'segment_id' in receipt
            sid = receipt.get('logical_source_id') if generalized else receipt.get('id')
            require(sid and receipt.get('status') == 'RAW_SEGMENT_READY', 'source is not complete: ' + file_id)
            require((receipt.get('segment_id') if generalized else receipt.get('id')) == file_id,
                    'receipt/file identity mismatch: ' + file_id)
            source = root / (file_id + '.jsonl.gz')
            bindings[source] = sha(source)
            require(bindings[source] == receipt['output_sha256'], 'raw checksum mismatch: ' + file_id)
            require(source.stat().st_size == receipt['output_bytes'], 'raw size mismatch: ' + file_id)
            identity = (receipt['repo_id'], receipt['revision'], receipt['path'])
            groups = {}
            for g in receipt['groups']:
                number = g['row_group']
                require(type(number) is int and number >= 0, 'invalid physical group')
                key = (*identity, number)
                require(key not in groups_seen, 'repeated physical group: ' + str(key))
                require(type(g['physical_row_offset']) is int and g['physical_row_offset'] >= 0 and
                        type(g['rows_read']) is int and g['rows_read'] > 0, 'invalid group bounds')
                groups_seen.add(key)
                groups[number] = g
            require(bool(groups), 'source has no physical groups')
            if 'prior_receipt_sha256' in receipt:
                prior = prior_by_source.get(sid)
                require(prior is not None, 'missing earlier tranche for ' + sid)
                require(prior[0] == receipt['prior_receipt_sha256'], 'prior receipt identity mismatch')
                require(prior[1] == identity, 'prior source revision/shard mismatch')
                excluded = receipt['exclude_row_groups']
                require(len(excluded) == len(set(excluded)) and set(excluded) == prior[2],
                        'prior group exclusions mismatch')
                require(not set(groups).intersection(excluded), 'prior groups were reused')
            if generalized:
                excluded = set(receipt.get('exclude_row_groups', []))
                historical = set()
                for ref in receipt.get('history', []):
                    path = Path(ref['path']).resolve(strict=True)
                    require(path in receipt_by_path, 'history receipt is not in an earlier input: ' + str(path))
                    previous_hash, previous_identity, previous_groups = receipt_by_path[path]
                    require(previous_hash == ref['sha256'], 'generalized history receipt hash mismatch')
                    require(previous_identity == identity, 'generalized history source identity mismatch')
                    historical.update(previous_groups)
                require(excluded == historical, 'generalized history group union mismatch')
                require(not set(groups).intersection(excluded), 'generalized history groups were reused')
            else:
                prior_by_source[sid] = (receipt_hash, identity, set(groups))
            receipt_by_path[receipt_path.resolve()] = (receipt_hash, identity, set(groups))
            require(type(receipt['rows_written']) is int and receipt['rows_written'] > 0,
                    'invalid source row count')
            tranche_rows += receipt['rows_written']
            files.append(dict(path=source, receipt_path=receipt_path, receipt=receipt, groups=groups,
                              source_id=sid, file_id=file_id, tranche=len(files), root_index=ordinal,
                              identity=identity))
        require(tranche_rows == summary['rows'], 'tranche row total mismatch')
        tranches.append(dict(index=ordinal, path=str(root), rows=tranche_rows,
                             sources=len(receipts), summary_sha256=bindings[summary_path]))
    return files, bindings, tranches


def audit(inputs, tokenizer_path, expected_tokenizer_sha256, plan_path, output):
    started = time.time()
    require(sha(tokenizer_path) == expected_tokenizer_sha256, 'tokenizer hash mismatch')
    files, bindings, tranches = preflight(inputs)
    bindings[tokenizer_path] = expected_tokenizer_sha256
    plan = bound_json(plan_path, bindings)
    expansion = plan.get('status') == 'FROZEN_UNIQUE_DATA_EXPANSION_PLAN_NOT_ADMITTED'
    if expansion:
        require(plan['source_repeat_cap'] == 2.0, 'unexpected source repeat cap')
        required = {x['source_id']: x['unique_prediction_tokens_required_at_two_epoch_cap'] for x in plan['sources']}
    else:  # retain the previously tested screen-quota format
        require(plan['sequence_length'] == 2048, 'unexpected packing length')
        required = {}
        for parents in plan['screen'].values():
            for parent in parents.values():
                for leaf, value in parent['leaves'].items():
                    required[leaf] = max(required.get(leaf, 0), (value['prediction_tokens'] + 1) // 2)
    for f in files:
        require((f['source_id'] if expansion else leaf_id(f['source_id'])) in required, 'source absent from expansion plan')
    output.mkdir(exist_ok=False)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    exact_seen, normalized_seen = {}, {}
    source_seen, source_families = defaultdict(set), defaultdict(set)
    source_counts, family_statuses, lengths = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    overlap, candidates = Counter(), Counter()
    file_counts = []
    # Deterministic ownership: source ID, then input argument order, then row.
    # Exact global duplicates count once; normalized nonidentical candidates stay.
    with (output / 'row-index.jsonl').open('x') as index:
        for f in sorted(files, key=lambda x: (x['source_id'], x['tranche'])):
            sid, tranche = f['source_id'], f['tranche']
            counts = source_counts[sid]
            physical_seen = set()
            nrows = 0
            with gzip.open(f['path'], 'rt', encoding='utf-8') as handle:
                for row_number, line in enumerate(handle):
                    row = json.loads(line)
                    text = row.get('text') or row.get('content')
                    require(isinstance(text, str) and bool(text.strip()), 'missing text')
                    p = row['_provenance']
                    require(p['source_id'] == sid and (p['repo_id'], p['revision'], p['shard']) == f['identity'],
                            'row provenance/source mismatch')
                    group = f['groups'].get(p['row_group'])
                    require(group is not None, 'row references an unrecorded group')
                    offset = p['row_in_group']
                    require(type(offset) is int and 0 <= offset < group['rows_read'], 'row outside group bounds')
                    require(p['physical_row'] == group['physical_row_offset'] + offset,
                            'physical row offset mismatch')
                    key = (p['row_group'], offset)
                    require(key not in physical_seen, 'repeated physical row')
                    physical_seen.add(key)
                    raw_hash = hashlib.sha256(text.encode()).hexdigest()
                    if sid.startswith('code_'):
                        require(row.get('content_sha256') == raw_hash, 'code text hash mismatch')
                    elif row.get('content_sha256'):
                        require(row['content_sha256'] == raw_hash, 'text hash mismatch')
                    normalized = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip().casefold()
                    candidate_hash = hashlib.sha256(normalized.encode()).hexdigest()
                    tokens = len(tokenizer.encode(text, add_special_tokens=False).ids) + 1
                    nrows += 1
                    counts['rows'] += 1
                    counts['encoded_tokens_including_one_eos'] += tokens
                    if raw_hash not in source_seen[sid]:
                        source_seen[sid].add(raw_hash)
                        counts['source_unique_encoded_tokens_including_one_eos'] += tokens
                    else:
                        counts['within_source_byte_identical_duplicates_across_all_tranches'] += 1
                    owner = exact_seen.get(raw_hash)
                    if owner is None:
                        exact_seen[raw_hash] = owner = dict(source_id=sid, tranche=tranche, row=row_number)
                        counts['globally_unique_documents_assigned'] += 1
                        counts['globally_unique_encoded_tokens_including_one_eos_assigned'] += tokens
                    else:
                        scope = 'cross_source' if owner['source_id'] != sid else (
                            'cross_tranche_same_source' if owner['tranche'] != tranche else 'within_file')
                        overlap[scope] += 1
                    previous = normalized_seen.setdefault(candidate_hash, raw_hash)
                    if previous != raw_hash:
                        candidates['normalized_nonidentical_rows'] += 1
                    family_key, status = family(row, sid)
                    family_statuses[sid][status] += 1
                    if family_key:
                        source_families[sid].add(family_key)
                    else:
                        counts['missing_family_rows'] += 1
                    bucket = ('lt128' if tokens < 128 else '128_to_2048' if tokens <= 2048 else
                              '2049_to_8192' if tokens <= 8192 else '8193_to_32768' if tokens <= 32768 else 'gt32768')
                    lengths[sid][bucket] += 1
                    index.write(json.dumps(dict(source_id=sid, tranche=tranche, row=row_number,
                                text_sha256=raw_hash, normalized_candidate_sha256=candidate_hash,
                                exact_dedup_owner=owner, encoded_tokens_including_one_eos=tokens,
                                family_candidate=family_key, family_status=status, provenance=p),
                                ensure_ascii=False) + '\n')
                    if nrows % 1000 == 0:
                        atomic_json(output / 'progress.json', dict(status='RUNNING', source=sid,
                                    tranche=tranche, source_file_rows=nrows, completed_files=len(file_counts),
                                    elapsed_seconds=time.time() - started))
            require(nrows == f['receipt']['rows_written'], 'source row count mismatch: ' + sid)
            file_counts.append(dict(source_id=sid, file_id=f['file_id'], tranche=tranche,
                                    root_index=f['root_index'], path=str(f['path']), rows=nrows,
                                    raw_file_sha256=bindings[f['path']]))
    # Check every bound artifact after the entire combined scan, including metadata.
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
        minimum = required[sid if expansion else leaf_id(sid)]
        summaries[sid] = dict(counts, byte_identical_unique_documents=len(source_seen[sid]),
                              provisional_family_candidates=len(source_families[sid]),
                              family_status=dict(family_statuses[sid]), token_length_histogram=dict(lengths[sid]),
                              packed_prediction_tokens_upper_bound_before_filtering=source_upper,
                              globally_deduplicated_assigned_prediction_tokens_upper_bound=assigned_upper,
                              min_unique_prediction_tokens_for_confirmation_at_two_epochs=minimum,
                              shortage_lower_bound_before_validation_and_quality_filters=max(0, minimum-assigned_upper))
    total_rows = sum(x['rows'] for x in tranches)
    require(total_rows == sum(x['rows'] for x in summaries.values()), 'combined row count mismatch')
    report = dict(status='COMBINED_RAW_INTAKE_MEASURED_NOT_ADMITTED', training_admitted=False,
                  checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  tokenizer_sha256=expected_tokenizer_sha256, expansion_plan_sha256=bindings[plan_path],
                  physical_group_disjointness='PASS', input_hashes_stable='PASS', tranches=tranches,
                  input_bindings={str(p): d for p, d in bindings.items()}, files=file_counts, sources=summaries,
                  rows=total_rows, globally_byte_identical_unique_documents=len(exact_seen),
                  byte_identical_overlap_rows=dict(overlap), normalized_nonidentical_candidates=dict(candidates),
                  row_index_sha256=sha(output / 'row-index.jsonl'), elapsed_seconds=time.time()-started,
                  limits=['Raw upper bounds include one EOS per document, before filtering and held-out reserves.',
                          'Exact global ownership is source ID then tranche input order then row; it is not quality ranking.',
                          'Families are provisional: registrable domains, repository aliases/forks and synthetic parents are unresolved.',
                          'No near-duplicate removal, old-FineWeb overlap, benchmark decontamination or quality admission is performed.',
                          'Normalized nonidentical candidates are retained; normalization may alter code and mathematics.',
                          'Two-epoch shortages are measured against the frozen 2.147B-token F2/F3 confirmation requirements.'])
    atomic_json(output / 'source-counts.json', summaries)
    atomic_json(output / 'report.json', report)
    atomic_json(output / 'progress.json', dict(status=report['status'], rows=total_rows,
                                             elapsed_seconds=report['elapsed_seconds']))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='append', type=Path, required=True, help='Completed tranches, oldest first; repeat this option.')
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--expected-tokenizer-sha256', required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.input, args.tokenizer, args.expected_tokenizer_sha256, args.plan, args.output)
    print(json.dumps({k: report[k] for k in ['status', 'rows', 'globally_byte_identical_unique_documents',
                                           'byte_identical_overlap_rows', 'elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
