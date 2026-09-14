"""Remove review-rejected train families from a bound token pack.

The immutable parent token streams are copied by document span.  Raw text is
not decoded or tokenized again.  Reserved partitions remain parent references.
"""
import argparse
import datetime
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import numpy as np


PARENT_STATUS = 'SELECTED_SOURCE_PACKS_COMPLETE_NOT_ADMITTED'
DECISION_STATUS = 'QUALITY_REVIEW_EXCLUSIONS_BOUND_NOT_APPLIED'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.next')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def span(array, tail, start, end):
    if not 0 <= start <= end <= len(array) + len(tail):
        raise ValueError('document span is outside the bound parent stream')
    pieces = []
    if start < len(array):
        pieces.append(np.asarray(array[start:min(end, len(array))], dtype=np.uint16))
    if end > len(array):
        pieces.append(np.asarray(tail[max(0, start-len(array)):end-len(array)], dtype=np.uint16))
    return pieces


def filter_source(item):
    source, output, excluded = item
    sid = source['source_id']; train = source['outputs']['train']; started = time.monotonic()
    for key in ('path', 'tail_path'):
        expected = train['sha256'] if key == 'path' else train['tail_sha256']
        if sha(train[key]) != expected:
            raise ValueError(f'parent train artifact changed: {sid}/{key}')
    if sha(source['document_index']) != source['document_index_sha256']:
        raise ValueError(f'parent document index changed: {sid}')
    array = np.load(train['path'], mmap_mode='r', allow_pickle=False)
    if array.dtype != np.uint16 or array.ndim != 1 or array.size != train['array_tokens']:
        raise ValueError(f'invalid parent train array: {sid}')
    tail_doc = json.loads(Path(train['tail_path']).read_text()); tail = tail_doc['token_ids']
    if len(tail) != train['tail_tokens'] or len(array) + len(tail) != train['encoded_tokens']:
        raise ValueError(f'parent tail accounting mismatch: {sid}')
    documents = []
    with gzip.open(source['document_index'], 'rt', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row['source_id'] == sid and row['partition'] == 'train':
                documents.append(row)
    if len(documents) != train['documents']:
        raise ValueError(f'parent train document count mismatch: {sid}')
    documents.sort(key=lambda row: row['start'])
    if any(a['end'] != b['start'] for a, b in zip(documents, documents[1:])) or documents[0]['start'] != 0 or documents[-1]['end'] != train['encoded_tokens']:
        raise ValueError(f'parent document spans are not contiguous: {sid}')
    kept = [row for row in documents if row['family_id'] not in excluded]
    removed = [row for row in documents if row['family_id'] in excluded]
    encoded = sum(row['end'] - row['start'] for row in kept); array_tokens = encoded // 2049 * 2049
    array_path = Path(output) / f'train-{sid}.blocks.npy'
    target = np.lib.format.open_memmap(array_path, mode='w+', dtype=np.uint16, shape=(array_tokens,))
    index_path = Path(output) / f'{sid}.train-documents.jsonl.gz'; cursor = written = 0; new_tail = []
    with gzip.open(index_path, 'xt', encoding='utf-8', compresslevel=1) as index:
        for row in kept:
            start = cursor
            for piece in span(array, tail, row['start'], row['end']):
                n = min(len(piece), max(0, array_tokens - written))
                if n:
                    target[written:written+n] = piece[:n]; written += n
                if n < len(piece):
                    new_tail.extend(int(v) for v in piece[n:])
                cursor += len(piece)
            updated = dict(row); updated['start'] = start; updated['end'] = cursor
            updated['first_document_target_offset'] = start + 1
            index.write(json.dumps(updated) + '\n')
    if written != array_tokens or cursor != encoded or len(new_tail) != encoded - array_tokens or len(new_tail) >= 2049:
        raise ValueError(f'filtered output accounting mismatch: {sid}')
    target.flush()
    with array_path.open('rb') as handle:
        import os
        os.fsync(handle.fileno())
    tail_path = Path(output) / f'{sid}.train-tail.json'
    write(tail_path, {'token_ids': new_tail, 'virtual_offset': array_tokens,
                      'training_admitted': False, 'parent_pack_sha256': train['sha256']})
    result = {
        'status': 'REVIEW_FILTERED_SOURCE_PACKED_NOT_ADMITTED', 'source_id': sid,
        'elapsed_seconds': time.monotonic() - started,
        'excluded_documents': len(removed),
        'excluded_families': len({row['family_id'] for row in removed}),
        'excluded_family_ids': sorted({row['family_id'] for row in removed}),
        'excluded_encoded_tokens': sum(row['end'] - row['start'] for row in removed),
        'output': {'path': str(array_path), 'sha256': sha(array_path), 'documents': len(kept),
                   'encoded_tokens': encoded, 'array_tokens': array_tokens,
                   'tail_tokens': len(new_tail), 'dtype': 'uint16', 'blocks': array_tokens // 2049,
                   'prediction_tokens': array_tokens // 2049 * 2048,
                   'tail_path': str(tail_path), 'tail_sha256': sha(tail_path)},
        'train_document_index': str(index_path), 'train_document_index_sha256': sha(index_path),
        'parent_reserved_outputs': {k: v for k, v in source['outputs'].items() if k != 'train'},
    }
    write(Path(output) / f'{sid}.report.json', result)
    return result


def run(args):
    started = time.monotonic()
    if not 1 <= args.workers <= 4:
        raise ValueError('workers must be 1..4')
    if sha(args.parent_report) != args.expected_parent_sha256 or sha(args.decisions) != args.expected_decisions_sha256:
        raise ValueError('input report identity mismatch')
    parent = json.loads(args.parent_report.read_text()); decisions = json.loads(args.decisions.read_text())
    if parent['status'] != PARENT_STATUS or decisions['status'] != DECISION_STATUS:
        raise ValueError('input stage incomplete')
    excluded = decisions['exclude_family_id_reasons']
    if not excluded or decisions['excluded_families'] != len(excluded):
        raise ValueError('empty or inconsistent family exclusion set')
    expected_bytes = sum(source['outputs']['train']['encoded_tokens'] * 2 for source in parent['sources'])
    if shutil.disk_usage(args.output.parent).free < 18 * 1024**3 + expected_bytes:
        raise ValueError('aggregate filtered-pack disk headroom insufficient')
    args.output.mkdir(exist_ok=False)
    write(args.output / 'launch.json', {'status': 'FILTERING_BOUND_TRAIN_PACK_NOT_ADMITTED',
          'parent_pack_sha256': args.expected_parent_sha256, 'decisions_sha256': args.expected_decisions_sha256,
          'excluded_families': len(excluded), 'workers': args.workers, 'training_admitted': False})
    with mp.get_context('fork').Pool(args.workers) as pool:
        results = list(pool.imap_unordered(filter_source,
            [(source, str(args.output), excluded) for source in parent['sources']]))
    matched = set()
    for result in results:
        with gzip.open(result['train_document_index'], 'rt', encoding='utf-8') as handle:
            retained = {json.loads(line)['family_id'] for line in handle}
        if retained & set(excluded):
            raise ValueError('excluded family remains in filtered pack')
        matched.update(result['excluded_family_ids'])
    if matched != set(excluded):
        raise ValueError('not every excluded family matched the selected train pack')
    if sha(args.parent_report) != args.expected_parent_sha256 or sha(args.decisions) != args.expected_decisions_sha256:
        raise ValueError('input report changed during filtering')
    final = {
        'status': 'QUALITY_REVIEW_FILTERED_PACKS_COMPLETE_NOT_ADMITTED',
        'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'elapsed_seconds': time.monotonic() - started, 'workers': args.workers,
        'parent_pack_sha256': args.expected_parent_sha256,
        'decisions_sha256': args.expected_decisions_sha256,
        'excluded_families': len(excluded),
        'excluded_documents': sum(r['excluded_documents'] for r in results),
        'excluded_encoded_tokens': sum(r['excluded_encoded_tokens'] for r in results),
        'retained_documents': sum(r['output']['documents'] for r in results),
        'retained_prediction_tokens': sum(r['output']['prediction_tokens'] for r in results),
        'sources': sorted(results, key=lambda r: r['source_id']),
        'tokenizer_sha256': parent['tokenizer_sha256'], 'training_admitted': False,
        'limitations': ['Only train families selected by the bound review decision are removed.',
                        'Reserved development and confirmation streams remain immutable parent references.',
                        'This pack does not itself grant corpus admission.'],
    }
    write(args.output / 'report.json', final)
    print(json.dumps({k: final[k] for k in ('status','elapsed_seconds','excluded_documents','retained_documents','retained_prediction_tokens')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-report', type=Path, required=True)
    parser.add_argument('--expected-parent-sha256', required=True)
    parser.add_argument('--decisions', type=Path, required=True)
    parser.add_argument('--expected-decisions-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    run(parser.parse_args())
