"""Pack quality-filtered family-disjoint confirmation data with loss masks."""
from collections import defaultdict
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


DOMAINS = {
    'dclm': 'general_web',
    'pdf_en': 'knowledge_reading',
    'finemath4': 'math', 'infiwebmath4': 'math',
    'code_cpp': 'code', 'code_java': 'code', 'code_javascript': 'code',
    'code_python': 'code', 'code_typescript': 'code',
    'multilingual_arb_Arab': 'multilingual', 'multilingual_cmn_Hani': 'multilingual',
    'multilingual_deu_Latn': 'multilingual', 'multilingual_fra_Latn': 'multilingual',
    'multilingual_jpn_Jpan': 'multilingual', 'multilingual_spa_Latn': 'multilingual',
}
DOMAIN_ORDER = ['general_web','knowledge_reading','math','code','multilingual']
SEQUENCE_LENGTH = 2048
EOS = 50279


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


def document_blocks(tokens, sequence_length=SEQUENCE_LENGTH, pad=EOS):
    """Yield independent document blocks; the first document token is context only."""
    if tokens.dtype != np.uint16 or tokens.ndim != 1 or len(tokens) < 2:
        raise ValueError('reserved document must contain context and at least one target')
    for offset in range(0, len(tokens) - 1, sequence_length):
        valid = min(sequence_length, len(tokens) - 1 - offset)
        block = np.full(sequence_length + 1, pad, dtype=np.uint16)
        available = min(sequence_length + 1, len(tokens) - offset)
        block[:available] = tokens[offset:offset+available]
        mask = np.zeros(sequence_length, dtype=np.bool_); mask[:valid] = True
        yield block, mask


def padding_blocks(total, batch_blocks=16):
    if total < 0 or batch_blocks <= 0:
        raise ValueError('invalid block count')
    return (-total) % batch_blocks


def pack_domain(item):
    domain, records, padding_only_blocks, output = item; started = time.monotonic()
    document_block_count = sum((record['end'] - record['start'] - 1 + SEQUENCE_LENGTH - 1) // SEQUENCE_LENGTH for record in records)
    blocks = document_block_count + padding_only_blocks
    token_path = Path(output) / f'confirmation-{domain}.blocks.npy'
    mask_path = Path(output) / f'confirmation-{domain}.loss-mask.npy'
    index_path = Path(output) / f'confirmation-{domain}.documents.jsonl.gz'
    tokens_out = np.lib.format.open_memmap(token_path, mode='w+', dtype=np.uint16,
                                           shape=(blocks * (SEQUENCE_LENGTH + 1),))
    masks_out = np.lib.format.open_memmap(mask_path, mode='w+', dtype=np.bool_,
                                          shape=(blocks * SEQUENCE_LENGTH,))
    arrays = {}; tails = {}; cursor = 0; valid_targets = 0
    with gzip.open(index_path, 'xt', encoding='utf-8', compresslevel=1) as index:
        for record in records:
            path = record['stream_path']
            if path not in arrays:
                array = np.load(path, mmap_mode='r', allow_pickle=False)
                if array.dtype != np.uint16 or array.ndim != 1:
                    raise ValueError('invalid reserved stream')
                arrays[path] = array
            array = arrays[path]
            tail_path = record.get('tail_path')
            if tail_path:
                if tail_path not in tails:
                    tails[tail_path] = np.asarray(json.loads(Path(tail_path).read_text())['token_ids'], dtype=np.uint16)
                tail = tails[tail_path]
            else:
                tail = np.empty(0, dtype=np.uint16)
            start, end = record['start'], record['end']
            if not 0 <= start <= end <= len(array) + len(tail):
                raise ValueError('reserved document span outside bound stream')
            pieces = []
            if start < len(array):
                pieces.append(np.asarray(array[start:min(end, len(array))], dtype=np.uint16))
            if end > len(array):
                pieces.append(np.asarray(tail[max(0, start-len(array)):end-len(array)], dtype=np.uint16))
            doc = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
            first = cursor
            for block, mask in document_blocks(doc):
                ts = cursor * (SEQUENCE_LENGTH + 1); ms = cursor * SEQUENCE_LENGTH
                tokens_out[ts:ts+SEQUENCE_LENGTH+1] = block
                masks_out[ms:ms+SEQUENCE_LENGTH] = mask
                valid_targets += int(mask.sum()); cursor += 1
            index.write(json.dumps({k: record[k] for k in ('source_id','family_id','text_sha256')} |
                                   {'block_start': first, 'block_end': cursor,
                                    'valid_target_tokens': len(doc)-1}) + '\n')
    for _ in range(padding_only_blocks):
        ts = cursor * (SEQUENCE_LENGTH + 1); ms = cursor * SEQUENCE_LENGTH
        tokens_out[ts:ts+SEQUENCE_LENGTH+1] = EOS
        masks_out[ms:ms+SEQUENCE_LENGTH] = False
        cursor += 1
    if cursor != blocks:
        raise ValueError('validation block accounting mismatch')
    tokens_out.flush(); masks_out.flush()
    return {
        'domain': domain, 'documents': len(records),
        'distinct_families': len({r['family_id'] for r in records}),
        'blocks': blocks, 'document_blocks': document_block_count,
        'padding_only_blocks': padding_only_blocks, 'valid_target_tokens': valid_targets,
        'token_path': str(token_path), 'token_sha256': sha(token_path),
        'mask_path': str(mask_path), 'mask_sha256': sha(mask_path),
        'document_index': str(index_path), 'document_index_sha256': sha(index_path),
        'elapsed_seconds': time.monotonic() - started,
    }


def run(args):
    started = time.monotonic()
    if args.partition != 'confirmation':
        raise ValueError('v1 packer is frozen to the confirmation partition')
    if not 1 <= args.workers <= 4:
        raise ValueError('workers must be 1..4')
    if sha(args.parent_report) != args.expected_parent_sha256:
        raise ValueError('parent pack identity mismatch')
    if sha(args.split_patch) != args.expected_split_patch_sha256:
        raise ValueError('quality split patch identity mismatch')
    parent = json.loads(args.parent_report.read_text())
    split = json.loads(args.split_patch.read_text())
    if parent['status'] != 'SELECTED_SOURCE_PACKS_COMPLETE_NOT_ADMITTED':
        raise ValueError('parent pack incomplete')
    if split['status'] != 'RESERVED_QUALITY_REPLACEMENTS_COMPLETE_NOT_APPLIED':
        raise ValueError('quality split patch incomplete')
    rejected = set(split['reserved_exclude_family_id_reasons'])
    moves = split['move_train_family_to_partition']
    if set(DOMAINS) != {s['source_id'] for s in parent['sources']}:
        raise ValueError('leaf source set changed')
    expected_bytes = 0; by_domain = defaultdict(list); bound = {}
    for source in parent['sources']:
        sid = source['source_id']; reserved = source['outputs'][args.partition]; train = source['outputs']['train']
        if sha(reserved['path']) != reserved['sha256']:
            raise ValueError(f'reserved stream changed: {sid}')
        bound[reserved['path']] = reserved['sha256']
        for path_key, hash_key in (('path','sha256'),('tail_path','tail_sha256')):
            if sha(train[path_key]) != train[hash_key]:
                raise ValueError(f'train stream changed: {sid}/{path_key}')
            bound[train[path_key]] = train[hash_key]
        if sha(source['document_index']) != source['document_index_sha256']:
            raise ValueError(f'document index changed: {sid}')
        bound[source['document_index']] = source['document_index_sha256']
        original = []; moved = []; all_reserved = []; all_train = []
        with gzip.open(source['document_index'], 'rt', encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                if row['partition'] == args.partition:
                    all_reserved.append(row)
                    if row['family_id'] not in rejected:
                        row['stream_path'] = reserved['path']; original.append(row)
                elif row['partition'] == 'train':
                    all_train.append(row)
                    if moves.get(row['family_id']) == args.partition:
                        row['stream_path'] = train['path']; row['tail_path'] = train['tail_path']; moved.append(row)
        all_reserved.sort(key=lambda row: row['start']); all_train.sort(key=lambda row: row['start'])
        if len(all_reserved) != reserved['documents'] or not all_reserved or all_reserved[0]['start'] != 0 or all_reserved[-1]['end'] != reserved['encoded_tokens'] or any(a['end'] != b['start'] for a,b in zip(all_reserved,all_reserved[1:])):
            raise ValueError(f'reserved document accounting mismatch: {sid}')
        if len(all_train) != train['documents'] or not all_train or all_train[0]['start'] != 0 or all_train[-1]['end'] != train['encoded_tokens'] or any(a['end'] != b['start'] for a,b in zip(all_train,all_train[1:])):
            raise ValueError(f'train document accounting mismatch: {sid}')
        docs = original + moved
        by_domain[DOMAINS[sid]].extend(docs)
        expected_bytes += sum(((d['end']-d['start']-1+SEQUENCE_LENGTH-1)//SEQUENCE_LENGTH) *
                              ((SEQUENCE_LENGTH+1)*2+SEQUENCE_LENGTH) for d in docs)
    natural_blocks = {domain:sum((d['end']-d['start']-1+SEQUENCE_LENGTH-1)//SEQUENCE_LENGTH
                                 for d in by_domain[domain]) for domain in DOMAIN_ORDER}
    # The frozen four-GPU protocol uses four sequences per GPU. Padding-only
    # masked blocks make every evaluation batch complete without repeating data.
    batch_blocks = 16
    padding = padding_blocks(sum(natural_blocks.values()), batch_blocks)
    expected_bytes += padding * ((SEQUENCE_LENGTH+1)*2+SEQUENCE_LENGTH)
    if shutil.disk_usage(args.output.parent).free < 17 * 1024**3 + expected_bytes:
        raise ValueError('aggregate validation-pack disk headroom insufficient')
    args.output.mkdir(exist_ok=False)
    write(args.output/'launch.json', {'status':'PACKING_DOCUMENT_MASKED_CONFIRMATION_NOT_ADMITTED',
          'parent_pack_sha256':args.expected_parent_sha256,'partition':args.partition,
          'workers':args.workers,'training_admitted':False})
    with mp.get_context('fork').Pool(args.workers) as pool:
        results = list(pool.imap_unordered(pack_domain,
            [(domain, by_domain[domain], padding if domain==DOMAIN_ORDER[0] else 0, str(args.output))
             for domain in DOMAIN_ORDER]))
    results.sort(key=lambda r: DOMAIN_ORDER.index(r['domain']))
    for result in results:
        if result['distinct_families'] < 1000 or result['valid_target_tokens'] < 1048576:
            raise ValueError(f"reserved minimum not met after document masking: {result['domain']}")
    expected_stats = split['final_reserved_stats'][args.partition]
    for result in results:
        expected = expected_stats[result['domain']]
        if result['distinct_families'] != expected['families'] or result['valid_target_tokens'] != expected['prediction_tokens']:
            raise ValueError(f"quality split accounting mismatch: {result['domain']}")
    for path, expected in bound.items():
        if sha(path) != expected:
            raise ValueError('bound parent artifact changed during validation packing')
    manifest = {
        'schema': 'p529m-packed-mixture-v3', 'purpose': 'document-masked-five-domain-confirmation',
        'sequence_length': SEQUENCE_LENGTH, 'tokenizer_sha256': parent['tokenizer_sha256'],
        'source_block_quotas': {r['domain']: r['blocks'] for r in results},
        'sources': [{'id':r['domain'],'revision':args.expected_parent_sha256,
                     'prior_prediction_tokens':0,'max_cumulative_epochs':'1',
                     'shards':[{'path':Path(r['token_path']).name,'sha256':r['token_sha256'],'blocks':r['blocks'],
                                'loss_mask':{'path':Path(r['mask_path']).name,'sha256':r['mask_sha256']}}]}
                    for r in results],
        'valid_target_tokens_by_domain': {r['domain']:r['valid_target_tokens'] for r in results},
        'distinct_families_by_domain': {r['domain']:r['distinct_families'] for r in results},
    }
    manifest_path = args.output/'confirmation-mixture.json'; write(manifest_path, manifest)
    final = {
        'status':'DOCUMENT_MASKED_FIVE_DOMAIN_CONFIRMATION_COMPLETE_NOT_ADMITTED',
        'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'elapsed_seconds':time.monotonic()-started,'parent_pack_sha256':args.expected_parent_sha256,
        'split_patch_sha256':args.expected_split_patch_sha256,
        'partition':args.partition,'domains':results,'manifest':str(manifest_path),
        'manifest_sha256':sha(manifest_path),'padded_prediction_tokens':sum(r['blocks']*SEQUENCE_LENGTH for r in results),
        'padding_only_blocks':sum(r['padding_only_blocks'] for r in results),
        'valid_target_tokens':sum(r['valid_target_tokens'] for r in results),
        'training_admitted':False,
        'limitations':['Every document is packed independently; first document tokens and padding have no loss target.',
                       'Confirmation data supports selection only and is not a sealed final test.',
                       'Family separation is bounded by the parent observed-family audit.'],
    }
    write(args.output/'report.json', final)
    print(json.dumps({k:final[k] for k in ('status','elapsed_seconds','padded_prediction_tokens','valid_target_tokens','manifest_sha256')}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent-report',type=Path,required=True)
    parser.add_argument('--expected-parent-sha256',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split-patch',type=Path,required=True);parser.add_argument('--expected-split-patch-sha256',required=True)
    parser.add_argument('--partition',default='confirmation');parser.add_argument('--workers',type=int,default=4)
    run(parser.parse_args())
