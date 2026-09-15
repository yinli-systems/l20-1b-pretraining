"""Pack the newly reserved F2 development or confirmation partition."""

import argparse
from collections import defaultdict
import datetime
import gzip
import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time

import numpy as np


ROOT = Path('/ssd/scxi253/pretrain500m-20260912-v1')
FAMILY = ROOT / 'data/family-split-f2-continuation-v1/report.json'
PACK = ROOT / 'data/selected-packs-f2-continuation-v1/report.json'
CORE = ROOT / 'source/validation-pack-v2/pack.py'
FAMILY_SHA = 'd02fe6a46088f19b85aa85061eabb878c9252c76751ef4ccf75006bb002ddda0'
CORE_SHA = 'd42dc3f01a701cb6b3bd45e5c70196d2d4691d2c6f841cfbed8db3dbd6534109'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    target = Path(path)
    next_path = target.with_suffix(target.suffix + '.next')
    next_path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    next_path.replace(target)


def core_module():
    if sha(CORE) != CORE_SHA:
        raise ValueError('frozen document-masked packer changed')
    spec = importlib.util.spec_from_file_location('f2_bound_validation_core', CORE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(args):
    started = time.monotonic()
    if args.partition not in ('development', 'confirmation') or not 1 <= args.workers <= 4:
        raise ValueError('invalid partition/workers')
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('fresh output path required')
    if sha(FAMILY) != FAMILY_SHA or sha(PACK) != args.expected_pack_sha256:
        raise ValueError('family or selected pack identity changed')
    family = json.loads(FAMILY.read_text())
    parent = json.loads(PACK.read_text())
    if family['status'] != 'FAMILY_CLOSURE_AND_RESERVED_ASSIGNMENTS_COMPLETE_NOT_ADMITTED' or not family['independent_under_observed_family_edges']:
        raise ValueError('family partition independence not established')
    if parent['status'] != 'SELECTED_SOURCE_PACKS_COMPLETE_NOT_ADMITTED' or parent['family_report_sha256'] != FAMILY_SHA:
        raise ValueError('selected source pack incomplete or mismatched')
    if family['training_admitted'] or parent['training_admitted']:
        raise ValueError('upstream admission state changed')
    core = core_module()
    if set(core.DOMAINS) != {source['source_id'] for source in parent['sources']}:
        raise ValueError('source-domain map changed')

    family_partitions = {}
    source_counts = defaultdict(lambda: {'documents': 0, 'encoded_tokens': 0})
    by_domain = defaultdict(list)
    bound = {}
    for source in parent['sources']:
        sid = source['source_id']
        if source['status'] != 'SELECTED_SOURCE_PACKED_NOT_ADMITTED':
            raise ValueError('source is not packed: ' + sid)
        stream = source['outputs'][args.partition]
        stream_path = Path(stream['path'])
        index_path = Path(source['document_index'])
        if sha(stream_path) != stream['sha256'] or sha(index_path) != source['document_index_sha256']:
            raise ValueError('reserved stream/index identity mismatch: ' + sid)
        array = np.load(stream_path, mmap_mode='r', allow_pickle=False)
        try:
            if array.dtype != np.dtype(np.uint16) or array.ndim != 1 or len(array) != stream['encoded_tokens']:
                raise ValueError('reserved stream shape/dtype mismatch: ' + sid)
        finally:
            del array
        bound[str(stream_path)] = stream['sha256']
        bound[str(index_path)] = source['document_index_sha256']
        previous_end = 0
        with gzip.open(index_path, 'rt', encoding='utf-8') as handle:
            for line in handle:
                record = json.loads(line)
                partition = record['partition']
                family_id = record['family_id']
                if family_id in family_partitions and family_partitions[family_id] != partition:
                    raise ValueError('document family crosses train/development/confirmation')
                family_partitions[family_id] = partition
                if partition != args.partition:
                    continue
                if record['source_id'] != sid or record['start'] != previous_end or record['end'] <= record['start'] + 1 or record['eos_id'] != 50279:
                    raise ValueError('reserved document identity/accounting mismatch: ' + sid)
                previous_end = record['end']
                source_counts[sid]['documents'] += 1
                source_counts[sid]['encoded_tokens'] += record['end'] - record['start']
                by_domain[core.DOMAINS[sid]].append(dict(
                    source_id=sid, family_id=family_id, text_sha256=record['text_sha256'],
                    start=record['start'], end=record['end'], stream_path=str(stream_path),
                ))
        if previous_end != stream['encoded_tokens'] or source_counts[sid] != {
                'documents': stream['documents'], 'encoded_tokens': stream['encoded_tokens']}:
            raise ValueError('reserved source total mismatch: ' + sid)
    for sid, stats in family['retained_sources'][args.partition].items():
        if source_counts[sid] != {key: stats[key] for key in ('documents', 'encoded_tokens')}:
            raise ValueError('family reserved totals differ from pack: ' + sid)

    natural_blocks = {domain: sum((record['end'] - record['start'] - 1 + 2047) // 2048
                                  for record in by_domain[domain]) for domain in core.DOMAIN_ORDER}
    padding = core.padding_blocks(sum(natural_blocks.values()), 16)
    total_bytes = (sum(natural_blocks.values()) + padding) * ((2049 * 2) + 2048)
    if shutil.disk_usage(args.output.parent).free < 17 * 1024**3 + total_bytes:
        raise ValueError('masked pack disk headroom insufficient')
    args.output.mkdir(exist_ok=False)
    write(args.output / 'launch.json', dict(
        status='F2_DOCUMENT_MASKED_RESERVED_PACK_RUNNING_NOT_ADMITTED', partition=args.partition,
        family_report_sha256=FAMILY_SHA, parent_pack_report_sha256=args.expected_pack_sha256,
        core_sha256=CORE_SHA, wrapper_sha256=sha(Path(__file__)),
        workers=args.workers, training_admitted=False,
    ))
    work = [(domain, by_domain[domain], padding if domain == core.DOMAIN_ORDER[0] else 0,
             str(args.output)) for domain in core.DOMAIN_ORDER]
    with mp.get_context('fork').Pool(args.workers) as pool:
        results = list(pool.imap_unordered(core.pack_domain, work))
    results.sort(key=lambda item: core.DOMAIN_ORDER.index(item['domain']))
    for result in results:
        domain = result['domain']
        expected = family['reserved_partitions']['document_prediction_tokens'][args.partition][domain]
        expected_families = family['reserved_partitions']['family_counts'][args.partition][domain]
        if result['valid_target_tokens'] != expected or result['distinct_families'] != expected_families:
            raise ValueError('reserved domain accounting mismatch: ' + domain)
        if result['valid_target_tokens'] < 1048576 or result['distinct_families'] < 1000:
            raise ValueError('reserved minimum not met after masking: ' + domain)
        for path_key, hash_key in (('token_path', 'token_sha256'), ('mask_path', 'mask_sha256'),
                                   ('document_index', 'document_index_sha256')):
            if sha(Path(result[path_key])) != result[hash_key]:
                raise ValueError('masked output identity mismatch: ' + domain)
    for path, expected in bound.items():
        if sha(Path(path)) != expected:
            raise ValueError('bound reserved source changed during packing')
    if sha(FAMILY) != FAMILY_SHA or sha(PACK) != args.expected_pack_sha256:
        raise ValueError('family or parent report changed during packing')

    manifest = dict(
        schema='p529m-packed-mixture-v3', purpose='document-masked-five-domain-' + args.partition,
        sequence_length=2048, tokenizer_sha256=parent['tokenizer_sha256'],
        source_block_quotas={result['domain']: result['blocks'] for result in results},
        sources=[dict(id=result['domain'], revision=args.expected_pack_sha256,
                      prior_prediction_tokens=0, max_cumulative_epochs='1',
                      shards=[dict(path=Path(result['token_path']).name,
                                   sha256=result['token_sha256'], blocks=result['blocks'],
                                   loss_mask=dict(path=Path(result['mask_path']).name,
                                                  sha256=result['mask_sha256']))])
                 for result in results],
        valid_target_tokens_by_domain={result['domain']: result['valid_target_tokens'] for result in results},
        distinct_families_by_domain={result['domain']: result['distinct_families'] for result in results},
    )
    manifest_path = args.output / (args.partition + '-mixture.json')
    write(manifest_path, manifest)
    report = dict(
        status='F2_DOCUMENT_MASKED_RESERVED_PACK_COMPLETE_NOT_ADMITTED',
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic() - started, partition=args.partition,
        family_report_sha256=FAMILY_SHA, parent_pack_report_sha256=args.expected_pack_sha256,
        core_sha256=CORE_SHA, wrapper_sha256=sha(Path(__file__)),
        domains=results, manifest=str(manifest_path),
        manifest_sha256=sha(manifest_path),
        padded_prediction_tokens=sum(result['blocks'] * 2048 for result in results),
        valid_target_tokens=sum(result['valid_target_tokens'] for result in results),
        training_admitted=False,
        limitations=['First document tokens and padding carry no loss target.',
                     'Family separation is limited to observed joins and verified candidate edges.',
                     'This is a development/confirmation pool, not a sealed final capability test.'],
    )
    write(args.output / 'report.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'partition', 'padded_prediction_tokens',
                                                 'valid_target_tokens', 'manifest_sha256')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', choices=('development', 'confirmation'), required=True)
    parser.add_argument('--expected-pack-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if args.output.exists():
            write(args.output / 'failure.json', dict(status='FAILED_NOT_ADMITTED',
                                                    error_type=type(error).__name__, error=str(error),
                                                    training_admitted=False))
        raise
