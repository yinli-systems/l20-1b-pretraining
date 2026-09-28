"""Build revised educational web pilot from pinned, already cached raw files.

Original metadata URL is retained as provenance. No source cursor is changed;
if the cached files cannot fill the quota this stops rather than lowering quality.
"""
from collections import Counter
import argparse
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from continuation_common import ROOT, RUN, BLOCK, GLOBAL, MIXES, PILOT_STEPS, atomic_json, lock, quotas, sha256
from continuation_web_quality import host_allowed, reject_reason
from prepare_continuation import build_source
from source_docs import REVISIONS, _clean_text


def cached_rows(directory, minimum_score):
    root = ROOT/f'data/raw-cache/HuggingFaceTB--smollm-corpus/{REVISIONS["smollm"]}/fineweb-edu-dedup'
    paths = sorted(root.glob('*.parquet'))
    if not paths:
        raise RuntimeError('no cached pinned web shards')
    inputs = []
    counters = Counter()
    try:
        for path in paths:
            # The upstream downloader records expected LFS hashes; recheck them.
            verified = json.loads(path.with_suffix(path.suffix+'.verified.json').read_text())
            digest = sha256(path)
            if digest != verified.get('sha256'):
                raise RuntimeError('cached web raw hash mismatch')
            inputs.append({'path':str(path),'sha256':digest,'bytes':path.stat().st_size})
            parquet = pq.ParquetFile(path,memory_map=True)
            for batch in parquet.iter_batches(batch_size=1024):
                for row in batch.to_pylist():
                    counters['raw_seen'] += 1
                    meta = row.get('metadata') or {}
                    url = meta.get('url','')
                    if meta.get('language') != 'en' or meta.get('language_score',0) < .9 or meta.get('int_score',0) < minimum_score:
                        counters['upstream_threshold'] += 1
                        continue
                    if not host_allowed(url):
                        counters['provenance_not_allowlisted'] += 1
                        continue
                    text = _clean_text(row.get('text'))
                    if text:
                        counters['eligible_to_filter'] += 1
                        yield {'text':text,'id':url,'raw_id':row.get('id'),
                               'int_score':meta['int_score'],'source':'web'}
                if counters['raw_seen'] % 32768 == 0:
                    atomic_json(directory/'raw-progress.json',dict(counters))
            parquet.close()
    finally:
        atomic_json(directory/'raw-inputs.json',{'inputs':inputs,'counters':dict(counters),
                                               'revision':REVISIONS['smollm'],'minimum_int_score':minimum_score})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minimum-score',type=int,choices=(3,4),required=True)
    args = parser.parse_args()
    handle = lock(RUN/'prepare.lock')
    directory = RUN/f'focused-web-score{args.minimum_score}-v4'
    directory.mkdir(parents=True,exist_ok=True)
    q = quotas(MIXES['A'],PILOT_STEPS*GLOBAL)
    target = math.ceil(q['fresh_web']*1.05+1)*BLOCK
    rows = cached_rows(directory,args.minimum_score)
    try:
        build_source('web',RUN,target,directory=directory,quality_filter=reject_reason,rows=rows,
                     metadata_extra={'source_minimum_int_score':args.minimum_score,
                                     'curation_program_sha256':sha256(Path(__file__)),
                                     'base_quality_policy_sha256':sha256(Path(__file__).with_name('continuation_quality.py'))})
    finally:
        rows.close()
        handle.close()


if __name__ == '__main__':
    main()
