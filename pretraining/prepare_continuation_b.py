#!/usr/bin/env python3
"""CPU-only, cached-source B preparation. Does not launch or approve training.

Preserves all A manifests and source cursors. B-only refill files share the
existing fresh-data exclusion DB under prepare.lock. No automatic download,
deletion, threshold relaxation, or recovery from a partial corpus is allowed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np

from continuation_common import ROOT, RUN, BLOCK, GLOBAL, MIXES, PILOT_STEPS, atomic_json, lock, owned, quotas, sha256
from prepare_continuation import spans, hits, contains_question, question_trie

BROOT = RUN/'b-candidate-v1'
# A and the parent already exist. Reserve B + its atomic replacement + 4 GiB.
MIN_FREE_BYTES = 2*13_200_800_000 + 4*2**30
BASES = {'web':RUN/'focused-web-score3-v4/manifest.json',
         'dclm':RUN/'refined-v3/dclm/manifest.json',
         'math':RUN/'refined-v3/math/manifest.json',
         'code':RUN/'curated-v2/code/manifest.json'}


def require_space(path):
    free = shutil.disk_usage(path).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f'B atomic-checkpoint reserve reached: {free} < {MIN_FREE_BYTES}')


def cached_rows(source, source_counts, inputs):
    """Read only full cached files whose hash matches pinned Hub metadata."""
    import pyarrow.parquet as pq
    from source_docs import REVISIONS, _clean_text, _redact_email_and_ipv4
    if source == 'narrative':
        from audit_continuation_narrative import REPO, REVISION, FIRST_FILE
        from continuation_narrative_quality import select_row
        path = ROOT/'data/raw-cache'/REPO.replace('/','--')/REVISION/FIRST_FILE['rfilename']
        if path.stat().st_size != FIRST_FILE['size'] or sha256(path) != FIRST_FILE['lfs']['sha256']:
            raise RuntimeError('cached narrative input differs from pinned LFS object')
        inputs.append({'path':str(path),'repo':REPO,'revision':REVISION,'sha256':FIRST_FILE['lfs']['sha256']})
        with gzip.open(path,'rt') as stream:
            for line in stream:
                source_counts['raw_seen'] += 1
                row,reason = select_row(json.loads(line))
                if reason:
                    source_counts['select_'+reason] += 1
                    continue
                yield row
        return
    if source == 'math':
        repo,rev,prefix = 'HuggingFaceTB/finemath',REVISIONS['finemath'],'finemath-4plus/'
    elif source == 'dclm':
        repo,rev,prefix = 'mlfoundations/dclm-baseline-1.0-parquet',REVISIONS['dclm'],'filtered/'
    else:
        raise ValueError(source)
    metadata_path = ROOT/'manifests/hub-filelists'/repo.replace('/','--')/f'{rev}.json'
    meta = json.loads(metadata_path.read_text())
    if meta['sha'] != rev:
        raise RuntimeError('source metadata revision mismatch')
    root = ROOT/'data/raw-cache'/repo.replace('/','--')/rev
    seen_files = 0
    for item in sorted(meta['siblings'],key=lambda x:x['rfilename']):
        name = item['rfilename']
        path = root/name
        if not name.startswith(prefix) or not name.endswith('.parquet') or not path.is_file():
            continue
        digest = item.get('lfs',{}).get('sha256')
        if not digest or path.stat().st_size != item['size'] or sha256(path) != digest:
            raise RuntimeError(f'cached source hash/size mismatch: {path}')
        inputs.append({'path':str(path),'repo':repo,'revision':rev,'sha256':digest})
        seen_files += 1
        parquet = pq.ParquetFile(path,memory_map=True)
        for batch in parquet.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                source_counts['raw_seen'] += 1
                if row.get('language','en')!='en' or row.get('language_score',1.) < .9:
                    source_counts['language_reject'] += 1
                    continue
                if source=='math' and row.get('int_score',4)<4:
                    source_counts['score_reject'] += 1
                    continue
                text = _clean_text(row.get('text'))
                if text:
                    yield {'text':_redact_email_and_ipv4(text) if source=='dclm' else text,
                           'id':str(row.get('url','')) if source=='math' else str(row.get('id',row.get('url',''))),
                           'source':source}
    if not seen_files:
        raise RuntimeError(f'no cached {source} file; downloading requires a separate capacity plan')


def build(source, target, quality_filter):
    from tokenizers import Tokenizer
    from pack_data import normalized_text, is_contaminated, ShardWriter, load_contamination_hashes
    from build_decontam import WORD_RE
    from curate_continuation import retain_sample, sample_record
    import importlib
    policy = importlib.import_module(quality_filter.__module__)
    directory = owned(BROOT/f'new-{source}')
    manifest = directory/'manifest.json'
    binding = {'source':source,'quality_version':policy.VERSION,
               'quality_policy_sha256':sha256(Path(policy.__file__)),
               'builder_sha256':sha256(Path(__file__)),
               'base_inputs_sha256':sha256(RUN/'base-inputs.json'),
               'benchmark_sha256':sha256(RUN/'benchmark.sqlite'),
               'tokenizer_sha256':sha256(ROOT/'tokenizer/tokenizer.json')}
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if any(previous[k]!=v for k,v in binding.items()) or previous['tokens'] < target:
            raise RuntimeError('existing B corpus changed or needs a new manifest')
        for shard in previous['shards']:
            if sha256(Path(shard['path']))!=shard['sha256']:
                raise RuntimeError('existing B shard changed')
        return manifest
    if directory.exists():
        raise RuntimeError('partial B corpus exists; retain and audit before retry')
    require_space(RUN)
    directory.mkdir(parents=True)
    old = sqlite3.connect(f'file:{ROOT}/manifests/full-dedup.sqlite?mode=ro',uri=True)
    dedup = sqlite3.connect(RUN/'fresh-dedup.sqlite')
    inventory = sqlite3.connect(directory/'documents.sqlite')
    inventory.execute('CREATE TABLE documents(hash BLOB PRIMARY KEY,id TEXT,tokens INTEGER,reason TEXT,metadata TEXT)')
    known = load_contamination_hashes(RUN/'benchmark.sqlite')
    benchmark = sqlite3.connect(f'file:{RUN}/benchmark.sqlite?mode=ro',uri=True)
    shorts = question_trie(row[0].split() for row in benchmark.execute('SELECT text FROM short_question_text'))
    benchmark.close()
    train = np.load(RUN/'indices/old-train.npy',mmap_mode='r')
    val = np.load(RUN/'indices/old-val.npy',mmap_mode='r')
    tokenizer = Tokenizer.from_file(str(ROOT/'tokenizer/tokenizer.json'))
    writer = ShardWriter(directory,'train',target,BLOCK*8192,BLOCK)
    counts,source_counts,inputs,samples = Counter(),Counter(),[],[]
    rows = cached_rows(source,source_counts,inputs)
    def progress():
        dedup.commit()
        inventory.commit()
        atomic_json(directory/'progress.json',{'stage':'building','source':source,'counters':dict(counts),
                    'source_counters':dict(source_counts),'tokens':writer.written+writer.buffered,
                    'target':writer.target_tokens,'time':time.time()})
        print(json.dumps({'source':source,'accepted':counts['accepted'],'tokens':writer.written+writer.buffered,
                          'target':writer.target_tokens,'time':time.time()}),flush=True)
    try:
        for row in rows:
            counts['seen'] += 1
            if counts['seen']%1000 == 0:
                progress()
            require_space(RUN)
            text = row['text']
            digest = hashlib.sha256(normalized_text(text).encode()).digest()
            if old.execute('SELECT 1 FROM documents WHERE hash=?',(digest,)).fetchone() or dedup.execute(
                    'SELECT 1 FROM documents WHERE hash=?',(digest,)).fetchone():
                counts['duplicate_document'] += 1
                continue
            if int.from_bytes(digest[:8],'big')%1000 == 0:
                counts['reserved_holdout'] += 1
                continue
            # Whole source book/document is checked BEFORE packing or truncation.
            if is_contaminated(text,known) or contains_question(WORD_RE.findall(text.lower()),shorts):
                counts['benchmark_match'] += 1
                continue
            ids = tokenizer.encode(text).ids
            if not 64 <= len(ids) <= 262144:
                counts['length_reject'] += 1
                continue
            reason = quality_filter(source,row.get('id',''),text,len(ids))
            if reason:
                counts['quality_'+reason] += 1
                continue
            if hits(val,np.unique(spans(ids,32))):
                counts['old_val_span'] += 1
                continue
            fingerprints = spans(ids)
            fingerprints = np.unique(fingerprints[fingerprints%np.uint64(1024)==0])
            n_old = hits(train,fingerprints)
            n_new = sum(dedup.execute('SELECT 1 FROM spans WHERE hash=?',(x.tobytes(),)).fetchone() is not None for x in fingerprints)
            if max(n_old,n_new)>=2 and max(n_old,n_new)>=.5*len(fingerprints):
                counts['near_copy_sampled'] += 1
                continue
            writer.append(ids+[0])
            identifier = row.get('id','')
            dedup.execute('INSERT INTO documents VALUES(?,?,?,?)',(digest,source,identifier,len(ids)))
            dedup.executemany('INSERT OR IGNORE INTO spans VALUES(?)',[(x.tobytes(),) for x in fingerprints])
            inventory.execute('INSERT INTO documents VALUES(?,?,?,?,?)',
                              (digest,identifier,len(ids),None,json.dumps(row.get('metadata',{}))))
            sample = sample_record(digest,identifier,len(ids),text)
            sample['metadata'] = row.get('metadata',{})
            retain_sample(samples,sample,40)
            counts['accepted'] += 1
            counts['accepted_document_input_tokens'] += len(ids)+1
            if counts['accepted']%100==0 or writer.complete:
                progress()
                atomic_json(directory/'quality-samples.local.json',samples)
            if writer.complete:
                break
        writer.flush()
        progress()
        atomic_json(directory/'quality-samples.local.json',samples)
        if not writer.complete:
            raise RuntimeError(f'cached {source} exhausted: {writer.written}/{writer.target_tokens}; no threshold relaxation')
    finally:
        rows.close()
        inventory.close()
        dedup.close()
        old.close()
    from source_docs import REVISIONS
    atomic_json(manifest,{**binding,'tokens':writer.written,'shards':writer.records,
                'source_revisions':REVISIONS,'raw_inputs':inputs,'inventory_sha256':sha256(directory/'documents.sqlite'),
                'counters':dict(counts),'source_counters':dict(source_counts),
                'trailing_accepted_tokens_dropped_for_alignment':counts['accepted_document_input_tokens']-writer.written,
                'stage':'built_pending_quality_review','training_enabled':False,'completed_unix':time.time()})
    return manifest


def main():
    import importlib
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',nargs='+',choices=('math','dclm','narrative','web','code'),
                        default=['math','dclm','narrative','web','code'])
    args = parser.parse_args()
    if len(set(args.sources))!=len(args.sources):
        parser.error('source list must not contain duplicates')
    policies = {'math':'continuation_math_quality','dclm':'continuation_general_quality',
                'narrative':'continuation_narrative_quality'}
    handle = lock(RUN/'prepare.lock')
    BROOT.mkdir(parents=True,exist_ok=True)
    q = quotas(MIXES['B'],GLOBAL*PILOT_STEPS)
    for source in args.sources:
        required = q['fresh_'+source]*BLOCK
        parts = [BASES[source]] if source in BASES else []
        available = sum(json.loads(p.read_text())['tokens'] for p in parts)
        if available<required:
            policy = importlib.import_module(policies[source])
            target = math.ceil(((required-available)*1.10+BLOCK)/BLOCK)*BLOCK
            parts.append(build(source,target,policy.reject_reason))
        manifests = [json.loads(p.read_text()) for p in parts]
        for key in ('base_inputs_sha256','benchmark_sha256','tokenizer_sha256'):
            if len({m[key] for m in manifests})!=1:
                raise RuntimeError('B composition input identities differ')
        shards = [item for m in manifests for item in m['shards']]
        for item in shards:
            if sha256(Path(item['path']))!=item['sha256'] or item['tokens']%BLOCK:
                raise RuntimeError('bad B component shard')
        result = {k:manifests[0][k] for k in ('source','base_inputs_sha256','benchmark_sha256','tokenizer_sha256')}
        result.update(shards=shards,tokens=sum(x['tokens'] for x in shards),required_input_tokens=required,
                      input_manifests=[{'path':str(p),'sha256':sha256(p)} for p in parts],
                      stage='composed_pending_quality_review',training_enabled=False)
        destination = BROOT/source/'manifest.json'
        if destination.exists():
            if json.loads(destination.read_text())!=result:
                raise RuntimeError('B composition is frozen')
        else:
            atomic_json(destination,result)
        print(json.dumps({'stage':'B_component_ready','source':source,'tokens':result['tokens'],'required':required}),flush=True)
    ready = {s:sha256(BROOT/s/'manifest.json') for s in ('math','dclm','narrative','web','code')
             if (BROOT/s/'manifest.json').exists()}
    atomic_json(BROOT/'status.json',{'stage':'all_pools_built_pending_quality_review' if len(ready)==5 else 'partial_pools_built',
                                   'training_enabled':False,'manifests':ready,'time':time.time()})
    handle.close()


if __name__=='__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        atomic_json(BROOT/'failure.json',{'type':type(error).__name__,'message':str(error),'time':time.time(),
                                        'automatic_restart':False,'training_enabled':False})
        raise
