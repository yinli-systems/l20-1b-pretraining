"""Build an isolated pilot corpus; no original data, weights, or cursor is modified.

Token-span dedup is deliberately labelled probabilistic, not a proof that all
near duplicates/paraphrases are removed. Raw source text stays on the GPU host.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
from pathlib import Path

import numpy as np

from continuation_common import (ROOT, RUN, BLOCK, GLOBAL, MIXES, PILOT_STEPS,
                                 atomic_json, lock, quotas, sha256)


def spans(ids, width=13):
    """Content-defined uint64 polynomial fingerprints, independent of offsets."""
    a = np.asarray(ids, dtype=np.uint64)
    n = len(a) - width + 1
    if n <= 0:
        return np.empty(0, dtype=np.uint64)
    h = np.zeros(n, dtype=np.uint64)
    for i in range(width):
        np.multiply(h, np.uint64(1_000_003), out=h)
        np.add(h, a[i:i+n] + np.uint64(1), out=h)
    # Avalanche before content-defined downsampling; this is not cryptographic.
    h ^= h >> np.uint64(30)
    h *= np.uint64(0xbf58476d1ce4e5b9)
    h ^= h >> np.uint64(27)
    h *= np.uint64(0x94d049bb133111eb)
    h ^= h >> np.uint64(31)
    return h


def hits(sorted_index, candidates):
    if len(sorted_index) == 0 or len(candidates) == 0:
        return 0
    indices = np.searchsorted(sorted_index, candidates)
    valid = indices < len(sorted_index)
    return int(np.count_nonzero(sorted_index[indices[valid]] == candidates[valid]))


def question_trie(questions):
    root = {}
    for words in questions:
        node = root
        for word in words:
            node = node.setdefault(word,{})
        node[None] = True
    return root


def contains_question(words, trie):
    """Exact normalized word matching without eight rolling Python hash loops."""
    for start,word in enumerate(words):
        node = trie.get(word)
        if node is None:
            continue
        if None in node:
            return True
        for next_word in words[start+1:start+12]:
            node = node.get(next_word)
            if node is None:
                break
            if None in node:
                return True
    return False


def atomic_npy(path, a):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.npy.tmp')
    with tmp.open('wb') as stream:
        np.save(stream, a, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def base_inputs(run):
    """Bind the baseline and create corpus sketches from immutable local shards."""
    destination = run / 'base-inputs.json'
    if destination.exists():
        result = json.loads(destination.read_text())
        for item in result['indices'].values():
            if sha256(Path(item['path'])) != item['sha256']:
                raise RuntimeError('corrupt frozen span index')
        return result
    run.mkdir(parents=True, exist_ok=True)
    result = {'started_unix': time.time(), 'indices': {}, 'old_manifests': {},
              'near_dedup_policy': {'old_train': '13-token hash, content-sampled 1/1024; reject >=2 hits and >=50% sampled-span overlap',
                                    'old_val': 'all 32-token hashes; reject any hit',
                                    'limitations': 'Not exhaustive near-dedup or paraphrase detection; 64-bit probabilistic fingerprints.'}}
    for split, width, modulus in [('val', 32, 1), ('train', 13, 1024)]:
        pieces = []
        for source in ('web', 'dclm', 'math', 'code'):
            manifest = ROOT / f'data/full-npy/{source}/manifest.json'
            result['old_manifests'][source] = sha256(manifest)
            m = json.loads(manifest.read_text())
            key = 'validation_shards' if split == 'val' else 'train_shards'
            for item in m[key]:
                p = Path(item['path'])
                cached = run / 'indices/parts' / f'{source}-{split}-{p.stem}.npy'
                marker = cached.with_suffix('.json')
                if cached.exists() and marker.exists():
                    record = json.loads(marker.read_text())
                    if record['input_sha256'] != item['sha256'] or sha256(cached) != record['sha256']:
                        raise RuntimeError('span index provenance mismatch')
                else:
                    if sha256(p) != item['sha256']:
                        raise RuntimeError(f'original shard mismatch: {p}')
                    a = np.load(p, mmap_mode='r', allow_pickle=False)
                    selected = []
                    for start in range(0, len(a), 1_000_000):
                        h = spans(a[start:min(len(a), start+1_000_000+width-1)], width)
                        selected.append(h[h % np.uint64(modulus) == 0])
                    atomic_npy(cached, np.unique(np.concatenate(selected)))
                    atomic_json(marker, {'input_sha256': item['sha256'], 'sha256': sha256(cached)})
                pieces.append(cached)
            print(json.dumps({'stage': 'baseline_span_index', 'source': source, 'split': split,
                              'parts': len(pieces), 'time': time.time()}), flush=True)
        a = np.unique(np.concatenate([np.load(p, allow_pickle=False) for p in pieces]))
        p = run / f'indices/old-{split}.npy'
        atomic_npy(p, a)
        result['indices'][split] = {'path': str(p), 'sha256': sha256(p), 'hashes': len(a)}
        del a
    result['original_dedup_sha256'] = sha256(ROOT / 'manifests/full-dedup.sqlite')
    result['tokenizer_sha256'] = sha256(ROOT / 'tokenizer/tokenizer.json')
    result['completed_unix'] = time.time()
    atomic_json(destination, result)
    return result


def benchmark_index(run):
    """Extend the old filter using actual scored docs; keep original filter intact."""
    destination = run / 'benchmark-filter.json'
    if destination.exists():
        record = json.loads(destination.read_text())
        if sha256(run / 'benchmark.sqlite') != record['sha256']:
            raise RuntimeError('benchmark filter mismatch')
        return
    original = sqlite3.connect(f'file:{ROOT}/manifests/decontam-13gram.sqlite?mode=ro', uri=True)
    db = sqlite3.connect(run / 'benchmark.sqlite')
    original.backup(db)
    original.close()
    db.execute('CREATE TABLE IF NOT EXISTS short_questions (n INTEGER, hash BLOB, PRIMARY KEY(n,hash))')
    db.execute('CREATE TABLE IF NOT EXISTS short_question_text (text TEXT PRIMARY KEY)')
    from build_decontam import strings, WORD_RE
    files = [ROOT / 'evaluations/final' / f'{s}.json' for s in ('zero_shot_core','mmlu_5shot','gsm8k_5shot')]
    boolq = sorted((ROOT / 'evaluations/comparison-20260911').rglob('*ours_boolq*.json'))
    if not boolq:
        boolq = [p for p in (ROOT / 'evaluations/comparison-20260911').rglob('*.json')
                 if 'ours_boolq' in str(p) and 'receipt' not in p.name]
    files += boolq
    provenance = []
    saw_boolq = False
    for p in files:
        value = json.loads(p.read_text())
        if 'samples' not in value:
            continue
        rows = 0
        pending = []
        short = []
        short_text = []
        for task, samples in value['samples'].items():
            saw_boolq |= task == 'boolq'
            for sample in samples:
                doc = sample['doc']
                for text in strings(doc):
                    w = WORD_RE.findall(text.lower())
                    pending.extend((hashlib.blake2b(' '.join(w[i:i+13]).encode(), digest_size=8).digest(),)
                                   for i in range(len(w)-12))
                for key in ('question','question_stem','ctx','sentence','goal'):
                    if isinstance(doc.get(key), str):
                        w = WORD_RE.findall(doc[key].lower())
                        if 5 <= len(w) < 13:
                            short.append((len(w), hashlib.blake2b(' '.join(w).encode(),digest_size=8).digest()))
                            short_text.append((' '.join(w),))
                rows += 1
                if len(pending) >= 50_000:
                    db.executemany('INSERT OR IGNORE INTO ngrams VALUES (?)', pending)
                    pending.clear()
        db.executemany('INSERT OR IGNORE INTO ngrams VALUES (?)', pending)
        db.executemany('INSERT OR IGNORE INTO short_questions VALUES (?,?)', short)
        db.executemany('INSERT OR IGNORE INTO short_question_text VALUES (?)', short_text)
        db.commit()
        provenance.append({'path': str(p), 'sha256': sha256(p), 'rows': rows})
        del value
    if not saw_boolq:
        raise RuntimeError('BoolQ raw samples not found; refuse incomplete decontamination')
    db.close()
    atomic_json(destination, {'sha256': sha256(run / 'benchmark.sqlite'), 'raw_inputs': provenance,
                             'short_question_min_words': 5, 'boolq_included': True,
                             'short_question_matcher':'normalized word trie v1',
                             'limitations': 'No guarantee against paraphrases or upstream contamination; no retroactive repair of 20B.'})


def source_rows(source, run):
    import source_docs as sd
    # New cursors only, never delete original raw cache or modify old progress.
    state = run / f'cursors/{source}.sqlite'
    state.parent.mkdir(parents=True, exist_ok=True)
    if not state.exists():
        old = sqlite3.connect(f'file:{ROOT}/manifests/source-files.sqlite?mode=ro', uri=True)
        new = sqlite3.connect(state)
        old.backup(new)
        old.close()
        new.close()
    if source in ('web','dclm','math','code'):
        return sd.iter_source(source, seed=42, state_db=state, delete_completed_raw=False)
    if source == 'web3':
        def rows():
            revision = '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
            for row in sd._parquet_rows('HuggingFaceFW/fineweb-edu','data/',revision,42,state,False):
                if row.get('language') != 'en' or row.get('language_score',0) < .9 or row.get('int_score',0) < 3:
                    continue
                text = sd._clean_text(row.get('text'))
                if text:
                    yield {'text': sd._redact_email_and_ipv4(text), 'id': str(row.get('id',row.get('url',''))),
                           'source': 'web3', 'int_score': row['int_score']}
        return rows()
    raise ValueError(f'source not enabled until source metadata audit: {source}')


def build_source(source, run, target, directory=None, quality_filter=None, rows=None, metadata_extra=None):
    from tokenizers import Tokenizer
    from pack_data import normalized_text, is_contaminated, ShardWriter, load_contamination_hashes
    from build_decontam import WORD_RE
    directory = directory or run / f'data/{source}'
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        if m['tokens'] < target:
            raise RuntimeError('pilot corpus is frozen; extension requires a new manifest')
        return
    directory.mkdir(parents=True, exist_ok=True)
    old = sqlite3.connect(f'file:{ROOT}/manifests/full-dedup.sqlite?mode=ro', uri=True)
    dedup = sqlite3.connect(run / 'fresh-dedup.sqlite')
    dedup.execute('CREATE TABLE IF NOT EXISTS documents (hash BLOB PRIMARY KEY, source TEXT, id TEXT, tokens INTEGER)')
    dedup.execute('CREATE TABLE IF NOT EXISTS spans (hash BLOB PRIMARY KEY)')
    known = load_contamination_hashes(run / 'benchmark.sqlite')
    benchmark = sqlite3.connect(f'file:{run}/benchmark.sqlite?mode=ro', uri=True)
    shorts = question_trie(row[0].split() for row in benchmark.execute('SELECT text FROM short_question_text'))
    benchmark.close()
    train_index = np.load(run / 'indices/old-train.npy',mmap_mode='r')
    val_index = np.load(run / 'indices/old-val.npy',mmap_mode='r')
    tokenizer = Tokenizer.from_file(str(ROOT / 'tokenizer/tokenizer.json'))
    writer = ShardWriter(directory, 'train', target, BLOCK * 8192, BLOCK)
    counters = Counter()
    quality_samples = []
    if writer.complete:
        raise RuntimeError('tokens without completion manifest require recovery audit')
    for row in source_rows(source,run) if rows is None else rows:
        counters['seen'] += 1
        text = row['text']
        digest = hashlib.sha256(normalized_text(text).encode()).digest()
        if old.execute('SELECT 1 FROM documents WHERE hash=?',(digest,)).fetchone() or dedup.execute(
                'SELECT 1 FROM documents WHERE hash=?',(digest,)).fetchone():
            counters['duplicate_document'] += 1
            continue
        # Keep the original split function reserved, even though old DB excludes known holdouts.
        if int.from_bytes(digest[:8],'big') % 1000 == 0:
            counters['reserved_holdout'] += 1
            continue
        contaminated = is_contaminated(text, known)
        if not contaminated and shorts:
            words = WORD_RE.findall(text.lower())
            contaminated = contains_question(words,shorts)
        if contaminated:
            counters['benchmark_match'] += 1
            continue
        ids = tokenizer.encode(text).ids
        if not 64 <= len(ids) <= 262_144:
            counters['length_reject'] += 1
            continue
        if quality_filter is not None:
            reason = quality_filter(source,row.get('id',''),text,len(ids))
            if reason:
                counters['quality_'+reason] += 1
                continue
        if hits(val_index, np.unique(spans(ids,32))):
            counters['old_val_span'] += 1
            continue
        sample = spans(ids)
        sample = np.unique(sample[sample % np.uint64(1024) == 0])
        n_old = hits(train_index,sample)
        n_new = sum(dedup.execute('SELECT 1 FROM spans WHERE hash=?',(h.tobytes(),)).fetchone() is not None
                    for h in sample)
        if max(n_old,n_new) >= 2 and max(n_old,n_new) >= .5 * len(sample):
            counters['near_copy_sampled'] += 1
            continue
        writer.append(ids + [0])
        dedup.execute('INSERT INTO documents VALUES (?,?,?,?)',(digest,source,row.get('id',''),len(ids)))
        dedup.executemany('INSERT OR IGNORE INTO spans VALUES (?)',[(h.tobytes(),) for h in sample])
        counters['accepted'] += 1
        if quality_filter is not None:
            from curate_continuation import retain_sample, sample_record
            retain_sample(quality_samples,sample_record(digest,row.get('id',''),len(ids),text),40)
        elif len(quality_samples) < 20:
            quality_samples.append({'id': row.get('id',''), 'sha256': digest.hex(), 'tokens': len(ids),
                                    'excerpt': text[:1000], 'int_score': row.get('int_score')})
        if counters['accepted'] % 500 == 0 or writer.complete:
            dedup.commit()
            progress = {'source':source, 'counters':dict(counters), 'tokens':writer.written+writer.buffered,
                        'target':writer.target_tokens, 'time':time.time()}
            atomic_json(directory/'progress.json',progress)
            atomic_json(directory/'quality-samples.local.json',quality_samples)
            print(json.dumps(progress),flush=True)
        if writer.complete:
            break
        if shutil.disk_usage(run).free < 42 * 2**30:
            raise RuntimeError('42 GiB reserve reached during data build; stop before checkpoint space is threatened')
    writer.flush()
    dedup.commit()
    dedup.close()
    old.close()
    if not writer.complete:
        raise RuntimeError(f'{source} exhausted: {writer.written}/{writer.target_tokens}')
    atomic_json(directory/'quality-samples.local.json',quality_samples)
    quality = {}
    if quality_filter is not None:
        import importlib
        policy = importlib.import_module(quality_filter.__module__)
        quality = {'quality_version':policy.VERSION,
                   'quality_policy_sha256':sha256(Path(policy.__file__))}
    atomic_json(manifest_path, {**(metadata_extra or {}),**quality,'source':source,'tokens':writer.written,'shards':writer.records,
                               'counters':dict(counters),'base_inputs_sha256':sha256(run/'base-inputs.json'),
                               'benchmark_sha256':sha256(run/'benchmark.sqlite'),
                               'tokenizer_sha256':sha256(ROOT/'tokenizer/tokenizer.json'),
                               'source_revisions': __import__('source_docs').REVISIONS,
                               'completed_unix':time.time()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('indices','benchmarks','pilot-a'),required=True)
    args = parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    handle = lock(RUN/'prepare.lock')
    base_inputs(RUN)
    if args.stage == 'indices':
        return
    benchmark_index(RUN)
    if args.stage == 'benchmarks':
        return
    q = quotas(MIXES['A'],PILOT_STEPS*GLOBAL)
    qb = quotas(MIXES['B'],PILOT_STEPS*GLOBAL)
    for source in ('math','web','dclm','code'):
        # Frozen pilot pool with 5% headroom; input block includes one extra target position.
        target = int(max(q[f'fresh_{source}'],qb.get(f'fresh_{source}',0))*1.05+1)*BLOCK
        build_source(source,RUN,target)
    atomic_json(RUN/'pilot-a-data-ready.json',{'stage':'data_built_pending_quality_review',
                'time':time.time(),'manifests':{s:sha256(RUN/f'data/{s}/manifest.json')
                for s in ('math','web','dclm','code')}})
    handle.close()


if __name__ == '__main__':
    main()
