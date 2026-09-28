"""Create a new reviewed-candidate corpus from immutable v1 token pools.

Recover document boundaries from the insertion-ordered provenance DB, verify
every decoded normalized SHA-256, and retain token IDs without re-tokenizing.
The last partially packed document is excluded. V1 files are never overwritten.
"""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from continuation_common import ROOT, RUN, BLOCK, atomic_json, lock, sha256
from continuation_quality import VERSION, reject_reason


class TokenStream:
    def __init__(self, arrays):
        self.arrays = arrays
        self.shard = self.offset = 0
        self.remaining = sum(len(a) for a in arrays)

    def take(self, n):
        if n < 0 or n > self.remaining:
            raise ValueError('document extends beyond recorded token pool')
        pieces = []
        remaining = n
        while remaining:
            a = self.arrays[self.shard]
            count = min(remaining, len(a)-self.offset)
            pieces.append(a[self.offset:self.offset+count])
            self.offset += count
            remaining -= count
            if self.offset == len(a):
                self.shard += 1
                self.offset = 0
        self.remaining -= n
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.uint16)


def verified_documents(stream, rows, decode, normalize, expected_tail=None):
    """Yield intact documents only; a partial final document never passes."""
    partial = False
    discarded = 0
    for digest, identifier, n in rows:
        if partial:
            if expected_tail is None:
                raise RuntimeError('provenance has documents after truncated final document')
            discarded += n+1
            continue
        if n+1 > stream.remaining:
            partial = True
            discarded += n+1-stream.remaining
            stream.take(stream.remaining)
            continue
        a = stream.take(n+1)
        if int(a[-1]) != 0:
            raise RuntimeError('document EOS boundary mismatch')
        text = decode(a[:-1].tolist())
        if hashlib.sha256(normalize(text).encode()).digest() != digest:
            raise RuntimeError('decoded document SHA-256 mismatch')
        yield digest, identifier, n, text, a
    if stream.remaining:
        raise RuntimeError('token pool has no matching document provenance')
    if expected_tail is not None and discarded != expected_tail:
        raise RuntimeError('recorded alignment tail does not match document inventory')


def sample_record(digest, identifier, n, text, reason=None):
    middle = max(0,len(text)//2-250)
    return {'sha256':digest.hex(), 'id':identifier, 'tokens':n, 'reason':reason,
            'head':text[:750], 'middle':text[middle:middle+750], 'tail':text[-350:]}


def retain_sample(samples, item, size):
    # Hash priority is deterministic and independent of document/source order.
    samples.append(item)
    samples.sort(key=lambda x:x['sha256'])
    del samples[size:]


def curate(source, input_path=None, directory=None, input_inventory=None, quality_filter=reject_reason):
    from tokenizers import Tokenizer
    from pack_data import normalized_text, ShardWriter
    input_path = input_path or RUN/f'data/{source}/manifest.json'
    m = json.loads(input_path.read_text())
    directory = directory or RUN/f'curated-v2/{source}'
    receipt = directory/'manifest.json'
    import importlib
    policy = importlib.import_module(quality_filter.__module__)
    policy_sha = sha256(Path(policy.__file__))
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old['input_manifest_sha256'] != sha256(input_path) or old['quality_policy_sha256'] != policy_sha:
            raise RuntimeError('completed curation binding changed')
        return
    directory.mkdir(parents=True, exist_ok=True)
    if list(directory.glob('train-*.npy')):
        raise RuntimeError('partial curation exists; audit before recovery')
    arrays = []
    for record in m['shards']:
        if sha256(Path(record['path'])) != record['sha256']:
            raise RuntimeError('input token pool changed')
        a = np.load(record['path'],mmap_mode='r',allow_pickle=False)
        if a.ndim != 1 or a.dtype != np.uint16 or len(a) != record['tokens']:
            raise RuntimeError('bad input token shard')
        arrays.append(a)
    if sha256(ROOT/'tokenizer/tokenizer.json') != m['tokenizer_sha256']:
        raise RuntimeError('tokenizer changed')
    tokenizer = Tokenizer.from_file(str(ROOT/'tokenizer/tokenizer.json'))
    decode = lambda ids:tokenizer.decode(ids,skip_special_tokens=False)
    if input_inventory is None:
        db = sqlite3.connect(f'file:{RUN}/fresh-dedup.sqlite?mode=ro', uri=True)
        rows = db.execute('SELECT hash,id,tokens FROM documents WHERE source=? ORDER BY rowid',(source,))
    else:
        if sha256(input_inventory) != m['inventory_sha256']:
            raise RuntimeError('input document inventory changed')
        db = sqlite3.connect(f'file:{input_inventory}?mode=ro', uri=True)
        rows = db.execute('SELECT hash,id,tokens FROM documents WHERE reason IS NULL ORDER BY rowid')
    writer = ShardWriter(directory,'train',m['tokens'],BLOCK*8192,BLOCK)
    counters = Counter()
    accepted, rejected = [], []
    # A compact document inventory permits another audit without retaining text.
    inventory = sqlite3.connect(directory/'documents.sqlite')
    inventory.execute('CREATE TABLE documents(hash BLOB PRIMARY KEY,id TEXT,tokens INTEGER,reason TEXT)')
    expected_tail = m.get('trailing_accepted_tokens_dropped_for_alignment') if input_inventory is not None else None
    for digest,identifier,n,text,a in verified_documents(TokenStream(arrays),rows,decode,normalized_text,expected_tail):
        counters['verified_documents'] += 1
        reason = quality_filter(source,identifier,text,n)
        inventory.execute('INSERT INTO documents VALUES(?,?,?,?)',(digest,identifier,n,reason))
        if reason:
            counters['rejected_documents'] += 1
            counters['rejected_input_tokens'] += n+1
            counters['reject_'+reason] += 1
            retain_sample(rejected,sample_record(digest,identifier,n,text,reason),20)
        else:
            writer.append(a)
            counters['accepted_documents'] += 1
            counters['accepted_document_input_tokens'] += n+1
            retain_sample(accepted,sample_record(digest,identifier,n,text),40)
        if counters['verified_documents'] % 5000 == 0:
            inventory.commit()
            print(json.dumps({'source':source,'stage':'curating','counters':dict(counters)}),flush=True)
    writer.flush()
    inventory.commit()
    inventory.close()
    db.close()
    atomic_json(directory/'quality-samples.local.json',{'sampling':'40 lowest SHA-256 values over all accepted documents; 20 over rejects',
                                                       'accepted':accepted,'rejected':rejected})
    result = {**{k:m[k] for k in ('source','base_inputs_sha256','benchmark_sha256','tokenizer_sha256','source_revisions')},
              'tokens':writer.written,'shards':writer.records,'quality_version':policy.VERSION,
              'quality_policy_sha256':policy_sha,'input_manifest_path':str(input_path),
              'input_manifest_sha256':sha256(input_path),'counters':dict(counters),
              'inventory_sha256':sha256(directory/'documents.sqlite'),
              'quality_samples_sha256':sha256(directory/'quality-samples.local.json'),
              'trailing_accepted_tokens_dropped_for_alignment':writer.buffered,
              'stage':'curated_pending_review','completed_unix':time.time(),
              'limitations':'Heuristic provenance/structure filter, not semantic certification. Original replay is not retrospectively cleaned.'}
    atomic_json(receipt,result)
    print(json.dumps(result),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources',nargs='+',choices=('math','web','dclm','code'),default=['math','web','dclm','code'])
    args = parser.parse_args()
    handle = lock(RUN/'prepare.lock')
    for source in args.sources:
        curate(source)
    handle.close()


if __name__ == '__main__':
    main()
