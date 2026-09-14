"""Bounded, stateless mixing of fixed-length prediction-token blocks.

Only explicitly listed, SHA-256-verified shards are consumed. This verifies
content identity, not corpus quality, licenses, deduplication or contamination.
"""
import bisect
from collections import Counter, OrderedDict
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


SCHEMA = 'p529m-packed-mixture-v2'
ALGORITHM = 'sha256-permuted-integer-period-v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def sha256(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('expected lowercase SHA-256')
    return value


def allocate(weights, blocks):
    """Hamilton allocation; ties resolve by canonical source id."""
    integer(blocks, 'blocks')
    if not weights or any(not isinstance(k, str) or not k for k in weights):
        raise ValueError('source ids must be nonempty strings')
    for value in weights.values():
        integer(value, 'weight')
    if sum(weights.values()) != 100:
        raise ValueError('integer percentages must sum to 100')
    out = {k: blocks * v // 100 for k, v in weights.items()}
    order = sorted(weights, key=lambda k: (-(blocks * weights[k] % 100), k))
    for k in order[:blocks - sum(out.values())]:
        out[k] += 1
    return out


class MixtureReader:
    def __init__(self, manifest_path, seed, target_tokens, expected_sha256):
        path = Path(manifest_path)
        sha256(expected_sha256)
        self.manifest_sha256 = digest(path)
        if self.manifest_sha256 != expected_sha256:
            raise ValueError('mixture manifest SHA-256 mismatch')
        doc = json.loads(path.read_text())
        if doc['schema'] != SCHEMA:
            raise ValueError('unsupported mixture schema')
        self.sequence_length = integer(doc['sequence_length'], 'sequence_length', 1)
        self.seed = integer(seed, 'seed')
        integer(target_tokens, 'target_tokens', 1)
        if target_tokens % self.sequence_length:
            raise ValueError('budget must contain complete prediction blocks')
        self.total_blocks = target_tokens // self.sequence_length
        self.weights = dict(sorted(doc['weights_percent'].items()))
        self.quotas = allocate(self.weights, self.total_blocks)
        active = {k for k, v in self.weights.items() if v}
        if len(doc['sources']) != len(active) or {s['id'] for s in doc['sources']} != active:
            raise ValueError('exactly one source is required for every positive weight')
        sha256(doc['tokenizer_sha256'])
        self.sources = {}
        content_seen = set()
        source_identity = []
        for source in sorted(doc['sources'], key=lambda s: s['id']):
            sid = source['id']
            revision = source['revision']
            if not isinstance(revision, str) or len(revision) not in (40, 64) or any(c not in '0123456789abcdef' for c in revision):
                raise ValueError('source revision must be a pinned commit or content hash')
            if not source['shards']:
                raise ValueError(f'empty source: {sid}')
            arrays, counts, shards = [], [], []
            # Hash order makes physical traversal independent of transport paths.
            for shard in sorted(source['shards'], key=lambda s: s['sha256']):
                expected = sha256(shard['sha256'])
                if expected in content_seen:
                    raise ValueError('duplicate shard content across the mixture')
                content_seen.add(expected)
                shard_path = path.parent / shard['path']
                if digest(shard_path) != expected:
                    raise ValueError(f'shard SHA-256 mismatch: {sid}/{shard_path.name}')
                array = np.load(shard_path, mmap_mode='r', allow_pickle=False)
                count = integer(shard['blocks'], 'shard blocks', 1)
                if array.dtype != np.uint16 or array.ndim != 1 or array.size != count * (self.sequence_length + 1):
                    raise ValueError(f'invalid packed shard: {sid}/{shard_path.name}')
                arrays.append(array)
                counts.append(count)
                shards.append({'sha256': expected, 'blocks': count})
            total = sum(counts)
            prior = integer(source['prior_prediction_tokens'], 'prior_prediction_tokens')
            cap = Decimal(str(source['max_cumulative_epochs']))
            if not cap.is_finite() or cap <= 0:
                raise ValueError('invalid cumulative epoch cap')
            consumed = prior + self.quotas[sid] * self.sequence_length
            if Decimal(consumed) > cap * total * self.sequence_length:
                raise ValueError(f'cumulative source epoch cap exceeded: {sid}')
            self.sources[sid] = {'arrays': arrays, 'ends': np.cumsum(counts).tolist(),
                                 'blocks': total, 'offset_seed': int(identity([self.seed, sid])[:16], 16)}
            source_identity.append({'id': sid, 'revision': revision, 'shards': shards,
                                    'prior_prediction_tokens': prior, 'max_cumulative_epochs': str(cap)})
        self.total_sequences = sum(s['blocks'] for s in self.sources.values())
        self.unique_prediction_tokens = self.total_sequences * self.sequence_length
        divisor = math.gcd(*[v for v in self.weights.values() if v])
        self.units = {k: v // divisor for k, v in self.weights.items() if v}
        self.period = sum(self.units.values())
        self.full_periods, self.remainder = divmod(self.total_blocks, self.period)
        self.tail = allocate(self.weights, self.remainder)
        self._cache = OrderedDict()
        self.fingerprint = identity({'algorithm': ALGORITHM, 'sequence_length': self.sequence_length,
                                     'seed': seed, 'target_tokens': target_tokens,
                                     'weights_percent': self.weights, 'quotas': self.quotas,
                                     'tokenizer_sha256': doc['tokenizer_sha256'], 'sources': source_identity})

    def _cycle(self, cycle):
        if cycle not in self._cache:
            counts = self.units if cycle < self.full_periods else self.tail
            items = [(identity([ALGORITHM, self.seed, cycle, sid, j]), sid, j)
                     for sid, count in counts.items() for j in range(count)]
            items.sort()
            ordinal = Counter()
            entries = []
            for _, sid, _ in items:
                entries.append((sid, cycle * self.units[sid] + ordinal[sid]))
                ordinal[sid] += 1
            self._cache[cycle] = entries
            if len(self._cache) > 64:
                self._cache.popitem(last=False)
        self._cache.move_to_end(cycle)
        return self._cache[cycle]

    def locate(self, global_block):
        integer(global_block, 'global_block')
        if global_block >= self.total_blocks:
            raise StopIteration('fixed mixture budget exhausted; no renormalization')
        cycle, offset = divmod(global_block, self.period)
        return self._cycle(cycle)[offset]

    def sequence(self, global_block):
        sid, ordinal = self.locate(global_block)
        source = self.sources[sid]
        epoch, position = divmod(ordinal, source['blocks'])
        physical = (position + source['offset_seed'] + epoch * 0x9E3779B1) % source['blocks']
        shard = bisect.bisect_right(source['ends'], physical)
        start = (physical - (source['ends'][shard - 1] if shard else 0)) * (self.sequence_length + 1)
        return torch.from_numpy(np.array(source['arrays'][shard][start:start + self.sequence_length + 1],
                                         dtype=np.int64, copy=True))

    def batch_for_step(self, step, rank, world_size, sequences_per_rank):
        integer(step, 'step')
        integer(world_size, 'world_size', 1)
        integer(rank, 'rank')
        integer(sequences_per_rank, 'sequences_per_rank', 1)
        if rank >= world_size:
            raise ValueError('rank outside world')
        base = step * world_size * sequences_per_rank + rank * sequences_per_rank
        if base + sequences_per_rank > self.total_blocks:
            raise StopIteration('incomplete batch at fixed budget boundary')
        values = torch.stack([self.sequence(i) for i in range(base, base + sequences_per_rank)])
        return values[:, :-1], values[:, 1:]

    def state_at(self, next_global_block):
        integer(next_global_block, 'next_global_block')
        if next_global_block > self.total_blocks:
            raise ValueError('cursor beyond fixed mixture budget')
        cycle, offset = divmod(next_global_block, self.period)
        counts = {k: cycle * self.units.get(k, 0) for k in self.weights}
        if offset:
            for sid, _ in self._cycle(cycle)[:offset]:
                counts[sid] += 1
        return {'fingerprint': self.fingerprint, 'next_global_block': next_global_block,
                'consumed_blocks': counts}

    def verify_state(self, state, next_global_block):
        if state != self.state_at(next_global_block):
            raise ValueError('mixture resume cursor or content identity mismatch')
