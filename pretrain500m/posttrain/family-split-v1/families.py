"""Bound corpus family closure and deterministic reserved partitions.

LSH only proposes pairs. Full, unhashed token shingles verify each edge.
"""
from collections import Counter, defaultdict
import hashlib
import re
import unicodedata

TOKENS = re.compile(r'[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]|[^\W_]+|[^\w\s]', re.UNICODE)
POLICY = dict(schema='p529m-family-split-v1', seed=20260914, bands=16,
              rows_per_band=4, maximum_bucket_rows=256, maximum_candidate_pairs=2000000,
              verified_jaccard_numerator=4, verified_jaccard_denominator=5,
              shingle_tokens=5, minimum_families_each_partition_each_domain=1000,
              minimum_prediction_tokens_each_partition_each_domain=1048576,
              partitions=['train', 'development', 'confirmation'],
              hot_buckets='quarantine all member families; never truncate and admit',
              reserved_minimum_interpretation='1000 families and 1Mi prediction tokens in EACH reserved partition',
              normalized_equality='NFC/whitespace/prose casefold; code preserves case and whitespace',
              scope='Observed corpus relationships only; not universal semantic independence')
DOMAINS = ['general_web', 'knowledge_reading', 'math', 'code', 'multilingual']


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if (self.size[a], -a) < (self.size[b], -b):
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]
        return True


def domain(source):
    if source == 'dclm': return 'general_web'
    if source == 'pdf_en': return 'knowledge_reading'
    if source in ('finemath4', 'infiwebmath4'): return 'math'
    if source.startswith('code_'): return 'code'
    if source.startswith('multilingual_'): return 'multilingual'
    return None


def normalized(text, code=False):
    text = unicodedata.normalize('NFC', text)
    if code: return text
    if not code: text = text.casefold()
    return re.sub(r'\s+', ' ', text).strip()


def shingles(text, code=False):
    tokens = TOKENS.findall(unicodedata.normalize('NFC', text) if code else unicodedata.normalize('NFC', text).casefold())
    return {tuple(tokens[i:i + 5]) for i in range(len(tokens) - 4)}


def verify(a, b):
    if not a or not b: return False, 0, len(a) + len(b)
    small, large = sorted((len(a), len(b)))
    if small * 5 < large * 4: return False, 0, None
    intersection = len(a & b)
    union = len(a) + len(b) - intersection
    return intersection * 5 >= union * 4, intersection, union


def candidates(records, progress=None, bucket_limit=256, pair_limit=2000000):
    n = len(records); pairs = set(); hot = set(); bucket_counts = []
    sketches = [bytes.fromhex(r['minhash64_u32_le_hex']) if r.get('minhash64_u32_le_hex') else None for r in records]
    if any(s is not None and len(s) != 256 for s in sketches): raise ValueError('invalid MinHash length')
    for band in range(16):
        buckets = defaultdict(list)
        for i, sketch in enumerate(sketches):
            if sketch is not None:
                buckets[(records[i]['source_id'].startswith('code_'), sketch[band*16:(band+1)*16])].append(i)
        count = 0
        for members in buckets.values():
            if len(members) > bucket_limit:
                hot.update(members); count += 1; continue
            for offset, a in enumerate(members):
                for b in members[offset+1:]: pairs.add(a*n+b)
            if len(pairs) > pair_limit: raise ValueError('candidate budget exceeded; nothing admitted')
        bucket_counts.append(count)
        if progress: progress(band+1, len(pairs), len(hot))
    # Verify even non-hot-bucket edges incident on quarantined rows so their
    # verified neighbours join the same excluded family transitively.
    return sorted(pairs), hot, bucket_counts


def join_keys(records, families, duplicates):
    owners = {}; counts = Counter()
    for i, r in enumerate(records):
        for field in ('text_sha256', 'normalized_text_sha256', 'legacy_normalized_family_sha256', 'canonical_family', 'numeric_template_sha256'):
            key = r.get(field)
            if not key: continue
            # Case preserving code and casefold prose are separate normalization scopes.
            if field == 'normalized_text_sha256': key = (r['source_id'].startswith('code_'), key)
            key = (field, key)
            if key in owners:
                counts[field] += families.union(i, owners[key])
                if field in ('text_sha256', 'normalized_text_sha256'): duplicates.union(i, owners[key])
            else: owners[key] = i
    return dict(counts)


def select(records, families, duplicates, hot):
    quarantined = defaultdict(set)
    for i, r in enumerate(records):
        reasons = r.get('exclusion_reasons', [])
        if reasons: quarantined[families.find(i)].update(reasons)
        if i in hot: quarantined[families.find(i)].add('unresolved_hot_lsh_bucket')
    reps = {}
    for i, r in enumerate(records):
        if not r['passes_filters_and_bound_exclusions'] or families.find(i) in quarantined: continue
        # Input order is frozen physical source/tranche/row order. An ineligible
        # earlier duplicate cannot suppress a later eligible representative.
        reps.setdefault(duplicates.find(i), i)
    selected = set(reps.values())
    return selected, quarantined


def partition(records, families, selected, minimum_families=1000, minimum_tokens=1048576):
    groups = defaultdict(list)
    for i in sorted(selected): groups[families.find(i)].append(i)
    names = {}; stats = {}
    for root, members in groups.items():
        identities = sorted(f"{records[i]['source_id']}:{records[i]['tranche']}:{records[i]['row']}:{records[i]['text_sha256']}" for i in members)
        names[root] = hashlib.sha256('\n'.join(identities).encode()).hexdigest()
        counts = Counter()
        for i in members:
            d = domain(records[i]['source_id'])
            if d: counts[d] += max(0, records[i]['encoded_tokens_including_one_eos'] - 1)
        stats[root] = counts
    counts = {p:Counter() for p in ('development', 'confirmation')}
    totals = {p:Counter() for p in counts}; assignments = {}
    order = sorted(groups, key=lambda root:hashlib.sha256(f"20260914:{names[root]}".encode()).digest())
    for root in order:
        def need(p):
            return sum(max(0, minimum_families-counts[p][d])/minimum_families + max(0, minimum_tokens-totals[p][d])/minimum_tokens for d in stats[root])
        choice = max(counts, key=lambda p:(need(p), p == 'development'))
        if need(choice) == 0:
            assignments[root] = 'train'; continue
        assignments[root] = choice
        for d, tokens in stats[root].items():
            counts[choice][d] += 1; totals[choice][d] += tokens
    deficits = {p:{d:{'families':max(0,minimum_families-counts[p][d]),'prediction_tokens':max(0,minimum_tokens-totals[p][d])} for d in DOMAINS} for p in counts}
    return assignments, names, {'family_counts':{p:dict(c) for p,c in counts.items()},
                               'document_prediction_tokens':{p:dict(c) for p,c in totals.items()},'deficits':deficits}
