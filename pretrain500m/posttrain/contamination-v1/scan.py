"""Unicode exact-span reservation audit. Outputs candidates, never admission."""
import argparse
from collections import Counter
import datetime
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import time
import unicodedata

import ahocorasick
from audit_raw_intake_v2 import preflight, require
from audit_raw_intake_v1 import atomic_json, sha


POLICY = dict(schema='p529m-unicode-reservation-span-v1', unicode_normalization='NFC',
              whitespace='collapse Unicode whitespace to one ASCII space, then strip',
              prose_case='casefold, then NFC', code_and_math_case='preserve',
              full_field_min_chars=64, long_span_chars=192, long_span_stride=96,
              minimum_distinct_nonspace_chars=12, short_primary_whole_document_min_chars=16,
              short_primary_minimum_distinct_nonspace_chars=8,
              short_primary_fields=['prompt', 'question', 'problem'],
              maximum_saved_hits_per_document=32, maximum_saved_owners_per_pattern=4,
              action='AUDIT_CANDIDATES_ONLY_NO_TRAINING_ADMISSION')


def normalize(text, mode):
    text = unicodedata.normalize('NFC', text)
    if mode == 'prose':
        text = unicodedata.normalize('NFC', text.casefold())
    return re.sub(r'\s+', ' ', text).strip()


def content_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def anchors(text):
    if len(text) < POLICY['full_field_min_chars']:
        return []
    size = POLICY['long_span_chars']
    if len(text) <= size:
        candidates = [text]
    else:
        starts = sorted(set(range(0, len(text)-size+1, POLICY['long_span_stride'])) | {len(text)-size})
        candidates = [text[i:i+size] for i in starts]
    return list(dict.fromkeys(s for s in candidates if len(set(s)-{' '}) >= POLICY['minimum_distinct_nonspace_chars']))


class Matcher:
    def __init__(self, references):
        self.automata = {mode: ahocorasick.Automaton() for mode in ['prose', 'literal']}
        self.patterns = []
        self.whole = {'prose': {}, 'literal': {}}
        self.coverage = Counter()
        self.by_reference_file = {}
        seen_refs = set()
        for row in references:
            require(row['training_admitted'] is False, 'reference must be marked non-training')
            require(row['reference_id'] not in seen_refs, 'duplicate reference identity')
            seen_refs.add(row['reference_id'])
            mode = 'literal' if row['domain'] in ['code', 'math', 'multilingual_math'] else 'prose'
            file_key = row['provenance']['repo'] + ':' + row['provenance']['path']
            coverage = self.by_reference_file.setdefault(file_key, Counter())
            for field in row['fields']:
                require(content_sha(field['text']) == field['sha256'], 'reference field digest mismatch')
                text = normalize(field['text'], mode)
                owner = dict(reference_id=row['reference_id'], field=field['name'], field_index=field['index'],
                             provenance=row['provenance'])
                patterns = anchors(text)
                whole_only = (not patterns and field['name'] in POLICY['short_primary_fields'] and
                              len(text) >= POLICY['short_primary_whole_document_min_chars'] and
                              len(set(text)-{' '}) >= POLICY['short_primary_minimum_distinct_nonspace_chars'])
                status = 'anchored_fields' if patterns else 'whole_document_only_fields' if whole_only else 'skipped_short_or_low_diversity_fields'
                coverage[status] += 1
                self.coverage[status] += 1
                if whole_only:
                    self.whole[mode].setdefault(text, []).append(owner)
                for pattern in patterns:
                    automaton = self.automata[mode]
                    if pattern in automaton:
                        self.patterns[automaton.get(pattern)]['owners'].append(owner)
                    else:
                        index = len(self.patterns)
                        self.patterns.append(dict(mode=mode, pattern=pattern, owners=[owner]))
                        automaton.add_word(pattern, index)
        for mode, automaton in self.automata.items():
            if len(automaton):
                automaton.make_automaton()
        self.coverage['reference_records'] = len(seen_refs)
        self.coverage['unique_anchor_patterns'] = len(self.patterns)
        require(bool(self.patterns), 'empty reference matcher')

    def matches(self, text):
        hits = []
        seen = set()
        truncated = False
        for mode, automaton in self.automata.items():
            normalized = normalize(text, mode)
            whole = self.whole[mode].get(normalized)
            if whole:
                if len(hits) < POLICY['maximum_saved_hits_per_document']:
                    hits.append(dict(mode=mode, match_kind='whole_document', normalized_start=0,
                                     normalized_end=len(normalized), matched_sha256=content_sha(normalized),
                                     reference_owners=whole[:POLICY['maximum_saved_owners_per_pattern']],
                                     reference_owner_count=len(whole)))
                else:
                    truncated = True
            if not len(automaton):
                continue
            for end, index in automaton.iter(normalized):
                if index in seen:
                    continue
                seen.add(index)
                if len(hits) >= POLICY['maximum_saved_hits_per_document']:
                    truncated = True
                    break
                pattern = self.patterns[index]
                start = end-len(pattern['pattern'])+1
                require(normalized[start:end+1] == pattern['pattern'], 'matcher span mismatch')
                hits.append(dict(mode=mode, match_kind='exact_normalized_span', normalized_start=start,
                                 normalized_end=end+1, matched_sha256=content_sha(pattern['pattern']),
                                 reference_owners=pattern['owners'][:POLICY['maximum_saved_owners_per_pattern']],
                                 reference_owner_count=len(pattern['owners'])))
        return hits, truncated


def scan(inputs, bundle, output):
    started = time.time()
    files, bindings, tranches = preflight(inputs)
    report_path = bundle/'report.json'
    reference_report_bytes = report_path.read_bytes()
    reference_report = json.loads(reference_report_bytes)
    references = bundle/'references.jsonl'
    reference_hash = sha(references)
    require(reference_hash == reference_report['references_sha256'], 'reference bundle hash mismatch')
    output.mkdir(exist_ok=False)
    with references.open() as handle:
        matcher = Matcher(json.loads(line) for line in handle)
    require(matcher.coverage['reference_records'] == reference_report['records'], 'reference count mismatch')
    require(sum(matcher.coverage[k] for k in ['anchored_fields', 'whole_document_only_fields',
                'skipped_short_or_low_diversity_fields']) == reference_report['fields'], 'reference field count mismatch')
    runtime = dict(python=__import__('sys').version, unicode_version=unicodedata.unidata_version,
                   pyahocorasick=importlib.metadata.version('pyahocorasick'), unicode_build=ahocorasick.unicode)
    require(runtime['pyahocorasick'] == '2.3.1' and runtime['unicode_build'], 'unqualified matcher runtime')
    atomic_json(output/'reference-coverage.json', dict(policy=POLICY, runtime=runtime, counts=dict(matcher.coverage),
                by_reference_file={k: dict(v) for k, v in matcher.by_reference_file.items()}))
    counts = Counter()
    source_counts = {}
    with (output/'candidate-exclusions.jsonl').open('x') as writer:
        for f in files:
            key = str(f['tranche']) + ':' + f['source_id']
            current = Counter()
            with gzip.open(f['path'], 'rt', encoding='utf-8') as handle:
                for ordinal, line in enumerate(handle):
                    row = json.loads(line)
                    text = row.get('text') or row.get('content')
                    require(isinstance(text, str), 'invalid raw text')
                    p = row['_provenance']
                    require(p['source_id'] == f['source_id'] and (p['repo_id'], p['revision'], p['shard']) == f['identity'],
                            'raw row/source identity mismatch')
                    group = f['groups'].get(p['row_group'])
                    require(group is not None and 0 <= p['row_in_group'] < group['rows_read'] and
                            p['physical_row'] == group['physical_row_offset']+p['row_in_group'], 'invalid raw row position')
                    hits, truncated = matcher.matches(text)
                    current['rows_scanned'] += 1
                    current['text_characters_scanned'] += len(text)
                    if hits:
                        current['candidate_exclusion_rows'] += 1
                        current['rows_with_truncated_saved_hits'] += int(truncated)
                        writer.write(json.dumps(dict(source_id=f['source_id'], tranche=f['tranche'], row=ordinal,
                                     text_sha256=content_sha(text), provenance=p, matches=hits,
                                     saved_matches_truncated=truncated, action='REVIEW_OR_EXCLUDE_BEFORE_ADMISSION'),
                                     ensure_ascii=False) + '\n')
                    if current['rows_scanned'] % 1000 == 0:
                        atomic_json(output/'progress.json', dict(status='RUNNING', source=key,
                                    rows_in_source=current['rows_scanned'], completed_files=len(source_counts),
                                    elapsed_seconds=time.time()-started))
            require(current['rows_scanned'] == f['receipt']['rows_written'], 'raw row count mismatch')
            source_counts[key] = dict(current)
            counts.update(current)
    for path, expected in bindings.items():
        require(sha(path) == expected, 'raw source changed during scan')
    for tranche in tranches:
        root = Path(tranche['path'])
        require(not (root/'writer.lock').exists(), 'new intake writer appeared during scan')
        for pattern in ['*.receipt.json', '*.jsonl.gz']:
            require(set(root.glob(pattern)) == {p for p in bindings if p.parent == root and p.match(pattern)},
                    'intake file set changed during scan')
    require(sha(references) == reference_hash and report_path.read_bytes() == reference_report_bytes,
            'reference bundle changed during scan')
    report = dict(status='SUPPLEMENTAL_EXACT_SPAN_AUDIT_COMPLETE_ADMISSION_PENDING', training_admitted=False,
                  checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), policy=POLICY,
                  policy_sha256=content_sha(json.dumps(POLICY, sort_keys=True)), runtime=runtime,
                  tranches=tranches, input_bindings={str(k):v for k,v in bindings.items()}, sources=source_counts,
                  counts=dict(counts), reference_counts=dict(matcher.coverage), references_sha256=reference_hash,
                  candidates_sha256=sha(output/'candidate-exclusions.jsonl'), elapsed_seconds=time.time()-started,
                  legacy_seven_task_audit_completed=False, old_fineweb_overlap_completed=False,
                  limits=['Candidate rows are not removed by this audit and no corpus admission is issued.',
                          'The chosen fixed exact-span policy is not semantic, paraphrase, translation, identifier-renaming or approximate decontamination.',
                          'Short non-primary fields and low-diversity references are skipped; short primary fields only match a complete document.',
                          '192-character anchors use stride 96 and a final tail anchor; arbitrary long-field overlaps shorter than 287 normalized characters may miss all anchors.',
                          'Normalization collapses whitespace, including code indentation; matches are overlap evidence, not proof of program equivalence.',
                          'Prose casefolds; code and mathematics preserve case, digits, punctuation and operators. NFC does not perform compatibility folding.',
                          'Hit offsets refer to normalized text. Saved hits/owners are capped; reference hashes permit reconstruction.',
                          'Only the listed input tranches and supplemental benchmark bundle are scanned. Legacy seven-task, near-duplicate and old-FineWeb checks remain required.'])
    atomic_json(output/'report.json', report)
    print(json.dumps({k:report[k] for k in ['status','counts','reference_counts','elapsed_seconds']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    scan(args.input, args.bundle, args.output)
