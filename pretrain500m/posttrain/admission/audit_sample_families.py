"""Inspect family metadata in downloaded samples without creating train splits.

This is an upper-bound sufficiency check, not a dedup or contamination audit.
Inspection samples are not representative of their full source distributions.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / 'research/data-mixture-v1'
SOURCES = {'fineweb_edu': 'general_web', 'dclm_100bt': 'general_web',
           'finepdfs_en': 'knowledge_reading', 'finemath4': 'math', 'infiwebmath4': 'math',
           'stackedu_python_text': 'code', 'fineweb2hq_zh': 'multilingual',
           'fineweb2hq_es': 'multilingual', 'cosmopedia2': 'synthetic'}


def normalized_text(text):
    # Only used as a conservative text-identity feature, not to modify training text.
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()


def hashed(text):
    return hashlib.sha256(text.encode()).hexdigest()


def canonical_url(value):
    parsed = urlsplit(value)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        return None
    host = parsed.hostname.lower().rstrip('.')
    if host.startswith('www.'):
        host = host[4:]
    port = parsed.port
    if port and not ((parsed.scheme == 'http' and port == 80) or (parsed.scheme == 'https' and port == 443)):
        host += ':' + str(port)
    query = sorted((k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
                   if not k.lower().startswith('utm_') and k.lower() not in ('fbclid', 'gclid'))
    # Treat http/https and a conventional www prefix as the same document family.
    return urlunsplit(('https', host, parsed.path or '/', urlencode(query), ''))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    design = json.loads((RESEARCH / 'experiment-design.json').read_text())
    text_owners = defaultdict(list)
    domains = defaultdict(lambda: {'rows': 0, 'family_keys': set(), 'missing_family_metadata': 0})
    records = []
    for sid, domain in SOURCES.items():
        path = RESEARCH / 'samples' / (sid + '.jsonl')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        families, hosts, missing, missing_parent, empty = set(), Counter(), 0, 0, 0
        for index, row in enumerate(rows):
            text = row.get('text')
            if not isinstance(text, str) or not text.strip():
                empty += 1
                continue
            text_owners[hashed(normalized_text(text))].append([sid, index])
            family = None
            if sid == 'stackedu_python_text':
                repo = row.get('repo_name', '').strip().lower()
                family = 'repository:' + repo if repo else None
            elif sid == 'cosmopedia2':
                # Prompt equality alone cannot show independence from the seed document.
                if not row.get('parent_document_id') and not row.get('seed_document_url'):
                    missing_parent += 1
                prompt = row.get('prompt')
                family = 'prompt:' + hashed(normalized_text(prompt)) if isinstance(prompt, str) and prompt.strip() else None
            else:
                try:
                    url = canonical_url(row.get('url', ''))
                except ValueError:
                    url = None
                if url:
                    family = 'url:' + url
                    hosts[urlsplit(url).hostname] += 1
            if family:
                families.add(family)
                domains[domain]['family_keys'].add(family)
            else:
                missing += 1
                domains[domain]['missing_family_metadata'] += 1
        domains[domain]['rows'] += len(rows)
        records.append({'source': sid, 'domain': domain, 'sample_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'rows': len(rows), 'nonempty_text_rows': len(rows) - empty,
                        'distinct_observed_family_keys_upper_bound': len(families),
                        'missing_primary_family_metadata': missing,
                        'synthetic_rows_without_seed_parent_identity': missing_parent,
                        'top_hosts_in_inspection_sample_only': hosts.most_common(5),
                        'population_quality_inference_allowed': False})
    minimum = design['development']['minimum_distinct_document_families_per_domain']
    domain_receipts = {}
    for domain in design['development']['domains']:
        item = domains[domain]
        count = len(item['family_keys'])
        domain_receipts[domain] = {'sample_rows': item['rows'], 'observed_family_key_upper_bound': count,
                                   'required_distinct_families': minimum,
                                   'minimum_additional_independent_families': max(0, minimum - count),
                                   'sufficiency_status': 'INSUFFICIENT' if count < minimum else 'STILL_REQUIRES_CLUSTER_AUDIT'}
    out = {'status': 'INSPECTION_POOL_ONLY_NOT_ADMITTED', 'sources': records,
           'domains': domain_receipts,
           'normalized_exact_duplicate_groups': [x for x in text_owners.values() if len(x) > 1],
           'limitations': ['URL keys are an upper bound: mirrors, near duplicates and parent relations can join them',
                           'this does not inspect or certify corpus licenses or factual quality',
                           'no benchmark content was used or exposed in this audit',
                           'English-only ASCII word n-grams in the old decontam builder do not cover multilingual contamination',
                           'synthetic prompt identity does not establish seed-document independence',
                           'no train/development/confirmation split was created from this inspection pool'],
           'training_admitted': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'domain_family_upper_bounds': {k: v['observed_family_key_upper_bound'] for k, v in domain_receipts.items()},
                      'normalized_duplicate_groups': len(out['normalized_exact_duplicate_groups']), 'training_admitted': False}))


if __name__ == '__main__':
    main()
