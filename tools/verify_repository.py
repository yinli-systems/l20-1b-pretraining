#!/usr/bin/env python3
"""CPU-only checks of the lossless layout and published numerical evidence.

This verifies stored evidence, not GPU predictions, data independence or seed
replication. Historical hashes are never rewritten to make this check pass.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]

class VerificationError(ValueError):
    """A stored artifact or reporting contract is inconsistent."""

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

def regular(root: Path, relative: str) -> Path:
    p = root / relative
    if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root.resolve()):
        raise VerificationError(f'Not a contained regular file: {relative}')
    return p

def close(actual: float, expected: float, label: str) -> None:
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12):
        raise VerificationError(f'{label}: {actual!r} != {expected!r}')

def verify_migration(root: Path) -> dict:
    manifest = load(root / 'docs/layout-migration.json')
    seen = set()
    for item in manifest['files']:
        if item['original_path'] in seen:
            raise VerificationError('Duplicate original migration entry')
        seen.add(item['original_path'])
        p = regular(root, item['current_path'])
        if sha256(p) != item['sha256']:
            raise VerificationError('Historical file changed: ' + item['original_path'])
    if len(seen) != 646:
        raise VerificationError('Expected the full 646-file pre-refresh inventory')
    return {'original_files_preserved': len(seen), 'hashes_unchanged': True}

def verify_results(root: Path) -> dict:
    base = root / 'results/2026-09-28'
    manifest = load(base / 'source-manifest.json')
    m = load(base / 'metrics.json')
    if sha256(base / 'metrics.json') != manifest['metrics_sha256']:
        raise VerificationError('Metrics hash mismatch')
    for relative, record in manifest['files'].items():
        if sha256(regular(base, relative)) != record['published_sha256']:
            raise VerificationError('Published source hash mismatch: ' + relative)
    total = 0
    for task, metric in m['tasks'].items():
        relative = f'scores/{task}-clusters.csv'
        record = manifest['cluster_score_files'][relative]
        path = regular(base, relative)
        if sha256(path) != record['sha256']:
            raise VerificationError('Cluster score hash mismatch: ' + relative)
        with path.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != metric['unique_images'] or len({r['image_sha256'] for r in rows}) != len(rows):
            raise VerificationError('Image cluster count/uniqueness mismatch: ' + task)
        count = 0; before = []; after = []
        for r in rows:
            n = int(r['questions']); a = float(r['before_sum']); b = float(r['after_sum'])
            if n < 1 or not all(math.isfinite(x) and -1e-12 <= x <= n + 1e-12 for x in [a, b]):
                raise VerificationError('Invalid per-cluster score: ' + task)
            if not re.fullmatch(r'[a-f0-9]{64}', r['image_sha256']):
                raise VerificationError('Invalid image hash')
            count += n; before.append(a); after.append(b)
        if count != metric['questions'] or count != record['questions']:
            raise VerificationError('Question count mismatch: ' + task)
        close(math.fsum(before)/count, metric['before'], task + ' before')
        close(math.fsum(after)/count, metric['after'], task + ' after')
        close(metric['after']-metric['before'], metric['delta'], task + ' delta')
        lo, hi = metric['paired_ci95']
        if not lo <= metric['delta'] <= hi:
            raise VerificationError('Declared interval does not bracket the observed delta')
        total += count
    if total != 15937 or total != m['evaluation']['total_questions']:
        raise VerificationError('Incomplete published-split evaluation')
    t = m['training']; ret = m['retention']
    if t['steps']*32 != t['new_image_events_used_in_saved_lineage'] or t['steps']*64 != t['global_image_events']:
        raise VerificationError('Training exposure count mismatch')
    if t['prepared_images'] < t['new_image_events_used_in_saved_lineage'] or t['target_reached']:
        raise VerificationError('Prepared/used/planned count conflation')
    close(ret['before']['old_QA']-ret['at_stop']['old_QA'], ret['old_QA_drop'], 'old QA decline')
    if ret['old_QA_drop'] <= ret['old_QA_allowed_absolute_drop'] or ret['checks_at_5120']['old_QA']:
        raise VerificationError('Retention failure is missing')
    if ret['consecutive_failures'] != 2 or t['stop_reason'] != 'retention_guard' or t['automatic_promotion']:
        raise VerificationError('Incorrect stop or promotion status')
    return {'questions': total, 'paired_image_clusters': sum(x['unique_images'] for x in m['tasks'].values()),
            'headline_means_recomputed': True, 'confidence_intervals_reinferred': False,
            'retention_failure_preserved': True, 'fresh_model_inference': False}

def verify_active_links(root: Path) -> dict:
    paths = [root/'README.md', root/'MODEL_CARD.md', root/'pretraining/README.md',
             root/'L20-VL-1.2B/README.md', *sorted((root/'docs').glob('*.md')),
             root/'results/README.md', root/'results/2026-09-28/README.md']
    checked = 0
    for path in paths:
        for target in re.findall(r'\]\(([^)\s]+)(?:\s+[^)]*)?\)', path.read_text(encoding='utf-8')):
            if urlsplit(target).scheme or target.startswith('#'):
                continue
            relative = unquote(target.split('#', 1)[0])
            dest = (path.parent / relative).resolve()
            if not dest.is_relative_to(root.resolve()) or not dest.exists():
                raise VerificationError(f'Broken active link in {path.relative_to(root)}: {target}')
            checked += 1
    return {'active_documents': len(paths), 'relative_links_checked': checked,
            'historical_documents_not_rewritten': True, 'external_links_not_network_checked': True}

def verify(root: Path = ROOT) -> dict:
    root = root.resolve()
    return {'status': 'verified', 'migration': verify_migration(root),
            'results': verify_results(root), 'documentation': verify_active_links(root)}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.root), indent=2))
    except (OSError, ValueError, KeyError) as e:
        print(json.dumps({'status': 'failed', 'error': str(e)}, indent=2))
        raise SystemExit(1) from e

if __name__ == '__main__':
    main()
