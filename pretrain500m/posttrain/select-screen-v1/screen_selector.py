"""Fail-closed two-seed selector for the frozen 529M CPT screen."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

RECIPES = (
    'F0_existing_web_control',
    'F1_broad_english_no_synthetic',
    'F2_reasoning_no_synthetic',
    'F3_broad_multilingual_no_synthetic',
)
SEEDS = (20260914, 20260915)
DOMAINS = ('general_web', 'knowledge_reading', 'math', 'code', 'multilingual')
BASE_SHA = '13aa21721e15c48cdfdafe30d8fdd41d9c95c90be766327661d96af1e90dd6cf'
PROTOCOL_SHA = '17c91660cfbbd48a3216243be9db9c5e8716c082dc8d8fcf02c75c582e84f993'
VALIDATION_SHA = '429c1ab33bdb8a92214ffdd2327462a05275b29fbd60abecc1d98b9fad634af3'
DDP_POLICY = {'bucket_cap_mb': 25, 'find_unused_parameters': True,
              'gradient_as_bucket_view': True, 'static_graph': False}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_run(path):
    path = path.resolve()
    required = ('run-manifest.json', 'metrics.jsonl', 'training-status.json',
                'model-final.pt', 'model-final.sha256')
    for name in required:
        if not (path / name).is_file():
            raise ValueError(f'missing {name}: {path}')
    manifest = json.loads((path / 'run-manifest.json').read_text())
    status = json.loads((path / 'training-status.json').read_text())
    rows = [json.loads(line) for line in (path / 'metrics.jsonl').read_text().splitlines()]
    steps = [row for row in rows if isinstance(row.get('mfu'), (int, float))]
    validations = [row for row in rows if isinstance(row.get('val_loss_equal_domain'), (int, float))]
    recipe = Path(manifest['training_manifest']).stem
    seed = manifest['seed']
    if recipe not in RECIPES or seed not in SEEDS:
        raise ValueError(f'unexpected recipe or seed: {recipe} {seed}')
    if manifest.get('initial_checkpoint_sha256') != BASE_SHA:
        raise ValueError('base checkpoint identity mismatch')
    if manifest.get('protocol_sha256') != PROTOCOL_SHA or manifest.get('validation_manifest_sha256') != VALIDATION_SHA:
        raise ValueError('protocol or validation identity mismatch')
    if manifest.get('target_tokens') != 536870912 or manifest.get('world_size') != 4:
        raise ValueError('screen budget or world size mismatch')
    if manifest.get('ddp_policy') != DDP_POLICY or manifest.get('checkpoint_mode') != 'model-only-final':
        raise ValueError('runtime policy mismatch')
    if status.get('status') != 'QUALIFICATION_COMPLETED' or status.get('step') != 256 or status.get('tokens') != 536870912:
        raise ValueError('run did not complete the frozen screen')
    if len(steps) != 256 or [row['step'] for row in steps] != list(range(1, 257)):
        raise ValueError('training metric steps are incomplete')
    if len(validations) != 2 or validations[0]['step'] != 0 or validations[-1]['step'] != 256:
        raise ValueError('initial or final validation is missing')
    if set(validations[-1].get('val_loss_by_domain', {})) != set(DOMAINS):
        raise ValueError('final validation domains differ')
    numeric = ('loss', 'grad_norm', 'mfu', 'step_seconds', 'tokens_per_second')
    if not all(math.isfinite(float(row[key])) for row in steps for key in numeric):
        raise ValueError('non-finite training metric')
    if any(row.get('ddp_buckets_rebuilt') for row in steps):
        raise ValueError('DDP buckets rebuilt')
    mfu10 = statistics.median(float(row['mfu']) for row in steps[-10:])
    if not mfu10 > 0.50 or not status.get('mfu_window_passed'):
        raise ValueError('MFU gate did not pass')
    parts = (path / 'model-final.sha256').read_text().split()
    if len(parts) != 2 or parts[1] != 'model-final.pt' or len(parts[0]) != 64:
        raise ValueError('invalid checkpoint digest file')
    if status.get('model_checkpoint_sha256') != parts[0] or status.get('model_checkpoint_resumable') is not False:
        raise ValueError('checkpoint status and digest disagree')
    if (path / 'model-final.pt').stat().st_size <= 0:
        raise ValueError('empty checkpoint')
    final = validations[-1]
    return {
        'recipe': recipe, 'seed': seed, 'path': str(path),
        'run_fingerprint_sha256': manifest['run_fingerprint_sha256'],
        'metrics_sha256': digest(path / 'metrics.jsonl'),
        'manifest_sha256': digest(path / 'run-manifest.json'),
        'status_sha256': digest(path / 'training-status.json'),
        'checkpoint_sha256': parts[0],
        'checkpoint_bytes': (path / 'model-final.pt').stat().st_size,
        'checkpoint_digest_evidence': 'runner digest after atomic save; status and digest file agree',
        'initial_equal_domain_loss': validations[0]['val_loss_equal_domain'],
        'final_equal_domain_loss': final['val_loss_equal_domain'],
        'final_loss_by_domain': final['val_loss_by_domain'],
        'mfu10_median': mfu10,
    }


def select(runs):
    if len(runs) != len(RECIPES) * len(SEEDS):
        raise ValueError('exactly eight frozen screen runs are required')
    keyed = {}
    for run in runs:
        key = (run['recipe'], run['seed'])
        if key in keyed:
            raise ValueError(f'duplicate run: {key}')
        keyed[key] = run
    expected = {(recipe, seed) for recipe in RECIPES for seed in SEEDS}
    if set(keyed) != expected:
        raise ValueError('recipe/seed matrix is incomplete')
    initial = {run['initial_equal_domain_loss'] for run in runs}
    if len(initial) != 1:
        raise ValueError('initial validation differs across runs')
    aggregate = {}
    for recipe in RECIPES:
        pair = [keyed[(recipe, seed)] for seed in SEEDS]
        losses = [run['final_equal_domain_loss'] for run in pair]
        aggregate[recipe] = {
            'mean_equal_domain_loss': statistics.mean(losses),
            'worst_seed_equal_domain_loss': max(losses),
            'seed_spread': max(losses) - min(losses),
            'mean_loss_by_domain': {domain: statistics.mean(run['final_loss_by_domain'][domain] for run in pair)
                                    for domain in DOMAINS},
        }
    candidates = RECIPES[1:]
    ranking = sorted(candidates, key=lambda recipe: (
        aggregate[recipe]['worst_seed_equal_domain_loss'],
        aggregate[recipe]['mean_equal_domain_loss'], recipe))
    winner = ranking[0]
    control = aggregate[RECIPES[0]]
    win = aggregate[winner]
    if not all(keyed[(winner, seed)]['final_equal_domain_loss'] < keyed[(RECIPES[0], seed)]['final_equal_domain_loss']
               for seed in SEEDS):
        raise ValueError('winner does not beat paired control for both seeds')
    win['relative_mean_loss_reduction_vs_control'] = 1 - win['mean_equal_domain_loss'] / control['mean_equal_domain_loss']
    return {
        'schema_version': 1,
        'status': 'PASS_FROZEN_TWO_SEED_SCREEN_SELECTION',
        'selection_rule': 'candidate with lower worst-seed equal-domain loss, then lower mean, then recipe id',
        'winner': winner,
        'ranking': ranking,
        'aggregate': aggregate,
        'runs': sorted(runs, key=lambda run: (run['recipe'], run['seed'])),
        'next_gate': 'held-out confirmation and expanded admitted training data; this screen is not final promotion',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = select([load_run(path) for path in args.run])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(args.output.name + f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    os.replace(temp, args.output)
    print(json.dumps({'status': result['status'], 'winner': result['winner'],
                      'ranking': result['ranking']}, sort_keys=True))


if __name__ == '__main__':
    main()
