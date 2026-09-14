"""Build an exact checkpoint plan after all four long CPT runs pass."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import torch

from evaluate import RECIPES, SEEDS


TARGET_TOKENS = 2_147_483_648
FINAL_STEP = 1024
INITIAL_SHA256 = '13aa21721e15c48cdfdafe30d8fdd41d9c95c90be766327661d96af1e90dd6cf'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_model_state(path, expected_protocol_sha256, expected_manifest_sha256):
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    expected = {
        'step': FINAL_STEP,
        'prediction_tokens': TARGET_TOKENS,
        'protocol_sha256': expected_protocol_sha256,
        'training_manifest_sha256': expected_manifest_sha256,
        'checkpoint_semantics': 'model-only; no optimizer, reader or RNG state; not resumable'}
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f'{path}: model checkpoint {key} mismatch')
    if state.get('origin') != {'kind': 'continued_pretraining',
                               'initial_checkpoint_sha256': INITIAL_SHA256}:
        raise ValueError(f'{path}: model checkpoint origin mismatch')
    if not isinstance(state.get('model'), dict) or not state['model']:
        raise ValueError(f'{path}: model state is empty')


def completed_run_dir(root, recipe, seed):
    parent = root / 'posttrain' / 'confirmations-v1'
    prefix = f'{recipe}-seed{seed}-'
    valid = []
    for run_dir in sorted(parent.glob(prefix + '*')):
        train = run_dir / 'train'
        status_path = train / 'training-status.json'
        manifest_path = train / 'run-manifest.json'
        model_path = train / 'model-final.pt'
        sidecar_path = train / 'model-final.sha256'
        if not all(path.is_file() for path in (status_path, manifest_path, model_path, sidecar_path)):
            continue
        status = json.loads(status_path.read_text())
        if (status.get('status') != 'QUALIFICATION_COMPLETED'
                or status.get('step') != FINAL_STEP
                or status.get('tokens') != TARGET_TOKENS
                or status.get('mfu_window_passed') is not True):
            continue
        valid.append(run_dir)
    if len(valid) != 1:
        raise RuntimeError(f'{recipe} seed {seed}: expected exactly one successful run, found {len(valid)}')
    return valid[0]


def successful_run(run_dir, recipe, seed, protocol_sha256, recipe_receipt):
    train = run_dir / 'train'
    status_path = train / 'training-status.json'
    manifest_path = train / 'run-manifest.json'
    model_path = train / 'model-final.pt'
    sidecar_path = train / 'model-final.sha256'
    status = json.loads(status_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    expected_manifest_sha = recipe_receipt['manifest_sha256']
    checks = {
        'seed': seed, 'target_tokens': TARGET_TOKENS, 'tokens_per_step': 2_097_152,
        'world_size': 4, 'microbatch': 4, 'accumulation': 64,
        'peak_lr': 0.0001, 'checkpoint_mode': 'model-only-final',
        'protocol_sha256': protocol_sha256,
        'data_manifest_sha256': expected_manifest_sha,
        'admission_sha256': recipe_receipt['admission_sha256'],
        'initial_checkpoint_sha256': INITIAL_SHA256}
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise ValueError(f'{run_dir}: run manifest {key} mismatch')
    actual_sha = digest(model_path)
    if status.get('model_checkpoint_sha256') != actual_sha:
        raise ValueError(f'{run_dir}: status/model digest mismatch')
    sidecar = sidecar_path.read_text().strip().split()
    if sidecar != [actual_sha, 'model-final.pt']:
        raise ValueError(f'{run_dir}: model sidecar mismatch')
    verify_model_state(model_path, protocol_sha256, expected_manifest_sha)
    match = re.search(r'(\d+)$', run_dir.name)
    if not match:
        raise ValueError(f'{run_dir}: missing terminal Slurm job id')
    return {'checkpoint_id': f'{recipe}_seed{seed}', 'recipe': recipe, 'seed': seed,
            'job_id': int(match.group(1)), 'checkpoint': str(model_path),
            'checkpoint_sha256': actual_sha,
            'training_status': str(status_path),
            'training_status_sha256': digest(status_path),
            'run_manifest': str(manifest_path),
            'run_manifest_sha256': digest(manifest_path)}

def build(root, protocol_path, expected_protocol_sha256, inputs_path, expected_inputs_sha256):
    if digest(protocol_path) != expected_protocol_sha256:
        raise ValueError('protocol SHA-256 mismatch')
    if digest(inputs_path) != expected_inputs_sha256:
        raise ValueError('confirmation inputs receipt SHA-256 mismatch')
    protocol = json.loads(protocol_path.read_text())
    inputs = json.loads(inputs_path.read_text())
    if (protocol.get('schema') != 'p529m-long-confirmation-v1'
            or protocol.get('status') != 'FROZEN'
            or tuple(protocol.get('recipes', ())) != RECIPES
            or tuple(protocol.get('seeds', ())) != SEEDS):
        raise ValueError('protocol identity mismatch')
    if inputs.get('status') != 'PASS_CONFIRMATION_INPUTS_ADMITTED':
        raise ValueError('confirmation inputs are not admitted')
    base = protocol['base_checkpoint']
    if base.get('sha256') != INITIAL_SHA256 or not Path(base['path']).is_file():
        raise ValueError('base checkpoint identity mismatch')
    # Complete the cheap status/file preflight for the entire grid before hashing
    # or loading any multi-gigabyte model checkpoint.
    completed = {(recipe, seed): completed_run_dir(root, recipe, seed)
                 for recipe in RECIPES for seed in SEEDS}
    checkpoints = [{'checkpoint_id': 'base', 'checkpoint': base['path'],
                    'checkpoint_sha256': base['sha256'], 'step': base['step']}]
    for recipe in RECIPES:
        receipt = inputs['recipes'][recipe]
        for seed in SEEDS:
            checkpoints.append(successful_run(completed[(recipe, seed)], recipe, seed,
                                              expected_protocol_sha256, receipt))
    return {'schema': 'p529m-long-confirmation-checkpoint-plan-v1',
            'status': 'PASS_EXACT_FOUR_RUN_CHECKPOINT_PLAN',
            'protocol': str(protocol_path), 'protocol_sha256': expected_protocol_sha256,
            'inputs_receipt': str(inputs_path), 'inputs_receipt_sha256': expected_inputs_sha256,
            'checkpoints': checkpoints,
            'training_completed': True,
            'claim_boundary': 'four model checkpoints admitted for frozen held-out loss evaluation; evaluation and selection remain pending'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--expected-protocol-sha256', required=True)
    parser.add_argument('--inputs-receipt', type=Path, required=True)
    parser.add_argument('--expected-inputs-receipt-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = build(args.root, args.protocol, args.expected_protocol_sha256,
                   args.inputs_receipt, args.expected_inputs_receipt_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(args.output.name + f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
    os.replace(temp, args.output)
    print(json.dumps({'status': output['status'], 'checkpoints': len(output['checkpoints'])}, sort_keys=True))


if __name__ == '__main__':
    main()
