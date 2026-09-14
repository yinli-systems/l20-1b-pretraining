"""Apply the frozen worst-seed then mean selection rule to long CPT results."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from evaluate import DOMAINS, RECIPES, SEEDS, expected_checkpoint_ids


RULE = 'lower worst-seed equal-domain loss, then lower two-seed mean, then recipe id'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def select(evaluation, protocol):
    if evaluation.get('status') != 'PASS_FOUR_RUN_HELDOUT_EVALUATION':
        raise ValueError('held-out evaluation is not complete')
    if (protocol.get('schema') != 'p529m-long-confirmation-v1'
            or protocol.get('status') != 'FROZEN'
            or protocol.get('selection', {}).get('rule') != RULE
            or tuple(protocol.get('recipes', ())) != RECIPES
            or tuple(protocol.get('seeds', ())) != SEEDS):
        raise ValueError('frozen protocol identity mismatch')
    results = evaluation.get('results', [])
    by_id = {row.get('checkpoint_id'): row for row in results}
    if len(results) != len(by_id) or set(by_id) != expected_checkpoint_ids():
        raise ValueError('evaluation does not contain the exact checkpoint grid')
    token_counts = None
    for checkpoint_id, row in by_id.items():
        loss = row.get('loss_equal_domain')
        losses = row.get('loss_by_domain', {})
        counts = row.get('target_tokens_by_domain', {})
        if not isinstance(loss, (int, float)) or not math.isfinite(loss):
            raise ValueError(f'non-finite loss for {checkpoint_id}')
        if set(losses) != set(DOMAINS) or set(counts) != set(DOMAINS):
            raise ValueError(f'domain coverage mismatch for {checkpoint_id}')
        if any(not math.isfinite(losses[d]) or counts[d] <= 0 for d in DOMAINS):
            raise ValueError(f'invalid domain measurement for {checkpoint_id}')
        ordered_counts = tuple(counts[d] for d in DOMAINS)
        if token_counts is None:
            token_counts = ordered_counts
        elif ordered_counts != token_counts:
            raise ValueError('checkpoint evaluations used different target tokens')
    recipe_rows = []
    for recipe in RECIPES:
        seed_rows = []
        for seed in SEEDS:
            checkpoint_id = f'{recipe}_seed{seed}'
            row = by_id[checkpoint_id]
            seed_rows.append({'seed': seed, 'checkpoint_id': checkpoint_id,
                              'checkpoint': row['checkpoint'],
                              'checkpoint_sha256': row['checkpoint_sha256'],
                              'loss_equal_domain': row['loss_equal_domain'],
                              'loss_by_domain': row['loss_by_domain']})
        losses = [row['loss_equal_domain'] for row in seed_rows]
        recipe_rows.append({'recipe': recipe, 'worst_seed_loss_equal_domain': max(losses),
                            'mean_two_seed_loss_equal_domain': sum(losses) / len(losses),
                            'seeds': seed_rows})
    ranked = sorted(recipe_rows, key=lambda row: (
        row['worst_seed_loss_equal_domain'], row['mean_two_seed_loss_equal_domain'], row['recipe']))
    for rank, row in enumerate(ranked, 1):
        row['rank'] = rank
    base = by_id['base']
    return {'schema': 'p529m-long-confirmation-selection-v1',
            'status': 'PASS_FROZEN_TWO_RECIPE_SELECTION',
            'selection_rule': RULE,
            'selected_recipe': ranked[0]['recipe'],
            'ranked_recipes': ranked,
            'base_loss_equal_domain': base['loss_equal_domain'],
            'target_tokens_by_domain': base['target_tokens_by_domain'],
            'formal_promotion': False,
            'claim_boundary': 'selection by frozen held-out language-model loss only; generative capability and market superiority remain unverified'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--expected-evaluation-sha256', required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--expected-protocol-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if digest(args.evaluation) != args.expected_evaluation_sha256:
        parser.error('evaluation SHA-256 mismatch')
    if digest(args.protocol) != args.expected_protocol_sha256:
        parser.error('protocol SHA-256 mismatch')
    evaluation = json.loads(args.evaluation.read_text())
    protocol = json.loads(args.protocol.read_text())
    if evaluation.get('protocol_sha256') != args.expected_protocol_sha256:
        parser.error('evaluation is not bound to the frozen protocol')
    output = select(evaluation, protocol)
    output.update({'evaluation': str(args.evaluation),
                   'evaluation_sha256': args.expected_evaluation_sha256,
                   'protocol': str(args.protocol),
                   'protocol_sha256': args.expected_protocol_sha256})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(args.output.name + f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
    os.replace(temp, args.output)
    print(json.dumps({'status': output['status'], 'selected_recipe': output['selected_recipe']}, sort_keys=True))


if __name__ == '__main__':
    main()
