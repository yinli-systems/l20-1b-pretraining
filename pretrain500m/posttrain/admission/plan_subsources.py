"""Derive exact leaf-source quotas without acquiring data or admitting a corpus."""
import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / 'research/data-mixture-v1'
spec = importlib.util.spec_from_file_location('research_quotas', RESEARCH / 'plan_tokens.py')
quotas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quotas)


def integer_percent(values):
    scaled = {k: Fraction(str(v)) * 100 for k, v in values.items()}
    if any(v.denominator != 1 for v in scaled.values()):
        raise ValueError('subsource weights require exact integer percentages')
    out = {k: int(v) for k, v in scaled.items()}
    if sum(out.values()) != 100:
        raise ValueError('subsource weights do not sum to 100')
    return out


def derive(design, tokens):
    seq = design['screen']['sequence_length']
    global_batch = design['screen']['global_prediction_tokens_per_step']
    if type(tokens) is not int or tokens <= 0 or tokens % global_batch:
        raise ValueError('token budget must be a positive number of full optimizer steps')
    subweights = {
        'math': integer_percent(design['sources']['math']['components']),
        'code': integer_percent(design['sources']['code']['pilot_languages']),
        'multilingual': design['sources']['multilingual']['within_source_percent'],
    }
    result = {}
    for recipe, weights in design['recipes_percent'].items():
        parents = quotas.allocate(weights, tokens // seq)
        groups = {}
        for group, blocks in parents.items():
            weights_inner = subweights.get(group, {group: 100})
            leaves = quotas.allocate(weights_inner, blocks) if blocks else {k: 0 for k in weights_inner}
            assert sum(leaves.values()) == blocks
            groups[group] = {'blocks': blocks, 'prediction_tokens': blocks * seq,
                             'leaves': {k: {'blocks': n, 'prediction_tokens': n * seq,
                                            'target_within_group_percent': weights_inner[k],
                                            'realized_within_group_percent': n / blocks * 100 if blocks else None}
                                        for k, n in leaves.items()}}
        result[recipe] = groups
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    design_path = RESEARCH / 'experiment-design.json'
    design = json.loads(design_path.read_text())
    out = {'status': 'QUOTA_DESIGN_ONLY_DATA_NOT_ADMITTED',
           'design_sha256': hashlib.sha256(design_path.read_bytes()).hexdigest(),
           'source_inventory_sha256': hashlib.sha256((RESEARCH / 'source-inventory.json').read_bytes()).hexdigest(),
           'sequence_length': design['screen']['sequence_length'],
           'rounding': 'Hamilton within each parent quota; ties by leaf source id',
           'unit': 'prediction tokens after filtering, deduplication, family splitting and packing',
           'seeds_share_pools_but_do_not_share_optimizer_state': True,
           'screen': derive(design, design['screen']['prediction_tokens_per_run']),
           'confirmation_candidate_options': derive(design, design['confirmation']['prediction_tokens_per_run']),
           'confirmation_note': 'all options budgeted; only two selected mixtures plus control would run',
           'requirements': [
               'record actual leaf provenance on each packed block',
               'materialize or hierarchically sample exact leaf quotas; arbitrary shard concatenation cannot enforce them',
               'include leaf pools and packing identity in parent manifest provenance',
               'verify source-specific cumulative repeat caps including any overlapping old FineWeb data',
               'use the same admitted pools for paired recipes and independent seeds',
               'deduplicate aliases, source parents and cross-source content before pool counts are called unique',
               'retain an independent family-disjoint development and confirmation reserve',
           ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'recipes': len(out['screen']), 'training_admitted': False}))


if __name__ == '__main__':
    main()
