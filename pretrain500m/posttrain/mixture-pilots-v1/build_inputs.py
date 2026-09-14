#!/usr/bin/env python3
"""Build hash-bound current-corpus manifests for the seven GPT-6 P1 pilots."""
import argparse
import datetime
import hashlib
import json
from decimal import Decimal
from pathlib import Path


TARGET_BLOCKS = 262_144
SEQUENCE_LENGTH = 2_048
TARGET_TOKENS = TARGET_BLOCKS * SEQUENCE_LENGTH
VALIDATION_TOKENS = 16_023_552
BASE_SHA256 = '13aa21721e15c48cdfdafe30d8fdd41d9c95c90be766327661d96af1e90dd6cf'
RESEARCH_PLAN_SHA256 = '58d56059111eb92dea1c0f9213a1e7ebbacabe7dea55c24cf31e366401dcc1a7'
VALIDATION_SHA256 = '429c1ab33bdb8a92214ffdd2327462a05275b29fbd60abecc1d98b9fad634af3'
REQUIRED_CHECKS = (
    'licenses', 'content_quality', 'cross_source_deduplication',
    'benchmark_decontamination', 'family_disjoint_splits',
)

# Percentages use basis points so quota construction never depends on floats.
RECIPES_BPS = {
    'R0_F2_current_control': {
        'fineweb_edu': 2500, 'dclm': 1500, 'pdf_en': 2500,
        'finemath4': 1500, 'infiwebmath4': 500,
        'code_python': 750, 'code_javascript': 300,
        'code_typescript': 150, 'code_cpp': 150, 'code_java': 150,
    },
    'R1_F1_current_reconsider': {
        'fineweb_edu': 3500, 'dclm': 2000, 'pdf_en': 2000,
        'finemath4': 1125, 'infiwebmath4': 375,
        'code_python': 500, 'code_javascript': 200,
        'code_typescript': 100, 'code_cpp': 100, 'code_java': 100,
    },
    'R2_current_retention_repair': {
        'fineweb_edu': 3500, 'dclm': 2500, 'pdf_en': 2000,
        'finemath4': 750, 'infiwebmath4': 250,
        'code_python': 500, 'code_javascript': 200,
        'code_typescript': 100, 'code_cpp': 100, 'code_java': 100,
    },
    'R3_current_balanced': {
        'fineweb_edu': 3000, 'dclm': 2500, 'pdf_en': 2000,
        'finemath4': 1125, 'infiwebmath4': 375,
        'code_python': 500, 'code_javascript': 200,
        'code_typescript': 100, 'code_cpp': 100, 'code_java': 100,
    },
    'R4_current_code_guard': {
        'fineweb_edu': 3000, 'dclm': 2000, 'pdf_en': 2000,
        'finemath4': 1125, 'infiwebmath4': 375,
        'code_python': 750, 'code_javascript': 300,
        'code_typescript': 150, 'code_cpp': 150, 'code_java': 150,
    },
}

ARMS = (
    ('r0-lr6e5', 'R0_F2_current_control', '0.00006'),
    ('r1-lr6e5', 'R1_F1_current_reconsider', '0.00006'),
    ('r2-lr6e5', 'R2_current_retention_repair', '0.00006'),
    ('r3-lr6e5', 'R3_current_balanced', '0.00006'),
    ('r4-lr6e5', 'R4_current_code_guard', '0.00006'),
    ('r3-lr3e5', 'R3_current_balanced', '0.00003'),
    ('r3-lr1e4', 'R3_current_balanced', '0.0001'),
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def exact_block_quotas(basis_points, total_blocks=TARGET_BLOCKS):
    if set(basis_points) != set(RECIPES_BPS['R0_F2_current_control']):
        raise ValueError('recipe source set changed')
    if sum(basis_points.values()) != 10_000 or any(value <= 0 for value in basis_points.values()):
        raise ValueError('recipe basis points must be positive and sum exactly to 10000')
    quotas = {source: total_blocks * value // 10_000 for source, value in basis_points.items()}
    missing = total_blocks - sum(quotas.values())
    ranked = sorted(
        basis_points,
        key=lambda source: (-(total_blocks * basis_points[source] % 10_000), source),
    )
    for source in ranked[:missing]:
        quotas[source] += 1
    if sum(quotas.values()) != total_blocks:
        raise AssertionError('largest-remainder quota construction failed')
    return quotas


def additional_block_capacity(source):
    unique_blocks = sum(shard['blocks'] for shard in source['shards'])
    cap = Decimal(str(source['max_cumulative_epochs']))
    prior_blocks = Decimal(source.get('prior_prediction_tokens', 0)) / Decimal(SEQUENCE_LENGTH)
    return int(cap * unique_blocks - prior_blocks)


def build(root, output, research_plan):
    root = root.resolve(strict=True)
    research_plan = research_plan.resolve(strict=True)
    if output.exists():
        raise ValueError('output already exists')
    if sha(research_plan) != RESEARCH_PLAN_SHA256:
        raise ValueError('GPT-6 research plan identity changed')

    parent_dir = root/'source/confirmation-inputs-expansion-v1'
    parent_receipt_path = parent_dir/'build-receipt.json'
    parent_receipt = json.loads(parent_receipt_path.read_text())
    if parent_receipt.get('status') != 'PASS_CONFIRMATION_INPUTS_ADMITTED':
        raise ValueError('parent confirmation inputs are not admitted')
    parent_recipe = 'F2_reasoning_no_synthetic'
    parent_manifest_path = parent_dir/'manifests'/(parent_recipe + '.json')
    parent_admission_path = parent_dir/'admissions'/(parent_recipe + '.admission.json')
    expected_parent = parent_receipt['recipes'][parent_recipe]
    if sha(parent_manifest_path) != expected_parent['manifest_sha256']:
        raise ValueError('parent manifest SHA-256 mismatch')
    if sha(parent_admission_path) != expected_parent['admission_sha256']:
        raise ValueError('parent admission SHA-256 mismatch')
    parent_manifest = json.loads(parent_manifest_path.read_text())
    parent_admission = json.loads(parent_admission_path.read_text())
    if parent_admission.get('status') != 'PASS' or any(
            parent_admission.get('checks', {}).get(check) != 'PASS' for check in REQUIRED_CHECKS):
        raise ValueError('parent corpus admission is incomplete')
    if parent_admission.get('training_manifest_sha256') != sha(parent_manifest_path):
        raise ValueError('parent admission does not bind its manifest')

    validation_path = root/'data/development-masked-v2/development-mixture.json'
    if sha(validation_path) != VALIDATION_SHA256:
        raise ValueError('development validation identity changed')
    base_path = root/'formal/run-v5/resume.pt'
    if not base_path.is_file() or sha(base_path) != BASE_SHA256:
        raise ValueError('immutable base checkpoint identity changed')

    parent_sources = {source['id']: source for source in parent_manifest['sources']}
    if set(parent_sources) != set(RECIPES_BPS['R0_F2_current_control']):
        raise ValueError('English parent source set changed')
    for source_id, source in parent_sources.items():
        for shard in source['shards']:
            path = Path(shard['path'])
            if not path.is_file():
                raise ValueError(f'missing packed shard: {source_id}: {path}')

    output.mkdir()
    manifests_dir = output/'manifests'
    admissions_dir = output/'admissions'
    manifests_dir.mkdir()
    admissions_dir.mkdir()
    protocol = {
        'schema': 'p529m-current-corpus-mixture-pilots-v1',
        'status': 'FROZEN_EXPLORATORY_CURRENT_ADMITTED_CORPUS',
        'data_scope': 'reweighted current admitted corpus; not the proposed fresh-corpus program',
        'target_prediction_tokens_per_run': TARGET_TOKENS,
        'steps_per_run': 256,
        'world_size_per_run': 4,
        'seed': 20260916,
        'arms': [
            {'arm_id': arm_id, 'recipe': recipe, 'peak_learning_rate': lr}
            for arm_id, recipe, lr in ARMS
        ],
        'base_checkpoint': {'path': str(base_path), 'sha256': BASE_SHA256, 'step': 7629},
        'training': {
            'architecture': 'deep', 'microbatch_per_gpu': 4, 'gradient_accumulation': 64,
            'global_prediction_tokens_per_step': 2_097_152,
            'warmup_prediction_tokens': 67_108_864,
            'schedule': 'warmup-stable-cosine-decay', 'deterministic': True,
            'checkpoint_mode': 'model-only-final',
        },
        'mfu_gate': {
            'dense_bf16_tflops_per_rtx5090': 209.5,
            'rolling_window_steps': 10, 'grace_steps': 5,
            'minimum_strictly_greater_than': 0.70,
        },
        'selection': {
            'development_signal': 'frozen five-domain equal-domain masked next-token loss',
            'purpose': 'mixture and learning-rate screening before independent capability evaluation',
            'automatic_promotion': False,
            'automatic_market_superiority_claim': False,
        },
        'research_plan_sha256': RESEARCH_PLAN_SHA256,
    }
    protocol_path = output/'protocol.json'
    write_json(protocol_path, protocol)
    protocol_sha = sha(protocol_path)

    built = {}
    for recipe, basis_points in RECIPES_BPS.items():
        quotas = exact_block_quotas(basis_points)
        for source_id, quota in quotas.items():
            capacity = additional_block_capacity(parent_sources[source_id])
            if quota > capacity:
                raise ValueError(f'{recipe}/{source_id} quota {quota} exceeds remaining cap {capacity}')
        manifest = {
            'schema': 'p529m-packed-mixture-v3',
            'protocol_id': protocol['schema'],
            'recipe': recipe,
            'sequence_length': SEQUENCE_LENGTH,
            'tokenizer_sha256': parent_manifest['tokenizer_sha256'],
            'source_block_quotas': quotas,
            'sources': [parent_sources[source_id] for source_id in sorted(quotas)],
            'evidence': {
                'parent_build_receipt_sha256': sha(parent_receipt_path),
                'parent_manifest_sha256': sha(parent_manifest_path),
                'parent_admission_sha256': sha(parent_admission_path),
                'development_manifest_sha256': VALIDATION_SHA256,
                'research_plan_sha256': RESEARCH_PLAN_SHA256,
                'protocol_sha256': protocol_sha,
            },
        }
        manifest_path = manifests_dir/(recipe + '.json')
        write_json(manifest_path, manifest)
        manifest_sha = sha(manifest_path)
        admission = {
            'schema': 'p529m-corpus-admission-v1', 'status': 'PASS', 'recipe': recipe,
            'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'training_manifest_sha256': manifest_sha,
            'validation_manifest_sha256': VALIDATION_SHA256,
            'protocol_sha256': protocol_sha,
            'checks': {check: 'PASS' for check in REQUIRED_CHECKS},
            'evidence': manifest['evidence'],
            'scope': parent_admission.get('scope', {}),
            'limitations': [
                'Admission is inherited only for exact already-admitted packed shards and bounded repeat caps.',
                'These pilots reweight the current corpus and are not fresh-corpus reproductions.',
                'Development loss is a screening signal and does not establish downstream or market superiority.',
            ],
        }
        admission_path = admissions_dir/(recipe + '.admission.json')
        write_json(admission_path, admission)
        built[recipe] = {
            'basis_points': basis_points,
            'source_block_quotas': quotas,
            'manifest_sha256': manifest_sha,
            'admission_sha256': sha(admission_path),
        }

    receipt = {
        'schema': 'p529m-current-corpus-mixture-pilot-inputs-v1',
        'status': 'PASS_INPUTS_ADMITTED_FOR_EXPLORATORY_PILOTS',
        'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'target_prediction_tokens_per_run': TARGET_TOKENS,
        'protocol_sha256': protocol_sha,
        'parent_build_receipt_sha256': sha(parent_receipt_path),
        'research_plan_sha256': RESEARCH_PLAN_SHA256,
        'recipes': built,
        'arms': protocol['arms'],
        'training_launched': False,
        'claim_boundary': protocol['data_scope'],
    }
    write_json(output/'build-receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--research-plan', type=Path, required=True)
    args = parser.parse_args()
    receipt = build(args.root, args.output, args.research_plan)
    print(json.dumps({'status': receipt['status'], 'recipes': list(receipt['recipes'])}, sort_keys=True))


if __name__ == '__main__':
    main()
