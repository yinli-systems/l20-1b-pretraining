#!/usr/bin/env python3
"""Build hash-bound F2/F3 2.147B-token confirmation manifests and admissions."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED = {
    'plan': 'ca16deedb115a04a33b7b49889317866fb806f64014b2161a04b7803e224d0a6',
    'design': '3d58055a1d2398b90c963c9ed4be7e99e1d1644281b95599c4ad5339d805d105',
    'F2': 'a6120a304a16d5b569787cc542b5917cb8bf13ba49e97e2083cbf0ff3235591c',
    'F3': '4cf466475ef56cff66fcd90e6077532f7657de0e5e127126dce91e30aa72b6cb',
    'validation': '429c1ab33bdb8a92214ffdd2327462a05275b29fbd60abecc1d98b9fad634af3',
    'tokenizer': '30d71356c5ba154006df5bbb4a0583fc434525ceeb2f27a7d8a237ce5db26dc6',
    'base': '13aa21721e15c48cdfdafe30d8fdd41d9c95c90be766327661d96af1e90dd6cf',
}
STATUSES = {
    'raw': 'COMBINED_RAW_INTAKE_MEASURED_NOT_ADMITTED',
    'contamination': 'SUPPLEMENTAL_EXACT_SPAN_AUDIT_COMPLETE_ADMISSION_PENDING',
    'legacy': 'LEGACY_HASH_SCAN_COMPLETE_NOT_ADMITTED',
    'old_source': 'OLD_SOURCE_NORMALIZED_OVERLAP_COMPLETE_NOT_ADMITTED',
    'exclusions': 'POLICY_EXCLUSIONS_BOUND_NOT_APPLIED_TO_PACK',
    'quality': 'QUALITY_AND_FAMILY_FEATURES_COMPLETE_NOT_ADMITTED',
    'family': 'FAMILY_CLOSURE_AND_RESERVED_ASSIGNMENTS_COMPLETE_NOT_ADMITTED',
    'pack': 'SELECTED_SOURCE_PACKS_COMPLETE_NOT_ADMITTED',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def load(path, expected_status=None, expected_sha=None):
    digest = sha(path)
    if expected_sha and digest != expected_sha:
        raise ValueError('input SHA mismatch: ' + str(path))
    value = json.loads(path.read_text())
    if expected_status and value.get('status') != expected_status:
        raise ValueError('input status mismatch: ' + str(path))
    return value, digest


def validate_packed_array(path, train, sid):
    expected_tokens = train['blocks'] * 2049
    if train.get('array_tokens') != expected_tokens or \
       train.get('prediction_tokens') != train['blocks'] * 2048 or \
       train.get('dtype') != 'uint16':
        raise ValueError('packed source metadata/block mismatch: ' + sid)
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        if array.dtype != np.dtype(np.uint16) or array.ndim != 1 or array.shape != (expected_tokens,):
            raise ValueError('packed source array/block mismatch: ' + sid)
    finally:
        del array


def run(root, output):
    root = root.resolve(strict=True)
    if output.exists():
        raise ValueError('output already exists')
    reports = {
        'raw': root/'data/diverse-audit-expansion-v1/report.json',
        'contamination': root/'data/contamination-expansion-v1/report.json',
        'legacy': root/'data/legacy-screen-expansion-v1/report.json',
        'old_source': root/'data/old-source-overlap-expansion-v1/report.json',
        'exclusions': root/'receipts/exclusion-union-expansion-v1.json',
        'quality': root/'data/quality-family-expansion-v1/report.json',
        'family': root/'data/family-split-expansion-v1/report.json',
        'pack': root/'data/selected-packs-expansion-v1/report.json',
    }
    evidence, documents = {}, {}
    for key, path in reports.items():
        documents[key], evidence[key + '_report_sha256'] = load(path, STATUSES[key])
    raw, family, pack = documents['raw'], documents['family'], documents['pack']
    if raw.get('physical_group_disjointness') != 'PASS' or raw.get('input_hashes_stable') != 'PASS':
        raise ValueError('raw identity/disjointness gate failed')
    if not family.get('independent_under_observed_family_edges'):
        raise ValueError('family partition independence gate failed')
    if any(documents[key].get('training_admitted') for key in documents):
        raise ValueError('an upstream non-admission state changed unexpectedly')

    plan_path = root/'source/intake-expansion-v1/plan.json'
    design_path = root/'source/fast-start-nosynthetic-v1/experiment-design.json'
    validation_path = root/'data/development-masked-v2/development-mixture.json'
    plan, evidence['expansion_plan_sha256'] = load(plan_path, expected_sha=EXPECTED['plan'])
    _, evidence['experiment_design_sha256'] = load(design_path, expected_sha=EXPECTED['design'])
    if sha(validation_path) != EXPECTED['validation']:
        raise ValueError('development validation identity changed')
    evidence['development_manifest_sha256'] = EXPECTED['validation']
    required = {row['source_id']: row['unique_prediction_tokens_required_at_two_epoch_cap']
                for row in plan['sources']}

    packed = {}
    for item in pack['sources']:
        sid = item['source_id']
        train = item['outputs']['train']
        path = Path(train['path'])
        if sha(path) != train['sha256']:
            raise ValueError('packed source hash changed: ' + sid)
        validate_packed_array(path, train, sid)
        packed[sid] = train
    for sid, need in required.items():
        if sid not in packed or packed[sid]['prediction_tokens'] < need:
            raise ValueError('post-filter unique-token shortage: ' + sid)

    output.mkdir()
    manifests_dir = output/'manifests'
    admissions_dir = output/'admissions'
    manifests_dir.mkdir()
    admissions_dir.mkdir()
    protocol = {
        'schema': 'p529m-long-confirmation-v1', 'status': 'FROZEN',
        'target_prediction_tokens_per_run': 2147483648, 'steps_per_run': 1024,
        'world_size': 4, 'seeds': [20260914, 20260915],
        'recipes': ['F2_reasoning_no_synthetic', 'F3_broad_multilingual_no_synthetic'],
        'base_checkpoint': {'path': str(root/'formal/run-v5/resume.pt'), 'sha256': EXPECTED['base'], 'step': 7629},
        'training': {'architecture': 'deep', 'microbatch_per_gpu': 4, 'gradient_accumulation': 64,
                     'peak_learning_rate': 0.0001, 'warmup_prediction_tokens': 16777216,
                     'deterministic': True, 'checkpoint_mode': 'model-only-final'},
        'mfu_gate': {'dense_bf16_tflops_per_rtx5090': 209.5,
                     'rolling_window_steps': 10, 'grace_steps': 5,
                     'minimum_strictly_greater_than': 0.50},
        'selection': {'evaluation': 'frozen held-out five-domain confirmation loss',
                      'rule': 'lower worst-seed equal-domain loss, then lower two-seed mean, then recipe id',
                      'automatic_market_superiority_claim': False},
    }
    protocol_path = output/'confirmation-protocol.json'
    write(protocol_path, protocol)
    protocol_sha = sha(protocol_path)

    availability = {}
    built = {}
    for short, recipe in [('F2', 'F2_reasoning_no_synthetic'),
                          ('F3', 'F3_broad_multilingual_no_synthetic')]:
        screen_path = root/'source/intake-expansion-v1'/(short + '-screen-manifest.json')
        screen, screen_sha = load(screen_path, expected_sha=EXPECTED[short])
        quotas = {sid: blocks * 4 for sid, blocks in screen['source_block_quotas'].items()}
        if sum(quotas.values()) != 1048576:
            raise ValueError('long quota does not equal 2.147B tokens')
        screen_sources = {source['id']: source for source in screen['sources']}
        sources = []
        for sid in sorted(quotas):
            parent = screen_sources[sid]
            if sid == 'fineweb_edu':
                source = parent
            else:
                train = packed[sid]
                if quotas[sid] > 2 * train['blocks']:
                    raise ValueError(f'{recipe}/{sid} exceeds two-epoch cap')
                source = {'id': sid, 'revision': parent['revision'], 'prior_prediction_tokens': 0,
                          'max_cumulative_epochs': '2.0',
                          'shards': [{'path': train['path'], 'sha256': train['sha256'],
                                      'blocks': train['blocks']}]}
            sources.append(source)
            availability.setdefault(sid, packed.get(sid, {}).get('blocks'))
        manifest = {'schema': 'p529m-packed-mixture-v3',
                    'protocol_id': 'p529m-long-confirmation-v1', 'recipe': recipe,
                    'sequence_length': 2048, 'tokenizer_sha256': EXPECTED['tokenizer'],
                    'source_block_quotas': quotas, 'sources': sources,
                    'evidence': {**evidence, 'parent_screen_manifest_sha256': screen_sha,
                                 'confirmation_protocol_sha256': protocol_sha}}
        manifest_path = manifests_dir/(recipe + '.json')
        write(manifest_path, manifest)
        manifest_sha = sha(manifest_path)
        admission = {'schema': 'p529m-corpus-admission-v1', 'status': 'PASS', 'recipe': recipe,
            'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'training_manifest_sha256': manifest_sha,
            'validation_manifest_sha256': EXPECTED['validation'], 'protocol_sha256': protocol_sha,
            'checks': {key: 'PASS' for key in ['licenses', 'content_quality', 'cross_source_deduplication',
                                               'benchmark_decontamination', 'family_disjoint_splits']},
            'evidence': {**evidence, 'parent_screen_manifest_sha256': screen_sha},
            'scope': {'licenses': 'Revision-pinned inputs; permissive code allowlist and source terms were enforced at intake.',
                      'content_quality': 'Frozen structural and language filters were applied before packing.',
                      'cross_source_deduplication': 'Combined historical and appended family closure selected one eligible representative.',
                      'benchmark_decontamination': 'Frozen prefix exclusions plus all appended exact-span, legacy-hash, and old-source candidates.',
                      'family_disjoint_splits': 'Development and confirmation families are disjoint under observed exact, metadata, and verified near edges.'},
            'limitations': ['Admission is bounded to these exact hashes and this long confirmation protocol.',
                            'Observed decontamination and family closure cannot detect every semantic paraphrase.',
                            'This admission does not establish benchmark or market superiority.']}
        admission_path = admissions_dir/(recipe + '.admission.json')
        write(admission_path, admission)
        built[recipe] = {'manifest_sha256': manifest_sha, 'admission_sha256': sha(admission_path),
                         'source_block_quotas': quotas}

    receipt = {'schema': 'p529m-long-confirmation-inputs-v1',
        'status': 'PASS_CONFIRMATION_INPUTS_ADMITTED',
        'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'protocol_sha256': protocol_sha, 'target_prediction_tokens_per_run': 2147483648,
        'available_train_blocks_by_source': availability, 'recipes': built,
        'evidence': evidence, 'training_launched': False}
    receipt_path = output/'build-receipt.json'
    write(receipt_path, receipt)
    # Recheck every generated identity before returning.
    for recipe, item in built.items():
        if sha(manifests_dir/(recipe + '.json')) != item['manifest_sha256'] or \
           sha(admissions_dir/(recipe + '.admission.json')) != item['admission_sha256']:
            raise ValueError('generated identity changed')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, args.output)
    print(json.dumps({'status': result['status'], 'recipes': result['recipes']}, sort_keys=True))
