"""Build exact, lineage-aware F2 0.537B continuation pilot inputs."""

import datetime
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path('/ssd/scxi253/pretrain500m-20260912-v1')
OUTPUT = ROOT / 'source/continuation-inputs-f2-v1/build-output'
EXPECTED = {
    'raw': '53b3e9b83c7bad3527bf6b5b01503a590fecbc954fbe72777b8d9606041ca3ea',
    'quality': '8876d4d764ceb9ea0b51df303121aceb4e7d4d2309da660aeebe33d3a8329388',
    'union': '43525be7a62ca52e0c8f51bb791d674f6835f7d356ce8401236c0bfdfd28b19b',
    'family': 'd02fe6a46088f19b85aa85061eabb878c9252c76751ef4ccf75006bb002ddda0',
    'pack': '07246fa006f377a97ba55827b4f511075c04c530575797ef61222c9583a4a7d9',
    'development': '56c789445533daed78e661d86b5cb21b1f33c45b0417a25983ca0935bed7a28d',
    'confirmation': '96b95978eebe6482a6dddb0b4af6b5790cce405ae475e425e4d76c25c43782b4',
    'screen': 'a6120a304a16d5b569787cc542b5917cb8bf13ba49e97e2083cbf0ff3235591c',
    'long': '1c9cce946fcb853e95ce42e5bb7446768101c5c5aae28973351f6f153c8ebca9',
    'old_admission': '9aa303d6a6e5bf5cfbae5c377562fb141fb4a3fb1331fc3527fdc96720270176',
    'audit_plan': '22539a83b664c5d397c2b3f48920d07cc117e7fb7bef1c330d0299fc8f9bc0d8',
    'rights': '07ae41dc8a4cf4d1597e79fed64ef035d75d57c961eaa5171a6464a2325eedf7',
    'quality_features': 'de3bb19ed8d6b3e2f2fffb2a1abcc0afc214862f81fb3b8f62b4367c5111c814',
    'tokenizer': '30d71356c5ba154006df5bbb4a0583fc434525ceeb2f27a7d8a237ce5db26dc6',
}
PARENTS = {
    20260914: ('F2_reasoning_no_synthetic-seed20260914-1589567',
               '69d9d8d3358f23815f2455f7e075645056f90fe01ed40a11525130a0e58ca085'),
    20260915: ('F2_reasoning_no_synthetic-seed20260915-1590004',
               'f6a4ff43aeeb892ce7dd3dd7ed69f022cef6af688f21b3f005f67850f6041cbc'),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path, key, status=None):
    if sha(path) != EXPECTED[key]:
        raise ValueError('bound input changed: ' + str(path))
    value = json.loads(path.read_text())
    if status and value.get('status') != status:
        raise ValueError('upstream stage incomplete: ' + str(path))
    return value


def write(path, value):
    target = Path(path)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    return sha(target)


def run():
    if os.getuid() != 1256 or ROOT.stat().st_uid != 1256 or ROOT.is_symlink():
        raise ValueError('owned ParaCloud root required')
    if (ROOT / 'OWNER.txt').read_text() != (
            'pretrain500m 2026-09-12 isolated run; owner task '
            '01a09290-b43f-7431-be8a-412ea5d37954\n'):
        raise ValueError('owner receipt changed')
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('fresh output path required')
    paths = {
        'raw': ROOT / 'data/diverse-audit-f2-continuation-v1/report.json',
        'quality': ROOT / 'data/quality-family-f2-continuation-v1/report.json',
        'union': ROOT / 'receipts/exclusion-union-f2-continuation-v1.json',
        'family': ROOT / 'data/family-split-f2-continuation-v1/report.json',
        'pack': ROOT / 'data/selected-packs-f2-continuation-v1/report.json',
        'development': ROOT / 'data/development-masked-f2-continuation-v1/report.json',
        'confirmation': ROOT / 'data/confirmation-masked-f2-continuation-v1/report.json',
        'screen': ROOT / 'source/intake-expansion-v1/F2-screen-manifest.json',
        'long': ROOT / 'source/confirmation-inputs-expansion-v1/manifests/F2_reasoning_no_synthetic.json',
        'old_admission': ROOT / 'source/confirmation-inputs-expansion-v1/admissions/F2_reasoning_no_synthetic.admission.json',
        'audit_plan': ROOT / 'source/intake-f2-continuation-v1/audit-plan.json',
        'rights': ROOT / 'source/continuation-inputs-f2-v1/source-rights-snapshot.json',
    }
    statuses = {
        'raw': 'COMBINED_RAW_INTAKE_MEASURED_NOT_ADMITTED',
        'quality': 'QUALITY_AND_FAMILY_FEATURES_COMPLETE_NOT_ADMITTED',
        'union': 'POLICY_EXCLUSIONS_BOUND_NOT_APPLIED_TO_PACK',
        'family': 'FAMILY_CLOSURE_AND_RESERVED_ASSIGNMENTS_COMPLETE_NOT_ADMITTED',
        'pack': 'SELECTED_SOURCE_PACKS_COMPLETE_NOT_ADMITTED',
        'development': 'F2_DOCUMENT_MASKED_RESERVED_PACK_COMPLETE_NOT_ADMITTED',
        'confirmation': 'F2_DOCUMENT_MASKED_RESERVED_PACK_COMPLETE_NOT_ADMITTED',
        'old_admission': 'PASS',
        'audit_plan': 'FROZEN_UNIQUE_DATA_EXPANSION_PLAN_NOT_ADMITTED',
        'rights': 'PINNED_SOURCE_CARDS_AND_SELECTED_LICENSE_POLICY_REVIEWED',
    }
    documents = {key: load(path, key, statuses.get(key)) for key, path in paths.items()}
    raw, quality, union, family, pack = (documents[key] for key in ('raw', 'quality', 'union', 'family', 'pack'))
    screen, long, old_admission, plan, rights = (documents[key] for key in
                                                  ('screen', 'long', 'old_admission', 'audit_plan', 'rights'))
    if raw['physical_group_disjointness'] != 'PASS' or raw['input_hashes_stable'] != 'PASS':
        raise ValueError('raw identity/disjointness failed')
    if quality['exclusion_plan_sha256'] != EXPECTED['union'] or family['input_bindings'][str(paths['union'])] != EXPECTED['union']:
        raise ValueError('quality/family exclusion identity differs')
    if family['input_bindings'][str(paths['quality'])] != EXPECTED['quality'] or not family['independent_under_observed_family_edges']:
        raise ValueError('family separation or quality binding failed')
    if pack['family_report_sha256'] != EXPECTED['family'] or pack['tokenizer_sha256'] != EXPECTED['tokenizer']:
        raise ValueError('pack family/tokenizer identity differs')
    if union['unique_excluded_text_hashes'] != 848 or union['raw_files_modified'] or union['family_expansion_completed']:
        raise ValueError('exclusion union state differs')
    for path, expected in union['input_bindings'].items():
        if sha(Path(path)) != expected:
            raise ValueError('benchmark/source exclusion binding changed: ' + path)
    for key in ('development', 'confirmation'):
        if documents[key]['parent_pack_report_sha256'] != EXPECTED['pack'] or sha(documents[key]['manifest']) != documents[key]['manifest_sha256']:
            raise ValueError('masked reserved pack identity differs: ' + key)
    if documents['development']['manifest_sha256'] != 'b9791aff2b537689c165f504f3c0fb262952c9d87762e8c2aa3388c6b9e3aa3b':
        raise ValueError('development masked manifest changed')
    if documents['confirmation']['manifest_sha256'] != '8d2f6add7a651401428da8684b5a3846facaec21fc3fa1705dc04065302e63bd':
        raise ValueError('confirmation masked manifest changed')
    if old_admission['training_manifest_sha256'] != EXPECTED['long'] or any(
            result != 'PASS' for result in old_admission['checks'].values()):
        raise ValueError('historical F2 source admission differs')
    if rights['historical_f2_admission_sha256'] != EXPECTED['old_admission'] or sha(
            ROOT / 'source/quality-family-expansion-v1/features.py') != EXPECTED['quality_features']:
        raise ValueError('selected source rights/license policy differs')
    if set(rights['code_allowlist']) != {'MIT', 'Apache-2.0', 'BSD-2-Clause', 'BSD-3-Clause',
                                         'ISC', 'CC0-1.0', 'Unlicense'}:
        raise ValueError('permissive code allowlist changed')
    if any(documents[key].get('training_admitted') for key in
           ('raw', 'quality', 'union', 'family', 'pack', 'development', 'confirmation')):
        raise ValueError('unexpected upstream admission state')
    if screen['recipe'] != 'F2_reasoning_no_synthetic' or long['recipe'] != screen['recipe'] or plan['source_repeat_cap'] != 2.0:
        raise ValueError('F2 recipe/lineage plan differs')
    quota = screen['source_block_quotas']
    long_quota = long['source_block_quotas']
    if sum(quota.values()) != 262144 or any(long_quota[sid] != quota[sid] * 4 for sid in quota):
        raise ValueError('F2 quota changed')
    by_long = {source['id']: source for source in long['sources']}
    by_pack = {source['source_id']: source for source in pack['sources']}
    needed = {item['source_id']: item for item in plan['sources']}
    if set(quota) != set(by_long) or set(quota) - {'fineweb_edu'} != set(needed):
        raise ValueError('F2 source set changed')
    sources = []
    for sid in sorted(quota):
        historical = by_long[sid]
        prior = historical['prior_prediction_tokens'] + long_quota[sid] * 2048
        if sid == 'fineweb_edu':
            selected = {**historical, 'prior_prediction_tokens': prior}
            count = sum(shard['blocks'] for shard in selected['shards'])
        else:
            train = by_pack[sid]['outputs']['train']
            path = Path(train['path'])
            if sha(path) != train['sha256']:
                raise ValueError('packed training source changed: ' + sid)
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            try:
                if array.dtype != np.dtype(np.uint16) or array.ndim != 1 or len(array) != train['blocks'] * 2049:
                    raise ValueError('packed training shape differs: ' + sid)
            finally:
                del array
            count = train['blocks']
            if needed[sid]['f2_long_exposure_blocks'] != long_quota[sid] or needed[sid]['f2_continuation_quota_blocks'] != quota[sid]:
                raise ValueError('frozen F2 lineage quotas differ: ' + sid)
            if count * 2048 < needed[sid]['unique_prediction_tokens_required_at_two_epoch_cap']:
                raise ValueError('two-epoch unique source shortage: ' + sid)
            selected = dict(id=sid, revision=historical['revision'],
                            prior_prediction_tokens=prior, max_cumulative_epochs='2.0',
                            shards=[dict(path=str(path), sha256=train['sha256'], blocks=count)])
        if Decimal(prior + quota[sid] * 2048) > Decimal(str(selected['max_cumulative_epochs'])) * count * 2048:
            raise ValueError('F2 cumulative source epoch cap exceeded: ' + sid)
        sources.append(selected)

    parents = {}
    for seed, (run_name, expected) in PARENTS.items():
        path = ROOT / 'posttrain/confirmations-v1' / run_name / 'train/model-final.pt'
        if sha(path) != expected:
            raise ValueError('F2 seed checkpoint changed: ' + str(seed))
        parents[str(seed)] = dict(path=str(path), sha256=expected,
                                  checkpoint_semantics='model-only-final; fresh optimizer required')
    validation = documents['development']
    if validation['padded_prediction_tokens'] % (4 * 4 * 2048):
        raise ValueError('masked validation does not form exact global batches')
    evidence = {key + '_sha256': value for key, value in EXPECTED.items() if key in paths}
    evidence['historical_f2_parent_checkpoint_sha256'] = {seed: value['sha256'] for seed, value in parents.items()}
    protocol = dict(schema='p529m-f2-continuation-pilot-v1', status='FROZEN',
                    recipe='F2_reasoning_no_synthetic', target_prediction_tokens_per_seed=536870912,
                    steps_per_seed=256, seeds=[20260914, 20260915], initial_checkpoints=parents,
                    optimizer_reset='fresh fused AdamW; no bit-exact continuation from model-only parents',
                    training=dict(world_size=4, gpu='single-node public RTX 5090', architecture='deep',
                                  sequence_length=2048, microbatch_per_gpu=4, accumulation=64,
                                  peak_lr_initial_hypothesis=0.00006, warmup_prediction_tokens=16777216,
                                  checkpoint_mode='full', save_every_steps=128,
                                  validate_before_training=True, validate_every_steps=128),
                    mfu_gate=dict(dense_bf16_tflops_per_gpu=209.5, grace_steps=5,
                                  rolling_window_steps=10, strictly_greater_than=0.70),
                    selection=dict(development='new observed-family-disjoint masked five-domain loss',
                                   confirmation='separate reserved masked five-domain loss after paired run',
                                   capability='independent paired capability/retention and contamination checks',
                                   automatic_market_superiority_claim=False),
                    source_block_quotas=quota, evidence=evidence)
    OUTPUT.mkdir(exist_ok=False)
    protocol_path = OUTPUT / 'f2-continuation-protocol.json'
    protocol_sha = write(protocol_path, protocol)
    manifest = dict(schema='p529m-packed-mixture-v3', protocol_id='p529m-f2-continuation-pilot-v1',
                    recipe=screen['recipe'], sequence_length=2048,
                    tokenizer_sha256=EXPECTED['tokenizer'], source_block_quotas=quota, sources=sources,
                    evidence={**evidence, 'protocol_sha256': protocol_sha,
                              'validation_manifest_sha256': validation['manifest_sha256']})
    manifest_path = OUTPUT / 'F2_reasoning_no_synthetic.continuation.json'
    manifest_sha = write(manifest_path, manifest)
    admission = dict(schema='p529m-corpus-admission-v1', status='PASS',
                     checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     recipe=screen['recipe'], scope='exact 0.537B prediction-token F2 continuation pilot only',
                     training_manifest_sha256=manifest_sha,
                     validation_manifest_sha256=validation['manifest_sha256'],
                     protocol_sha256=protocol_sha,
                     checks={key: 'PASS' for key in ('licenses', 'content_quality',
                                                   'cross_source_deduplication',
                                                   'benchmark_decontamination',
                                                   'family_disjoint_splits')},
                     evidence=evidence,
                     check_scopes=dict(
                         licenses='Pinned dataset card terms, prior admitted F2 sources and selected-code permissive allowlist; attribution provenance retained.',
                         content_quality='Frozen structural, source-score, language and syntax filters; not a factual-truth certification.',
                         cross_source_deduplication='Combined exact, metadata and verified candidate-family closure selected one eligible representative.',
                         benchmark_decontamination='Frozen prefix union excluded all observed exact-span, legacy-hash and old-source normalized candidates before packing.',
                         family_disjoint_splits='New development/confirmation masks and train source indexes are disjoint under observed family joins.'),
                     limitations=['Observed family/benchmark screens have bounded recall.',
                                  'This admission does not establish model improvement or market superiority.',
                                  'Model/dataset publication has separate source terms and attribution review.'])
    admission_path = OUTPUT / 'F2_reasoning_no_synthetic.continuation.admission.json'
    admission_sha = write(admission_path, admission)
    receipt = dict(schema='p529m-f2-continuation-inputs-v1',
                   status='PASS_F2_CONTINUATION_INPUTS_ADMITTED_NOT_TRAINED',
                   checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   target_prediction_tokens_per_seed=536870912,
                   protocol_sha256=protocol_sha, manifest_sha256=manifest_sha,
                   admission_sha256=admission_sha,
                   validation_manifest_sha256=validation['manifest_sha256'],
                   confirmation_manifest_sha256=documents['confirmation']['manifest_sha256'],
                   source_block_quotas=quota, parents=parents,
                   training_launched=False, formal_promotion=False)
    write(OUTPUT / 'build-receipt.json', receipt)
    print(json.dumps({key: receipt[key] for key in ('status', 'protocol_sha256', 'manifest_sha256',
                                                  'admission_sha256', 'training_launched')}))


if __name__ == '__main__':
    try:
        run()
    except Exception as error:
        if OUTPUT.exists():
            write(OUTPUT / 'failure.json', dict(status='FAILED_NOT_TRAINING_ADMITTED',
                                                error_type=type(error).__name__, error=str(error),
                                                training_launched=False))
        raise
