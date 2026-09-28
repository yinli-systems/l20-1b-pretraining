"""Freeze a declared candidate set before any new matched-protocol scores exist."""
import argparse
import json
from pathlib import Path
import time

from run_efficiency_evaluation import PROTOCOL, sha256, read_plan

PROGRAMS = ('run_efficiency_evaluation.py','strict_efficiency_hf.py',
            'analyze_comparison.py','evaluate_comparison.py','summarize_efficiency_frontier.py')
CONVERTER = 'convert_trusted_torch_checkpoint.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--environment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--version',default='efficiency-20260912-v1-plan-v2')
    parser.add_argument('--authorized-pause',type=Path)
    parser.add_argument('--prefetched-first',action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inventory = json.loads(args.inventory.read_text())
    environment = json.loads(args.environment.read_text())
    comparison_path = Path('reports/metrics/tinyllama-comparison-20260911.json')
    receipt_path = Path('reports/receipts/tinyllama-comparison-20260911.json')
    comparison = json.loads(comparison_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if sha256(receipt_path) != comparison['evaluation_receipt_sha256']:
        raise ValueError('Prior comparison receipt changed')
    scan_path = Path('reports/metrics/datadecide-upstream-scan-20260912.json')
    scan = json.loads(scan_path.read_text())
    if scan['ranking'][0]['recipe'] != 'DCLM-Baseline (QC 20%)':
        raise ValueError('Upstream selection differs from declared choice')
    jobs = [j for j in inventory['jobs'] if j['resolution_status'] == 'resolved'
            and not j.get('reuse_existing_if_verified')]
    for job in jobs:
        if job['adapter'] not in environment['adapter_admission']:
            raise ValueError(f'Unadmitted adapter: {job["id"]}')
        if job['adapter'] == 'author_open_lm' and (job['config'].get('attn_name') != 'torch_attn' or job['config'].get('ffn_type') != 'swiglu_torch'):
            raise ValueError('OpenLM requires an unavailable backend')
    # Predeclared first tier: meaningful near-budget comparisons before the rest
    # of the curve. No ordering or candidate selection based on our new scores.
    priority = ['dd_dclm_baseline_qc_20p_10000','dd_dclm_baseline_qc_20p_12500','dd_dclm_baseline_qc_20p_15000',
                'tinyllama_early_10b','tinyllama_early_21b','pythia_1b','weborganizer_dclm',
                'dd_fineweb_edu_12500','dd_fineweb_edu_15000']
    jobs.sort(key=lambda j:priority.index(j['id']) if j['id'] in priority else len(priority))
    if args.prefetched_first:
        jobs.sort(key=lambda j:0 if j['id']=='pythia_1b' else 1)
    original_tokens = 19_999_703_040
    ours_n = receipt['jobs']['ours_boolq']['actual_config']['model_num_parameters']
    tiny_n = receipt['jobs']['tinyllama_3t_core']['actual_config']['model_num_parameters']
    original_hashes = {'evaluations/final/zero_shot_core.json':receipt['original_result_sha256']['zero_shot_core'],
                       'evaluations/comparison-20260911/ours_boolq.json':receipt['jobs']['ours_boolq']['sha256']}
    references = [
        {'id':'ours_20b','compute':6*ours_n*original_tokens,'score':comparison['ours_primary_macro']*100,
         'parameters':ours_n,'tokens':original_tokens,'category':'natural_corpus_base',
         'evidence_sha256':original_hashes,
         'six_without_boolq':comparison['comparisons']['tinyllama_3t']['six_task_sensitivity_excluding_boolq']['ours']*100},
        {'id':'tinyllama_3t','compute':6*tiny_n*3_000_000_000_000,
         'score':comparison['comparisons']['tinyllama_3t']['primary_macro']['baseline']*100,
         'parameters':tiny_n,'tokens':3_000_000_000_000,'category':'natural_corpus_base',
         'token_kind':'nominal_repo_label',
         'evidence_sha256':{'evaluations/comparison-20260911/tinyllama_3t_core.json':receipt['jobs']['tinyllama_3t_core']['sha256']},
         'six_without_boolq':comparison['comparisons']['tinyllama_3t']['six_task_sensitivity_excluding_boolq']['baseline']*100},
    ]
    conversions = {}
    for job in jobs:
        source = next((item for item in job['files'] if item['path'] == 'pytorch_model.bin'),None)
        if job['repo'] == 'TinyLlama/tinyLlama-intermediate-checkpoints' and source:
            conversions[job['id']] = {'source_path':'pytorch_model.bin','source_sha256':source['sha256'],
                'output_path':'model.safetensors','expected_tensors':201,'expected_numel':1_100_048_384,
                'container_image':'python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea'}
    programs = PROGRAMS + ((CONVERTER,) if conversions else ())
    plan = {'version':args.version,'created_unix':time.time(),'protocol':PROTOCOL,
            'jobs':jobs,'reference_points':references,
            'program_sha256':{name:sha256(Path(__file__).with_name(name)) for name in programs},
            'adapter_file_sha256':environment['adapter_file_sha256'],
            'system_source_root':environment['system_source_root'],
            'system_source_sha256':environment['system_source_sha256'],
            'package_versions':environment['package_versions'],
            'provenance_sha256':{'inventory':sha256(args.inventory),'adapter_environment':sha256(args.environment),
                                  'upstream_scan':sha256(scan_path),'prior_comparison_summary':sha256(comparison_path),
                                  'prior_comparison_receipt':sha256(receipt_path),'plan_builder':sha256(__file__)},
            'blocked':[j for j in inventory['jobs'] if j['resolution_status'] != 'resolved']+inventory['unresolved_requests'],
            'new_training_tokens':0,
            'gpu_admission':'Only after continuation B normal completion at step190, its GPU lock releases and no CUDA process remains.',
            'failure_policy':'Stop at first failed/incomplete job; retain evidence; no silent reruns or substitutions.',
            'cache_policy':'Delete only this queue\u0027s enumerated verified reproducible downloads after a verified result and summary; never training data/checkpoints.',
            'limitations':['Declared subset only, not all DataDecide checkpoints or a world ranking.',
                           'Upstream OLMES scores only select recipe; never mixed into the matched-protocol chart.',
                           '6ND uses actual total parameter count consistently; excludes data curation, teacher and experimental-search compute.',
                           'DataDecide published 1.1768B excludes input embeddings; actual total is 1.279854592B.',
                           'Both DataDecide 18.02B and 21.63B neighbors are retained; no interpolation claimed as measurement.',
                           'DataDecide 14.42B and TinyLlama 10B lower-compute points are included to test dominance below our compute budget.',
                           'BoolQ was not explicitly decontaminated in our original data: six-task sensitivity required.',
                           'Phi teacher/synthetic category remains separate. WebOrganizer mixture was selected on related target benchmarks.',
                           'CPU smoke is not actual GPU/benchmark admission; each worker verifies real weights and loading keys.',
                           'Checkpoint branch token labels can be rounded; no exact 75x or global #1 claim.']}
    if conversions:
        plan['runtime_weight_conversions'] = conversions
        plan['limitations'].append('Hash-pinned TinyLlama protocol-5 state dicts are statically audited and converted in a networkless read-only-root container; exact tensor equality is required and receipted before evaluation.')
    if args.authorized_pause:
        pause = json.loads(args.authorized_pause.read_text())
        if pause['reason'] != 'user_requested_evaluation_priority' or pause['status']['stage'] != 'checkpointed_stop':
            raise ValueError('Not a safe user-authorized pause')
        plan['authorized_training_pause'] = {'reason':pause['reason'],'step':pause['checkpoint']['step'],
                                             'checkpoint':pause['checkpoint'],'receipt_sha256':sha256(args.authorized_pause)}
        plan['gpu_admission'] = 'User requested evaluation first: only the hash-bound checkpointed B pause, released GPU lock and no active CUDA process admit evaluation. No automatic training resume.'
    if args.prefetched_first:
        plan['queue_order_reason'] = 'Evaluate already hash-verified and CPU-smoked Pythia first to obtain the first result without waiting for another download; candidate set and scoring unchanged.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(plan,stream,indent=2)
    read_plan(args.output)
    print(json.dumps({'jobs':len(jobs),'reused_reference_points':len(references),
                      'blocked_requests':len(plan['blocked']),'plan_sha256':sha256(args.output)}),flush=True)


if __name__ == '__main__':
    main()
