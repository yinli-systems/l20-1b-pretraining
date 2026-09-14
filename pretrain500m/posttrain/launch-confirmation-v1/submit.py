#!/usr/bin/env python3
"""Submit the four admitted long-confirmation jobs exactly once."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path('/ssd/scxi253/pretrain500m-20260912-v1')
INPUTS = ROOT/'source/confirmation-inputs-expansion-v1'
RUNNER = ROOT/'source/confirmation-runner-v1/run.sbatch'
RECEIPT = ROOT/'receipts/confirmation-v1-submission.json'
STORAGE_GATE = 17_049_915_392
OWNER = 'pretrain500m 2026-09-12 isolated run; owner task 01a09290-b43f-7431-be8a-412ea5d37954'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if RECEIPT.exists():
        raise ValueError('submission receipt already exists')
    build = json.loads((INPUTS/'build-receipt.json').read_text())
    if build['status'] != 'PASS_CONFIRMATION_INPUTS_ADMITTED':
        raise ValueError('confirmation inputs are not admitted')
    free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    if free < STORAGE_GATE:
        raise ValueError(f'aggregate checkpoint storage gate failed: {free} < {STORAGE_GATE}')
    queue = subprocess.run(['squeue', '-h', '-u', os.environ['USER'], '-o', '%A|%T|%j'],
                           check=True, text=True, capture_output=True).stdout
    active = [line for line in queue.splitlines() if line.split('|')[-1].startswith('p529m-cf-')]
    if active:
        raise ValueError('active confirmation job already exists: ' + ';'.join(active))
    jobs = []
    for short, recipe in [('F2', 'F2_reasoning_no_synthetic'),
                          ('F3', 'F3_broad_multilingual_no_synthetic')]:
        for seed in [20260914, 20260915]:
            name = f'p529m-cf-{short}-s{str(seed)[-2:]}'
            command = ['sbatch', '--parsable', '--partition=gpu_5090',
                       '--job-name=' + name,
                       '--export=ALL,RECIPE=' + recipe + ',RUN_SEED=' + str(seed), str(RUNNER)]
            last = None
            for attempt in range(3):
                result = subprocess.run(command, text=True, capture_output=True)
                if result.returncode == 0:
                    job_id = int(result.stdout.strip().split(';')[0])
                    jobs.append({'job_id': job_id, 'job_name': name, 'recipe': recipe,
                                 'seed': seed, 'submit_attempt': attempt + 1})
                    break
                last = result.stderr.strip() or result.stdout.strip()
                time.sleep(2)
            else:
                receipt = {'schema': 'p529m-long-confirmation-submission-v1',
                    'status': 'PARTIAL_SUBMISSION_REQUIRES_RECOVERY',
                    'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'owner': OWNER, 'jobs': jobs, 'failed_recipe': recipe,
                    'failed_seed': seed, 'scheduler_error': last,
                    'free_bytes_before_submit': free, 'storage_gate_bytes': STORAGE_GATE}
                RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
                raise RuntimeError(last)
    artifacts = {'runner': sha(RUNNER), 'build_receipt': sha(INPUTS/'build-receipt.json'),
                 'protocol': sha(INPUTS/'confirmation-protocol.json')}
    for recipe in ['F2_reasoning_no_synthetic', 'F3_broad_multilingual_no_synthetic']:
        artifacts[recipe + '_manifest'] = sha(INPUTS/'manifests'/(recipe + '.json'))
        artifacts[recipe + '_admission'] = sha(INPUTS/'admissions'/(recipe + '.admission.json'))
    capacity = subprocess.run(['sinfo', '-p', 'gpu_5090,hp_5090', '-N', '-o', '%P %N %t %G'],
                              text=True, capture_output=True).stdout
    receipt = {'schema': 'p529m-long-confirmation-submission-v1',
        'status': 'SUBMITTED_PENDING_ALLOCATION',
        'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'owner': OWNER, 'jobs': jobs, 'world_size_per_job': 4, 'submitted_gpu_total': 16,
        'target_prediction_tokens_per_run': 2147483648,
        'mfu_rule': 'rolling ten-step median after five-step grace must be strictly greater than 0.50',
        'free_bytes_before_submit': free, 'storage_gate_bytes': STORAGE_GATE,
        'storage_gate_passed': True, 'artifacts': artifacts,
        'capacity_snapshot': capacity, 'training_started': False,
        'claim_boundary': 'submitted confirmation jobs; allocation, MFU, completion, held-out selection and promotion remain unverified'}
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': receipt['status'], 'jobs': jobs}, sort_keys=True))


if __name__ == '__main__':
    main()
