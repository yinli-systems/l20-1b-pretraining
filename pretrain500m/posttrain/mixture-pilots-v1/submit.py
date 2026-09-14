#!/usr/bin/env python3
"""Submit the seven frozen four-RTX-5090 mixture/LR pilots exactly once."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path('/ssd/scxi253/pretrain500m-20260912-v1')
INPUTS = ROOT/'source/mixture-pilot-inputs-v1'
RUNNER = ROOT/'source/mixture-pilots-v1/run.sbatch'
RECEIPT = ROOT/'receipts/mixture-pilots-v1-submission.json'
STORAGE_GATE = 50 * 1024**3
EXPECTED_STATUS = 'PASS_INPUTS_ADMITTED_FOR_EXPLORATORY_PILOTS'
JOB_PREFIX = 'p529m-p1-'
OWNER = 'pretrain500m current-corpus P1 pilots; Codex task 44ca'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def command_output(command):
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout


def main():
    if RECEIPT.exists():
        raise ValueError('submission receipt already exists')
    build_path = INPUTS/'build-receipt.json'
    build = json.loads(build_path.read_text())
    if build.get('status') != EXPECTED_STATUS or build.get('training_launched') is not False:
        raise ValueError('pilot inputs are not in the expected pre-launch state')
    free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    if free < STORAGE_GATE:
        raise ValueError(f'aggregate model-output storage gate failed: {free} < {STORAGE_GATE}')
    queue = command_output(['squeue', '-h', '-u', os.environ['USER'], '-o', '%A|%T|%j'])
    active = [line for line in queue.splitlines() if line.split('|')[-1].startswith(JOB_PREFIX)]
    if active:
        raise ValueError('active mixture-pilot job already exists: ' + ';'.join(active))

    artifacts = {'runner': sha(RUNNER), 'build_receipt': sha(build_path),
                 'protocol': sha(INPUTS/'protocol.json')}
    for recipe, item in build['recipes'].items():
        manifest_path = INPUTS/'manifests'/(recipe + '.json')
        admission_path = INPUTS/'admissions'/(recipe + '.admission.json')
        if sha(manifest_path) != item['manifest_sha256'] or sha(admission_path) != item['admission_sha256']:
            raise ValueError('generated pilot input identity changed: ' + recipe)
        artifacts[recipe + '_manifest'] = item['manifest_sha256']
        artifacts[recipe + '_admission'] = item['admission_sha256']

    jobs = []
    for arm in build['arms']:
        suffix = arm['arm_id'].replace('lr', 'l')
        name = JOB_PREFIX + suffix
        exports = ','.join([
            'ALL', 'RECIPE=' + arm['recipe'], 'ARM_ID=' + arm['arm_id'],
            'PEAK_LR=' + arm['peak_learning_rate'], 'RUN_SEED=20260916',
        ])
        command = [
            'sbatch', '--parsable', '--partition=gpu_5090', '--qos=gpugpu',
            '--job-name=' + name, '--export=' + exports, str(RUNNER),
        ]
        last_error = None
        for attempt in range(3):
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode == 0:
                job_id = int(result.stdout.strip().split(';')[0])
                jobs.append({**arm, 'job_id': job_id, 'job_name': name,
                             'seed': 20260916, 'submit_attempt': attempt + 1})
                break
            last_error = result.stderr.strip() or result.stdout.strip()
            time.sleep(2)
        else:
            write_receipt('PARTIAL_SUBMISSION_REQUIRES_RECOVERY', jobs, free, artifacts,
                          scheduler_error=last_error, failed_arm=arm)
            raise RuntimeError(last_error)

    write_receipt('SUBMITTED_PENDING_ALLOCATION', jobs, free, artifacts)
    print(json.dumps({'status': 'SUBMITTED_PENDING_ALLOCATION', 'jobs': jobs}, sort_keys=True))


def write_receipt(status, jobs, free, artifacts, **extra):
    capacity = subprocess.run(
        ['sinfo', '-h', '-p', 'gpu_5090', '-o', '%P|%a|%D|%t|%G'],
        text=True, capture_output=True,
    ).stdout
    receipt = {
        'schema': 'p529m-current-corpus-mixture-pilot-submission-v1',
        'status': status,
        'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'owner': OWNER, 'jobs': jobs,
        'world_size_per_job': 4, 'submitted_gpu_total': 4 * len(jobs),
        'target_prediction_tokens_per_run': 536_870_912,
        'mfu_rule': 'after five-step grace, every checked rolling ten-step median must be strictly greater than 0.70',
        'free_bytes_before_submit': free, 'storage_gate_bytes': STORAGE_GATE,
        'storage_gate_passed': free >= STORAGE_GATE,
        'artifacts': artifacts, 'capacity_snapshot': capacity,
        'training_started': False,
        'claim_boundary': 'submitted current-corpus exploratory pilots; allocation, MFU, completion, capability and superiority remain unverified',
        **extra,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
