"""Migrate the inspected v4 queue failure to resumable-download runner v5."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path('/home/hhai/pretrain')
OUT = ROOT/'evaluations/efficiency-20260912-v1'
CONT = ROOT/'continuation/20260911-v1'
OLD_PLAN = OUT/'efficiency-evaluation-plan-20260912-v4.json'
NEW_PLAN = OUT/'efficiency-evaluation-plan-20260912-v5.json'
RECEIPT = OUT/'receipt.json'
ACTIVE_RUNNER = OUT/'run_efficiency_evaluation.py'
STAGED_RUNNER = OUT/'run_efficiency_evaluation-plan-v5.staged.py'
ARCHIVE_RUNNER = OUT/'run_efficiency_evaluation-plan-v4.py'
ARCHIVE_RECEIPT = OUT/'receipt-plan-v4-failed-qc7-fw3-15k.json'
ARCHIVE_LOG = OUT/'queue-plan-v4-failed-qc7-fw3-15k.log'
MIGRATION = OUT/'plan-v4-to-v5-migration.json'
FAILED_JOB = 'dd_dclm_baseline_qc_7p_fw3_15000'
PARTIAL_NAME = 'model-00001-of-00002.safetensors.part'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict, exclusive=False) -> None:
    if exclusive and path.exists():
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(value,stream,indent=2,ensure_ascii=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        if exclusive and path.exists():
            raise FileExistsError(path)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def assert_no_writer() -> None:
    active = subprocess.check_output(
        ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True
    ).strip()
    if active:
        raise RuntimeError('CUDA process active')
    for proc in Path('/proc').iterdir():
        try:
            if not proc.name.isdigit() or int(proc.name) == os.getpid():
                continue
            command = (proc/'cmdline').read_bytes()
            if b'run_efficiency_evaluation' in command or b'run_continuation.py' in command:
                raise RuntimeError(f'Relevant writer active: {proc.name}')
        except (FileNotFoundError,PermissionError):
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-plan-sha256',required=True)
    parser.add_argument('--old-receipt-sha256',required=True)
    parser.add_argument('--old-queue-log-sha256',required=True)
    parser.add_argument('--old-runner-sha256',required=True)
    parser.add_argument('--new-runner-sha256',required=True)
    parser.add_argument('--partial-sha256',required=True)
    parser.add_argument('--partial-size',required=True,type=int)
    args = parser.parse_args()

    for path in (NEW_PLAN,ARCHIVE_RUNNER,ARCHIVE_RECEIPT,ARCHIVE_LOG,MIGRATION):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    with (OUT/'queue.lock').open('r+') as queue_lock,(CONT/'gpu.lock').open('r+') as gpu_lock:
        fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert_no_writer()
        queue_log = OUT/'queue-v4.log'
        if sha256(OLD_PLAN) != args.old_plan_sha256 or sha256(RECEIPT) != args.old_receipt_sha256:
            raise ValueError('Plan or receipt changed')
        if sha256(queue_log) != args.old_queue_log_sha256:
            raise ValueError('Queue log changed')
        if sha256(ACTIVE_RUNNER) != args.old_runner_sha256 or sha256(STAGED_RUNNER) != args.new_runner_sha256:
            raise ValueError('Runner changed')

        old_plan = json.loads(OLD_PLAN.read_text())
        old_receipt = json.loads(RECEIPT.read_text())
        failed = old_receipt.get('jobs',{}).get(FAILED_JOB,{})
        if old_receipt.get('status') != 'failed' or old_receipt.get('current_job') != FAILED_JOB:
            raise ValueError('Not the inspected failed queue')
        expected_error = f'Download failed for {FAILED_JOB}/model-00001-of-00002.safetensors: ChunkedEncodingError; partial retained'
        if failed.get('status') != 'failed' or failed.get('error') != expected_error:
            raise ValueError('Failure mode changed')
        completed = {name:value for name,value in old_receipt['jobs'].items() if value.get('status') == 'complete'}
        if len(completed) != 21:
            raise ValueError('Completed-job inventory changed')
        for name,item in completed.items():
            if sha256(OUT/'results'/f'{name}.json') != item['result_sha256'] or sha256(OUT/'results'/f'{name}-summary.json') != item['summary_sha256']:
                raise ValueError(f'Completed evidence changed: {name}')

        job = next(item for item in old_plan['jobs'] if item['id'] == FAILED_JOB)
        model_dir = OUT/'models'/FAILED_JOB
        owner = json.loads((model_dir/'ownership.json').read_text())
        identity = {'job':job['id'],'repo':job['repo'],'revision':job['revision'],'files':job['files']}
        if owner != identity:
            raise ValueError('Download ownership changed')
        partial = model_dir/PARTIAL_NAME
        if partial.is_symlink() or not partial.is_file() or partial.stat().st_size != args.partial_size or sha256(partial) != args.partial_sha256:
            raise ValueError('Partial download changed')
        expected_size = next(item['size'] for item in job['files'] if item['path']+'.part' == PARTIAL_NAME)
        if not 0 < args.partial_size < expected_size:
            raise ValueError('Partial size is not resumable')

        new_plan = copy.deepcopy(old_plan)
        new_plan['version'] = 'efficiency-20260912-v1-plan-v5'
        new_plan['created_unix'] = time.time()
        new_plan['supersedes_plan_sha256'] = args.old_plan_sha256
        new_plan['program_sha256']['run_efficiency_evaluation.py'] = args.new_runner_sha256
        new_plan.setdefault('limitations',[]).append(
            'Transport-only v5 resumes hash-bound partial downloads only after exact HTTP Content-Range validation; final declared size and SHA-256/git-blob verification remain mandatory.'
        )
        atomic_json(NEW_PLAN,new_plan,exclusive=True)

        shutil.copy2(ACTIVE_RUNNER,ARCHIVE_RUNNER)
        shutil.copy2(RECEIPT,ARCHIVE_RECEIPT)
        shutil.copy2(queue_log,ARCHIVE_LOG)
        if sha256(ARCHIVE_RUNNER) != args.old_runner_sha256 or sha256(ARCHIVE_RECEIPT) != args.old_receipt_sha256 or sha256(ARCHIVE_LOG) != args.old_queue_log_sha256:
            raise RuntimeError('Failure archive verification failed')
        os.replace(STAGED_RUNNER,ACTIVE_RUNNER)
        try:
            subprocess.run([sys.executable,str(ACTIVE_RUNNER),'--plan',str(NEW_PLAN),'--check'],check=True,timeout=600)
        except Exception:
            shutil.copy2(ARCHIVE_RUNNER,ACTIVE_RUNNER)
            raise

        new_receipt = copy.deepcopy(old_receipt)
        historical = copy.deepcopy(failed)
        historical['archived_receipt_sha256'] = args.old_receipt_sha256
        historical['partial_sha256_at_migration'] = args.partial_sha256
        historical['partial_size_at_migration'] = args.partial_size
        new_receipt.setdefault('failed_attempts',[]).append({FAILED_JOB:historical})
        new_receipt['jobs'] = completed
        new_receipt['plan_sha256'] = sha256(NEW_PLAN)
        new_receipt.update(status='migrated_ready',updated_unix=time.time())
        for key in ('current_job','error','pid'):
            new_receipt.pop(key,None)
        atomic_json(RECEIPT,new_receipt)

        migration = {
            'schema':1,'status':'verified_ready','old_plan_sha256':args.old_plan_sha256,
            'new_plan_sha256':sha256(NEW_PLAN),'old_receipt_sha256':args.old_receipt_sha256,
            'archived_receipt_sha256':sha256(ARCHIVE_RECEIPT),
            'archived_queue_log_sha256':sha256(ARCHIVE_LOG),
            'archived_runner_sha256':sha256(ARCHIVE_RUNNER),
            'new_runner_sha256':sha256(ACTIVE_RUNNER),'completed_jobs_preserved':sorted(completed),
            'failed_job_requeued':FAILED_JOB,'partial_preserved_for_resume':str(partial),
            'partial_size':args.partial_size,'partial_sha256':args.partial_sha256,
            'protocol_changed':False,'training_resumed':False,'completed_unix':time.time(),
        }
        atomic_json(MIGRATION,migration,exclusive=True)
        print(json.dumps(migration,separators=(',',':')))


if __name__ == '__main__':
    main()
