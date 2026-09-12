"""Migrate one inspected failed v3 queue to sandboxed-weight-conversion v4.

This is a one-shot, fail-closed migration. It preserves completed raw results,
archives the failed receipt and old runner, and never resumes training.
"""
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
OLD_PLAN = OUT/'efficiency-evaluation-plan-20260912-v3.json'
NEW_PLAN = OUT/'efficiency-evaluation-plan-20260912-v4.json'
RECEIPT = OUT/'receipt.json'
ARCHIVE_RECEIPT = OUT/'receipt-plan-v3-failed-tiny10b.json'
ARCHIVE_RUNNER = OUT/'run_efficiency_evaluation-plan-v3.py'
STAGED_RUNNER = OUT/'run_efficiency_evaluation-plan-v4.staged.py'
MIGRATION = OUT/'plan-v3-to-v4-migration.json'
CONVERTER = OUT/'convert_trusted_torch_checkpoint.py'
CONVERSION_DIR = OUT/'conversion-tiny10b'
IMAGE = 'python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea'


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(path: Path, value: dict, exclusive=False) -> None:
    if exclusive and path.exists():
        raise FileExistsError(path)
    fd, tmp = tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(value,stream,indent=2,ensure_ascii=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        if exclusive and path.exists():
            raise FileExistsError(path)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def assert_no_worker() -> None:
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active:
        raise RuntimeError('CUDA process active')
    for proc in Path('/proc').iterdir():
        try:
            if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
            command=(proc/'cmdline').read_bytes()
            if b'run_efficiency_evaluation' in command or b'run_continuation.py' in command:
                raise RuntimeError(f'Relevant process active: {proc.name}')
        except (FileNotFoundError,PermissionError):
            continue


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-plan-sha256',required=True)
    parser.add_argument('--old-receipt-sha256',required=True)
    parser.add_argument('--conversion-receipt-sha256',required=True)
    parser.add_argument('--staged-runner-sha256',required=True)
    parser.add_argument('--converter-sha256',required=True)
    args=parser.parse_args()
    for path in (NEW_PLAN,ARCHIVE_RECEIPT,ARCHIVE_RUNNER,MIGRATION):
        if path.exists():raise FileExistsError(path)
    with (OUT/'queue.lock').open('r+') as queue_lock,(CONT/'gpu.lock').open('r+') as gpu_lock:
        fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert_no_worker()
        if sha256(OLD_PLAN)!=args.old_plan_sha256 or sha256(RECEIPT)!=args.old_receipt_sha256:
            raise ValueError('Old plan/receipt changed')
        if sha256(STAGED_RUNNER)!=args.staged_runner_sha256 or sha256(CONVERTER)!=args.converter_sha256:
            raise ValueError('Staged program changed')
        old_plan=json.loads(OLD_PLAN.read_text());old_receipt=json.loads(RECEIPT.read_text())
        if old_receipt.get('status')!='failed' or old_receipt.get('current_job')!='tinyllama_early_10b' or old_receipt['jobs']['tinyllama_early_10b'].get('status')!='failed':
            raise ValueError('Not the inspected v3 TinyLlama failure')
        completed={name:value for name,value in old_receipt['jobs'].items() if value.get('status')=='complete'}
        if len(completed)!=4 or (OUT/'results/tinyllama_early_10b.json').exists() or (OUT/'results/tinyllama_early_10b-summary.json').exists():
            raise ValueError('Unexpected completed/failed result inventory')
        for name,item in completed.items():
            if sha256(OUT/'results'/f'{name}.json')!=item['result_sha256'] or sha256(OUT/'results'/f'{name}-summary.json')!=item['summary_sha256']:
                raise ValueError(f'Completed evidence changed: {name}')
        active_runner=OUT/'run_efficiency_evaluation.py'
        for name,digest in old_plan['program_sha256'].items():
            if sha256(OUT/name)!=digest:raise ValueError(f'Old program changed: {name}')
        conversion_receipt=CONVERSION_DIR/'conversion-receipt.json';converted=CONVERSION_DIR/'model.safetensors'
        if sha256(conversion_receipt)!=args.conversion_receipt_sha256:
            raise ValueError('Conversion receipt changed')
        converted_info=json.loads(conversion_receipt.read_text())
        if converted_info.get('status')!='exact_tensor_equality_verified' or converted_info.get('output_sha256')!=sha256(converted):
            raise ValueError('Converted artifact not verified')
        failed_job=next(job for job in old_plan['jobs'] if job['id']=='tinyllama_early_10b')
        source=next(item for item in failed_job['files'] if item['path']=='pytorch_model.bin')
        model_dir=OUT/'models/tinyllama_early_10b'
        if source['sha256']!=converted_info['source_sha256'] or sha256(model_dir/'pytorch_model.bin')!=source['sha256']:
            raise ValueError('Conversion source does not match frozen job')
        owner=json.loads((model_dir/'ownership.json').read_text())
        if owner!={'job':failed_job['id'],'repo':failed_job['repo'],'revision':failed_job['revision'],'files':failed_job['files']}:
            raise ValueError('Failed download ownership changed')

        new_plan=copy.deepcopy(old_plan);new_plan['version']='efficiency-20260912-v1-plan-v4'
        new_plan['created_unix']=time.time();new_plan['supersedes_plan_sha256']=args.old_plan_sha256
        new_plan['program_sha256']['run_efficiency_evaluation.py']=args.staged_runner_sha256
        new_plan['program_sha256']['convert_trusted_torch_checkpoint.py']=args.converter_sha256
        conversions={}
        for job in new_plan['jobs']:
            source_item=next((item for item in job['files'] if item['path']=='pytorch_model.bin'),None)
            if job['repo']=='TinyLlama/tinyLlama-intermediate-checkpoints' and source_item:
                conversions[job['id']]={'source_path':'pytorch_model.bin','source_sha256':source_item['sha256'],
                    'output_path':'model.safetensors','expected_tensors':201,'expected_numel':1_100_048_384,
                    'container_image':IMAGE}
        new_plan['runtime_weight_conversions']=conversions
        new_plan.setdefault('limitations',[]).append('Hash-pinned TinyLlama protocol-5 state dicts are statically audited and converted in a networkless read-only-root container; exact tensor equality is required and receipted before evaluation.')
        atomic_json(NEW_PLAN,new_plan,exclusive=True)

        shutil.copy2(active_runner,ARCHIVE_RUNNER)
        if sha256(ARCHIVE_RUNNER)!=old_plan['program_sha256']['run_efficiency_evaluation.py']:
            raise ValueError('Old runner archive failed')
        os.replace(STAGED_RUNNER,active_runner)
        try:
            subprocess.run([sys.executable,str(active_runner),'--plan',str(NEW_PLAN),'--check'],check=True,timeout=600)
        except Exception:
            shutil.copy2(ARCHIVE_RUNNER,active_runner)
            raise
        os.replace(converted,model_dir/'model.safetensors')
        os.replace(conversion_receipt,model_dir/'conversion-receipt.json')

        shutil.copy2(RECEIPT,ARCHIVE_RECEIPT)
        if sha256(ARCHIVE_RECEIPT)!=args.old_receipt_sha256:
            raise ValueError('Failed receipt archive mismatch')
        new_receipt={'plan_sha256':sha256(NEW_PLAN),'created_unix':time.time(),
            'jobs':completed,'new_training_tokens':0,'status':'migrated_ready',
            'B_checkpoint':old_receipt.get('B_checkpoint'),
            'imported_completed_jobs':sorted(completed),
            'superseded_receipt_sha256':args.old_receipt_sha256,
            'failure_preserved_in':str(ARCHIVE_RECEIPT)}
        atomic_json(RECEIPT,new_receipt)
        migration={'schema':1,'status':'verified_ready','old_plan_sha256':args.old_plan_sha256,
            'new_plan':str(NEW_PLAN),'new_plan_sha256':sha256(NEW_PLAN),
            'old_receipt_sha256':args.old_receipt_sha256,
            'archived_receipt_sha256':sha256(ARCHIVE_RECEIPT),
            'archived_runner_sha256':sha256(ARCHIVE_RUNNER),
            'new_runner_sha256':sha256(active_runner),'converter_sha256':sha256(CONVERTER),
            'conversion_receipt_sha256':sha256(model_dir/'conversion-receipt.json'),
            'converted_weight_sha256':sha256(model_dir/'model.safetensors'),
            'completed_jobs_preserved':sorted(completed),'failed_job_requeued':'tinyllama_early_10b',
            'training_resumed':False,'completed_unix':time.time()}
        atomic_json(MIGRATION,migration,exclusive=True)
        print(json.dumps(migration,separators=(',',':')))


if __name__=='__main__':main()
