#!/usr/bin/env python3
"""Produce a fail-closed final receipt for the frozen efficiency evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

TASKS = {'hellaswag','piqa','winogrande','openbookqa','arc_easy','arc_challenge','boolq'}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(value,stream,indent=2,ensure_ascii=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir',type=Path,required=True)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()

    root = args.evaluation_dir.resolve()
    plan_path = root/args.plan
    output = root/args.output
    receipt_path = root/'receipt.json'
    frontier_path = root/'frontier-summary.json'
    plan = json.loads(plan_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    frontier = json.loads(frontier_path.read_text())
    plan_hash = sha256(plan_path)
    jobs = {job['id']:job for job in plan['jobs']}
    if len(jobs) != 36 or len(plan['jobs']) != 36:
        raise ValueError('Expected exactly 36 unique declared jobs')
    if receipt.get('status') != 'complete' or receipt.get('plan_sha256') != plan_hash:
        raise ValueError('Queue is not complete under the selected plan')
    if receipt.get('new_training_tokens') != 0 or set(receipt.get('jobs',{})) != set(jobs):
        raise ValueError('Receipt inventory or training-token boundary changed')
    if frontier.get('status') != 'declared_subset_complete' or frontier.get('plan_sha256') != plan_hash:
        raise ValueError('Frontier is not complete under the selected plan')
    if len(frontier.get('points',[])) != len(jobs) + len(plan.get('reference_points',[])):
        raise ValueError('Frontier point count mismatch')

    result_bindings = {}
    for name, job in jobs.items():
        item = receipt['jobs'][name]
        if item.get('status') != 'complete' or item.get('error'):
            raise ValueError(f'Job not complete: {name}')
        raw_path = root/'results'/f'{name}.json'
        summary_path = root/'results'/f'{name}-summary.json'
        raw_hash = sha256(raw_path)
        summary_hash = sha256(summary_path)
        if raw_hash != item.get('result_sha256') or summary_hash != item.get('summary_sha256'):
            raise ValueError(f'Result binding mismatch: {name}')
        summary = json.loads(summary_path.read_text())
        if summary.get('job') != name or summary.get('result_sha256') != raw_hash:
            raise ValueError(f'Summary identity mismatch: {name}')
        if summary.get('training_tokens') != job['training_tokens'] or set(summary.get('tasks',{})) != TASKS:
            raise ValueError(f'Summary protocol mismatch: {name}')
        if any(task.get('protocol_alignment') != 'PASS' for task in summary['tasks'].values()):
            raise ValueError(f'Per-example protocol alignment failed: {name}')
        result_bindings[name] = {'result_sha256':raw_hash,'summary_sha256':summary_hash}

        model_dir = root/'models'/name
        entries = list(model_dir.iterdir())
        if len(entries) != 1 or entries[0].name != 'ownership.json' or entries[0].is_symlink():
            raise ValueError(f'Rebuildable model cleanup incomplete: {name}')
        owner = json.loads(entries[0].read_text())
        expected_owner = {'job':job['id'],'repo':job['repo'],'revision':job['revision'],'files':job['files']}
        if owner != expected_owner:
            raise ValueError(f'Model ownership receipt mismatch: {name}')

    continuation = root.parent.parent/'continuation/20260911-v1'
    b_status = json.loads((continuation/'pilot-B/status.json').read_text())
    if b_status.get('stage') != 'checkpointed_stop' or b_status.get('branch_step') != 61:
        raise ValueError('Continuation B pause boundary changed')

    final = {
        'schema':1,'status':'verified_complete','verified_unix':time.time(),
        'plan':str(plan_path),'plan_sha256':plan_hash,'queue_receipt_sha256':sha256(receipt_path),
        'frontier_summary_sha256':sha256(frontier_path),'frontier_svg_sha256':sha256(root/'frontier.svg'),
        'declared_jobs':len(jobs),'complete_jobs':len(result_bindings),'noncomplete_jobs':0,
        'raw_results':len(list((root/'results').glob('*.json'))) - len(list((root/'results').glob('*-summary.json'))),
        'summaries':len(list((root/'results').glob('*-summary.json'))),
        'frontier_points':len(frontier['points']),'new_training_tokens':receipt['new_training_tokens'],
        'historical_failed_attempts':len(receipt.get('failed_attempts',[])),
        'continuation_B':{'stage':b_status['stage'],'branch_step':b_status['branch_step']},
        'rebuildable_model_payloads_removed':True,'result_bindings':result_bindings,
        'limitations':[
            'Verification covers the frozen declared subset, not a global census of all models.',
            'Paired confidence intervals measure benchmark-item uncertainty, not training-seed variance.',
            'The original corpus did not explicitly decontaminate BoolQ; six-task sensitivity remains required.',
        ],
    }
    if final['raw_results'] != 36 or final['summaries'] != 36:
        raise ValueError('Final result file count mismatch')
    atomic_json(output,final)
    print(json.dumps({'status':final['status'],'output':str(output),'sha256':sha256(output),
                      'complete_jobs':final['complete_jobs'],'frontier_points':final['frontier_points']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
