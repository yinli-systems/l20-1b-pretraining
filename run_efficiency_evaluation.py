#!/usr/bin/env python3
"""Serial, hash-bound seven-task baseline evaluations after continuation B finishes.

Never stops training, modifies its environment/checkpoints, or trusts remote model
code. Downloaded weights are disposable only after a verified result + analysis.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/hhai/pretrain')
OUT = ROOT / 'evaluations/efficiency-20260912-v1'
CONTINUATION = ROOT / 'continuation/20260911-v1'
TASKS = ('hellaswag','piqa','winogrande','openbookqa','arc_easy','arc_challenge','boolq')
PROTOCOL = {'lm_eval_version':'0.4.9','dtype':'bfloat16','batch_size':'auto:4',
            'max_length':2048,'num_fewshot':0,'random_seed':42,'numpy_random_seed':42,
            'torch_random_seed':42,'fewshot_random_seed':1234,'limit':None,'apply_chat_template':False,
            'tasks':list(TASKS), 'bootstrap_repetitions':10000,'bootstrap_seed':20260912}
POST_B_RESERVE = 11 * 1024**3  # B HF export, result files, and at least 4 GiB slack.
REQUIRED_PROGRAMS = {'run_efficiency_evaluation.py','strict_efficiency_hf.py',
                     'analyze_comparison.py','evaluate_comparison.py','summarize_efficiency_frontier.py'}
CONVERTER_PROGRAM = 'convert_trusted_torch_checkpoint.py'


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def verify_loading_info(info):
    failures = {key: info.get(key, []) for key in
                ('missing_keys','unexpected_keys','mismatched_keys','error_msgs') if info.get(key)}
    if failures:
        raise ValueError(f'Checkpoint loading mismatch: {failures}')


def admit_B_checkpoint(plan, status, checkpoint):
    pause = plan.get('authorized_training_pause')
    if pause:
        if pause.get('reason') != 'user_requested_evaluation_priority':
            raise ValueError('Unrecognized pause authorization')
        if status.get('stage') != 'checkpointed_stop' or status.get('branch_step') != pause['step']:
            raise RuntimeError('B is not at the authorized checkpointed pause')
        if checkpoint != pause['checkpoint']:
            raise RuntimeError('Paused B checkpoint receipt changed')
    elif status.get('stage') != 'pilot_complete_pending_evaluation' or status.get('branch_step') != 190:
        raise RuntimeError('B did not finish normally; do not take over after an early stop/failure')
    if checkpoint['step'] != status['branch_step'] or sha256(CONTINUATION/'pilot-B/resume.pth') != checkpoint['sha256']:
        raise RuntimeError('B checkpoint is not verified')


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def read_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan['protocol'] != PROTOCOL:
        raise ValueError('Protocol changed')
    required = REQUIRED_PROGRAMS | ({CONVERTER_PROGRAM} if plan.get('runtime_weight_conversions') else set())
    if set(plan['program_sha256']) != required:
        raise ValueError('Incomplete program bindings')
    ids = [job['id'] for job in plan['jobs']]
    if len(ids) != len(set(ids)) or any(not re.fullmatch('[a-z0-9_]+', x) for x in ids):
        raise ValueError('Invalid or duplicate job IDs')
    for name, digest in plan['program_sha256'].items():
        if Path(name).name != name or sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError(f'Program changed: {name}')
    for job in plan['jobs']:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', job['repo']):
            raise ValueError('Invalid model repository')
        if not re.fullmatch('[0-9a-f]{40}', job['revision']):
            raise ValueError('Mutable model revision')
        names = [f['path'] for f in job['files']]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate artifact names')
        if any(Path(name).name != name or name in ('.', '..') for name in names):
            raise ValueError('Unsafe model file path')
        if job['adapter'] not in ('transformers_builtin','author_hf_olmo','author_open_lm'):
            raise ValueError('Unadmitted adapter')
        if job['adapter'] != 'transformers_builtin' and not plan.get('adapter_file_sha256',{}).get(job['adapter']):
            raise ValueError('Missing author adapter bindings')
        if job['adapter'] == 'author_open_lm' and (job['config'].get('attn_name') != 'torch_attn' or job['config'].get('ffn_type') != 'swiglu_torch'):
            raise ValueError('Unadmitted OpenLM backend')
        if not isinstance(job['training_tokens'],int) or job['training_tokens'] <= 0:
            raise ValueError('Unknown training compute axis')
    for job_id, conversion in plan.get('runtime_weight_conversions', {}).items():
        job = next((candidate for candidate in plan['jobs'] if candidate['id'] == job_id), None)
        if job is None or conversion.get('source_path') != 'pytorch_model.bin' or conversion.get('output_path') != 'model.safetensors':
            raise ValueError('Invalid runtime weight conversion target')
        source = next((item for item in job['files'] if item['path'] == conversion['source_path']), None)
        if source is None or not source.get('sha256') or conversion.get('source_sha256') != source['sha256']:
            raise ValueError('Runtime conversion source is not hash-bound')
        if conversion.get('expected_tensors') != 201 or conversion.get('expected_numel') != 1_100_048_384:
            raise ValueError('Unadmitted TinyLlama state-dict shape')
        if conversion.get('container_image') != 'python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea':
            raise ValueError('Unpinned conversion container')
    return plan


def validate_runtime(plan, adapter='transformers_builtin'):
    versions = dict(plan['package_versions'])
    if adapter == 'author_hf_olmo':
        versions.update(numpy='1.26.4',scipy='1.15.3')
    for name, expected in versions.items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f'Package version changed: {name}')
    root = Path('/home/hhai/pretrain/.venv/lib/python3.12/site-packages')
    if plan['system_source_root'] != str(root):
        raise ValueError('Unexpected source root')
    if not plan['system_source_sha256']:
        raise ValueError('Missing harness/transformers source bindings')
    for relative, expected in plan['system_source_sha256'].items():
        path = root/relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or path.is_symlink() or sha256(path) != expected:
            raise ValueError(f'Runtime source changed: {relative}')


def validate_adapter(plan, adapter):
    for relative, expected in plan.get('adapter_file_sha256',{}).get(adapter,{}).items():
        artifact_path = OUT/relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or artifact_path.is_symlink():
            raise ValueError('Unsafe adapter path')
        if sha256(artifact_path) != expected:
            raise ValueError(f'Adapter dependency changed: {relative}')


def verify_file(path, item):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != item['size']:
        raise ValueError(f'Invalid artifact size/path: {path}')
    sha = hashlib.sha256()
    git = hashlib.sha1(f'blob {item["size"]}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            sha.update(block)
            git.update(block)
    if item.get('sha256'):
        valid = sha.hexdigest() == item['sha256']
    else:
        valid = git.hexdigest() == item['git_blob_sha1']
    if not valid:
        raise ValueError(f'Artifact hash mismatch: {path}')
    return sha.hexdigest()


def require_space(directory, pending_bytes, reserve=POST_B_RESERVE):
    free = shutil.disk_usage(directory).free
    if free < pending_bytes + reserve:
        raise RuntimeError(f'Insufficient disk: {free} available, {pending_bytes+reserve} required')
    return free


def append_download_response(response, tmp, expected_size, offset):
    """Write one HTTP response while proving any resumed byte range is exact."""
    if response.status_code == 206:
        content_range = response.headers.get('Content-Range', '')
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
        if not match:
            raise ValueError('Missing or malformed Content-Range')
        start, end, total = map(int, match.groups())
        if start != offset or end < start or total != expected_size:
            raise ValueError('Unexpected resumed byte range')
        declared = response.headers.get('Content-Length')
        if declared is not None and int(declared) != end - start + 1:
            raise ValueError('Range Content-Length mismatch')
        mode = 'ab'
    elif response.status_code == 200:
        response.raise_for_status()
        offset = 0
        mode = 'wb'
        declared = response.headers.get('Content-Length')
        if declared is not None and int(declared) != expected_size:
            raise ValueError('Full-download Content-Length mismatch')
    else:
        response.raise_for_status()
        raise ValueError(f'Unexpected HTTP status {response.status_code}')
    with tmp.open(mode) as stream:
        for block in response.iter_content(4*1024**2):
            if not block:
                continue
            stream.write(block)
            if stream.tell() > expected_size:
                raise ValueError('Download exceeded declared artifact size')
        stream.flush()
        os.fsync(stream.fileno())
    return tmp.stat().st_size


def download_model(job, directory, reserve=POST_B_RESERVE):
    import requests
    directory.mkdir(parents=True, exist_ok=True)
    owner = directory / 'ownership.json'
    identity = {'job':job['id'], 'repo':job['repo'], 'revision':job['revision'], 'files':job['files']}
    if owner.exists():
        if json.loads(owner.read_text()) != identity:
            raise ValueError('Download directory ownership mismatch')
    else:
        if list(directory.iterdir()):
            raise ValueError('Unowned nonempty download directory')
        atomic_json(owner, identity)
    missing = sum(f['size'] for f in job['files'] if not (directory/f['path']).exists())
    require_space(directory, missing, reserve)
    hashes = {}
    for item in job['files']:
        path = directory / item['path']
        if not path.exists():
            tmp = directory / (item['path']+'.part')
            endpoint = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com').rstrip('/')
            if tmp.exists() and (tmp.is_symlink() or not tmp.is_file() or tmp.stat().st_size > item['size']):
                raise ValueError('Unsafe or oversized partial download')
            for attempt in range(8):
                try:
                    # A cache-busting query avoids stale CDN redirects; never log signed URLs.
                    url = f'{endpoint}/{job["repo"]}/resolve/{job["revision"]}/{item["path"]}?download=true&attempt={time.time_ns()}'
                    offset = tmp.stat().st_size if tmp.exists() else 0
                    headers = {'Range':f'bytes={offset}-'} if offset else {}
                    with requests.get(url,headers=headers,stream=True,timeout=(20,60)) as response:
                        size = append_download_response(response,tmp,item['size'],offset)
                    if size != item['size']:
                        raise EOFError(f'Incomplete artifact: {size}/{item["size"]}')
                    verify_file(tmp, item)
                    os.replace(tmp, path)
                    break
                except Exception as exc:
                    if attempt == 7:
                        raise RuntimeError(f'Download failed for {job["id"]}/{item["path"]}: {type(exc).__name__}; partial retained') from None
                    time.sleep(min(3 * (2 ** attempt), 30))
        hashes[item['path']] = verify_file(path, item)
    return hashes


def verify_converted_weights(plan, job, directory):
    spec = plan.get('runtime_weight_conversions', {}).get(job['id'])
    if not spec:
        return None
    source_item = next(item for item in job['files'] if item['path'] == spec['source_path'])
    verify_file(directory/spec['source_path'], source_item)
    receipt_path = directory/'conversion-receipt.json'
    output_path = directory/spec['output_path']
    if not receipt_path.is_file() or receipt_path.is_symlink() or not output_path.is_file() or output_path.is_symlink():
        raise FileNotFoundError('Verified converted runtime weights are missing')
    receipt = json.loads(receipt_path.read_text())
    expected = {
        'status':'exact_tensor_equality_verified', 'source_sha256':spec['source_sha256'],
        'tensor_count':spec['expected_tensors'], 'numel':spec['expected_numel'], 'dtype':'float32'}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError('Converted-weight receipt does not match plan')
    if receipt.get('output_bytes') != output_path.stat().st_size or receipt.get('output_sha256') != sha256(output_path):
        raise ValueError('Converted runtime weight hash/size mismatch')
    if receipt.get('required_launcher_boundary') != 'network=none, read-only root, cap-drop=all, no-new-privileges, pids<=64':
        raise ValueError('Conversion sandbox boundary changed')
    return {'path':spec['output_path'], 'size':receipt['output_bytes'],
            'sha256':receipt['output_sha256'], 'receipt_sha256':sha256(receipt_path)}


def prepare_converted_weights(plan, job, directory):
    spec = plan.get('runtime_weight_conversions', {}).get(job['id'])
    if not spec:
        return None
    try:
        return verify_converted_weights(plan, job, directory)
    except FileNotFoundError:
        pass
    if (directory/spec['output_path']).exists() or (directory/'conversion-receipt.json').exists():
        raise RuntimeError('Partial conversion retained; inspect before retry')
    image = spec['container_image']
    digest = subprocess.check_output(['docker','image','inspect',image,'--format','{{index .RepoDigests 0}}'],text=True).strip()
    if digest != image:
        raise RuntimeError('Conversion container digest mismatch')
    stage = OUT/'derived'/job['id']
    if stage.exists() and list(stage.iterdir()):
        raise RuntimeError('Nonempty conversion stage retained')
    stage.mkdir(parents=True,exist_ok=True)
    converter = OUT/CONVERTER_PROGRAM
    if sha256(converter) != plan['program_sha256'][CONVERTER_PROGRAM]:
        raise ValueError('Conversion program changed')
    command = ['docker','run','--rm','--network','none','--read-only','--cap-drop','ALL',
        '--security-opt','no-new-privileges','--pids-limit','64','--memory','12g','--memory-swap','12g',
        '--cpus','2','--user',f'{os.getuid()}:{os.getgid()}','--env','HOME=/tmp',
        '--env','PYTHONDONTWRITEBYTECODE=1','--env','PYTHONPATH=/site',
        '--tmpfs','/tmp:rw,noexec,nosuid,size=536870912',
        '--mount',f'type=bind,src={ROOT/".venv/lib/python3.12/site-packages"},dst=/site,readonly',
        '--mount',f'type=bind,src={directory},dst=/input,readonly',
        '--mount',f'type=bind,src={converter},dst=/convert.py,readonly',
        '--mount',f'type=bind,src={stage},dst=/output',image,'python','/convert.py',
        '--source',f'/input/{spec["source_path"]}','--source-sha256',spec['source_sha256'],
        '--output',f'/output/{spec["output_path"]}','--receipt','/output/conversion-receipt.json',
        '--expected-tensors',str(spec['expected_tensors']),'--expected-numel',str(spec['expected_numel'])]
    with (OUT/f'{job["id"]}-conversion.log').open('x') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    os.replace(stage/spec['output_path'], directory/spec['output_path'])
    os.replace(stage/'conversion-receipt.json', directory/'conversion-receipt.json')
    stage.rmdir()
    return verify_converted_weights(plan,job,directory)


def release_owned_downloads(job, directory):
    """Only discard this run's enumerated, hash-verified reproducible artifacts."""
    owner = json.loads((directory/'ownership.json').read_text())
    if owner != {'job':job['id'],'repo':job['repo'],'revision':job['revision'],'files':job['files']}:
        raise ValueError('Cannot clean unowned downloads')
    if directory.is_symlink() or directory.parent.resolve() != (OUT/'models').resolve():
        raise ValueError('Unexpected cache path')
    for item in job['files']:
        verify_file(directory/item['path'], item)
    removed = []
    for item in job['files']:
        path = directory/item['path']
        path.unlink()
        removed.append(str(path))
    return removed


def release_converted_weights(plan, job, directory):
    converted = verify_converted_weights(plan,job,directory)
    if converted is None:
        return []
    paths = [directory/converted['path'], directory/'conversion-receipt.json']
    for path in paths:
        path.unlink()
        if path.exists() or path.is_symlink():
            raise RuntimeError('Converted-weight cleanup not verified')
    return [str(path) for path in paths]


def analyze_job(result, job):
    import numpy as np
    from analyze_comparison import PRIMARY, read_verified, compare_task
    from evaluate_comparison import ORIGINALS
    original = read_verified(ROOT/'evaluations/final/zero_shot_core.json', ORIGINALS['zero_shot_core'])
    boolq = read_verified(ROOT/'evaluations/comparison-20260911/ours_boolq.json',
                          '683f549e320a3ab3f42a2e218b5ea9732641f26ca281048ce81e7a68ce0b7c4c')
    rng = np.random.default_rng(PROTOCOL['bootstrap_seed'])
    comparisons, draws = {}, []
    for task in TASKS:
        record, sample = compare_task(boolq if task == 'boolq' else original, result, task,
                                      PRIMARY[task], rng, PROTOCOL['bootstrap_repetitions'])
        comparisons[task] = record
        draws.append(sample)
    def macro(tasks):
        values = np.mean([draws[TASKS.index(task)] for task in tasks], axis=0)
        return {'ours':float(np.mean([comparisons[t]['ours'] for t in tasks])),
                'baseline':float(np.mean([comparisons[t]['baseline'] for t in tasks])),
                'ours_minus_baseline_pp':float(np.mean([comparisons[t]['difference_pp'] for t in tasks])),
                'paired_95_ci_pp':(np.quantile(values,[.025,.975])*100).tolist()}
    return {'job':job['id'], 'tasks':comparisons, 'seven_task_macro':macro(TASKS),
            'six_task_without_boolq':macro([t for t in TASKS if t != 'boolq']),
            'total_parameters':result['config']['model_num_parameters'],
            'training_tokens':job['training_tokens'], 'token_kind':job['token_kind'],
            'training_flops_proxy_6ND':6*result['config']['model_num_parameters']*job['training_tokens'],
            'category':job['category'], 'pareto_scope':'candidate-set only; not global SOTA',
            'limitations':['6ND is an approximate proxy, not measured hardware FLOPs; data/teacher/experimentation compute excluded.',
                          'Original 20B data did not explicitly decontaminate BoolQ; report the six-task sensitivity.',
                          'Paired intervals reflect benchmark sample uncertainty, not training seeds or multiple comparisons.',
                          'Tokenizer-dependent input truncation and default BOS behavior can differ.']}


def worker(plan, name):
    os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    job = next(j for j in plan['jobs'] if j['id'] == name)
    directory = OUT/'models'/name
    for item in job['files']:
        verify_file(directory/item['path'],item)
    verify_converted_weights(plan,job,directory)
    validate_runtime(plan,job['adapter'])
    validate_adapter(plan,job['adapter'])
    if job['adapter'] == 'author_hf_olmo':
        import hf_olmo  # Author package, never silently map this architecture to Llama.
    elif job['adapter'] == 'author_open_lm':
        import open_lm.hf
    elif job['adapter'] != 'transformers_builtin':
        raise ValueError('Unadmitted adapter')
    import torch
    from lm_eval import evaluator
    from strict_efficiency_hf import StrictHFLM
    if importlib.metadata.version('lm_eval') != PROTOCOL['lm_eval_version']:
        raise ValueError('Harness version mismatch')
    model = StrictHFLM(pretrained=str(directory),device='cuda',batch_size='auto:4',dtype='bfloat16',
                 max_length=2048,trust_remote_code=False)
    # Catch unavailable/wrong architecture and non-finite actual-checkpoint logits
    # before spending time on the benchmark. No sampling/selection on benchmark outcomes.
    probe = torch.tensor([[2,3,4,5]],device='cuda')
    with torch.inference_mode():
        probe_logits = model.model(input_ids=probe,use_cache=False).logits
    if not torch.isfinite(probe_logits).all():
        raise ValueError('Non-finite actual-checkpoint logits')
    del probe_logits,probe
    result = evaluator.simple_evaluate(model=model,tasks=list(TASKS),num_fewshot=0,
        batch_size='auto:4',device='cuda',limit=None,random_seed=42,numpy_random_seed=42,
        torch_random_seed=42,fewshot_random_seed=1234,log_samples=True,apply_chat_template=False)
    if not result or set(result.get('results',{})) != set(TASKS):
        raise ValueError('Incomplete seven-task results')
    result['efficiency_environment'] = {name:importlib.metadata.version(name) for name in
                                       ('torch','transformers','numpy','scipy','datasets','lm_eval')}
    result['efficiency_loading_info'] = model.efficiency_loading_info
    result_path = OUT/'results'/f'{name}.json'
    if result_path.exists():
        raise FileExistsError('Unreceipted result retained')
    atomic_json(result_path,result)
    # Verify exact same protocol and per-example alignment before accepting any score.
    from analyze_comparison import read_verified
    result = read_verified(result_path,sha256(result_path))
    summary = analyze_job(result,job)
    summary['result_sha256'] = sha256(result_path)
    summary['job_identity'] = job
    atomic_json(OUT/'results'/f'{name}-summary.json',summary)
    print(json.dumps({'stage':'evaluated','job':name,'macro':summary['seven_task_macro']}),flush=True)
    del model,result
    gc.collect()
    torch.cuda.empty_cache()


def queue(plan, plan_path):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'results').mkdir(exist_ok=True)
    (OUT/'models').mkdir(exist_ok=True)
    with (OUT/'queue.lock').open('a') as queue_lock:
        fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt_path = OUT/'receipt.json'
        identity = sha256(plan_path)
        validate_runtime(plan)
        for adapter in {job['adapter'] for job in plan['jobs']}:
            validate_adapter(plan,adapter)
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {
            'plan_sha256':identity,'created_unix':time.time(),'jobs':{},'new_training_tokens':0}
        if receipt['plan_sha256'] != identity:
            raise ValueError('Queue plan changed')
        receipt.update(status='waiting_for_B_gpu_lock',pid=os.getpid(),updated_unix=time.time())
        atomic_json(receipt_path,receipt)
        print(json.dumps({'stage':receipt['status'],'jobs':len(plan['jobs'])}),flush=True)
        # This is a normal queued GPU job, not a service that polls/restarts training.
        with (CONTINUATION/'gpu.lock').open('r+') as gpu_lock:
            fcntl.flock(gpu_lock,fcntl.LOCK_EX)
            if sha256(plan_path) != identity:
                raise ValueError('Plan changed while waiting for B')
            status = json.loads((CONTINUATION/'pilot-B/status.json').read_text())
            checkpoint = json.loads((CONTINUATION/'pilot-B/resume.json').read_text())
            admit_B_checkpoint(plan,status,checkpoint)
            active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if active:
                raise RuntimeError('Another CUDA process is active despite cooperative lock')
            receipt['B_checkpoint'] = checkpoint
            for job in plan['jobs']:
                name = job['id']
                prior = receipt['jobs'].get(name,{})
                if prior.get('status') == 'complete':
                    if sha256(OUT/'results'/f'{name}.json') != prior['result_sha256'] or sha256(OUT/'results'/f'{name}-summary.json') != prior['summary_sha256']:
                        raise ValueError('Completed result changed')
                    continue
                if prior.get('status') in ('failed','running','downloading'):
                    raise RuntimeError('Previous incomplete job requires inspection; no silent rerun')
                receipt.update(status='running',current_job=name,updated_unix=time.time())
                receipt['jobs'][name] = {'status':'downloading','started_unix':time.time()}
                atomic_json(receipt_path,receipt)
                directory = OUT/'models'/name
                try:
                    hashes = download_model(job,directory)
                    converted = prepare_converted_weights(plan,job,directory)
                    receipt['jobs'][name].update(status='running',model_file_sha256=hashes,
                                                 converted_runtime_weight=converted)
                    atomic_json(receipt_path,receipt)
                    with (OUT/f'{name}.log').open('a') as log:
                        environment = dict(os.environ)
                        environment.pop('PYTHONPATH',None)
                        if job['adapter'] == 'author_hf_olmo':
                            environment['PYTHONPATH'] = str(OUT/'vendor')
                        elif job['adapter'] == 'author_open_lm':
                            environment['PYTHONPATH'] = os.pathsep.join([str(OUT/'open-lm-author-clean'),str(OUT/'openlm-deps')])
                        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--plan',str(plan_path),'--worker',name],
                                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200,env=environment)
                    result_path = OUT/'results'/f'{name}.json'
                    summary_path = OUT/'results'/f'{name}-summary.json'
                    summary = json.loads(summary_path.read_text())
                    if summary['result_sha256'] != sha256(result_path):
                        raise ValueError('Summary/result binding failed')
                    receipt['jobs'][name].update(status='complete',completed_unix=time.time(),
                        result_sha256=sha256(result_path),summary_sha256=sha256(summary_path),
                        seven_task_macro=summary['seven_task_macro'])
                    atomic_json(receipt_path,receipt)
                    receipt['jobs'][name]['converted_runtime_weights_removed'] = release_converted_weights(plan,job,directory)
                    receipt['jobs'][name]['regenerable_downloads_removed'] = release_owned_downloads(job,directory)
                    atomic_json(receipt_path,receipt)
                    subprocess.run([sys.executable,str(Path(__file__).with_name('summarize_efficiency_frontier.py')),
                                    '--plan',str(plan_path)],check=True,timeout=300)
                except Exception as exc:
                    if receipt['jobs'][name].get('status') == 'complete':
                        receipt.update(status='postprocessing_failed',postprocessing_error=str(exc),updated_unix=time.time())
                    else:
                        receipt['jobs'][name].update(status='failed',error=str(exc),failed_unix=time.time())
                        receipt.update(status='failed',updated_unix=time.time())
                    atomic_json(receipt_path,receipt)
                    raise
            receipt.update(status='complete',completed_unix=time.time())
            receipt.pop('current_job',None)
            atomic_json(receipt_path,receipt)
            subprocess.run([sys.executable,str(Path(__file__).with_name('summarize_efficiency_frontier.py')),
                            '--plan',str(plan_path)],check=True,timeout=300)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--queue',action='store_true')
    group.add_argument('--worker')
    group.add_argument('--check',action='store_true')
    args = parser.parse_args()
    plan = read_plan(args.plan)
    if args.check:
        validate_runtime(plan)
        for adapter in {job['adapter'] for job in plan['jobs']}:
            validate_adapter(plan,adapter)
        for point in plan['reference_points']:
            for relative, digest in point['evidence_sha256'].items():
                if sha256(ROOT/relative) != digest:
                    raise ValueError('Original/reference evidence changed')
        print(json.dumps({'status':'validated_plan','jobs':len(plan['jobs']),'plan_sha256':sha256(args.plan)}))
    elif args.worker:
        worker(plan,args.worker)
    else:
        try:
            queue(plan,args.plan)
        except Exception as exc:
            path = OUT/'receipt.json'
            if path.exists():
                receipt = json.loads(path.read_text())
                if receipt.get('pid') == os.getpid():
                    receipt.update(status=('postprocessing_failed' if receipt.get('status') == 'postprocessing_failed' else 'failed'),
                                   error=str(exc),updated_unix=time.time())
                    atomic_json(path,receipt)
            raise


if __name__ == '__main__':
    main()
