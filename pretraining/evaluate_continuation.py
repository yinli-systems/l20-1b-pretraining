#!/usr/bin/env python3
"""Export a frozen pilot and evaluate it without altering old evidence or training.

The CUDA lock is shared with the trainer. Only forward passes are performed.
Unreceipted/changed artifacts fail closed; completed suites are hash-verified.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from continuation_common import ROOT, RUN, PARENT, PILOT_STEPS, STEP_TOKENS, atomic_json, lock, owned, sha256
from evaluate_comparison import CORE, ORIGINALS, OUR_MODEL, OUTPUT, PROTOCOL


def jobs(phase='all'):
    suites = [{'name':'core', 'tasks':list(CORE), 'num_fewshot':0}]
    if phase == 'all':
        suites += [{'name':task+'_5shot', 'tasks':[task], 'num_fewshot':5} for task in ('mmlu','gsm8k')]
    if phase not in ('all','core'):
        raise ValueError('unknown phase')
    return suites


def checkpoint_identity(branch):
    directory = RUN/f'pilot-{branch}'
    path = directory/'resume.pth'
    receipt = json.loads(path.with_suffix('.json').read_text())
    status = json.loads((directory/'status.json').read_text())
    if status['stage'] != 'pilot_complete_pending_evaluation' or receipt['step'] != PILOT_STEPS:
        raise RuntimeError('only completed, protection-passing pilots may enter this evaluation')
    if path.stat().st_size != receipt['bytes'] or sha256(path) != receipt['sha256']:
        raise RuntimeError('pilot checkpoint integrity mismatch')
    pre = json.loads((RUN/'preflight.json').read_text())
    if receipt['parent_sha256'] != pre['parent']['sha256']:
        raise RuntimeError('pilot parent differs from frozen baseline')
    return {'checkpoint_sha256':receipt['sha256'], 'parent_sha256':receipt['parent_sha256'],
            'step':receipt['step'], 'branch_prediction_tokens':PILOT_STEPS*STEP_TOKENS,
            'data_plan_sha256':sha256(directory/'data-plan.json')}


def baseline_hashes():
    hashes = {str(ROOT/'evaluations/final'/f'{name}.json'):digest for name,digest in ORIGINALS.items()}
    path = OUTPUT/'receipt.json'
    receipt = json.loads(path.read_text())
    if receipt['status'] != 'complete' or receipt['protocol'] != PROTOCOL:
        raise RuntimeError('baseline comparison protocol is not complete/frozen')
    hashes[str(path)] = sha256(path)
    for job in receipt['jobs'].values():
        if job['status'] != 'complete':
            raise RuntimeError('missing baseline evaluation')
        hashes[job['result']] = job['sha256']
    for path,digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise RuntimeError(f'baseline evidence changed: {path}')
    return hashes


def parity(checkpoint, hf):
    """Fixed 4x256-token forward-only numerical check before benchmark evaluation."""
    import numpy as np
    import torch
    from litgpt import GPT, Config
    from transformers import AutoModelForCausalLM
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(4)
    native = torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    with torch.device('meta'):
        model = GPT(Config.from_file(PARENT.parent/'model_config.yaml'))
    model.to(dtype=torch.bfloat16).to_empty(device='cuda')
    model.load_state_dict(native['model'],strict=True)
    model.reset_parameters()  # LitGPT regenerates non-persistent RoPE buffers only.
    model.eval()
    val = json.loads((RUN/'validation/manifest.json').read_text())['sets']['original_mix_100']
    if sha256(Path(val['path'])) != val['sha256']:
        raise RuntimeError('parity input changed')
    x = torch.from_numpy(np.load(val['path'],mmap_mode='r')[:4,:256].copy().astype('int64')).cuda()
    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
        reference = model(x).float()
    del model,native
    gc.collect()
    torch.cuda.empty_cache()
    exported = AutoModelForCausalLM.from_pretrained(hf,torch_dtype=torch.bfloat16,local_files_only=True).cuda().eval()
    with torch.inference_mode():
        result = exported(x).logits.float()
    rmse = float((reference-result).square().mean().sqrt())
    agreement = float((reference.argmax(-1)==result.argmax(-1)).float().mean())
    record = {'rmse':rmse,'max_abs':float((reference-result).abs().max()),
              'argmax_agreement':agreement,'tokens':int(x.numel()),'training_tokens':0,
              'thresholds':{'rmse_max':.05,'argmax_agreement_min':.98},'passed':rmse<=.05 and agreement>=.98}
    if not record['passed']:
        atomic_json(hf/'failed-parity.json',record)
        raise RuntimeError(f'export numerical parity failed: {record}')
    del exported,reference,result,x
    gc.collect()
    torch.cuda.empty_cache()
    return record


def export(branch, directory, identity):
    import torch
    from litgpt import Config
    from litgpt.scripts.convert_lit_checkpoint import copy_weights_llama
    hf = directory/'hf'
    if hf.exists():
        m = json.loads((hf/'manifest.json').read_text())
        if m['identity'] != identity or not m['parity']['passed']:
            raise RuntimeError('existing export identity/parity changed')
        for name,digest in m['files'].items():
            if sha256(hf/name) != digest:
                raise RuntimeError(f'export file changed: {name}')
        return hf
    stage = owned(directory/'hf-building')
    if stage.exists():
        raise RuntimeError('partial export exists; retain and audit before retry')
    # A second B checkpoint plus its atomic replacement and 4 GiB workspace remain.
    # The parent and A are already on disk; reserve B plus its later replacement.
    if shutil.disk_usage(directory).free < 4_500_000_000 + 2*13_200_800_000 + 4*2**30:
        raise RuntimeError('insufficient export + B atomic-checkpoint reserve')
    stage.mkdir()
    checkpoint = RUN/f'pilot-{branch}/resume.pth'
    state = torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if state['branch_step'] != identity['step'] or state['data_plan_sha256'] != identity['data_plan_sha256']:
        raise RuntimeError('native checkpoint metadata changed')
    weights = {}
    copy_weights_llama(Config.from_file(PARENT.parent/'model_config.yaml'),weights,state['model'])
    if sum(x.numel() for x in weights.values()) != 1_100_048_384:
        raise RuntimeError('export parameter count mismatch')
    with (stage/'pytorch_model.bin').open('xb') as stream:
        torch.save(weights,stream)  # Ordinary weights-only-compatible torch serialization.
        stream.flush()
        os.fsync(stream.fileno())
    loaded = torch.load(stage/'pytorch_model.bin',map_location='cpu',mmap=True,weights_only=True)
    if loaded.keys()!=weights.keys() or any(not torch.equal(weights[k],loaded[k]) for k in weights):
        raise RuntimeError('export tensor round-trip mismatch')
    del loaded,weights,state
    gc.collect()
    names = ('config.json','model_config.yaml','tokenizer.json','tokenizer_config.json','generation_config.json')
    for name in names:
        shutil.copyfile(OUR_MODEL/name,stage/name)
        if sha256(OUR_MODEL/name)!=sha256(stage/name):
            raise RuntimeError('tokenizer/config copy mismatch')
    if sha256(stage/'tokenizer.json') != json.loads((RUN/'preflight.json').read_text())['parent']['tokenizer_sha256']:
        raise RuntimeError('tokenizer differs from frozen training tokenizer')
    record = parity(checkpoint,stage)
    atomic_json(stage/'manifest.json',{'identity':identity,'created_unix':time.time(),
                'files':{p.name:sha256(p) for p in stage.iterdir() if p.is_file()},
                'parity':record,'tensor_round_trip_exact':True,
                'exporter_sha256':sha256(Path(__file__))})
    os.rename(stage,hf)
    return hf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch',choices=('A','B'),required=True)
    parser.add_argument('--phase',choices=('core','all'),default='all')
    parser.add_argument('--plan',action='store_true')
    args = parser.parse_args()
    if args.plan:
        print(json.dumps({'branch':args.branch,'protocol':PROTOCOL,'jobs':jobs(args.phase),'training_tokens':0},indent=2))
        return
    directory = owned(RUN/f'evaluation-{args.branch}-v1')
    directory.mkdir(parents=True,exist_ok=True)
    handle = lock(RUN/'gpu.lock')
    import torch
    if importlib.metadata.version('lm_eval') != PROTOCOL['lm_eval_version'] or torch.cuda.get_device_name()!='NVIDIA L20':
        raise RuntimeError('unexpected evaluation environment')
    path = directory/'receipt.json'
    identity = checkpoint_identity(args.branch)
    inputs = baseline_hashes()
    print(json.dumps({'stage':'exporting','branch':args.branch,'time':time.time()}),flush=True)
    hf = export(args.branch,directory,identity)
    identity['export_manifest_sha256'] = sha256(hf/'manifest.json')
    receipt = json.loads(path.read_text()) if path.exists() else {
        'branch':args.branch,'protocol':PROTOCOL,'identity':identity,'input_result_sha256':inputs,
        'runner_sha256':sha256(Path(__file__)),'jobs':{},'created_unix':time.time()}
    if any(receipt[k]!=v for k,v in {'identity':identity,'protocol':PROTOCOL,'input_result_sha256':inputs,
                                    'runner_sha256':sha256(Path(__file__))}.items()):
        raise RuntimeError('evaluation identity/protocol changed across restart')
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from evaluate_comparison import save_json
    for job in jobs(args.phase):
        name = job['name']
        destination = directory/f'{name}.json'
        previous = receipt['jobs'].get(name,{})
        if previous.get('status')=='complete':
            if previous['job']!=job or sha256(destination)!=previous['sha256']:
                raise RuntimeError('completed suite changed')
            continue
        if destination.exists():
            raise RuntimeError('unreceipted result will not be overwritten')
        receipt.update(status='running',current_job=name)
        receipt['jobs'][name] = {'status':'running','job':job,'started_unix':time.time()}
        atomic_json(path,receipt)
        print('START '+name,flush=True)
        try:
            model = HFLM(pretrained=str(hf),device='cuda',batch_size='auto:4',dtype='bfloat16',max_length=2048)
            result = evaluator.simple_evaluate(model=model,tasks=job['tasks'],num_fewshot=job['num_fewshot'],
                         batch_size='auto:4',device='cuda',limit=None,random_seed=42,numpy_random_seed=42,
                         torch_random_seed=42,fewshot_random_seed=1234,log_samples=True)
            if not result or not result.get('results') or not result.get('samples'):
                raise RuntimeError('incomplete result')
            save_json(destination,result)
            receipt['jobs'][name].update(status='complete',completed_unix=time.time(),
                        result=str(destination),sha256=sha256(destination),results=result['results'])
            atomic_json(path,receipt)
            del model,result
            gc.collect()
            torch.cuda.empty_cache()
            print('COMPLETE '+name,flush=True)
            analyzer = Path(__file__).with_name('analyze_continuation_evaluation.py')
            if name == 'core' and analyzer.exists():
                subprocess.run([sys.executable,str(analyzer),'--branch',args.branch,'--phase','core'],check=True)
        except Exception as error:
            receipt.update(status='failed')
            receipt['jobs'][name].update(status='failed',error=repr(error),failed_unix=time.time())
            atomic_json(path,receipt)
            raise
    receipt.update(status='complete' if args.phase=='all' else 'core_complete',completed_unix=time.time())
    receipt.pop('current_job',None)
    atomic_json(path,receipt)
    analyzer = Path(__file__).with_name('analyze_continuation_evaluation.py')
    if analyzer.exists():
        subprocess.run([sys.executable,str(analyzer),'--branch',args.branch,'--phase',args.phase],check=True)
    handle.close()
    print('FINISHED '+receipt['status'],flush=True)


if __name__=='__main__':
    main()
