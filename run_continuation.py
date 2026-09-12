"""Single-L20 continuation with a frozen parent, local schedule, and global ledger.

Preflight is forward-only. Training requires a separately reviewed corpus receipt.
Every automatic run stops at the pilot boundary; no automatic winner promotion.
"""
from __future__ import annotations

import argparse
import bisect
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time

import numpy as np

from continuation_common import (ROOT, RUN, PARENT, PARENT_TOKENS, SEQ, BLOCK, MICRO,
    ACCUM, GLOBAL, STEP_TOKENS, CAP, PILOT_STEPS, BRANCH_STEPS, MIXES, Budget,
    atomic_json, learning_rate, lock, owned, quotas, sha256)
from prepare_continuation import atomic_npy

EXPECTED_TOKENIZER = 'e755dfeeed1acab465839adc0988811983a3287db230a8d9bc0f74704208609c'
EXPECTED_HF_WEIGHTS = 'bc7c43425cf538d5dcea333fdd1513faef0a994679acc08681505a879b3cd1f3'
EXPECTED_CONFIG = '030d868464d4466007747b8f28c8fe5a43884b5aa4e28580d01b86df27b8ff8e'


def parent_identity():
    tok = ROOT / 'tokenizer/tokenizer.json'
    if sha256(tok) != EXPECTED_TOKENIZER:
        raise RuntimeError('tokenizer differs from the evaluated model')
    hf = ROOT / 'evaluations/final/hf/pytorch_model.bin'
    if sha256(hf) != EXPECTED_HF_WEIGHTS:
        raise RuntimeError('evaluated HF weights changed')
    config = PARENT.parent / 'model_config.yaml'
    if sha256(config) != EXPECTED_CONFIG:
        raise RuntimeError('native architecture changed')
    return {'checkpoint':str(PARENT),'sha256':sha256(PARENT),
            'bytes':PARENT.stat().st_size,'tokens':PARENT_TOKENS,
            'tokenizer_sha256':EXPECTED_TOKENIZER,'hf_weights_sha256':EXPECTED_HF_WEIGHTS,
            'config_sha256':EXPECTED_CONFIG}


def fixed_validation():
    from data_module import HighQualityEnglish
    directory = RUN / 'validation'
    receipt = directory/'manifest.json'
    if receipt.exists():
        data = json.loads(receipt.read_text())
        for item in data['sets'].values():
            if sha256(Path(item['path'])) != item['sha256']:
                raise RuntimeError('fixed validation changed')
        return data
    directory.mkdir(parents=True,exist_ok=True)
    module = HighQualityEnglish(data_path=ROOT/'data/packed',seed=42,num_workers=2)
    module.connect(batch_size=MICRO,max_seq_length=SEQ)
    module.prepare_data()
    batches = []
    loader = module.val_dataloader()
    for i,batch in enumerate(loader):
        if i == 100:
            break
        batches.append(batch.numpy().astype(np.uint16))
    if len(batches) != 100 or any(b.shape != (MICRO,BLOCK) for b in batches):
        raise RuntimeError('original 100-batch probe cannot be reconstructed')
    sets = {'original_mix_100':np.concatenate(batches)}
    for source in ('web','dclm','math','code'):
        m = json.loads((ROOT/f'data/full-npy/{source}/manifest.json').read_text())
        arrays = [np.load(item['path'],mmap_mode='r').reshape(-1,BLOCK) for item in m['validation_shards']]
        all_rows = np.concatenate(arrays)
        # Independently named per-source probes, not substituted into original PPL.
        indices = np.random.default_rng(4201).permutation(len(all_rows))[:96]
        if len(indices) != 96:
            raise RuntimeError(f'not enough held-out {source} sequences')
        sets[source] = all_rows[indices]
    result = {'sets':{},'source':'original val only','micro_batch':MICRO,'sequence_length':SEQ,
              'selection_seed':4201,'old_mix_reconstructed_seed':42,'created_unix':time.time()}
    for name,values in sets.items():
        path = directory/f'{name}.npy'
        atomic_npy(path,values)
        result['sets'][name] = {'path':str(path),'sha256':sha256(path),'shape':list(values.shape)}
    atomic_json(receipt,result)
    return result


def load_native(checkpoint, check_parent=False):
    import torch
    from litgpt import GPT, Config
    t = time.monotonic()
    state = torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if check_parent and (state['step_count'] != 19148 or state['iter_num'] != 1627580):
        raise RuntimeError('unexpected parent step/iteration counters')
    config = Config.from_file(PARENT.parent/'model_config.yaml')
    with torch.device('meta'):
        model = GPT(config)
    model.to_empty(device='cuda')
    model.load_state_dict(state['model'],strict=True)
    model.reset_parameters()  # Regenerate non-persistent RoPE buffers only.
    n = sum(p.numel() for p in model.parameters())
    if n != 1_100_048_384:
        raise RuntimeError(f'wrong model size: {n}')
    for key,value in model.state_dict().items():
        if not torch.equal(value,state['model'][key].to('cuda')):
            raise RuntimeError(f'weight copy mismatch: {key}')
    optimizer = torch.optim.AdamW(model.parameters(),lr=4e-5,betas=(.9,.95),weight_decay=.1,fused=True)
    optimizer.load_state_dict(state['optimizer'])
    if len(optimizer.state) != len(list(model.parameters())):
        raise RuntimeError('optimizer state incomplete')
    steps = set()
    for p,values in optimizer.state.items():
        steps.add(int(values['step'].item()))
        for name in ('exp_avg','exp_avg_sq'):
            if values[name].dtype != torch.float32 or values[name].shape != p.shape or values[name].device != p.device:
                raise RuntimeError('optimizer tensor format mismatch')
    expected_step = 19148 if check_parent else 19148+state['branch_step']
    if steps != {expected_step}:
        raise RuntimeError(f'inconsistent Adam counters: {steps}')
    for g in optimizer.param_groups:
        if tuple(g['betas']) != (.9,.95) or g['weight_decay'] != .1 or not g['fused']:
            raise RuntimeError('optimizer hyperparameters changed')
    metadata = {k:v for k,v in state.items() if k not in ('model','optimizer','train_dataloader')}
    del state
    gc.collect()
    torch.cuda.synchronize()
    print(json.dumps({'stage':'loaded_native','seconds':time.monotonic()-t,'parameters':n,
                      'adam_steps':list(steps),'weights_verified_exact':True,
                      'gpu_allocated_gib':torch.cuda.memory_allocated()/2**30}),flush=True)
    return model,optimizer,metadata


def validation(model, manifest):
    import torch
    from litgpt.utils import chunked_cross_entropy
    results = {}
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        for name,item in manifest['sets'].items():
            a = np.load(item['path'],mmap_mode='r')
            values = []
            for start in range(0,len(a),MICRO):
                batch = torch.tensor(a[start:start+MICRO].astype(np.int64),device='cuda')
                logits = model(batch[:,:SEQ].contiguous())
                loss = chunked_cross_entropy(logits,batch[:,1:].contiguous())
                values.append(loss.float())
                del logits
            mean = float(torch.stack(values).mean())
            if not math.isfinite(mean):
                raise RuntimeError('nonfinite validation loss')
            results[name] = {'loss':mean,'ppl':math.exp(mean),'prediction_tokens':len(a)*SEQ}
    model.train()
    return results


def preflight():
    import torch
    output = RUN/'preflight.json'
    identity = parent_identity()
    if output.exists():
        old = json.loads(output.read_text())
        if old['parent'] != identity or old['validation_manifest_sha256'] != sha256(RUN/'validation/manifest.json'):
            raise RuntimeError('preflight binding changed')
        print(json.dumps({'stage':'preflight_already_complete','receipt':str(output)}),flush=True)
        return
    val = fixed_validation()
    model,optimizer,_ = load_native(PARENT,check_parent=True)
    a = np.load(val['sets']['original_mix_100']['path'])[:MICRO]
    probe = torch.tensor(a[:,:SEQ].astype(np.int64),device='cuda')
    model.eval()
    with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        eager = model(probe).detach()
        compiled = torch.compile(model)
        accelerated = compiled(probe).detach()
        # This checks the same native model on actual held-out inputs, not a benchmark score.
        difference = (accelerated.float()-eager.float()).abs()
        max_difference = float(difference.max())
        rmse = float(difference.square().mean().sqrt())
        agreement = float((eager.argmax(-1)==accelerated.argmax(-1)).float().mean())
    del eager,accelerated,difference,probe
    if rmse > .05 or agreement < .98:
        raise RuntimeError(f'eager/compiled output gate failed: rmse={rmse},argmax={agreement}')
    results = validation(compiled,val)
    record = {'parent':identity,'completed_unix':time.time(),'new_training_tokens':0,
              'validation_manifest_sha256':sha256(RUN/'validation/manifest.json'),
              'validation':results,'eager_vs_compile':{'max_abs':max_difference,'rmse':rmse,'argmax_agreement':agreement},
              'optimizer_inherited':True,'native_weight_copy_exact':True,
              'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name()},
              'historical_validation_loss':2.4247145652770996,
              'historical_probe_delta':results['original_mix_100']['loss']-2.4247145652770996,
              'policy':'Compare future runs to these frozen inputs and freshly measured native baseline.'}
    atomic_json(output,record)
    print(json.dumps(record),flush=True)
    del compiled,model,optimizer


def make_plan(branch):
    """No-replacement per component, exact largest-remainder token mixture."""
    review_path = RUN / f'quality-review-{branch}.json'
    if not review_path.exists():
        raise RuntimeError('corpus quality review missing; no training authorized by this gate')
    review = json.loads(review_path.read_text())
    if review.get('decision') != 'pass' or review.get('branch') != branch:
        raise RuntimeError('corpus quality gate not passed')
    if review.get('require_program_binding'):
        for name,digest in review['program_sha256'].items():
            if sha256(Path(__file__).with_name(name)) != digest:
                raise RuntimeError(f'reviewed program changed: {name}')
    directory = owned(RUN/f'pilot-{branch}',RUN)
    directory.mkdir(parents=True,exist_ok=True)
    destination = directory/'data-plan.json'
    if destination.exists():
        plan = json.loads(destination.read_text())
        if plan['review_sha256'] != sha256(review_path) or sha256(directory/'order.npy') != plan['order_sha256']:
            raise RuntimeError('frozen training order changed')
        for component in plan['components']:
            for item in component['records']:
                if sha256(Path(item['path'])) != item['sha256']:
                    raise RuntimeError(f'resume input changed: {item["path"]}')
        return plan
    q = quotas(MIXES[branch],PILOT_STEPS*GLOBAL)
    components = []
    rows = []
    for index,(name,count) in enumerate(sorted(q.items())):
        source = name.split('_',1)[1]
        if name.startswith('replay_'):
            path = ROOT/f'data/full-npy/{source}/manifest.json'
        else:
            path = owned(Path(review.get('manifest_paths',{}).get(source,str(RUN/f'data/{source}/manifest.json'))),RUN)
        manifest = json.loads(path.read_text())
        if name.startswith('fresh_') and review['manifests'].get(source) != sha256(path):
            raise RuntimeError(f'{source}: data no longer matches quality review')
        records = manifest['train_shards'] if name.startswith('replay_') else manifest['shards']
        total = 0
        for item in records:
            if sha256(Path(item['path'])) != item['sha256']:
                raise RuntimeError(f'shard changed: {item["path"]}')
            if item['tokens'] % BLOCK:
                raise RuntimeError('unaligned shard')
            total += item['tokens']//BLOCK
        if total < count:
            raise RuntimeError(f'insufficient unique blocks for {name}: {total} < {count}')
        rng = random.Random('continuation-20260911-v1/'+name)
        selected = rng.sample(range(total),count)
        rows.extend((index,i) for i in selected)
        components.append({'name':name,'sequences':count,'records':records,
                           'manifest_path':str(path),'manifest_sha256':sha256(path)})
    random.Random(42).shuffle(rows)
    atomic_npy(directory/'order.npy',np.asarray(rows,dtype=np.int64))
    plan = {'branch':branch,'components':components,'prediction_tokens':len(rows)*SEQ,
            'order_sha256':sha256(directory/'order.npy'),'review_sha256':sha256(review_path),
            'micro_batch':MICRO,'global_batch':GLOBAL,'pilot_steps':PILOT_STEPS}
    atomic_json(destination,plan)
    return plan


class PlannedData:
    def __init__(self, plan, order_path):
        self.order = np.load(order_path,mmap_mode='r')
        self.sources = []
        for component in plan['components']:
            arrays = [np.load(item['path'],mmap_mode='r') for item in component['records']]
            sizes = np.cumsum([len(a)//BLOCK for a in arrays]).tolist()
            self.sources.append((arrays,sizes))

    def batch(self, offset):
        out = np.empty((MICRO,BLOCK),dtype=np.int64)
        for i,(source,index) in enumerate(self.order[offset:offset+MICRO]):
            arrays,sizes = self.sources[int(source)]
            shard = bisect.bisect_right(sizes,int(index))
            local = int(index) - (sizes[shard-1] if shard else 0)
            out[i] = arrays[shard][local*BLOCK:(local+1)*BLOCK]
        return out


def save_resume(path,model,optimizer,step,identity,plan_sha,spent):
    import torch
    owned(path,RUN)
    # torch.save streams individual GPU storages to host, no full CPU clone.
    expected = 3*1_100_048_384*4
    if shutil.disk_usage(path.parent).free < expected + 4*2**30:
        raise RuntimeError('insufficient free space for atomic resumable checkpoint')
    temporary = path.with_suffix('.pth.tmp')
    if temporary.exists():
        raise RuntimeError('uncommitted checkpoint temp exists; inspect before retry')
    with temporary.open('xb') as stream:
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                    'branch_step':step,'parent':identity,'data_plan_sha256':plan_sha,
                    'branch_tokens':step*STEP_TOKENS,'experiment_charged_tokens_at_save':spent,
                    'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state()},stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)
    atomic_json(path.with_suffix('.json'),{'path':str(path),'step':step,'sha256':sha256(path),
                 'bytes':path.stat().st_size,'saved_unix':time.time(),'parent_sha256':identity['sha256']})


def train(branch):
    import torch
    from litgpt.utils import chunked_cross_entropy
    pre = json.loads((RUN/'preflight.json').read_text())
    identity = parent_identity()
    if identity != pre['parent']:
        raise RuntimeError('parent does not match preflight')
    val = fixed_validation()
    if sha256(RUN/'validation/manifest.json') != pre['validation_manifest_sha256']:
        raise RuntimeError('validation manifest changed')
    plan = make_plan(branch)
    directory = RUN/f'pilot-{branch}'
    checkpoint = directory/'resume.pth'
    plan_sha = sha256(directory/'data-plan.json')
    launch = {'branch':branch,'parent':identity,'data_plan_sha256':plan_sha,
              'program_sha256':{name:sha256(Path(__file__).with_name(name)) for name in
                               ('run_continuation.py','continuation_common.py','prepare_continuation.py','data_module.py')},
              'micro_batch':MICRO,'global_batch':GLOBAL,'accumulation':ACCUM,'sequence_length':SEQ,
              'seed':42,'precision':'bf16-mixed','optimizer_state_inherited':True,
              'optimizer':{'type':'AdamW','fused':True,'betas':[.9,.95],'weight_decay':.1,'clip_norm':1.0},
              'schedule':{'branch_steps':BRANCH_STEPS,'hold_steps':172,'start_lr':4e-5,'end_lr':4e-6},
              'automatic_stop_steps':PILOT_STEPS,'experiment_budget_cap':CAP,
              'torch':torch.__version__,'cuda':torch.version.cuda,
              'validation_manifest_sha256':pre['validation_manifest_sha256']}
    launch_path = directory/'launch-manifest.json'
    if launch_path.exists() and json.loads(launch_path.read_text()) != launch:
        raise RuntimeError('launch configuration changed across resume')
    atomic_json(launch_path,launch)
    if checkpoint.exists():
        receipt = json.loads(checkpoint.with_suffix('.json').read_text())
        if sha256(checkpoint) != receipt['sha256']:
            raise RuntimeError('resume checkpoint hash mismatch')
        model,optimizer,meta = load_native(checkpoint)
        if meta['parent'] != identity or meta['data_plan_sha256'] != plan_sha:
            raise RuntimeError('resume provenance mismatch')
        start = meta['branch_step']
        torch.set_rng_state(meta['rng_cpu'])
        torch.cuda.set_rng_state(meta['rng_cuda'])
    else:
        model,optimizer,_ = load_native(PARENT,check_parent=True)
        start = 0
    data = PlannedData(plan,directory/'order.npy')
    compiled = torch.compile(model)
    compiled.train()
    optimizer.zero_grad(set_to_none=True)
    budget = Budget(RUN/'budget.sqlite')
    stop = False
    def request_stop(_sig,_frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    baseline_loss = pre['validation']['original_mix_100']['loss']
    status = {'branch':branch,'stage':'initializing_training','branch_step':start,'parent_tokens':PARENT_TOKENS,
              'experiment_charged_tokens':budget.spent,'time':time.time(),
              'note':'First compiled backward/optimizer update is not complete yet.'}
    atomic_json(directory/'status.json',status)
    print(json.dumps(status),flush=True)
    total_start = time.monotonic()
    with (directory/'metrics.jsonl').open('a',buffering=1) as log:
        for step in range(start,PILOT_STEPS):
            charge = budget.reserve(branch,step)
            t = time.monotonic()
            lr = learning_rate(step)
            for group in optimizer.param_groups:
                group['lr'] = lr
            loss_sum = torch.zeros((),device='cuda')
            for micro in range(ACCUM):
                offset = step*GLOBAL+micro*MICRO
                batch = torch.from_numpy(data.batch(offset)).pin_memory().to('cuda',non_blocking=True)
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                    logits = compiled(batch[:,:SEQ].contiguous())
                    loss = chunked_cross_entropy(logits,batch[:,1:].contiguous())
                (loss/ACCUM).backward()
                loss_sum += loss.detach()
                del logits,loss
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            mean_loss = float(loss_sum/ACCUM)
            if not math.isfinite(mean_loss):
                raise RuntimeError('nonfinite loss; optimizer not updated')
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            budget.complete(charge)
            elapsed = time.monotonic()-t
            status = {'branch':branch,'stage':'training','branch_step':step+1,'pilot_steps':PILOT_STEPS,
                      'branch_prediction_tokens':(step+1)*STEP_TOKENS,
                      'lineage_prediction_tokens':PARENT_TOKENS+(step+1)*STEP_TOKENS,
                      'experiment_charged_tokens':budget.spent,'budget_cap':CAP,
                      'loss':mean_loss,'lr':lr,'grad_norm_before_clip':float(gradient_norm),
                      'step_seconds':elapsed,'training_tokens_per_second':STEP_TOKENS/elapsed,
                      'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'time':time.time()}
            log.write(json.dumps(status)+'\n')
            print(json.dumps(status),flush=True)
            atomic_json(directory/'status.json',status)
            final = step+1 == PILOT_STEPS
            if (step+1)%50 == 0 or final or stop:
                results = validation(compiled,val)
                metric = {'stage':'validation','branch':branch,'step':step+1,'sets':results,'time':time.time()}
                log.write(json.dumps(metric)+'\n')
                print(json.dumps(metric),flush=True)
                # Retain the diagnostic artifact even if the candidate fails protection.
                save_resume(checkpoint,model,optimizer,step+1,identity,plan_sha,budget.spent)
                if results['original_mix_100']['loss'] > baseline_loss+math.log(1.01):
                    status.update(stage='stopped_preservation_gate',validation=results)
                    atomic_json(directory/'status.json',status)
                    return
                if final or stop:
                    status.update(stage='pilot_complete_pending_evaluation' if final else 'checkpointed_stop',
                                  validation=results,checkpoint=str(checkpoint),
                                  wall_seconds=time.monotonic()-total_start)
                    atomic_json(directory/'status.json',status)
                    return
    budget.close()


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('preflight','train'),required=True)
    parser.add_argument('--branch',choices=('A','B'),default='A')
    args = parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    handle = lock(RUN/'gpu.lock')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(4)
    if torch.cuda.get_device_name() != 'NVIDIA L20':
        raise RuntimeError('unexpected GPU; repeat resource preflight')
    try:
        if args.phase == 'preflight':
            preflight()
        else:
            train(args.branch)
    except BaseException as error:
        atomic_json(RUN/f'{args.phase}-{args.branch}-failure.json',
                    {'type':type(error).__name__,'message':str(error),'time':time.time(),
                     'automatic_restart':False})
        raise
    finally:
        handle.close()


if __name__ == '__main__':
    main()
