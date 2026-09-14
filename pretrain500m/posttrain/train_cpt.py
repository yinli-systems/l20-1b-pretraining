"""DDP BF16 pretraining with atomic complete-state resume and dated receipts."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from data import PackedReader
from model import Config,LM


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()


def lr_for_step(step,total,peak,warmup,decay):
    if step<warmup:return peak*(step+1)/warmup
    if step<total-decay:return peak
    progress=(step-(total-decay))/decay
    return peak*0.5*(1+math.cos(math.pi*min(1,progress)))


def atomic_save(value,path):
    tmp=path.with_suffix(path.suffix+'.next')
    torch.save(value,tmp);os.replace(tmp,path)


def atomic_json(value,path):
    tmp=path.with_suffix(path.suffix+'.next')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--val-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--target-tokens',type=int,default=16_000_000_000)
    p.add_argument('--microbatch',type=int,default=2);p.add_argument('--accumulation',type=int,default=64)
    p.add_argument('--architecture',choices=('wide','deep'),default='wide')
    p.add_argument('--peak-lr',type=float,default=6e-4);p.add_argument('--warmup-tokens',type=int,default=104_857_600)
    p.add_argument('--save-every',type=int,default=1000);p.add_argument('--validate-every',type=int,default=250)
    p.add_argument('--seed',type=int,default=20260912);p.add_argument('--resume',action='store_true')
    p.add_argument('--init-model-checkpoint',type=Path)
    p.add_argument('--expected-init-sha256')
    p.add_argument('--max-data-epochs',type=float,default=4.0)
    p.add_argument('--dense-bf16-tflops-per-gpu',type=float,required=True)
    p.add_argument('--min-mfu',type=float,default=0.50)
    p.add_argument('--mfu-grace-steps',type=int,default=5)
    p.add_argument('--mfu-window',type=int,default=10)
    p.add_argument('--deterministic',action='store_true')
    p.add_argument('--protocol-file',type=Path)
    p.add_argument('--validate-before-training',action='store_true')
    p.add_argument('--no-save-final',action='store_true')
    p.add_argument('--pilot-steps',type=int);a=p.parse_args()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if a.resume and a.init_model_checkpoint:
        p.error('--resume and --init-model-checkpoint are mutually exclusive')
    if a.expected_init_sha256 and not a.init_model_checkpoint:
        p.error('--expected-init-sha256 requires --init-model-checkpoint')
    torch.cuda.set_device(local);dist.init_process_group('nccl',device_id=torch.device('cuda',local))
    if a.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(4);torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed)
    c=Config() if a.architecture=='wide' else Config(hidden_size=1280,intermediate_size=3584,
      num_hidden_layers=26,num_attention_heads=20,num_key_value_heads=5)
    raw=LM(c).cuda();params=sum(x.numel() for x in raw.parameters());assert params==c.parameter_count()
    init_sha=None
    if a.init_model_checkpoint:
        verification=[None]
        if rank==0:
            try:
                init_sha=digest(a.init_model_checkpoint)
                if a.expected_init_sha256 and init_sha!=a.expected_init_sha256:
                    raise RuntimeError(
                        f'initial checkpoint sha256 mismatch: expected {a.expected_init_sha256}, got {init_sha}'
                    )
                verification[0]={'sha256':init_sha}
            except Exception as exc:
                verification[0]={'error':f'{type(exc).__name__}: {exc}'}
        dist.broadcast_object_list(verification,src=0)
        if 'error' in verification[0]:
            raise RuntimeError(verification[0]['error'])
        init_sha=verification[0]['sha256']
        initial=torch.load(a.init_model_checkpoint,map_location='cpu',weights_only=False,mmap=True)
        if 'model' not in initial:
            raise RuntimeError('initial checkpoint does not contain a model state')
        raw.load_state_dict(initial['model'],strict=True)
        del initial
    opt=torch.optim.AdamW(raw.parameters(),lr=a.peak_lr,betas=(0.9,0.95),eps=1e-8,weight_decay=0.1,fused=True)
    step=0;a.output_dir.mkdir(parents=True,exist_ok=True);resume_path=a.output_dir/'resume.pt'
    train=PackedReader(a.data_dir,a.seed,repeat=True);val=PackedReader(a.val_dir,a.seed)
    tokens_per_step=a.microbatch*a.accumulation*world*c.max_position_embeddings
    total_steps=a.target_tokens//tokens_per_step
    target_tokens=total_steps*tokens_per_step
    warmup=max(1,a.warmup_tokens//tokens_per_step);decay=max(1,total_steps//10)
    data_epochs=target_tokens/train.unique_prediction_tokens
    if data_epochs>a.max_data_epochs:
        raise ValueError(f'target requires {data_epochs:.6f} data epochs, above limit {a.max_data_epochs}')
    state=None
    if a.resume:
        state=torch.load(resume_path,map_location='cpu',weights_only=False)
        raw.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);step=state['step']
        random.setstate(state['python_rng'][rank]);np.random.set_state(state['numpy_rng'][rank])
        torch.set_rng_state(state['torch_rng'][rank]);torch.cuda.set_rng_state(state['cuda_rng'][rank],local)
    model=torch.compile(raw);model=DDP(model,device_ids=[local],gradient_as_bucket_view=True)
    data_manifest=a.data_dir.parent/'manifest.json'
    protocol_sha=digest(a.protocol_file) if a.protocol_file else None
    data_manifest_sha=digest(data_manifest) if data_manifest.is_file() else None
    model_source=Path(os.environ.get('P529M_MODEL_SOURCE',Path(__file__).parent.parent))
    code_hashes={'train_cpt.py':digest(Path(__file__)),
      **{name:digest(model_source/name) for name in ('model.py','data.py')}}
    manifest={'status':'pilot' if a.pilot_steps else 'formal',
      'from_scratch':a.init_model_checkpoint is None,
      'initial_checkpoint':str(a.init_model_checkpoint) if a.init_model_checkpoint else None,
      'initial_checkpoint_sha256':init_sha,
      'parameters_total':params,'model_config':c.hf_dict(),'world_size':world,'microbatch':a.microbatch,
      'accumulation':a.accumulation,'tokens_per_step':tokens_per_step,'target_tokens':target_tokens,
      'requested_target_tokens':a.target_tokens,'optimizer':'AdamW','betas':[.9,.95],'weight_decay':.1,
      'peak_lr':a.peak_lr,'schedule':'warmup-stable-cosine-decay','warmup_steps':warmup,'decay_steps':decay,
      'seed':a.seed,'data_sequences':train.total_sequences,
      'unique_prediction_tokens':train.unique_prediction_tokens,'effective_data_epochs':data_epochs,
      'validation_sequences':val.total_sequences,
      'data_directory':str(a.data_dir),'validation_directory':str(a.val_dir),
      'data_manifest':str(data_manifest) if data_manifest.is_file() else None,
      'data_manifest_sha256':data_manifest_sha,
      'protocol_file':str(a.protocol_file) if a.protocol_file else None,'protocol_sha256':protocol_sha,
      'deterministic_algorithms':a.deterministic,
      'estimated_model_flops_per_token':raw.flops_per_token(c.max_position_embeddings),
      'dense_bf16_tflops_per_gpu':a.dense_bf16_tflops_per_gpu,'mfu_minimum':a.min_mfu,
      'mfu_formula':'tokens_per_second * estimated_model_flops_per_token / (world_size * dense_bf16_flops_per_gpu)',
      'deadline_utc':'2026-09-13T22:20:31Z'}
    fingerprint_fields={key:manifest[key] for key in (
      'parameters_total','model_config','world_size','microbatch','accumulation','tokens_per_step',
      'target_tokens','optimizer','betas','weight_decay','peak_lr','schedule','warmup_steps','decay_steps',
      'seed','data_sequences','unique_prediction_tokens','effective_data_epochs','validation_sequences',
      'data_manifest_sha256','protocol_sha256','deterministic_algorithms')}
    if a.init_model_checkpoint:
        fingerprint_fields['initial_checkpoint_sha256']=init_sha
    run_fingerprint=hashlib.sha256(json.dumps(fingerprint_fields,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    manifest['run_fingerprint_sha256']=run_fingerprint;manifest['code_sha256']=code_hashes
    if state is not None and state.get('run_fingerprint')!=run_fingerprint:
        raise RuntimeError('resume checkpoint run fingerprint does not match the current frozen run')
    if rank==0:
        atomic_json(manifest,a.output_dir/'run-manifest.json')
        launch={'unix':time.time(),'slurm_job_id':os.getenv('SLURM_JOB_ID'),'resume':a.resume,
                'step_at_launch':step,'run_fingerprint_sha256':run_fingerprint,'code_sha256':code_hashes}
        with (a.output_dir/'launches.jsonl').open('a') as f:f.write(json.dumps(launch,sort_keys=True)+'\n')
    max_steps=min(total_steps,step+a.pilot_steps) if a.pilot_steps else total_steps
    started=time.monotonic();metrics=a.output_dir/'metrics.jsonl';mfu_samples=[];run_steps=0
    if a.validate_before_training:
        raw.eval();values=[]
        with torch.no_grad():
            for i in range(4):
                x,y=val.batch_for_step(i,rank,world,a.microbatch);x=x.cuda();y=y.cuda()
                with torch.autocast('cuda',dtype=torch.bfloat16):values.append(raw(x,y).float())
        value=torch.stack(values).mean();dist.all_reduce(value);value/=world;raw.train()
        if rank==0:
            with metrics.open('a') as f:
                f.write(json.dumps({'step':step,'val_loss':value.item(),
                    'val_ppl':math.exp(value.item()),'phase':'before_training','unix':time.time()})+'\n')
    stop_requested=False;stop_reason=None
    def request_stop(_signum,_frame):
        nonlocal stop_requested,stop_reason
        stop_requested=True;stop_reason='signal'
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    while step<max_steps:
        run_steps+=1
        step_started=time.monotonic()
        lr=lr_for_step(step,total_steps,a.peak_lr,warmup,decay)
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);loss_sum=0.0
        for accumulation in range(a.accumulation):
            x,y=train.batch_for_step(step*a.accumulation+accumulation,rank,world,a.microbatch)
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
            ctx=model.no_sync() if accumulation<a.accumulation-1 else contextlib.nullcontext()
            with ctx,torch.autocast('cuda',dtype=torch.bfloat16):loss=model(x,y)/a.accumulation
            loss.backward();loss_sum+=loss.detach().float()
        norm=torch.nn.utils.clip_grad_norm_(raw.parameters(),1.0,error_if_nonfinite=True);opt.step();step+=1
        loss_tensor=loss_sum.clone();dist.all_reduce(loss_tensor);loss_value=loss_tensor.item()/world
        now=time.monotonic();duration=now-step_started
        rate=tokens_per_step/duration
        mfu=rate*raw.flops_per_token(c.max_position_embeddings)/(world*a.dense_bf16_tflops_per_gpu*1e12)
        if rank==0:
            record={'step':step,'prediction_tokens':step*tokens_per_step,'loss':loss_value,'lr':lr,
                    'grad_norm':norm.item(),'step_seconds':duration,'tokens_per_second':rate,'mfu':mfu,
                    'elapsed_seconds':now-started,'unix':time.time()}
            with metrics.open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            if run_steps>a.mfu_grace_steps:
                mfu_samples.append(mfu)
                if len(mfu_samples)>=a.mfu_window and statistics.median(mfu_samples[-a.mfu_window:])<=a.min_mfu:
                    stop_requested=True;stop_reason='mfu_below_minimum'
        stop_flag=torch.tensor(int(stop_requested),device='cuda');dist.all_reduce(stop_flag,op=dist.ReduceOp.MAX)
        stop_requested=bool(stop_flag.item())
        if rank==0 and stop_requested and stop_reason is None:stop_reason='peer_signal'
        if step%a.validate_every==0 or step==max_steps:
            raw.eval();values=[]
            with torch.no_grad():
                for i in range(4):
                    x,y=val.batch_for_step(i,rank,world,a.microbatch);x=x.cuda();y=y.cuda()
                    with torch.autocast('cuda',dtype=torch.bfloat16):values.append(raw(x,y).float())
            value=torch.stack(values).mean();dist.all_reduce(value);value/=world;raw.train()
            if rank==0:
                with metrics.open('a') as f:f.write(json.dumps({'step':step,'val_loss':value.item(),'val_ppl':math.exp(value.item()),'unix':time.time()})+'\n')
        if step%a.save_every==0 or (step==max_steps and not a.no_save_final) or stop_requested:
            rng={'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(local)}
            gathered=[None]*world if rank==0 else None;dist.gather_object(rng,gathered,dst=0)
            if rank==0:
                state={'model':raw.state_dict(),'optimizer':opt.state_dict(),'step':step,
                       'run_fingerprint':run_fingerprint,
                       **{k:[x[k] for x in gathered] for k in rng}}
                atomic_save(state,resume_path)
                (a.output_dir/'resume.sha256').write_text(digest(resume_path)+'  resume.pt\n')
            dist.barrier()
        if stop_requested:
            reason_holder=[stop_reason if rank==0 else None]
            dist.broadcast_object_list(reason_holder,src=0)
            stop_reason=reason_holder[0]
            if rank==0:
                status={'status':'CHECKPOINTED_STOP','reason':stop_reason,'step':step,
                        'tokens':step*tokens_per_step,'checkpoint_sha256':digest(resume_path),'unix':time.time()}
                atomic_json(status,a.output_dir/'training-status.json');print(json.dumps(status),flush=True)
            break
    if rank==0 and not stop_requested:
        status={'status':'COMPLETED','step':step,'tokens':step*tokens_per_step,'unix':time.time()}
        if resume_path.is_file():status['checkpoint_sha256']=digest(resume_path)
        atomic_json(status,a.output_dir/'training-status.json');print(json.dumps(status),flush=True)
    exit_code=65 if stop_reason=='mfu_below_minimum' else (75 if stop_requested else 0)
    dist.destroy_process_group()
    if exit_code:raise SystemExit(exit_code)

if __name__=='__main__':main()
