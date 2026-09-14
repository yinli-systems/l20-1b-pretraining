"""Prospective v6 multi-node qualification and throughput runner."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import random
import signal
import shutil
import statistics
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
MODEL_SOURCE = Path(os.environ.get('P529M_MODEL_SOURCE', Path(__file__).resolve().parents[2])).resolve()
sys.path.append(str(MODEL_SOURCE))
from model import Config,LM
from mixture import MixtureReader
from resume import SCHEMA as RESUME_SCHEMA, resolve_origin, run_fingerprint as fingerprint_for, verify_resume

VALIDATION_DOMAINS=('general_web','knowledge_reading','math','code','multilingual')


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
    with tmp.open('wb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def atomic_json(value,path):
    tmp=path.with_suffix(path.suffix+'.next')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def masked_domain_sums(logits,targets,masks,source_ids):
    """Return exact token-loss sums/counts in the frozen five-domain order."""
    if logits.shape[:-1]!=targets.shape or targets.shape!=masks.shape or len(source_ids)!=targets.shape[0]:
        raise ValueError('masked validation batch shapes differ')
    losses=torch.nn.functional.cross_entropy(
        logits.float().reshape(-1,logits.shape[-1]),targets.reshape(-1),reduction='none').view_as(targets)
    sums=torch.zeros(len(VALIDATION_DOMAINS),dtype=torch.float64,device=losses.device)
    counts=torch.zeros(len(VALIDATION_DOMAINS),dtype=torch.float64,device=losses.device)
    for row,source in enumerate(source_ids):
        if source not in VALIDATION_DOMAINS:raise ValueError(f'unknown validation domain: {source}')
        index=VALIDATION_DOMAINS.index(source);mask=masks[row]
        sums[index]+=losses[row][mask].double().sum();counts[index]+=mask.sum().double()
    return sums,counts


def evaluate_masked_domains(raw,val,rank,world,microbatch):
    raw.eval();sums=torch.zeros(len(VALIDATION_DOMAINS),dtype=torch.float64,device='cuda')
    counts=torch.zeros_like(sums)
    batches=val.total_blocks//(world*microbatch)
    if batches*world*microbatch!=val.total_blocks:raise ValueError('validation blocks do not form exact global batches')
    with torch.no_grad():
        for i in range(batches):
            x,y,masks,sources=val.batch_for_step(i,rank,world,microbatch,include_loss_mask=True)
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);masks=masks.cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=raw(x)
            batch_sums,batch_counts=masked_domain_sums(logits,y,masks,sources)
            sums+=batch_sums;counts+=batch_counts
    dist.all_reduce(sums);dist.all_reduce(counts);raw.train()
    if torch.any(counts<=0):raise ValueError('every validation domain must contain target tokens')
    losses=sums/counts;equal=losses.mean();weighted=sums.sum()/counts.sum()
    return {'val_loss':equal.item(),'val_ppl':math.exp(equal.item()),
            'val_loss_objective':'equal-domain mean masked next-token loss',
            'val_loss_equal_domain':equal.item(),'val_loss_token_weighted':weighted.item(),
            'val_loss_by_domain':{domain:losses[i].item() for i,domain in enumerate(VALIDATION_DOMAINS)},
            'val_target_tokens_by_domain':{domain:int(counts[i].item()) for i,domain in enumerate(VALIDATION_DOMAINS)}}


def model_only_state(raw,step,prediction_tokens,origin,run_fingerprint,protocol_sha,training_manifest_sha):
    return {'model':{k:v.detach().cpu() for k,v in raw.state_dict().items()},'step':step,
            'prediction_tokens':prediction_tokens,'origin':origin,
            'run_fingerprint':run_fingerprint,'protocol_sha256':protocol_sha,
            'training_manifest_sha256':training_manifest_sha,
            'checkpoint_semantics':'model-only; no optimizer, reader or RNG state; not resumable'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mixture-manifest',type=Path,required=True)
    p.add_argument('--expected-mixture-sha256',required=True)
    p.add_argument('--validation-manifest',type=Path,required=True)
    p.add_argument('--expected-validation-sha256',required=True)
    p.add_argument('--validation-tokens',type=int,required=True)
    p.add_argument('--validation-microbatch',type=int)
    p.add_argument('--admission-receipt',type=Path,required=True)
    p.add_argument('--expected-admission-sha256',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--target-tokens',type=int,default=16_000_000_000)
    p.add_argument('--microbatch',type=int,default=2);p.add_argument('--accumulation',type=int,default=64)
    p.add_argument('--architecture',choices=('wide','deep'),default='wide')
    p.add_argument('--peak-lr',type=float,default=6e-4);p.add_argument('--warmup-tokens',type=int,default=104_857_600)
    p.add_argument('--save-every',type=int,default=1000);p.add_argument('--validate-every',type=int,default=250)
    p.add_argument('--seed',type=int,default=20260912);p.add_argument('--resume',action='store_true')
    p.add_argument('--init-model-checkpoint',type=Path)
    p.add_argument('--expected-init-sha256')
    p.add_argument('--dense-bf16-tflops-per-gpu',type=float,required=True)
    p.add_argument('--min-mfu',type=float,default=0.50)
    p.add_argument('--mfu-grace-steps',type=int,default=5)
    p.add_argument('--mfu-window',type=int,default=10)
    p.add_argument('--deterministic',action='store_true')
    p.add_argument('--protocol-file',type=Path,required=True)
    p.add_argument('--validate-before-training',action='store_true')
    p.add_argument('--pilot-steps',type=int,required=True)
    p.add_argument('--checkpoint-mode',choices=('full','model-only-final','none'),default='full')
    p.add_argument('--ddp-find-unused-parameters',action=argparse.BooleanOptionalAction,default=False)
    a=p.parse_args()
    validation_microbatch=a.validation_microbatch or a.microbatch
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if a.resume and a.init_model_checkpoint:
        p.error('--resume and --init-model-checkpoint are mutually exclusive')
    if a.expected_init_sha256 and not a.init_model_checkpoint:
        p.error('--expected-init-sha256 requires --init-model-checkpoint')
    if not a.resume and not (a.init_model_checkpoint and a.expected_init_sha256):
        p.error('fresh CPT requires a pinned initialization checkpoint')
    if a.checkpoint_mode != 'full' and a.resume:
        p.error('model-only and no-checkpoint runs are explicitly non-resumable')
    if not 1 <= a.pilot_steps <= 512:
        p.error('this runner supports only 1..512 qualification steps per launch')
    if min(a.microbatch,validation_microbatch,a.accumulation,a.target_tokens,a.validation_tokens,
           a.save_every,a.validate_every,a.mfu_window) <= 0:
        p.error('batch, budget, checkpoint, validation and MFU window sizes must be positive')
    if not math.isfinite(a.min_mfu) or a.min_mfu < 0.50 or a.mfu_grace_steps < 0 or not math.isfinite(a.dense_bf16_tflops_per_gpu) or a.dense_bf16_tflops_per_gpu <= 0:
        p.error('MFU must be strictly above at least 0.50 with a finite positive peak denominator')
    if not math.isfinite(a.peak_lr) or a.peak_lr <= 0 or a.warmup_tokens < 0:
        p.error('invalid learning-rate schedule')
    if digest(a.admission_receipt) != a.expected_admission_sha256:
        p.error('data admission receipt SHA-256 mismatch')
    admission=json.loads(a.admission_receipt.read_text())
    if admission.get('schema') != 'p529m-corpus-admission-v1' or admission.get('status') != 'PASS':
        p.error('a completed corpus admission receipt is required')
    for key in ('licenses','content_quality','cross_source_deduplication','benchmark_decontamination','family_disjoint_splits'):
        if admission.get('checks',{}).get(key) != 'PASS':
            p.error(f'corpus admission incomplete: {key}')
    if (admission.get('training_manifest_sha256') != a.expected_mixture_sha256 or
        admission.get('validation_manifest_sha256') != a.expected_validation_sha256 or
        admission.get('protocol_sha256') != digest(a.protocol_file)):
        p.error('admission receipt does not match the data and protocol')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    checkpoint_reference=a.output_dir/'resume.pt' if a.resume else a.init_model_checkpoint
    required_free=(2*1024**3 if a.checkpoint_mode=='none' else
                   (2*528748800*2+2*1024**3 if a.checkpoint_mode=='model-only-final'
                    else 2*checkpoint_reference.stat().st_size+2*1024**3))
    if shutil.disk_usage(a.output_dir).free < required_free:
        p.error('insufficient disk headroom for atomic checkpoint plus 2 GiB margin')
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
        # DDP synchronizes rank-0 parameters during construction. Reading the
        # multi-GiB checkpoint once avoids four identical reads on each node.
        load_status=[None]
        if rank==0:
            try:
                initial=torch.load(a.init_model_checkpoint,map_location='cpu',weights_only=False,mmap=True)
                if 'model' not in initial:
                    raise RuntimeError('initial checkpoint does not contain a model state')
                raw.load_state_dict(initial['model'],strict=True)
                del initial
                load_status[0]={'loaded_on_rank':0}
            except Exception as exc:
                load_status[0]={'error':f'{type(exc).__name__}: {exc}'}
        dist.broadcast_object_list(load_status,src=0)
        if 'error' in load_status[0]:raise RuntimeError(load_status[0]['error'])
    opt=torch.optim.AdamW(raw.parameters(),lr=a.peak_lr,betas=(0.9,0.95),eps=1e-8,weight_decay=0.1,fused=True)
    step=0;a.output_dir.mkdir(parents=True,exist_ok=True);resume_path=a.output_dir/'resume.pt'
    if not a.resume and any(a.output_dir.iterdir()):
        raise ValueError('fresh run requires an empty output directory')
    tokens_per_step=a.microbatch*a.accumulation*world*c.max_position_embeddings
    if a.target_tokens % tokens_per_step or a.validation_tokens % (world*validation_microbatch*c.max_position_embeddings):
        raise ValueError('training and validation budgets must be exactly divisible by global batches')
    total_steps=a.target_tokens//tokens_per_step
    target_tokens=a.target_tokens
    readers=[None]
    if rank==0:
        try:
            train=MixtureReader(a.mixture_manifest,a.seed,target_tokens,a.expected_mixture_sha256,
                                verify_content=True)
            val=MixtureReader(a.validation_manifest,a.seed,a.validation_tokens,a.expected_validation_sha256,
                              verify_content=True)
            readers[0]={'verified':True}
        except Exception as exc:
            readers[0]={'error':f'{type(exc).__name__}: {exc}'}
    dist.broadcast_object_list(readers,src=0)
    if 'error' in readers[0]:raise RuntimeError(readers[0]['error'])
    if rank!=0:
        train=MixtureReader(a.mixture_manifest,a.seed,target_tokens,a.expected_mixture_sha256,
                            verify_content=False)
        val=MixtureReader(a.validation_manifest,a.seed,a.validation_tokens,a.expected_validation_sha256,
                          verify_content=False)
    if train.sequence_length != c.max_position_embeddings or val.sequence_length != c.max_position_embeddings:
        raise ValueError('packed sequence length does not match the model')
    if train.has_loss_masks or not val.has_loss_masks:
        raise ValueError('training must be unmasked and validation must have complete document masks')
    if set(val.sources)!=set(VALIDATION_DOMAINS):
        raise ValueError('validation manifest must contain the frozen five domains exactly')
    train_doc=json.loads(a.mixture_manifest.read_text());val_doc=json.loads(a.validation_manifest.read_text())
    if train_doc['tokenizer_sha256'] != val_doc['tokenizer_sha256']:
        raise ValueError('training and validation tokenizer mismatch')
    train_hashes={s['sha256'] for source in train_doc['sources'] for s in source['shards']}
    val_hashes={s['sha256'] for source in val_doc['sources'] for s in source['shards']}
    if train_hashes & val_hashes:
        raise ValueError('training and validation share packed shard content')
    if any(val.quotas[sid] > source['blocks'] for sid,source in val.sources.items()):
        raise ValueError('validation cannot repeat packed blocks')
    warmup=max(1,a.warmup_tokens//tokens_per_step);decay=max(1,total_steps//10)
    data_epochs=target_tokens/train.unique_prediction_tokens
    state=None
    if a.resume:
        expected_resume=(a.output_dir/'resume.sha256').read_text().split()[0]
        if digest(resume_path) != expected_resume:
            raise ValueError('resume checkpoint SHA-256 mismatch')
        state=torch.load(resume_path,map_location='cpu',weights_only=False)
    origin=resolve_origin(initial_sha256=init_sha,checkpoint=state)
    data_manifest=a.mixture_manifest
    protocol_sha=digest(a.protocol_file) if a.protocol_file else None
    data_manifest_sha=digest(data_manifest) if data_manifest.is_file() else None
    model_source=MODEL_SOURCE
    code_hashes={'train_cpt.py':digest(Path(__file__)),
      'model.py':digest(model_source/'model.py'),
      **{name:digest(Path(__file__).parent/name) for name in ('mixture.py','resume.py')}}
    manifest={'status':'qualification_only',
      'from_scratch':False,
      'initial_checkpoint':str(a.init_model_checkpoint) if a.init_model_checkpoint else None,
      'initial_checkpoint_sha256':origin['initial_checkpoint_sha256'],
      'origin':origin,
      'parameters_total':params,'model_config':c.hf_dict(),'world_size':world,'microbatch':a.microbatch,
      'accumulation':a.accumulation,'tokens_per_step':tokens_per_step,'target_tokens':target_tokens,
      'requested_target_tokens':a.target_tokens,'optimizer':'AdamW','betas':[.9,.95],'weight_decay':.1,
      'peak_lr':a.peak_lr,'schedule':'warmup-stable-cosine-decay','warmup_steps':warmup,'decay_steps':decay,
      'seed':a.seed,'data_sequences':train.total_sequences,
      'unique_prediction_tokens':train.unique_prediction_tokens,'effective_data_epochs':data_epochs,
      'validation_sequences':val.total_sequences,
      'training_manifest':str(a.mixture_manifest),'validation_manifest':str(a.validation_manifest),
      'training_reader_fingerprint':train.fingerprint,'validation_reader_fingerprint':val.fingerprint,
      'validation_prediction_tokens':a.validation_tokens,
      'validation_microbatch':validation_microbatch,
      'validation_manifest_sha256':a.expected_validation_sha256,
      'content_verification':'rank 0 full SHA-256 before all-rank reader construction',
      'admission_sha256':a.expected_admission_sha256,
      'source_prediction_token_quotas':{k:v*c.max_position_embeddings for k,v in train.quotas.items()},
      'data_manifest':str(data_manifest) if data_manifest.is_file() else None,
      'data_manifest_sha256':data_manifest_sha,
      'protocol_file':str(a.protocol_file) if a.protocol_file else None,'protocol_sha256':protocol_sha,
      'deterministic_algorithms':a.deterministic,
      'ddp_policy':{'find_unused_parameters':a.ddp_find_unused_parameters,'static_graph':False,
                    'gradient_as_bucket_view':True,'bucket_cap_mb':25},
      'estimated_model_flops_per_token':raw.flops_per_token(c.max_position_embeddings),
      'dense_bf16_tflops_per_gpu':a.dense_bf16_tflops_per_gpu,'mfu_minimum':a.min_mfu,
      'mfu_formula':'tokens_per_second * estimated_model_flops_per_token / (world_size * dense_bf16_flops_per_gpu)',
      'runtime':{'python':platform.python_version(),'torch':str(torch.__version__),
                 'numpy':np.__version__,'cuda':torch.version.cuda,
                 'gpu_name':torch.cuda.get_device_name(local),
                 'gpu_capability':list(torch.cuda.get_device_capability(local)),
                 'cudnn':torch.backends.cudnn.version()},
      'formal_promotion':False,'checkpoint_mode':a.checkpoint_mode,
      'checkpoint_semantics':('non-resumable model weights at successful final step only'
                              if a.checkpoint_mode=='model-only-final' else
                              ('no checkpoint; metrics-only throughput benchmark'
                               if a.checkpoint_mode=='none' else 'full resumable state'))}
    fingerprint_fields={key:manifest[key] for key in (
      'parameters_total','model_config','world_size','microbatch','accumulation','tokens_per_step',
      'target_tokens','optimizer','betas','weight_decay','peak_lr','schedule','warmup_steps','decay_steps',
      'seed','data_sequences','unique_prediction_tokens','effective_data_epochs','validation_sequences',
      'data_manifest_sha256','protocol_sha256','deterministic_algorithms','ddp_policy')}
    fingerprint_fields.update({key:manifest[key] for key in (
      'training_reader_fingerprint','validation_reader_fingerprint','validation_prediction_tokens',
      'validation_microbatch','validation_manifest_sha256','admission_sha256','runtime',
      'dense_bf16_tflops_per_gpu','mfu_minimum')})
    fingerprint_fields.update({'code_sha256':code_hashes,'validate_every':a.validate_every,
                               'validate_before_training':a.validate_before_training,
                               'mfu_window':a.mfu_window,'mfu_grace_steps':a.mfu_grace_steps,
                               'checkpoint_mode':a.checkpoint_mode})
    run_fingerprint=fingerprint_for(fingerprint_fields,origin)
    manifest['run_fingerprint_sha256']=run_fingerprint;manifest['code_sha256']=code_hashes
    if state is not None:
        step=verify_resume(state,run_fingerprint,train,tokens_per_step//c.max_position_embeddings,total_steps,world)
        raw.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
    # V4 jobs observed no unused parameters on every rank. V5 defaults to the
    # DDP fast path, while retaining an explicit A/B benchmark switch.
    model=torch.compile(raw);model=DDP(model,device_ids=[local],gradient_as_bucket_view=True,
                                     find_unused_parameters=a.ddp_find_unused_parameters,
                                     static_graph=False,bucket_cap_mb=25)
    if state is not None:
        random.setstate(state['python_rng'][rank]);np.random.set_state(state['numpy_rng'][rank])
        torch.set_rng_state(state['torch_rng'][rank]);torch.cuda.set_rng_state(state['cuda_rng'][rank],local)
        del state
    if rank==0:
        atomic_json(manifest,a.output_dir/'run-manifest.json')
        launch={'unix':time.time(),'slurm_job_id':os.getenv('SLURM_JOB_ID'),'resume':a.resume,
                'step_at_launch':step,'run_fingerprint_sha256':run_fingerprint,'code_sha256':code_hashes}
        with (a.output_dir/'launches.jsonl').open('a') as f:f.write(json.dumps(launch,sort_keys=True)+'\n')
    max_steps=min(total_steps,step+a.pilot_steps) if a.pilot_steps else total_steps
    started=time.monotonic();metrics=a.output_dir/'metrics.jsonl';mfu_samples=[];run_steps=0
    if a.validate_before_training:
        validation=evaluate_masked_domains(raw,val,rank,world,validation_microbatch)
        if rank==0:
            with metrics.open('a') as f:
                f.write(json.dumps({'step':step,**validation,'phase':'before_training','unix':time.time()})+'\n')
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
                    'elapsed_seconds':now-started,'unix':time.time(),
                    'ddp_buckets_rebuilt':bool(model._has_rebuilt_buckets)}
            if model._has_rebuilt_buckets:raise RuntimeError('fixed-reduction policy unexpectedly rebuilt buckets')
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
            validation=evaluate_masked_domains(raw,val,rank,world,validation_microbatch)
            if rank==0:
                with metrics.open('a') as f:f.write(json.dumps({'step':step,**validation,'unix':time.time()})+'\n')
        should_save_full=(a.checkpoint_mode=='full' and (step%a.save_every==0 or step==max_steps or stop_requested))
        should_save_model=(a.checkpoint_mode=='model-only-final' and step==max_steps and not stop_requested)
        if should_save_full:
            rng={'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(local)}
            gathered=[None]*world if rank==0 else None;dist.gather_object(rng,gathered,dst=0)
            if rank==0:
                state={'model':raw.state_dict(),'optimizer':opt.state_dict(),'step':step,
                       'resume_schema':RESUME_SCHEMA,'origin':origin,
                       'reader_state':train.state_at(step*tokens_per_step//c.max_position_embeddings),
                       'run_fingerprint':run_fingerprint,
                       **{k:[x[k] for x in gathered] for k in rng}}
                atomic_save(state,resume_path)
                sha_path=a.output_dir/'resume.sha256';sha_tmp=sha_path.with_suffix('.next')
                sha_tmp.write_text(digest(resume_path)+'  resume.pt\n');os.replace(sha_tmp,sha_path)
            dist.barrier()
        elif should_save_model:
            if rank==0:
                model_path=a.output_dir/'model-final.pt'
                state=model_only_state(raw,step,step*tokens_per_step,origin,run_fingerprint,
                                       protocol_sha,a.expected_mixture_sha256)
                atomic_save(state,model_path)
                sha_path=a.output_dir/'model-final.sha256';sha_tmp=sha_path.with_suffix('.next')
                sha_tmp.write_text(digest(model_path)+'  model-final.pt\n');os.replace(sha_tmp,sha_path)
            dist.barrier()
        if stop_requested:
            reason_holder=[stop_reason if rank==0 else None]
            dist.broadcast_object_list(reason_holder,src=0)
            stop_reason=reason_holder[0]
            if rank==0:
                status={'status':('CHECKPOINTED_STOP' if a.checkpoint_mode=='full' else 'STOPPED_WITHOUT_RESUMABLE_CHECKPOINT'),
                        'reason':stop_reason,'step':step,'tokens':step*tokens_per_step,'unix':time.time(),
                        'checkpoint_mode':a.checkpoint_mode}
                if resume_path.is_file():status['checkpoint_sha256']=digest(resume_path)
                atomic_json(status,a.output_dir/'training-status.json');print(json.dumps(status),flush=True)
            break
    if rank==0 and not stop_requested:
        status={'status':'QUALIFICATION_COMPLETED','formal_promotion':False,
                'mfu_window_passed':len(mfu_samples)>=a.mfu_window and statistics.median(mfu_samples[-a.mfu_window:])>a.min_mfu,
                'step':step,'tokens':step*tokens_per_step,'unix':time.time()}
        model_path=a.output_dir/'model-final.pt'
        if resume_path.is_file():status['checkpoint_sha256']=digest(resume_path)
        if model_path.is_file():
            status['model_checkpoint_sha256']=digest(model_path)
            status['model_checkpoint_resumable']=False
        atomic_json(status,a.output_dir/'training-status.json');print(json.dumps(status),flush=True)
    exit_code=65 if stop_reason=='mfu_below_minimum' else (75 if stop_requested else 0)
    dist.destroy_process_group()
    if exit_code:raise SystemExit(exit_code)

if __name__=='__main__':main()
