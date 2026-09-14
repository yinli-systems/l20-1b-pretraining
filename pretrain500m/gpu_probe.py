"""Synthetic throughput qualification; never reports benchmark quality."""
import argparse
import contextlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import Config,LM


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--microbatch',type=int,default=2)
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--accumulation',type=int,default=4)
    p.add_argument('--compile',action='store_true')
    p.add_argument('--deep',action='store_true')
    a=p.parse_args()
    rank=int(os.getenv('RANK','0'));world=int(os.getenv('WORLD_SIZE','1'))
    local=int(os.getenv('LOCAL_RANK','0'))
    torch.cuda.set_device(local);torch.set_num_threads(4)
    if world>1:dist.init_process_group('nccl')
    torch.manual_seed(42)
    c=Config()
    if a.deep:
        c=Config(hidden_size=1280,intermediate_size=3584,num_hidden_layers=26,
                 num_attention_heads=20,num_key_value_heads=5)
    raw=LM(c).cuda()
    params=sum(p.numel() for p in raw.parameters())
    assert params==c.parameter_count()
    opt=torch.optim.AdamW(raw.parameters(),lr=0.001,betas=(0.9,0.95),weight_decay=0.1,fused=True)
    model=torch.compile(raw) if a.compile else raw
    if world>1:model=DDP(model,device_ids=[local],gradient_as_bucket_view=True)
    torch.manual_seed(100+rank)
    x=torch.randint(0,c.vocab_size,(a.microbatch,c.max_position_embeddings),device='cuda')
    y=torch.randint(0,c.vocab_size,x.shape,device='cuda')
    if world>1:dist.barrier()
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic();times=[];losses=[]
    for step in range(a.warmup+a.steps):
        torch.cuda.synchronize();t=time.monotonic();opt.zero_grad(set_to_none=True)
        for i in range(a.accumulation):
            ctx=model.no_sync() if world>1 and i<a.accumulation-1 else contextlib.nullcontext()
            with ctx, torch.autocast('cuda',dtype=torch.bfloat16):
                loss=model(x,y)/a.accumulation
                loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(raw.parameters(),1.0,error_if_nonfinite=True)
        opt.step();torch.cuda.synchronize()
        duration=time.monotonic()-t
        assert torch.isfinite(loss)
        if step>=a.warmup:times.append(duration);losses.append(loss.item()*a.accumulation)
        if rank==0:print(json.dumps({'step':step,'seconds':duration,'loss':loss.item()*a.accumulation,'grad_norm':norm.item()}),flush=True)
    times_t=torch.tensor(times,device='cuda')
    if world>1:dist.all_reduce(times_t,op=dist.ReduceOp.MAX)
    times=times_t.cpu().tolist()
    if rank==0:
        tokens=a.microbatch*c.max_position_embeddings*a.accumulation*world
        rates=sorted(tokens/t for t in times)
        props=torch.cuda.get_device_properties(local)
        record={'evidence_class':'synthetic_throughput_only','job_id':os.getenv('SLURM_JOB_ID'),
                'hostname':platform.node(),'world_size':world,'config':c.hf_dict(),'parameters':params,
                'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':props.name,
                'capability':torch.cuda.get_device_capability(),'gpu_memory_bytes':props.total_memory,
                'compile':a.compile,'microbatch':a.microbatch,'accumulation':a.accumulation,
                'warmup_steps':a.warmup,'measurement_steps':a.steps,'seconds':times,'losses':losses,
                'tokens_per_optimizer_step':tokens,'tokens_per_second':tokens*len(times)/sum(times),
                'tokens_per_second_p10':rates[int((len(rates)-1)*.1)],
                'tokens_per_second_p50':statistics.median(rates),'tokens_per_second_p90':rates[int((len(rates)-1)*.9)],
                'total_elapsed_seconds':time.monotonic()-started,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'estimated_model_flops_per_token':raw.flops_per_token(c.max_position_embeddings),
                'dense_bf16_peak_flops_per_gpu':None,'mfu':None,
                'limitations':['No corpus I/O, validation or checkpoint overhead included in steady-state throughput.',
                               'Synthetic repeated random batch does not establish language learning or quality.',
                               'MFU stays unset until exact device precision peak is sourced.']}
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f:json.dump(record,f,indent=2)
        print(json.dumps(record),flush=True)
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
