"""Bounded four-GPU reducer-recreation probe, separate from model qualification.

Synthetic FP32 inputs isolate DDP reducer lifecycle; this does not establish
full-model recovery, a data-quality result, or sustained training efficiency.
"""
import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from recovery_trace import fingerprint


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(512, 512, bias=False) for _ in range(8)])

    def forward(self, value):
        for layer in self.layers:
            value = value + torch.tanh(layer(value)) / 4
        return value.square().mean()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def segment(fixed, rank, local, start, end, checkpoint=None):
    torch.manual_seed(20260914)
    raw = Probe().cuda(local)
    opt = torch.optim.AdamW(raw.parameters(), lr=0.001, fused=True)
    if checkpoint is not None:
        raw.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['optimizer'])
    model = DDP(raw, device_ids=[local], bucket_cap_mb=1,
                find_unused_parameters=fixed, static_graph=False, gradient_as_bucket_view=True)
    traces = []
    for step in range(start, end):
        opt.zero_grad(set_to_none=True)
        for micro in range(2):
            generator = torch.Generator().manual_seed(12345 + step * 100 + rank * 2 + micro)
            x = torch.randn(64, 512, generator=generator).cuda(local)
            with model.no_sync() if micro == 0 else contextlib.nullcontext():
                loss = model(x) / 2
                loss.backward()
        grad = fingerprint({n: p.grad for n, p in raw.named_parameters()})
        opt.step()
        traces.append({'step': step + 1, 'gradient': grad,
                       'buckets_rebuilt': bool(model._has_rebuilt_buckets),
                       'bucket_sizes': model._get_ddp_logging_data().get('bucket_sizes')})
    result = cpu_tree({'model': raw.state_dict(), 'optimizer': opt.state_dict()})
    del model, raw, opt
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return result, traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', device_id=torch.device('cuda', local))
    rank = dist.get_rank()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    results = {}
    for fixed, name in ((False, 'default_rebuilding'), (True, 'fixed_buckets')):
        continuous, a = segment(fixed, rank, local, 0, 4)
        intermediate, b = segment(fixed, rank, local, 0, 2)
        resumed, c = segment(fixed, rank, local, 2, 4, intermediate)
        final_match = fingerprint(continuous) == fingerprint(resumed)
        gradients_match = [x['gradient'] == y['gradient'] for x, y in zip(a, b + c)]
        item = {'rank': rank, 'model_optimizer_exact': final_match,
                'gradient_exact_by_step': gradients_match,
                'continuous_buckets': [{k: v for k, v in x.items() if k != 'gradient'} for x in a],
                'split_buckets': [{k: v for k, v in x.items() if k != 'gradient'} for x in b + c]}
        all_ranks = [None] * dist.get_world_size()
        dist.all_gather_object(all_ranks, item)
        results[name] = all_ranks
    fixed_pass = all(x['model_optimizer_exact'] and all(x['gradient_exact_by_step'])
                     and not any(b['buckets_rebuilt'] for b in x['continuous_buckets'] + x['split_buckets'])
                     for x in results['fixed_buckets'])
    default_diverges = any(not x['model_optimizer_exact'] for x in results['default_rebuilding'])
    report = {'status': 'FIXED_BUCKET_PROBE_PASS' if fixed_pass else 'FAIL',
              'default_reducer_recreation_divergence_reproduced': default_diverges,
              'scope': 'Synthetic FP32 reducer recreation in one process per GPU; full-model process-restart qualification still required',
              'formal_quality_gain': False, 'full_model_recovery_qualified': False,
              'results': results}
    if rank == 0:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'results'}), flush=True)
    dist.destroy_process_group()
    if not fixed_pass:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
