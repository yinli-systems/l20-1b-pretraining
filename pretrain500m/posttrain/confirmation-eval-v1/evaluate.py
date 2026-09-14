"""Exact distributed masked-loss confirmation for frozen CPT checkpoints."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.distributed as dist

MODEL_SOURCE = Path(os.environ['P529M_MODEL_SOURCE']).resolve()
sys.path.append(str(MODEL_SOURCE))
from model import Config, LM
from mixture import MixtureReader

DOMAINS = ('general_web', 'knowledge_reading', 'math', 'code', 'multilingual')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def masked_domain_sums(logits, targets, masks, source_ids):
    losses = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction='none').view_as(targets)
    sums = torch.zeros(len(DOMAINS), dtype=torch.float64, device=logits.device)
    counts = torch.zeros_like(sums)
    for row, source in enumerate(source_ids):
        if source not in DOMAINS:
            raise ValueError(f'unknown confirmation domain: {source}')
        index = DOMAINS.index(source)
        sums[index] += losses[row][masks[row]].double().sum()
        counts[index] += masks[row].sum().double()
    return sums, counts


def summarize(sums, counts):
    if torch.any(counts <= 0):
        raise ValueError('every confirmation domain needs target tokens')
    losses = sums / counts
    equal = losses.mean().item()
    return {'loss_equal_domain': equal, 'ppl_equal_domain': math.exp(equal),
            'loss_token_weighted': (sums.sum() / counts.sum()).item(),
            'loss_by_domain': {domain: losses[i].item() for i, domain in enumerate(DOMAINS)},
            'target_tokens_by_domain': {domain: int(counts[i].item()) for i, domain in enumerate(DOMAINS)}}


def load_and_broadcast(raw, checkpoint, expected_sha, rank):
    outcome = [None]
    if rank == 0:
        try:
            actual = digest(checkpoint)
            if actual != expected_sha:
                raise ValueError(f'checkpoint digest mismatch: {checkpoint}')
            state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
            if 'model' not in state:
                raise ValueError('checkpoint has no model state')
            raw.load_state_dict(state['model'], strict=True)
            outcome[0] = {'sha256': actual}
            del state
        except Exception as exc:
            outcome[0] = {'error': f'{type(exc).__name__}: {exc}'}
    dist.broadcast_object_list(outcome, src=0)
    if 'error' in outcome[0]:
        raise RuntimeError(outcome[0]['error'])
    for parameter in raw.parameters():
        dist.broadcast(parameter.data, src=0)
    for buffer in raw.buffers():
        dist.broadcast(buffer.data, src=0)


def evaluate(raw, reader, rank, world, microbatch):
    batches = reader.total_blocks // (world * microbatch)
    if batches * world * microbatch != reader.total_blocks:
        raise ValueError('confirmation blocks do not form exact global batches')
    sums = torch.zeros(len(DOMAINS), dtype=torch.float64, device='cuda')
    counts = torch.zeros_like(sums)
    raw.eval()
    with torch.no_grad():
        for step in range(batches):
            x, y, mask, sources = reader.batch_for_step(step, rank, world, microbatch, include_loss_mask=True)
            x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True); mask = mask.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = raw(x)
            batch_sums, batch_counts = masked_domain_sums(logits, y, mask, sources)
            sums += batch_sums; counts += batch_counts
    dist.all_reduce(sums); dist.all_reduce(counts)
    return summarize(sums, counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation-manifest', type=Path, required=True)
    parser.add_argument('--expected-validation-sha256', required=True)
    parser.add_argument('--validation-tokens', type=int, required=True)
    parser.add_argument('--checkpoint', type=Path, action='append', required=True)
    parser.add_argument('--checkpoint-id', action='append', required=True)
    parser.add_argument('--checkpoint-sha256', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--microbatch', type=int, default=1)
    args = parser.parse_args()
    if not (len(args.checkpoint) == len(args.checkpoint_id) == len(args.checkpoint_sha256)):
        parser.error('checkpoint fields must have equal lengths')
    if len(set(args.checkpoint_id)) != len(args.checkpoint_id):
        parser.error('checkpoint ids must be unique')
    rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK']); world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local); dist.init_process_group('nccl', device_id=torch.device('cuda', local))
    torch.set_num_threads(4)
    verified = [None]
    if rank == 0:
        try:
            actual = digest(args.validation_manifest)
            if actual != args.expected_validation_sha256:
                raise ValueError('confirmation manifest digest mismatch')
            verified[0] = {'sha256': actual}
        except Exception as exc:
            verified[0] = {'error': f'{type(exc).__name__}: {exc}'}
    dist.broadcast_object_list(verified, src=0)
    if 'error' in verified[0]: raise RuntimeError(verified[0]['error'])
    reader = MixtureReader(args.validation_manifest, 0, args.validation_tokens,
                           args.expected_validation_sha256, verify_content=rank == 0)
    if set(reader.sources) != set(DOMAINS) or not reader.has_loss_masks:
        raise ValueError('confirmation reader identity mismatch')
    config = Config(hidden_size=1280, intermediate_size=3584, num_hidden_layers=26,
                    num_attention_heads=20, num_key_value_heads=5)
    results = []
    for checkpoint_id, checkpoint, expected_sha in zip(args.checkpoint_id, args.checkpoint, args.checkpoint_sha256):
        raw = LM(config).cuda()
        load_and_broadcast(raw, checkpoint, expected_sha, rank)
        result = evaluate(raw, reader, rank, world, args.microbatch)
        result.update({'checkpoint_id': checkpoint_id, 'checkpoint': str(checkpoint),
                       'checkpoint_sha256': expected_sha})
        results.append(result)
        del raw
        torch.cuda.empty_cache()
        dist.barrier()
    if rank == 0:
        base = next(result for result in results if result['checkpoint_id'] == 'base')
        candidates = [result for result in results if result['checkpoint_id'] != 'base']
        for result in candidates:
            result['relative_equal_domain_loss_reduction_vs_base'] = 1 - result['loss_equal_domain'] / base['loss_equal_domain']
            result['relative_loss_change_vs_base_by_domain'] = {
                domain: result['loss_by_domain'][domain] / base['loss_by_domain'][domain] - 1 for domain in DOMAINS}
        passed = (len(candidates) == 2 and all(result['loss_equal_domain'] < base['loss_equal_domain'] for result in candidates)
                  and all(max(result['relative_loss_change_vs_base_by_domain'].values()) <= 0.005 for result in candidates))
        output = {'schema_version': 1,
                  'status': 'PASS_TWO_SEED_HELDOUT_CONFIRMATION' if passed else 'FAIL_HELDOUT_CONFIRMATION',
                  'world_size': world, 'microbatch': args.microbatch,
                  'validation_manifest': str(args.validation_manifest),
                  'validation_manifest_sha256': args.expected_validation_sha256,
                  'validation_prediction_tokens': args.validation_tokens,
                  'results': results,
                  'gate': 'both F3 seeds improve equal-domain loss versus base and no domain regresses by more than 0.5%',
                  'claim_boundary': 'held-out language-model loss confirmation; not generative task accuracy or final promotion'}
        temp = args.output.with_name(args.output.name + f'.tmp.{os.getpid()}')
        temp.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n'); os.replace(temp, args.output)
        print(json.dumps({'status': output['status'], 'losses': {r['checkpoint_id']: r['loss_equal_domain'] for r in results}}, sort_keys=True))
    dist.barrier(); dist.destroy_process_group()
    if rank == 0 and not passed: raise SystemExit(66)


if __name__ == '__main__':
    main()
