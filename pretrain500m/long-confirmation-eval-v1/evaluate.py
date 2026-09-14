"""Exact distributed masked-loss evaluation for four frozen long CPT runs."""
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
RECIPES = ('F2_reasoning_no_synthetic', 'F3_broad_multilingual_no_synthetic')
SEEDS = (20260914, 20260915)


def expected_checkpoint_ids():
    return {'base', *(f'{recipe}_seed{seed}' for recipe in RECIPES for seed in SEEDS)}


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
    parser.add_argument('--protocol-file', type=Path, required=True)
    parser.add_argument('--expected-protocol-sha256', required=True)
    parser.add_argument('--checkpoint-plan', type=Path, required=True)
    parser.add_argument('--expected-checkpoint-plan-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--microbatch', type=int, default=1)
    args = parser.parse_args()
    if digest(args.checkpoint_plan) != args.expected_checkpoint_plan_sha256:
        parser.error('checkpoint plan SHA-256 mismatch')
    checkpoint_plan = json.loads(args.checkpoint_plan.read_text())
    if (checkpoint_plan.get('schema') != 'p529m-long-confirmation-checkpoint-plan-v1'
            or checkpoint_plan.get('status') != 'PASS_EXACT_FOUR_RUN_CHECKPOINT_PLAN'
            or checkpoint_plan.get('protocol_sha256') != args.expected_protocol_sha256):
        parser.error('checkpoint plan identity mismatch')
    checkpoints = checkpoint_plan.get('checkpoints', [])
    checkpoint_ids = [row.get('checkpoint_id') for row in checkpoints]
    if len(checkpoint_ids) != len(set(checkpoint_ids)):
        parser.error('checkpoint ids must be unique')
    if set(checkpoint_ids) != expected_checkpoint_ids():
        parser.error('checkpoint ids must be base plus the exact frozen recipe/seed grid')
    rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK']); world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local); dist.init_process_group('nccl', device_id=torch.device('cuda', local))
    torch.set_num_threads(4)
    verified = [None]
    if rank == 0:
        try:
            actual = digest(args.validation_manifest)
            if actual != args.expected_validation_sha256:
                raise ValueError('confirmation manifest digest mismatch')
            protocol_sha = digest(args.protocol_file)
            if protocol_sha != args.expected_protocol_sha256:
                raise ValueError('confirmation protocol digest mismatch')
            protocol = json.loads(args.protocol_file.read_text())
            if (protocol.get('schema') != 'p529m-long-confirmation-v1'
                    or protocol.get('status') != 'FROZEN'
                    or tuple(protocol.get('recipes', ())) != RECIPES
                    or tuple(protocol.get('seeds', ())) != SEEDS
                    or protocol.get('selection', {}).get('rule')
                    != 'lower worst-seed equal-domain loss, then lower two-seed mean, then recipe id'):
                raise ValueError('confirmation protocol identity mismatch')
            verified[0] = {'validation_sha256': actual, 'protocol_sha256': protocol_sha}
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
    for entry in checkpoints:
        checkpoint_id = entry['checkpoint_id']
        checkpoint = Path(entry['checkpoint'])
        expected_sha = entry['checkpoint_sha256']
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
        passed = len(candidates) == 4 and {r['checkpoint_id'] for r in results} == expected_checkpoint_ids()
        output = {'schema_version': 2,
                  'status': 'PASS_FOUR_RUN_HELDOUT_EVALUATION' if passed else 'FAIL_INCOMPLETE_HELDOUT_EVALUATION',
                  'world_size': world, 'microbatch': args.microbatch,
                  'validation_manifest': str(args.validation_manifest),
                  'validation_manifest_sha256': args.expected_validation_sha256,
                  'validation_prediction_tokens': args.validation_tokens,
                  'protocol': str(args.protocol_file),
                  'protocol_sha256': args.expected_protocol_sha256,
                  'checkpoint_plan': str(args.checkpoint_plan),
                  'checkpoint_plan_sha256': args.expected_checkpoint_plan_sha256,
                  'results': results,
                  'gate': 'base plus the exact frozen two-recipe by two-seed grid evaluated on all five domains',
                  'claim_boundary': 'complete held-out language-model loss measurement; selection and promotion are separate'}
        temp = args.output.with_name(args.output.name + f'.tmp.{os.getpid()}')
        temp.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n'); os.replace(temp, args.output)
        print(json.dumps({'status': output['status'], 'losses': {r['checkpoint_id']: r['loss_equal_domain'] for r in results}}, sort_keys=True))
    dist.barrier(); dist.destroy_process_group()
    if rank == 0 and not passed: raise SystemExit(66)


if __name__ == '__main__':
    main()
