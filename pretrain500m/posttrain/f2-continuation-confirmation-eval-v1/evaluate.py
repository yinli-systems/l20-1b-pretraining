"""Exact paired masked-loss confirmation for frozen F2 continuation checkpoints."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

MODEL_SOURCE = Path(os.environ['P529M_MODEL_SOURCE']).resolve()
sys.path.append(str(MODEL_SOURCE))
from model import Config, LM
from mixture import MixtureReader

DOMAINS = ('general_web', 'knowledge_reading', 'math', 'code', 'multilingual')
SEEDS = (20260914, 20260915)


def expected_checkpoint_ids():
    return [f'{role}_seed{seed}' for seed in SEEDS for role in ('parent', 'continuation')]


def validate_plan(plan):
    if plan.get('schema') != 'p529m-f2-continuation-paired-confirmation-plan-v1' or plan.get('status') != 'FROZEN_BEFORE_CONFIRMATION_READ':
        raise ValueError('paired confirmation plan identity mismatch')
    if tuple(plan.get('domains', ())) != DOMAINS:
        raise ValueError('confirmation domain identity mismatch')
    entries = plan.get('checkpoints', ())
    if [r.get('checkpoint_id') for r in entries] != expected_checkpoint_ids():
        raise ValueError('confirmation checkpoint identity/order mismatch')
    for entry in entries:
        if entry.get('checkpoint_id') != f"{entry.get('role')}_seed{entry.get('seed')}":
            raise ValueError('confirmation checkpoint role/seed mismatch')
        if len(entry.get('checkpoint_sha256', '')) != 64 or not Path(entry.get('checkpoint', '')).is_absolute():
            raise ValueError('confirmation checkpoint hash/path mismatch')
    gate = plan.get('gate', {})
    if gate.get('both_seeds_lower_equal_domain_masked_loss_vs_matched_parent') is not True or gate.get('max_relative_domain_regression_vs_matched_parent') != 0.005:
        raise ValueError('confirmation gate identity mismatch')
    if plan.get('confirmation_padded_prediction_tokens') != 18022400 or plan.get('confirmation_valid_target_tokens') != 11045058:
        raise ValueError('confirmation target count mismatch')
    return entries


def paired_gate(results, max_domain_regression):
    by_id = {row['checkpoint_id']: row for row in results}
    if set(by_id) != set(expected_checkpoint_ids()):
        raise ValueError('paired confirmation result grid mismatch')
    pairs = []
    for seed in SEEDS:
        parent = by_id[f'parent_seed{seed}']; continuation = by_id[f'continuation_seed{seed}']
        relative = 1 - continuation['loss_equal_domain'] / parent['loss_equal_domain']
        changes = {domain: continuation['loss_by_domain'][domain] / parent['loss_by_domain'][domain] - 1 for domain in DOMAINS}
        passed = relative > 0 and max(changes.values()) <= max_domain_regression
        pairs.append({'seed': seed, 'parent_checkpoint_sha256': parent['checkpoint_sha256'],
                      'continuation_checkpoint_sha256': continuation['checkpoint_sha256'],
                      'relative_equal_domain_loss_reduction': relative,
                      'relative_loss_change_by_domain': changes, 'passed': passed})
    return pairs, all(row['passed'] for row in pairs)


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
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--microbatch', type=int, default=1)
    args = parser.parse_args()
    if digest(args.plan) != args.expected_plan_sha256:
        parser.error('paired confirmation plan SHA-256 mismatch')
    plan = json.loads(args.plan.read_text())
    entries = validate_plan(plan)
    validation_manifest = Path(plan['confirmation_manifest'])
    expected_validation_sha256 = plan['confirmation_manifest_sha256']
    validation_tokens = plan['confirmation_padded_prediction_tokens']
    rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK']); world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local); dist.init_process_group('nccl', device_id=torch.device('cuda', local))
    torch.set_num_threads(4)
    verified = [None]
    if rank == 0:
        try:
            actual = digest(validation_manifest)
            if actual != expected_validation_sha256:
                raise ValueError('confirmation manifest digest mismatch')
            verified[0] = {'sha256': actual}
        except Exception as exc:
            verified[0] = {'error': f'{type(exc).__name__}: {exc}'}
    dist.broadcast_object_list(verified, src=0)
    if 'error' in verified[0]: raise RuntimeError(verified[0]['error'])
    reader = MixtureReader(validation_manifest, 0, validation_tokens,
                           expected_validation_sha256, verify_content=rank == 0)
    if set(reader.sources) != set(DOMAINS) or not reader.has_loss_masks:
        raise ValueError('confirmation reader identity mismatch')
    config = Config(hidden_size=1280, intermediate_size=3584, num_hidden_layers=26,
                    num_attention_heads=20, num_key_value_heads=5)
    results = []
    for entry in entries:
        checkpoint_id = entry['checkpoint_id']; checkpoint = Path(entry['checkpoint']); expected_sha = entry['checkpoint_sha256']
        raw = LM(config).cuda()
        load_and_broadcast(raw, checkpoint, expected_sha, rank)
        result = evaluate(raw, reader, rank, world, args.microbatch)
        if sum(result['target_tokens_by_domain'].values()) != plan['confirmation_valid_target_tokens']:
            raise ValueError('confirmation valid target count mismatch')
        result.update({'checkpoint_id': checkpoint_id, 'checkpoint': str(checkpoint),
                       'checkpoint_sha256': expected_sha})
        results.append(result)
        del raw
        torch.cuda.empty_cache()
        dist.barrier()
    if rank == 0:
        pairs, passed = paired_gate(results, plan['gate']['max_relative_domain_regression_vs_matched_parent'])
        output = {'schema_version': 1,
                  'status': 'PASS_PAIRED_F2_CONTINUATION_CONFIRMATION' if passed else 'FAIL_PAIRED_F2_CONTINUATION_CONFIRMATION',
                  'world_size': world, 'microbatch': args.microbatch,
                  'plan': str(args.plan), 'plan_sha256': args.expected_plan_sha256,
                  'validation_manifest': str(validation_manifest),
                  'validation_manifest_sha256': expected_validation_sha256,
                  'validation_prediction_tokens': validation_tokens,
                  'results': results,
                  'pairs': pairs, 'gate': plan['gate'],
                  'claim_boundary': 'reserved five-domain language-model masked loss; capability, retention, contamination and market comparison separate'}
        temp = args.output.with_name(args.output.name + f'.tmp.{os.getpid()}')
        temp.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n'); os.replace(temp, args.output)
        print(json.dumps({'status': output['status'], 'pairs': pairs}, sort_keys=True))
    dist.barrier(); dist.destroy_process_group()
    if rank == 0 and not passed: raise SystemExit(66)


if __name__ == '__main__':
    main()
