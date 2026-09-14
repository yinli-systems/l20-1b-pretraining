"""CPU correctness checks; these do not qualify NCCL, CUDA or GPU MFU."""
from collections import Counter
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mixture import MixtureReader, allocate, digest

SCHEMA = 'p529m-packed-mixture-v2'
from resume import SCHEMA as RESUME_SCHEMA, resolve_origin, run_fingerprint, verify_resume
from train_cpt import VALIDATION_DOMAINS, masked_domain_sums, model_only_state


def corpus(tmp_path, weights=None, blocks=1000, cap=2, prior=0):
    weights = weights or {'web': 35, 'pdf': 20, 'math': 15, 'code': 15,
                          'synthetic': 5, 'multilingual': 10, 'zero': 0}
    sources = []
    for i, sid in enumerate(sorted(k for k, v in weights.items() if v)):
        path = tmp_path / f'{sid}.npy'
        np.save(path, (np.arange(blocks * 4) + i * 5000).astype(np.uint16))
        sources.append({'id': sid, 'revision': str(i) * 40,
                        'max_cumulative_epochs': cap, 'prior_prediction_tokens': prior,
                        'shards': [{'path': path.name, 'sha256': digest(path), 'blocks': blocks}]})
    doc = {'schema': SCHEMA, 'sequence_length': 3, 'tokenizer_sha256': 'a' * 64,
           'weights_percent': weights, 'sources': sources}
    path = tmp_path / 'mixture.json'
    path.write_text(json.dumps(doc))
    return path, doc


def reader(path, blocks, seed=20260914):
    return MixtureReader(path, seed, blocks * 3, digest(path))


@pytest.mark.parametrize('blocks', [1, 3, 19, 20, 21, 37, 100, 1024])
def test_exact_quotas_and_every_prefix_cursor(tmp_path, blocks):
    path, doc = corpus(tmp_path)
    r = reader(path, blocks)
    counts = Counter()
    logical = set()
    for i in range(blocks):
        state = r.state_at(i)
        assert state['consumed_blocks'] == {k: counts[k] for k in doc['weights_percent']}
        pair = r.locate(i)
        assert pair not in logical
        logical.add(pair)
        counts[pair[0]] += 1
    assert {k: counts[k] for k in doc['weights_percent']} == allocate(doc['weights_percent'], blocks)
    assert r.state_at(blocks)['consumed_blocks'] == r.quotas
    with pytest.raises(StopIteration):
        r.sequence(blocks)


def test_disjoint_ranks_equal_the_global_stream_and_resume(tmp_path):
    path, _ = corpus(tmp_path)
    expected = reader(path, 1024)
    sequences = torch.stack([expected.sequence(i) for i in range(1024)])
    assert len(set(tuple(row.tolist()) for row in sequences)) == 1024
    for world in (1, 2, 4, 8):
        r = reader(path, 1024)
        rows = []
        for step in range(1024 // (world * 2)):
            if step == 11:
                state = r.state_at(step * world * 2)
                r = reader(path, 1024)
                r.verify_state(state, step * world * 2)
            for rank in range(world):
                x, y = r.batch_for_step(step, rank, world, 2)
                assert torch.equal(x[:, 1:], y[:, :-1])
                rows.extend(torch.cat([x, y[:, -1:]], dim=1))
        assert torch.equal(torch.stack(rows), sequences)


def test_order_independent_and_seed_changes_stream(tmp_path):
    path, doc = corpus(tmp_path)
    a = reader(path, 1024)
    doc['sources'].reverse()
    doc['weights_percent'] = dict(reversed(list(doc['weights_percent'].items())))
    path.write_text(json.dumps(doc))
    b, c = reader(path, 1024), reader(path, 1024, seed=20260915)
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != c.fingerprint
    assert all(torch.equal(a.sequence(i), b.sequence(i)) for i in range(80))
    assert any(not torch.equal(a.sequence(i), c.sequence(i)) for i in range(80))


def test_cache_is_bounded_and_eviction_does_not_change_stream(tmp_path):
    path, _ = corpus(tmp_path, blocks=10, cap=10000)
    r = reader(path, 100000)
    prefix = [r.locate(i) for i in range(20)]
    for i in range(0, 100000, 20):
        r.locate(i)
    assert len(r._cache) <= 64
    assert prefix == [r.locate(i) for i in range(20)]


@pytest.mark.parametrize('bad', ['missing', 'duplicate_id', 'duplicate_content', 'dtype', 'partial', 'cap', 'prior', 'revision'])
def test_rejects_invalid_data_and_never_renormalizes(tmp_path, bad):
    path, doc = corpus(tmp_path, blocks=100)
    if bad == 'missing':
        doc['sources'].pop()
    elif bad == 'duplicate_id':
        doc['sources'][-1] = deepcopy(doc['sources'][0])
    elif bad == 'duplicate_content':
        doc['sources'][-1]['shards'] = deepcopy(doc['sources'][0]['shards'])
    elif bad in ('dtype', 'partial'):
        shard = doc['sources'][0]['shards'][0]
        array = np.arange(400 if bad == 'dtype' else 399, dtype=np.uint32 if bad == 'dtype' else np.uint16)
        np.save(tmp_path / shard['path'], array)
        shard['sha256'] = digest(tmp_path / shard['path'])
    elif bad == 'cap':
        doc['sources'][0]['max_cumulative_epochs'] = 0.01
    elif bad == 'prior':
        doc['sources'][0]['prior_prediction_tokens'] = 600
    else:
        doc['sources'][0]['revision'] = 'main'
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        reader(path, 100)


def test_manifest_and_shard_hash_drift_rejected(tmp_path):
    path, doc = corpus(tmp_path)
    with pytest.raises(ValueError, match='manifest SHA-256'):
        MixtureReader(path, 0, 300, '0' * 64)
    shard = tmp_path / doc['sources'][0]['shards'][0]['path']
    values = np.load(shard)
    values[0] += 1
    np.save(shard, values)
    with pytest.raises(ValueError, match='shard SHA-256'):
        reader(path, 100)


def test_nonverifying_rank_still_binds_manifest_and_array_shape(tmp_path):
    path,doc=corpus(tmp_path,{'web':100},blocks=10,cap=100)
    assert MixtureReader(path,0,30,digest(path),verify_content=False).total_blocks==10
    shard=tmp_path/doc['sources'][0]['shards'][0]['path']
    values=np.load(shard);values[0]+=1;np.save(shard,values)
    # Rank zero catches content drift. Other ranks may skip the expensive full
    # digest only after rank zero has broadcast a successful verification.
    with pytest.raises(ValueError,match='shard SHA-256'):MixtureReader(path,0,30,digest(path))
    assert MixtureReader(path,0,30,digest(path),verify_content=False).total_blocks==10


@pytest.mark.parametrize('weights', [{'a': 99}, {'a': -1, 'b': 101}, {'a': 100.0}, {'a': True, 'b': 99}])
def test_invalid_weight_inputs(weights):
    with pytest.raises(ValueError):
        allocate(weights, 100)


def test_single_source_epoch_traversal_and_declared_replay(tmp_path):
    path, _ = corpus(tmp_path, {'web': 100}, blocks=17, cap=2)
    r = reader(path, 34)
    first = [tuple(r.sequence(i).tolist()) for i in range(17)]
    second = [tuple(r.sequence(i).tolist()) for i in range(17, 34)]
    assert len(set(first)) == 17
    assert set(first) == set(second)
    assert first != second


def test_input_boundaries(tmp_path):
    path, _ = corpus(tmp_path)
    r = reader(path, 21)
    for args in [(-1, 0, 1, 1), (0, 2, 2, 1), (0, 0, 0, 1)]:
        with pytest.raises(ValueError):
            r.batch_for_step(*args)
    with pytest.raises(StopIteration):
        r.batch_for_step(5, 1, 2, 2)
    with pytest.raises(ValueError):
        MixtureReader(path, 0, 5, digest(path))
    with pytest.raises(ValueError):
        r.verify_state(r.state_at(20), 19)


@pytest.mark.parametrize('recipe', ['M0_existing_web_control', 'M1_broad_english', 'M2_reasoning', 'M3_broad_multilingual'])
def test_real_screen_quotas_match_the_research_plan(tmp_path, recipe):
    root = Path(__file__).resolve().parents[2] / 'research' / 'data-mixture-v1'
    design = json.loads((root / 'experiment-design.json').read_text())
    plan = json.loads((root / 'token-budget-plan.json').read_text())
    assert plan['design_sha256'] == digest(root / 'experiment-design.json')
    path, _ = corpus(tmp_path, design['recipes_percent'][recipe], cap=10000)
    r = reader(path, plan['prediction_tokens_per_run'] // 2048)
    assert r.quotas == plan['recipes'][recipe]['blocks']
    assert r.state_at(r.total_blocks)['consumed_blocks'] == r.quotas


def equal_nested(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_nested(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_nested(x, y)
    else:
        assert a == b


def test_cpu_optimizer_and_rng_resume_are_bit_exact(tmp_path):
    path, _ = corpus(tmp_path, {'web': 50, 'code': 50}, blocks=100)
    r = reader(path, 24)
    fields = {'world_size': 1, 'reader': r.fingerprint, 'steps': 12, 'lr': 0.001}
    origin = resolve_origin('b' * 64)
    fp = run_fingerprint(fields, origin)

    def fresh():
        torch.manual_seed(812)
        random.seed(34)
        np.random.seed(56)
        model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Dropout(0.25), torch.nn.Linear(8, 3))
        return model, torch.optim.AdamW(model.parameters(), lr=0.001)

    def advance(model, opt, first, end, data):
        losses = []
        for step in range(first, end):
            x, y = data.batch_for_step(step, 0, 1, 2)
            scale = 1 + random.random() / 100 + float(np.random.random()) / 100
            opt.param_groups[0]['lr'] = 0.001 * (12 - step) / 12
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x.float() / 10000) * scale, y.float() / 10000)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return losses

    model, opt = fresh()
    uninterrupted = advance(model, opt, 0, 12, r)
    expected_model, expected_opt = deepcopy(model.state_dict()), deepcopy(opt.state_dict())
    model, opt = fresh()
    split_losses = advance(model, opt, 0, 5, r)
    state = {'resume_schema': RESUME_SCHEMA, 'origin': origin, 'run_fingerprint': fp,
             'reader_state': r.state_at(10), 'step': 5, 'model': model.state_dict(),
             'optimizer': opt.state_dict(), 'python_rng': [random.getstate()],
             'numpy_rng': [np.random.get_state()], 'torch_rng': [torch.get_rng_state()],
             'cuda_rng': [None]}
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=False)
    resumed_reader = reader(path, 24)
    resumed_origin = resolve_origin(checkpoint=state)
    assert resumed_origin == origin
    resumed_fp = run_fingerprint(fields, resumed_origin)
    step = verify_resume(state, resumed_fp, resumed_reader, 2, 12, 1)
    model, opt = fresh()
    model.load_state_dict(state['model'])
    opt.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng'][0])
    np.random.set_state(state['numpy_rng'][0])
    torch.set_rng_state(state['torch_rng'][0])
    split_losses += advance(model, opt, step, 12, resumed_reader)
    assert split_losses == uninterrupted
    equal_nested(model.state_dict(), expected_model)
    equal_nested(opt.state_dict(), expected_opt)
    for changed in ({**fields, 'world_size': 2}, {**fields, 'lr': 0.002}):
        with pytest.raises(ValueError, match='fingerprint'):
            verify_resume(state, run_fingerprint(changed, resumed_origin), resumed_reader, 2, 12, 1)
    broken = deepcopy(state)
    broken['reader_state']['consumed_blocks']['web'] += 1
    with pytest.raises(ValueError, match='cursor'):
        verify_resume(broken, fp, resumed_reader, 2, 12, 1)
    with pytest.raises(ValueError, match='legacy checkpoint'):
        resolve_origin(checkpoint={'run_fingerprint': fp})


def test_explicit_leaf_quotas_preserve_nested_rounding(tmp_path):
    path, doc = corpus(tmp_path, {'fineweb': 50, 'math_a': 25, 'math_b': 25}, blocks=20, cap=100)
    doc['schema'] = 'p529m-packed-mixture-v3'
    doc.pop('weights_percent')
    doc['source_block_quotas'] = {'fineweb': 7, 'math_a': 8, 'math_b': 6}
    path.write_text(json.dumps(doc))
    reader = MixtureReader(path, 20260914, 21 * 3, digest(path))
    assert reader.quotas == doc['source_block_quotas']
    assert Counter(reader.locate(i)[0] for i in range(21)) == doc['source_block_quotas']
    assert reader.state_at(21)['consumed_blocks'] == doc['source_block_quotas']


def test_explicit_quota_sum_and_sources_fail_closed(tmp_path):
    path, doc = corpus(tmp_path, {'a': 50, 'b': 50}, blocks=10, cap=100)
    doc['schema'] = 'p529m-packed-mixture-v3'
    doc.pop('weights_percent')
    for quotas in ({'a': 5, 'b': 4}, {'a': 5, 'missing': 5}):
        doc['source_block_quotas'] = quotas
        path.write_text(json.dumps(doc))
        with pytest.raises(ValueError): MixtureReader(path, 0, 30, digest(path))


def test_masked_validation_batches_preserve_source_and_mask_identity(tmp_path):
    path, doc = corpus(tmp_path, {'general_web': 50, 'math': 50}, blocks=10, cap=100)
    doc['schema'] = 'p529m-packed-mixture-v3'
    doc.pop('weights_percent')
    doc['source_block_quotas'] = {'general_web': 2, 'math': 2}
    for source in doc['sources']:
        mask_path = tmp_path / f"{source['id']}.mask.npy"
        mask = np.ones(10 * 3, dtype=np.bool_)
        mask[::4] = False
        np.save(mask_path, mask)
        source['shards'][0]['loss_mask'] = {
            'path': mask_path.name, 'sha256': digest(mask_path),
        }
    path.write_text(json.dumps(doc))
    r = MixtureReader(path, 20260914, 12, digest(path))
    x, y, masks, sources = r.batch_for_step(0, 0, 1, 2, include_loss_mask=True)
    assert x.shape == y.shape == masks.shape == (2, 3)
    assert all(source in doc['source_block_quotas'] for source in sources)
    assert masks.dtype == torch.bool
    with pytest.raises(ValueError, match='loss masks'):
        unmasked = tmp_path / 'unmasked'
        unmasked.mkdir()
        unmasked_path, _ = corpus(unmasked, {'web': 100}, blocks=10, cap=100)
        reader(unmasked_path, 1).batch_for_step(0, 0, 1, 1, include_loss_mask=True)


def test_partial_or_changed_loss_masks_fail_closed(tmp_path):
    path, doc = corpus(tmp_path, {'web': 100}, blocks=10, cap=100)
    source = doc['sources'][0]
    source['shards'].append(deepcopy(source['shards'][0]))
    source['shards'][1]['path'] = 'second.npy'
    np.save(tmp_path / 'second.npy', np.arange(40, dtype=np.uint16) + 1000)
    source['shards'][1]['sha256'] = digest(tmp_path / 'second.npy')
    mask_path = tmp_path / 'web.mask.npy'
    np.save(mask_path, np.ones(30, dtype=np.bool_))
    source['shards'][0]['loss_mask'] = {'path': mask_path.name, 'sha256': digest(mask_path)}
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='every shard'):
        reader(path, 1)
    source['shards'][1]['loss_mask'] = deepcopy(source['shards'][0]['loss_mask'])
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='loss-mask SHA-256|invalid loss mask'):
        source['shards'][0]['loss_mask']['sha256'] = '0' * 64
        path.write_text(json.dumps(doc))
        reader(path, 1)


def test_masked_domain_sums_are_token_exact_and_domain_ordered():
    logits = torch.tensor([
        [[4.0, 0.0], [0.0, 4.0], [4.0, 0.0]],
        [[0.0, 4.0], [4.0, 0.0], [0.0, 4.0]],
    ])
    targets = torch.tensor([[0, 1, 1], [1, 0, 0]])
    masks = torch.tensor([[True, False, True], [True, True, False]])
    sums, counts = masked_domain_sums(logits, targets, masks, ['math', 'general_web'])
    assert counts[VALIDATION_DOMAINS.index('math')].item() == 2
    assert counts[VALIDATION_DOMAINS.index('general_web')].item() == 2
    assert sums[VALIDATION_DOMAINS.index('math')] > sums[VALIDATION_DOMAINS.index('general_web')]
    with pytest.raises(ValueError, match='unknown validation domain'):
        masked_domain_sums(logits[:1], targets[:1], masks[:1], ['unknown'])


def test_model_only_state_omits_every_resumable_training_state():
    model=torch.nn.Linear(3,2)
    state=model_only_state(model,7,1234,{'initial_checkpoint_sha256':'a'*64},'b'*64,'c'*64,'d'*64)
    assert set(state)=={'model','step','prediction_tokens','origin','run_fingerprint','protocol_sha256',
                        'training_manifest_sha256','checkpoint_semantics'}
    assert not {'optimizer','reader_state','python_rng','numpy_rng','torch_rng','cuda_rng'} & set(state)
    assert all(t.device.type=='cpu' for t in state['model'].values())
    assert 'not resumable' in state['checkpoint_semantics']
