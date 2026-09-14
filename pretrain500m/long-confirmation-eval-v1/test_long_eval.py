import copy

import pytest
import torch

import build_plan
from build_plan import INITIAL_SHA256, TARGET_TOKENS, build, digest
from evaluate import DOMAINS, RECIPES, SEEDS, expected_checkpoint_ids, masked_domain_sums, summarize
from select_long import RULE, select


def protocol():
    return {'schema': 'p529m-long-confirmation-v1', 'status': 'FROZEN',
            'recipes': list(RECIPES), 'seeds': list(SEEDS),
            'selection': {'rule': RULE}}


def evaluation(losses):
    counts = {domain: 10 + i for i, domain in enumerate(DOMAINS)}
    rows = [{'checkpoint_id': 'base', 'checkpoint': '/base', 'checkpoint_sha256': 'b' * 64,
             'loss_equal_domain': 3.0, 'loss_by_domain': {d: 3.0 for d in DOMAINS},
             'target_tokens_by_domain': counts}]
    for recipe in RECIPES:
        for seed in SEEDS:
            value = losses[(recipe, seed)]
            rows.append({'checkpoint_id': f'{recipe}_seed{seed}', 'checkpoint': f'/{recipe}/{seed}',
                         'checkpoint_sha256': (recipe + str(seed)).encode().hex().ljust(64, '0')[:64],
                         'loss_equal_domain': value, 'loss_by_domain': {d: value for d in DOMAINS},
                         'target_tokens_by_domain': counts})
    return {'status': 'PASS_FOUR_RUN_HELDOUT_EVALUATION', 'results': rows}


def test_expected_checkpoint_grid_is_exact():
    assert expected_checkpoint_ids() == {
        'base',
        'F2_reasoning_no_synthetic_seed20260914',
        'F2_reasoning_no_synthetic_seed20260915',
        'F3_broad_multilingual_no_synthetic_seed20260914',
        'F3_broad_multilingual_no_synthetic_seed20260915'}


def test_masked_domain_accounting_and_summary():
    logits = torch.zeros(2, 2, 3)
    targets = torch.tensor([[0, 1], [2, 1]])
    masks = torch.tensor([[True, False], [True, True]])
    sums, counts = masked_domain_sums(logits, targets, masks, ('general_web', 'math'))
    assert counts[DOMAINS.index('general_web')].item() == 1
    assert counts[DOMAINS.index('math')].item() == 2
    assert summarize(torch.tensor([2., 4., 6., 8., 10.], dtype=torch.float64),
                     torch.tensor([2., 2., 2., 2., 2.], dtype=torch.float64))['loss_equal_domain'] == 3.0


def test_selection_minimizes_worst_seed_before_mean():
    losses = {
        (RECIPES[0], SEEDS[0]): 2.0, (RECIPES[0], SEEDS[1]): 2.2,
        (RECIPES[1], SEEDS[0]): 2.1, (RECIPES[1], SEEDS[1]): 2.1}
    result = select(evaluation(losses), protocol())
    assert result['selected_recipe'] == RECIPES[1]
    assert result['ranked_recipes'][0]['worst_seed_loss_equal_domain'] == 2.1


def test_recipe_id_is_final_tiebreaker():
    losses = {(recipe, seed): 2.0 for recipe in RECIPES for seed in SEEDS}
    assert select(evaluation(losses), protocol())['selected_recipe'] == min(RECIPES)


def test_selection_rejects_missing_run_and_target_mismatch():
    losses = {(recipe, seed): 2.0 for recipe in RECIPES for seed in SEEDS}
    missing = evaluation(losses)
    missing['results'].pop()
    with pytest.raises(ValueError, match='exact checkpoint grid'):
        select(missing, protocol())
    mismatch = evaluation(losses)
    mismatch['results'][-1] = copy.deepcopy(mismatch['results'][-1])
    mismatch['results'][-1]['target_tokens_by_domain']['math'] += 1
    with pytest.raises(ValueError, match='different target tokens'):
        select(mismatch, protocol())


def make_completed_grid(tmp_path):
    root = tmp_path / 'root'
    protocol_path = tmp_path / 'protocol.json'
    inputs_path = tmp_path / 'inputs.json'
    base = tmp_path / 'base.pt'
    base.write_bytes(b'base')
    protocol_row = protocol()
    protocol_row['base_checkpoint'] = {'path': str(base), 'sha256': INITIAL_SHA256, 'step': 7629}
    protocol_path.write_text(__import__('json').dumps(protocol_row))
    protocol_sha = digest(protocol_path)
    recipes = {}
    for recipe in RECIPES:
        recipes[recipe] = {'manifest_sha256': (recipe.encode().hex() + '0' * 64)[:64],
                           'admission_sha256': (recipe.encode().hex() + '1' * 64)[:64]}
    inputs_path.write_text(__import__('json').dumps(
        {'status': 'PASS_CONFIRMATION_INPUTS_ADMITTED', 'recipes': recipes}))
    for recipe in RECIPES:
        for seed in SEEDS:
            run = root / 'posttrain' / 'confirmations-v1' / f'{recipe}-seed{seed}-{seed}' / 'train'
            run.mkdir(parents=True)
            model = run / 'model-final.pt'
            torch.save({'model': {'weight': torch.ones(1)}, 'step': 1024,
                        'prediction_tokens': TARGET_TOKENS,
                        'origin': {'kind': 'continued_pretraining',
                                   'initial_checkpoint_sha256': INITIAL_SHA256},
                        'run_fingerprint': 'f' * 64, 'protocol_sha256': protocol_sha,
                        'training_manifest_sha256': recipes[recipe]['manifest_sha256'],
                        'checkpoint_semantics': 'model-only; no optimizer, reader or RNG state; not resumable'}, model)
            model_sha = digest(model)
            (run / 'model-final.sha256').write_text(f'{model_sha}  model-final.pt\n')
            (run / 'training-status.json').write_text(__import__('json').dumps(
                {'status': 'QUALIFICATION_COMPLETED', 'step': 1024, 'tokens': TARGET_TOKENS,
                 'mfu_window_passed': True, 'model_checkpoint_sha256': model_sha}))
            (run / 'run-manifest.json').write_text(__import__('json').dumps(
                {'seed': seed, 'target_tokens': TARGET_TOKENS, 'tokens_per_step': 2097152,
                 'world_size': 4, 'microbatch': 4, 'accumulation': 64, 'peak_lr': 0.0001,
                 'checkpoint_mode': 'model-only-final', 'protocol_sha256': protocol_sha,
                 'data_manifest_sha256': recipes[recipe]['manifest_sha256'],
                 'admission_sha256': recipes[recipe]['admission_sha256'],
                 'initial_checkpoint_sha256': INITIAL_SHA256}))
    return root, protocol_path, inputs_path, protocol_sha


def test_checkpoint_plan_requires_exact_successful_grid(tmp_path):
    root, protocol_path, inputs_path, protocol_sha = make_completed_grid(tmp_path)
    result = build(root, protocol_path, protocol_sha, inputs_path, digest(inputs_path))
    assert result['status'] == 'PASS_EXACT_FOUR_RUN_CHECKPOINT_PLAN'
    assert {row['checkpoint_id'] for row in result['checkpoints']} == expected_checkpoint_ids()
    missing = next((root / 'posttrain' / 'confirmations-v1').glob(
        f'{RECIPES[1]}-seed{SEEDS[1]}-*'))
    (missing / 'train' / 'training-status.json').unlink()
    with pytest.raises(RuntimeError, match='expected exactly one successful run'):
        build(root, protocol_path, protocol_sha, inputs_path, digest(inputs_path))


def test_incomplete_grid_never_hashes_a_model(tmp_path, monkeypatch):
    root, protocol_path, inputs_path, protocol_sha = make_completed_grid(tmp_path)
    missing = next((root / 'posttrain' / 'confirmations-v1').glob(
        f'{RECIPES[1]}-seed{SEEDS[1]}-*'))
    (missing / 'train' / 'training-status.json').unlink()
    original_digest = build_plan.digest
    model_hashes = []

    def monitored(path):
        if str(path).endswith('model-final.pt'):
            model_hashes.append(str(path))
        return original_digest(path)

    monkeypatch.setattr(build_plan, 'digest', monitored)
    with pytest.raises(RuntimeError, match='expected exactly one successful run'):
        build_plan.build(root, protocol_path, protocol_sha, inputs_path, original_digest(inputs_path))
    assert model_hashes == []
