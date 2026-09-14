from copy import deepcopy
import pytest
from screen_selector import RECIPES, SEEDS, select


def matrix():
    values = {
        'F0_existing_web_control': (3.31, 3.32),
        'F1_broad_english_no_synthetic': (2.70, 2.71),
        'F2_reasoning_no_synthetic': (2.69, 2.69),
        'F3_broad_multilingual_no_synthetic': (2.57, 2.58),
    }
    return [
        {'recipe': recipe, 'seed': seed, 'initial_equal_domain_loss': 3.4,
         'final_equal_domain_loss': values[recipe][index],
         'final_loss_by_domain': {domain: values[recipe][index] for domain in
             ('general_web', 'knowledge_reading', 'math', 'code', 'multilingual')}}
        for recipe in RECIPES for index, seed in enumerate(SEEDS)
    ]


def test_selects_by_worst_seed_before_mean():
    runs = matrix()
    result = select(runs)
    assert result['winner'] == 'F3_broad_multilingual_no_synthetic'
    f2 = next(run for run in runs if run['recipe'].startswith('F2') and run['seed'] == SEEDS[0])
    f2['final_equal_domain_loss'] = 2.0
    f2['final_loss_by_domain'] = {key: 2.0 for key in f2['final_loss_by_domain']}
    assert select(runs)['winner'] == 'F3_broad_multilingual_no_synthetic'


@pytest.mark.parametrize('mutation', ('missing', 'duplicate', 'base_drift', 'control_regression'))
def test_fails_closed(mutation):
    runs = matrix()
    if mutation == 'missing':
        runs.pop()
    elif mutation == 'duplicate':
        runs[-1] = deepcopy(runs[0])
    elif mutation == 'base_drift':
        runs[-1]['initial_equal_domain_loss'] = 3.5
    else:
        for run in runs:
            if run['recipe'].startswith('F3'):
                run['final_equal_domain_loss'] = 3.5
        # Force F3 to rank first while failing its paired-control requirement.
        for recipe in RECIPES[1:3]:
            for run in runs:
                if run['recipe'] == recipe:
                    run['final_equal_domain_loss'] = 3.6
    with pytest.raises(ValueError):
        select(runs)
