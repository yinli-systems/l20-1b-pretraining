from collections import Counter

from patch import DOMAINS,choose_replacements


def family(partition,domain,tokens=100):
    values=Counter();values[domain]=tokens
    return {'partition':partition,'tokens':values,'documents':1}


def test_replacements_are_disjoint_and_restore_each_partition():
    families={}
    for partition in ('development','confirmation'):
        for domain in DOMAINS:
            for i in range(2):families[f'{partition}-{domain}-{i}']=family(partition,domain)
    for domain in DOMAINS:
        for i in range(4):families[f'train-{domain}-{i}']=family('train',domain)
    rejected={'development-math-0','confirmation-code-0'}
    moves,stats=choose_replacements(families,rejected,minimum_families=2,minimum_tokens=100)
    assert len(moves)==2 and set(moves.values())=={'development','confirmation'}
    assert all(stats[p][d]['families']>=2 for p in stats for d in DOMAINS)
