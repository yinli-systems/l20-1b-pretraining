import math
from pathlib import Path

import numpy as np
import pytest

from continuation_common import (Budget, BRANCH_STEPS, CAP, GLOBAL, MIXES, PILOT_STEPS,
                                 STEP_TOKENS, learning_rate, owned, quotas)
from prepare_continuation import atomic_npy, contains_question, hits, question_trie, spans


def test_budget_is_global_and_retries_charged(tmp_path):
    b = Budget(tmp_path/'budget.sqlite',cap=3*STEP_TOKENS)
    charge = b.reserve('A',0)
    b.complete(charge)
    b.reserve('A',0)  # Replay after a lost checkpoint costs again.
    b.close()
    b = Budget(tmp_path/'budget.sqlite',cap=3*STEP_TOKENS)
    b.reserve('B',0)
    with pytest.raises(RuntimeError):
        b.reserve('B',1)
    assert b.spent == 3*STEP_TOKENS
    with pytest.raises(ValueError):
        Budget(tmp_path/'budget.sqlite',cap=4*STEP_TOKENS)


def test_counts_and_schedule():
    assert (PILOT_STEPS*2+1534)*STEP_TOKENS == 1_999_134_720 <= CAP
    for mix in MIXES.values():
        q = quotas(mix,PILOT_STEPS*GLOBAL)
        assert sum(q.values()) == 96900
        assert all(abs(q[k]/sum(q.values())-w) < 1/sum(q.values()) for k,w in mix.items())
    assert learning_rate(0) == learning_rate(171) == 4e-5
    assert learning_rate(189) > 3.99e-5  # Pilot is not its own cooldown.
    assert math.isclose(learning_rate(BRANCH_STEPS-1),4e-6)
    assert all(learning_rate(i+1) <= learning_rate(i) for i in range(BRANCH_STEPS-1))
    with pytest.raises(ValueError):
        learning_rate(BRANCH_STEPS)


def test_output_boundary(tmp_path):
    assert owned(tmp_path/'run/new',tmp_path/'run') == tmp_path/'run/new'
    for p in (tmp_path/'run',tmp_path/'old',tmp_path/'run/../../old'):
        with pytest.raises(ValueError):
            owned(p,tmp_path/'run')
    (tmp_path/'run').mkdir()
    (tmp_path/'run/link').symlink_to(tmp_path/'old')
    with pytest.raises(ValueError):
        owned(tmp_path/'run/link/file',tmp_path/'run')


def test_span_index_shift_invariance_and_queries():
    a = np.arange(100,dtype=np.uint16)
    assert np.array_equal(spans(a[11:80]),spans(a)[11:68])
    index = np.unique(spans(a))
    assert hits(index,spans(a[30:70])) == 28
    assert hits(index,np.array([],dtype=np.uint64)) == 0
    assert hits(np.array([],dtype=np.uint64),spans(a)) == 0
    assert len(spans(a[:12])) == 0
    assert hits(index,spans(a+1000)) == 0


def test_short_question_matcher_exact_and_boundary():
    questions = ['is there a bank in this town'.split(), 'what is the capital of france'.split()]
    trie = question_trie(questions)
    assert contains_question('someone asked is there a bank in this town yesterday'.split(),trie)
    assert contains_question(questions[1],trie)
    assert not contains_question('what is the capital of germany'.split(),trie)
    assert not contains_question('this is not the question'.split(),trie)
    # Twelve words is the last short-question length; no missing last word.
    q = [str(i) for i in range(12)]
    assert contains_question(['prefix']+q+['suffix'],question_trie([q]))
    assert not contains_question(q[:-1],question_trie([q]))


def test_plan_and_resume_detects_changed_tokens(tmp_path,monkeypatch):
    import run_continuation as rc
    from continuation_common import atomic_json, sha256
    root = tmp_path/'original'
    run = tmp_path/'run'
    monkeypatch.setattr(rc,'ROOT',root)
    monkeypatch.setattr(rc,'RUN',run)
    monkeypatch.setattr(rc,'PILOT_STEPS',1)
    monkeypatch.setattr(rc,'GLOBAL',6)
    monkeypatch.setattr(rc,'MIXES',{'A':{'fresh_web':.5,'replay_web':.5}})
    manifests = {}
    for base,is_old,value in [(root/'data/full-npy/web',True,7),(run/'data/web',False,11)]:
        p = base/'train.npy'
        atomic_npy(p,np.full(12*2049,value,dtype=np.uint16))
        record = {'path':str(p),'tokens':12*2049,'sha256':sha256(p)}
        m = base/'manifest.json'
        atomic_json(m,{'train_shards' if is_old else 'shards':[record]})
        if not is_old:
            manifests['web'] = sha256(m)
    with pytest.raises(RuntimeError,match='review missing'):
        rc.make_plan('A')
    atomic_json(run/'quality-review-A.json',{'decision':'pass','branch':'A','manifests':manifests})
    plan = rc.make_plan('A')
    ds = rc.PlannedData(plan,run/'pilot-A/order.npy')
    a = ds.batch(0)
    assert a.shape == (6,2049)
    assert sorted(a[:,0].tolist()) == [7,7,7,11,11,11]
    assert rc.make_plan('A') == plan
    atomic_npy(run/'data/web/train.npy',np.full(12*2049,999,dtype=np.uint16))
    with pytest.raises(RuntimeError,match='resume input changed'):
        rc.make_plan('A')


def test_quality_manifest_cannot_escape_run(tmp_path,monkeypatch):
    import run_continuation as rc
    from continuation_common import atomic_json
    run = tmp_path/'run'
    monkeypatch.setattr(rc,'RUN',run)
    monkeypatch.setattr(rc,'PILOT_STEPS',1)
    monkeypatch.setattr(rc,'GLOBAL',6)
    monkeypatch.setattr(rc,'MIXES',{'A':{'fresh_web':1.0}})
    atomic_json(run/'quality-review-A.json',{'decision':'pass','branch':'A','manifests':{},
                'manifest_paths':{'web':str(tmp_path/'unrelated/manifest.json')}})
    with pytest.raises(ValueError,match='outside the dedicated run'):
        rc.make_plan('A')
