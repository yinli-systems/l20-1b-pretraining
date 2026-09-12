import json

import pytest

import evaluate_continuation as ec


def test_suites_keep_full_frozen_protocol():
    assert ec.jobs('core') == [{'name':'core','tasks':list(ec.CORE),'num_fewshot':0}]
    assert len(ec.jobs('all')) == 3
    assert len(ec.CORE) == 9 and 'boolq' in ec.CORE
    assert ec.jobs()[1:] == [{'name':x+'_5shot','tasks':[x],'num_fewshot':5} for x in ('mmlu','gsm8k')]
    assert ec.PROTOCOL['limit'] is None
    with pytest.raises(ValueError):
        ec.jobs('reduced')


def test_checkpoint_identity_fails_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(ec,'RUN',tmp_path)
    directory = tmp_path/'pilot-A'
    directory.mkdir()
    checkpoint = directory/'resume.pth'
    checkpoint.write_bytes(b'fake checkpoint')
    (directory/'data-plan.json').write_text('{}')
    status = directory/'status.json'
    status.write_text(json.dumps({'stage':'pilot_complete_pending_evaluation'}))
    receipt = {'bytes':checkpoint.stat().st_size,'step':190,'sha256':ec.sha256(checkpoint),'parent_sha256':'parent'}
    (directory/'resume.json').write_text(json.dumps(receipt))
    (tmp_path/'preflight.json').write_text(json.dumps({'parent':{'sha256':'parent'}}))
    assert ec.checkpoint_identity('A')['branch_prediction_tokens'] == 198451200
    status.write_text(json.dumps({'stage':'stopped_preservation_gate'}))
    with pytest.raises(RuntimeError,match='protection-passing'):
        ec.checkpoint_identity('A')
    status.write_text(json.dumps({'stage':'pilot_complete_pending_evaluation'}))
    checkpoint.write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='integrity mismatch'):
        ec.checkpoint_identity('A')
