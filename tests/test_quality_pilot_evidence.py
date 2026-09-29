"""CPU checks of the completed real-pilot evidence, not GPU reproduction."""
from pathlib import Path
import hashlib,json,math
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'L20-VL-1.2B/experiments/quality-pilot-20260929'
def load(rel):return json.loads((P/rel).read_text())
def test_bound_startup_and_endpoint_files():
    for root in (P,P/'endpoint'):
        for name,want in json.loads((root/'manifest.json').read_text())['files'].items():
            path=root/name
            assert not path.is_symlink() and path.resolve().is_relative_to(root.resolve())
            assert hashlib.sha256(path.read_bytes()).hexdigest()==want

def test_real_update_sequence_and_exposure():
    rows=[json.loads(x) for x in (P/'endpoint/train.jsonl').read_text().splitlines()]
    assert [x['step'] for x in rows]==list(range(1,129))
    assert sum(x['new_image_events'] for x in rows)==4096
    assert sum(x['image_events'] for x in rows)==8192
    assert all(math.isfinite(v) for x in rows for v in x['losses'].values())

def test_unique_plan_and_optimizer_increment():
    plan=load('evidence/plan.json');ids=[x for step in plan['plans'] for group in step[:8] for x in group]
    assert len(ids)==len(set(ids))==4096
    audit=load('endpoint/independent-state-audit.json')
    assert audit['optimizer_states']==183
    assert audit['Adam_step_increment_min']==audit['Adam_step_increment_max']==128

def test_regression_and_scope_not_hidden():
    h=load('endpoint/headline.json')
    assert h['old_QA_after']<h['old_QA_before'] and h['old_QA_after']>=h['old_QA_floor']
    assert h['all_four_retention_checkpoints_pass']
    assert not h['automatic_promotion'] and not h['bulk_8_4M_training_started']
    assert not h['A448_started'] and not h['H896_started']
    assert h['remaining_arm_queue_blocked_not_retried'] and h['confirmation_not_accessed']
