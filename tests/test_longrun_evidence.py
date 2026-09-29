"""Validate published long-run startup evidence without training a model."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'L20-VL-1.2B/experiments/longtrain-20260929'
def read(name):return json.loads((P/name).read_text())
def test_all_export_hashes_match():
    for name,want in read('manifest.json')['files'].items():
        p=P/name;assert not p.is_symlink() and p.resolve().is_relative_to(P.resolve())
        assert hashlib.sha256(p.read_bytes()).hexdigest()==want

def test_actual_saved_optimizer_updates():
    a=read('startup-audit.json')
    assert a['optimizer_states']==183 and a['optimizer_step_increment_min']==a['optimizer_step_increment_max']==16
    assert a['new_distinct_images_in_saved_checkpoint']==512 and a['last_consumed_data_catalog']==15 and a['catalog_chain_verified']
    assert all(x['changed_tensors']==x['tensors'] and x['all_finite'] for x in a['parameter_changes'].values())

def test_budget_is_not_a_completed_scale_claim():
    c=read('config.json');a=read('startup-audit.json')
    assert c['max_updates']*32==c['max_new_unique_images']==131072
    assert c['maximum_wall_seconds']==21600 and c['campaign_unique_image_goal']==8400000
    assert not a['8_4M_campaign_completed'] and not a['highest_quality_proven'] and not a['new_model_benchmark_improvement_claimed']
    assert c['old_QA_anchor']==.5625 and c['old_QA_floor']==.5125

def test_finite_sequential_losses_and_qualified_source():
    rows=[json.loads(x) for x in (P/'first16-updates.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1,17))
    assert sum(r['new_image_events'] for r in rows)==512
    assert all(math.isfinite(r['full_weighted_objective']) and math.isfinite(r['gradient_norm']) for r in rows)
    q=read('qualification.json');assert q['pass'] and q['CPU_contract_tests']==30
    assert hashlib.sha256((P/'long_train.py').read_bytes()).hexdigest()==q['trainer_sha256']
    assert hashlib.sha256((P/'stream_sources.py').read_bytes()).hexdigest()==q['stream_source_sha256']
