"""CPU-only verification of published evidence, not model inference."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def verify(root=ROOT):
    root=root.resolve();m=json.loads((root/'manifest.json').read_text())
    for rel,want in m['files'].items():
        p=root/rel
        if p.is_symlink() or not p.resolve().is_relative_to(root) or not p.is_file():raise ValueError('unsafe evidence path')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=want:raise ValueError('evidence hash mismatch: '+rel)
    s=json.loads((root/'fresh-v2/summary.json').read_text());q=json.loads((root/'fresh-v2/qualification.json').read_text());h=json.loads((root/'headline.json').read_text())
    assert s['status']=='complete' and s['qualification_pass'] and q['pass'] and len(q['rows'])==16
    assert all(r['true_tokens_equal'] and r['wrong_tokens_equal'] and r['diagnostics_max_abs']==0 for r in q['rows'])
    low=s['original224_regression'];assert low['images']==8 and low['true_tokens_equal'] and low['wrong_tokens_equal'] and low['diagnostics_max_abs']==0
    a=sum(r['seconds'] for r in s['timing'] if r['mode']=='reference');b=sum(r['seconds'] for r in s['timing'] if r['mode']=='prepared')
    assert math.isclose(a/b,s['speed_ratio'],rel_tol=1e-12)
    assert math.isclose(a/b,h['throughput_ratio'],rel_tol=1e-12)
    assert math.isclose(100*(1-b/a),h['wall_time_reduction_percent'],rel_tol=1e-12)
    assert s['weights_unchanged'] and s['optimizer_steps']==h['optimizer_steps']==0
    assert not h['new_training_launched'] and not h['bulk_scale_completed']
    policy=json.loads((root/'fresh-v2/policy.json').read_text())
    for name,want in policy['source_hashes'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==want
    v1=json.loads((root/'v1-regression-decision.json').read_text());assert v1['original224_8image_regression']=='failed'
    assert hashlib.sha256((root/'prepared_eval-v1.py').read_bytes()).hexdigest()==v1['v1_source_sha256']
    return {'status':'verified','bound_files':len(m['files']),'fresh_GPU_reproduced_by_this_verifier':False,'optimizer_updates':0}
if __name__=='__main__':print(json.dumps(verify(),indent=2))
