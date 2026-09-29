"""Verify published records only; no model loading or inference."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def verify(root=ROOT):
    manifest=json.loads((root/'manifest.json').read_text())
    for rel,want in manifest['files'].items():
        path=(root/rel).resolve()
        if not path.is_relative_to(root.resolve()) or path.is_symlink() or not path.is_file():raise ValueError('invalid evidence path')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=want:raise ValueError('hash mismatch: '+rel)
    s=json.loads((root/'evidence/summary.json').read_text());d=s['data']
    assert sum(d['counts'].values())==9386
    assert sum(v for k,v in d['counts'].items() if k.endswith('/train'))==7400
    assert d['checks']['ledger_split_mismatches']==d['checks']['old_holdout_to_train']==d['checks']['original_heldout_labels_changed']==0
    assert d['training_numeric_labels_canonicalized']==1186
    assert s['integrity']['optimizer_steps']==0 and s['integrity']['old_trainable_parameters_unchanged']
    assert s['CPU_tests']['pass'] and s['CPU_tests']['tests']==26
    assert s['cuda_graph']['full_model']['True']['greedy_identical']==64
    assert s['cuda_graph']['full_model']['True']['nll_max_absolute_delta']==0
    assert all(s['cuda_graph']['guard_checks'].values())
    g=s['cuda_graph']['gradient_checks'][-1]
    assert g['gradient_tensors']==184 and g['relative_gradient_l2']==0 and g['frozen_encoder_has_no_gradients']
    assert s['prefetch']['file_descriptors_before']==s['prefetch']['file_descriptors_after']==40
    assert s['scale']['new_training_exposures']==0 and not s['scale']['completed_8400000_image_run']
    assert not s['training_blocker']['blocked_trainer_creation_retried_or_rerouted']
    return {'status':'verified','bound_files':len(manifest['files']),'training_updates':0,'fresh_GPU_reproduction':False}
if __name__=='__main__':print(json.dumps(verify(),indent=2))
