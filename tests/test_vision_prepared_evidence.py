"""Cheap CPU checks of the exact public GPU evidence package."""
from pathlib import Path
import importlib.util,json,math,shutil
import pytest
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'L20-VL-1.2B/experiments/vision-scale50-20260929'
P=BASE/'prepared-evaluation'
def verifier(folder,name):
    spec=importlib.util.spec_from_file_location(name,folder/'verify.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_recovered_original_evidence():
    assert verifier(BASE/'original-execution','old_vision_verify').verify()['status']=='verified'
def test_prepared_evidence_hashes_and_contracts():
    assert verifier(P,'new_vision_verify').verify()['status']=='verified'
def test_export_recovery_matches_transcribed_headlines():
    report=json.loads((BASE/'export-recovery-20260929.json').read_text())
    assert report['remote_and_Mac_archive_hashes_match'] and all(report['transcribed_summary_cross_checks'].values())
def test_raw_timing_recomputes_headline():
    s=json.loads((P/'fresh-v2/summary.json').read_text());h=json.loads((P/'headline.json').read_text())
    assert len(s['timing'])==8
    a=sum(r['seconds'] for r in s['timing'] if r['mode']=='reference');b=sum(r['seconds'] for r in s['timing'] if r['mode']=='prepared')
    assert math.isclose(a/b,h['throughput_ratio'],rel_tol=1e-12)
def test_candidate_failure_not_erased():
    r=json.loads((P/'v1-regression-decision.json').read_text())
    assert r['original224_8image_regression']=='failed' and r['prior_failed_diagnostic_retained']
def test_no_training_or_accuracy_promotion():
    h=json.loads((P/'headline.json').read_text());assert h['optimizer_steps']==0 and h['weights_unchanged']
    assert not h['new_training_launched'] and not h['bulk_scale_completed'] and h['not_new_accuracy_or_training_speed']
def test_generated_ids_and_diagnostics_match():
    q=json.loads((P/'fresh-v2/qualification.json').read_text())
    assert q['pass'] and len(q['rows'])==16
    assert all(r['true_tokens_equal'] and r['wrong_tokens_equal'] and r['diagnostics_max_abs']==0 for r in q['rows'])
def test_altered_evidence_is_rejected(tmp_path):
    copy=tmp_path/'case';shutil.copytree(P,copy);(copy/'headline.json').write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'):verifier(copy,'tampered_vision_verify').verify(copy)
