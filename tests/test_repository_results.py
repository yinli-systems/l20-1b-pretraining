from pathlib import Path
import json
import math
import pytest
from tools.verify_repository import ROOT, VerificationError, close, regular, verify_migration, verify_results, verify_active_links


def metrics():
    return json.loads((ROOT/'results/2026-09-28/metrics.json').read_text())

def test_all_historical_bytes_survive_migration():
    assert verify_migration(ROOT)['original_files_preserved'] == 646

def test_headline_scores_recompute_from_paired_counts():
    r = verify_results(ROOT)
    assert r['questions'] == 15937 and r['paired_image_clusters'] == 6774
    assert r['fresh_model_inference'] is False

def test_active_documentation_has_no_broken_relative_links():
    assert verify_active_links(ROOT)['relative_links_checked'] > 40

def test_prepared_used_and_target_are_different():
    t = metrics()['training']
    assert t['new_image_events_used_in_saved_lineage'] == 163840
    assert t['prepared_images'] == 165376
    assert t['maximum_new_image_target'] == 250000
    assert not t['target_reached']

def test_retention_failure_is_not_hidden():
    m = metrics()
    assert not m['retention']['checks_at_4864']['old_QA']
    assert not m['retention']['checks_at_5120']['old_QA']
    assert m['retention']['consecutive_failures'] == 2
    assert not m['training']['automatic_promotion']

def test_chartqa_unresolved_and_ai2d_confounded():
    m = metrics()
    assert m['tasks']['chartqa']['paired_ci95'][0] < 0 < m['tasks']['chartqa']['paired_ci95'][1]
    assert any('format' in x.lower() for x in m['limitations'])
    assert m['evaluation']['known_previous_ChartQA_overlap_questions'] == 2

def test_numerical_mismatch_fails_closed():
    with pytest.raises(VerificationError):
        close(0.2, 0.3, 'tamper')
    with pytest.raises(VerificationError):
        close(math.nan, 0.3, 'nan')

def test_missing_or_escaping_artifact_rejected(tmp_path):
    with pytest.raises(VerificationError):
        regular(tmp_path, 'missing.json')
    outside = tmp_path.parent/'outside-evidence.txt'
    outside.write_text('x')
    with pytest.raises(VerificationError):
        regular(tmp_path, '../outside-evidence.txt')

def test_symlink_not_accepted_as_regular_evidence(tmp_path):
    p = tmp_path/'evidence.json';p.write_text('{}')
    q = tmp_path/'link.json';q.symlink_to(p)
    with pytest.raises(VerificationError):
        regular(tmp_path, 'link.json')
