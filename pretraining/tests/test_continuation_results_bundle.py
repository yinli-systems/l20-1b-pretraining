"""Compact published evidence can be verified without raw benchmark text/GPU."""
import hashlib
import json
from pathlib import Path

import pytest

from continuation_common import PILOT_STEPS, STEP_TOKENS, CAP

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('phase,receipt_name',[
    ('core','continuation-evaluation-A-20260911.json'),
    ('all','continuation-evaluation-A-complete-20260911.json'),
])
def test_A_summary_binds_its_own_receipt_snapshot(phase,receipt_name):
    summary=json.loads((ROOT/f'reports/metrics/continuation-A-{phase}-20260911.json').read_text())
    receipt=ROOT/'reports/receipts'/receipt_name
    assert hashlib.sha256(receipt.read_bytes()).hexdigest()==summary['evaluation_receipt_sha256']
    assert summary['identity_verification']['status']=='declared_artifacts_verified'
    assert summary['ours_primary_macro']==pytest.approx(.5100527379898937)
    comparison=summary['comparisons']['original20B']['primary_macro']
    assert comparison['paired_95_ci_pp'][0]<0<comparison['paired_95_ci_pp'][1]
    assert summary['claims']=={'sota_asserted':False,'automatic_training_promotion':False}


def test_B_admission_has_budget_coverage_and_exact_programs():
    review=json.loads((ROOT/'reports/receipts/continuation-quality-review-B-20260911.json').read_text())
    assert review['branch']=='B' and review['decision']=='pass'
    assert review['automatic_promotion'] is False and review['semantic_error_rate'] is None
    assert review['pilot_prediction_tokens']==PILOT_STEPS*STEP_TOKENS
    assert review['experiment_prediction_token_cap']==CAP
    assert review['narrative_source']['final_intact_books']==98
    for source,required in review['required_input_tokens'].items():
        assert review['available_input_tokens'][source]>=required
    for name,digest in review['program_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
