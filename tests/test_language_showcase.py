"""Keep the public comparison honest without running new model inference."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from tools import build_language_showcase as showcase

ROOT=Path(__file__).resolve().parents[1]

def test_full_frozen_checkpoint_inventory():
    data=showcase.read_source(ROOT)
    assert len(data['checkpoints'])==36
    assert len({x['job'] for x in data['checkpoints']})==36
    assert data['ours']['parameters']==1_100_048_384
    assert data['ours']['training_tokens']==19_999_703_040

def test_six_task_sensitivity_is_visible():
    table=showcase.main_table(showcase.read_source(ROOT))
    assert '6-task' in table and '**49.94**' in table and '**51.26**' in table
    readme=(ROOT/'README.md').read_text()
    assert 'not explicitly decontaminated' in readme and 'BoolQ' in readme

def test_stronger_baselines_are_not_hidden():
    data=showcase.read_source(ROOT);lookup={x['job']:x for x in data['checkpoints']}
    for job in ['phi_1_5','falcon_rw_1b','tinyllama_2_5t','dd_dclm_baseline_qc_7p_fw3_12500','weborganizer_domain_mix']:
        assert job in showcase.SELECTED
        assert lookup[job]['seven_task_macro']['baseline']*100>data['ours']['seven_task_macro_percent']

def test_per_task_scores_reconstruct_frozen_macros():
    data=showcase.read_source(ROOT)
    for row in data['checkpoints']:
        for key in ['ours','baseline']:
            scores=[row['tasks'][t][key] for t in data['protocol']['tasks']]
            assert sum(scores)/7==pytest.approx(row['seven_task_macro'][key],abs=1e-12)
            six=[row['tasks'][t][key] for t in data['protocol']['tasks'] if t!='boolq']
            assert sum(six)/6==pytest.approx(row['six_task_without_boolq'][key],abs=1e-12)

def test_all_252_task_protocol_alignments_are_retained():
    rows=showcase.read_source(ROOT)['checkpoints']
    assert sum(len(x['tasks']) for x in rows)==252
    assert all(t['protocol_alignment']=='PASS' for row in rows for t in row['tasks'].values())

def test_frozen_source_tampering_is_rejected(tmp_path):
    for name in [showcase.SOURCE,'pretraining/reproducibility/manifest.json']:
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    path=tmp_path/showcase.SOURCE;data=json.loads(path.read_text());data['ours']['seven_task_macro_percent']=99
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Frozen language evidence hash mismatch'):
        showcase.read_source(tmp_path)

def test_generated_readme_and_figure_match_evidence():
    result=showcase.run(True,ROOT)
    assert result['status']=='verified' and not result['new_model_inference']

def test_exposure_is_not_sold_as_training_speed():
    text=(ROOT/'README.md').read_text()
    assert '50× less reported token exposure' in text
    assert 'does **not** establish 50× lower cost' in text
    assert 'intervals cross zero' in text

def test_vlm_retention_tradeoff_remains_visible():
    text=(ROOT/'README.md').read_text()
    assert '56.25% → 50.86%' in text and 'not promoted' in text
    assert 'AI2D includes' in text and 'answer-format repair' in text

def test_exact_baseline_identity_and_revisions_available():
    data=json.loads((ROOT/'results/language-model/comparison.json').read_text())
    assert len(data['checkpoints'])==36 and not data['new_inference_performed']
    assert all(len(r['revision'])==40 and '/' in r['repo'] for r in data['checkpoints'])


def test_repository_identity_and_model_identity_are_separate():
    text=(ROOT/'CITATION.cff').read_text()
    assert 'cff-version: 1.2.0' in text and 'type: software' in text
    assert 'yinli-systems/l20-pretraining-lab' in text
    readme=(ROOT/'README.md').read_text()
    assert 'AliceYin/L20-1B-20B-Base' in readme
    assert 'git clone https://github.com/yinli-systems/l20-pretraining-lab.git' in readme
