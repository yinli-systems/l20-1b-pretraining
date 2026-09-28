import json

import pytest

import prepare_continuation_b_verified as verifier


def test_replaced_valid_index_is_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(verifier,'RUN',tmp_path)
    index = tmp_path/'indices/old-train.npy'
    index.parent.mkdir()
    index.write_bytes(b'original')
    receipt = {'indices':{'train':{'path':str(index),'sha256':verifier.sha256(index)}}}
    (tmp_path/'base-inputs.json').write_text(json.dumps(receipt))
    index.write_bytes(b'valid looking but changed')
    with pytest.raises(RuntimeError,match='train exclusion index mismatch'):
        verifier.verify_inputs()
