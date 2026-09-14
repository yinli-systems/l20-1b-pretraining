"""Exercise cross-tranche accounting and rejection paths on owned fixture data."""
import gzip
import hashlib
import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from audit_raw_intake_v2 import audit, preflight, sha


def write_json(path, value):
    path.write_text(json.dumps(value))


def segment(root, sid, group, texts, prior=None):
    receipt = dict(id=sid, status='RAW_SEGMENT_READY', repo_id='owned-fixture/' + sid,
                   revision='frozen-fixture-v1', path='fixture.parquet',
                   groups=[dict(row_group=group, physical_row_offset=group*100, rows_read=len(texts))])
    if prior:
        receipt.update(prior_receipt_sha256=sha(prior),
                       exclude_row_groups=[g['row_group'] for g in json.loads(prior.read_text())['groups']])
    raw = root / (sid + '.jsonl.gz')
    with gzip.open(raw, 'wt') as handle:
        for i, text in enumerate(texts):
            row = dict(text=text, url='https://owned-fixture.example/document/' + str(i),
                       _provenance=dict(source_id=sid, repo_id=receipt['repo_id'], revision=receipt['revision'],
                                        shard=receipt['path'], row_group=group, row_in_group=i, physical_row=group*100+i))
            handle.write(json.dumps(row) + '\n')
    receipt.update(rows_written=len(texts), output_sha256=sha(raw), output_bytes=raw.stat().st_size)
    write_json(root / (sid + '.receipt.json'), receipt)


def summary(root):
    receipts = [json.loads(p.read_text()) for p in root.glob('*.receipt.json')]
    write_json(root / 'intake-summary.json', dict(status='RAW_INTAKE_FINISHED_NOT_ADMITTED',
               sources_blocked=0, sources_ready=len(receipts), rows=sum(r['rows_written'] for r in receipts)))


@pytest.fixture
def pool(tmp_path):
    first, second = tmp_path/'first', tmp_path/'second'
    first.mkdir(); second.mkdir()
    lower, upper = 'alpha '*2049, 'ALPHA '*2049
    segment(first, 'dclm', 0, [lower])
    segment(first, 'pdf_en', 0, [lower])
    segment(second, 'dclm', 1, [lower, upper], first/'dclm.receipt.json')
    summary(first); summary(second)
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, 'alpha': 1, 'ALPHA': 2}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    token_path = tmp_path/'tokenizer.json'
    tokenizer.save(str(token_path))
    quotas = tmp_path/'quotas.json'
    write_json(quotas, {'sequence_length': 2048, 'screen': {'M1': {'text': {'leaves': {
        'dclm': {'prediction_tokens': 8192}, 'finepdfs_edu': {'prediction_tokens': 8192}}}}}})
    return first, second, token_path, quotas, tmp_path/'audit'


def run(pool):
    first, second, tokenizer, quotas, output = pool
    return audit([first, second], tokenizer, sha(tokenizer), quotas, output)


def mutate_receipt(path, **updates):
    value = json.loads(path.read_text()); value.update(updates); write_json(path, value)


def test_cross_tranche_global_dedup_and_normalized_candidate_is_retained(pool):
    report = run(pool)
    assert report['rows'] == 4
    assert report['globally_byte_identical_unique_documents'] == 2
    assert report['byte_identical_overlap_rows'] == {'cross_tranche_same_source': 1, 'cross_source': 1}
    assert report['normalized_nonidentical_candidates'] == {'normalized_nonidentical_rows': 1}
    assert report['sources']['dclm']['globally_deduplicated_assigned_prediction_tokens_upper_bound'] == 4096
    assert report['sources']['dclm']['shortage_lower_bound_before_validation_and_quality_filters'] == 0
    assert report['sources']['pdf_en']['globally_deduplicated_assigned_prediction_tokens_upper_bound'] == 0
    assert report['sources']['pdf_en']['shortage_lower_bound_before_validation_and_quality_filters'] == 4096
    index = [json.loads(line) for line in (pool[-1]/'row-index.jsonl').read_text().splitlines()]
    assert index[1]['exact_dedup_owner'] == {'source_id': 'dclm', 'tranche': 0, 'row': 0}
    assert report['training_admitted'] is False


@pytest.mark.parametrize('damage,match', [
    ('group', 'repeated physical group'), ('prior', 'prior receipt identity mismatch'),
    ('exclusion', 'prior group exclusions mismatch'), ('writer', 'still has a writer'),
    ('blocked', 'blocked sources'), ('missing_raw', 'raw/receipt file set mismatch'),
    ('raw_checksum', 'raw checksum mismatch'), ('duplicate_input', 'duplicate intake directory')])
def test_input_rejections_before_output(pool, damage, match):
    first, second, _, _, output = pool
    p = second/'dclm.receipt.json'
    if damage == 'group':
        mutate_receipt(p, groups=[{'row_group': 0, 'physical_row_offset': 0, 'rows_read': 2}])
    elif damage == 'prior':
        mutate_receipt(p, prior_receipt_sha256='0'*64)
    elif damage == 'exclusion':
        mutate_receipt(p, exclude_row_groups=[])
    elif damage == 'writer':
        (second/'writer.lock').write_text('123')
    elif damage == 'blocked':
        mutate_receipt(second/'intake-summary.json', sources_blocked=1)
    elif damage == 'missing_raw':
        (second/'dclm.jsonl.gz').unlink()
    elif damage == 'raw_checksum':
        with (second/'dclm.jsonl.gz').open('ab') as handle:
            handle.write(b'changed')
    with pytest.raises(ValueError, match=match):
        if damage == 'duplicate_input':
            preflight([first, first])
        else:
            run(pool)
    assert not output.exists()


@pytest.mark.parametrize('damage,match', [
    ('physical_row', 'physical row offset mismatch'), ('source_id', 'row provenance/source mismatch'),
    ('duplicate_row', 'repeated physical row'), ('content_sha256', 'text hash mismatch')])
def test_bad_rows_do_not_produce_success_report(pool, damage, match):
    raw = pool[1]/'dclm.jsonl.gz'
    with gzip.open(raw, 'rt') as handle:
        rows = [json.loads(line) for line in handle]
    if damage == 'physical_row':
        rows[0]['_provenance']['physical_row'] = 9999
    elif damage == 'source_id':
        rows[0]['_provenance']['source_id'] = 'pdf_en'
    elif damage == 'duplicate_row':
        rows[1]['_provenance'] = rows[0]['_provenance'].copy()
    else:
        rows[0]['content_sha256'] = hashlib.sha256(b'incorrect').hexdigest()
    with gzip.open(raw, 'wt') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    mutate_receipt(pool[1]/'dclm.receipt.json', output_sha256=sha(raw), output_bytes=raw.stat().st_size)
    with pytest.raises(ValueError, match=match):
        run(pool)
    assert not (pool[-1]/'report.json').exists()


def test_receipt_change_during_scan_is_rejected(pool, monkeypatch):
    import audit_raw_intake_v2 as module
    original = module.family
    changed = False

    def change_receipt(row, sid):
        nonlocal changed
        if not changed:
            mutate_receipt(pool[0]/'dclm.receipt.json', changed_after_preflight=True)
            changed = True
        return original(row, sid)

    monkeypatch.setattr(module, 'family', change_receipt)
    with pytest.raises(ValueError, match='input changed during audit'):
        run(pool)
    assert not (pool[-1]/'report.json').exists()
