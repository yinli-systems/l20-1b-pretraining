import gzip
import hashlib
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import intake


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    table = pa.table({'text': ['Owned test document number '+str(i) for i in range(12)],
                      'url': ['https://fixture.example/doc/'+str(i) for i in range(12)]})
    sink = io.BytesIO(); pq.write_table(table, sink, row_group_size=2)
    body = sink.getvalue()

    class MemoryHTTP(io.BytesIO):
        def __init__(self, url):
            super().__init__(body)
            self.total, self.transferred, self.ranges = len(body), 0, []

        def read(self, size=-1):
            data = super().read(size)
            self.transferred += len(data)
            return data

    prior_dir = tmp_path/'data/diverse-intake-v2';prior_dir.mkdir(parents=True)
    prior = dict(id='dclm', repo_id='owned/fixture', revision='fixture-v1', path='file.parquet',
                 status='RAW_SEGMENT_READY', groups=[{'row_group': 1}], exclude_row_groups=[0])
    prior_path = prior_dir/'dclm.receipt.json';prior_path.write_text(json.dumps(prior))
    output=tmp_path/'data/diverse-intake-v3';output.mkdir()
    monkeypatch.setattr(intake, 'ROOT', tmp_path);monkeypatch.setattr(intake, 'OUTPUT', output)
    monkeypatch.setattr(intake, 'MIN_FREE', 0)
    monkeypatch.setattr(intake.http_ranges, 'BoundedHTTPFile', MemoryHTTP)
    segment = {k:prior[k] for k in ['id','repo_id','revision','path']}
    segment.update(bytes=len(body), prior_receipt_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),
                   exclude_row_groups=[1], all_excluded_row_groups=[0,1], additional_row_groups=3)
    return segment, prior_path, output


def test_all_older_groups_are_excluded_but_immediate_link_is_preserved(fixture):
    segment, _, output = fixture
    result = intake.acquire(segment)
    assert result['status'] == 'RAW_SEGMENT_READY'
    assert result['exclude_row_groups'] == [1]
    assert result['all_excluded_row_groups'] == [0,1]
    assert len(result['groups']) == 3
    assert not {0,1}.intersection(g['row_group'] for g in result['groups'])
    with gzip.open(output/'dclm.jsonl.gz','rt') as handle:
        rows=[json.loads(line) for line in handle]
    assert len(rows)==6
    assert all(row['_provenance']['physical_row']>=4 for row in rows)


@pytest.mark.parametrize('damage,match', [('cumulative','cumulative row-group exclusion mismatch'),
    ('immediate','prior row-group exclusion mismatch'),('receipt','prior receipt changed'),
    ('identity','prior source identity mismatch')])
def test_incorrect_lineage_fails_without_ready_output(fixture, damage, match):
    segment, prior, output=fixture
    if damage=='cumulative':segment['all_excluded_row_groups']=[1]
    elif damage=='immediate':segment['exclude_row_groups']=[0,1]
    elif damage=='receipt':prior.write_text(prior.read_text()+' ')
    else:segment['repo_id']='different/fixture'
    result=intake.acquire(segment)
    assert result['status']=='BLOCKED' and match in result['error']
    assert not (output/'dclm.jsonl.gz').exists()


def test_more_than_two_prior_tranches_keep_transitive_exclusions(fixture):
    segment, prior_path, _=fixture
    prior=json.loads(prior_path.read_text());prior.update(groups=[{'row_group':2}],exclude_row_groups=[1],all_excluded_row_groups=[0,1])
    prior_path.write_text(json.dumps(prior))
    segment.update(prior_receipt_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),
                   exclude_row_groups=[2],all_excluded_row_groups=[0,1,2],additional_row_groups=2)
    result=intake.acquire(segment)
    assert result['status']=='RAW_SEGMENT_READY'
    assert all(g['row_group']>=3 for g in result['groups'])


def test_frozen_combined_auditor_accepts_three_disjoint_tranches(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'raw-audit-v2'))
    from test_audit_raw_intake_v2 import segment, summary, mutate_receipt
    from audit_raw_intake_v2 import preflight
    roots = [tmp_path/str(i) for i in range(3)]
    for i, root in enumerate(roots):
        root.mkdir()
        segment(root, 'dclm', i, ['Owned document '+str(i)],
                roots[i-1]/'dclm.receipt.json' if i else None)
        if i==2:
            mutate_receipt(root/'dclm.receipt.json', all_excluded_row_groups=[0,1])
        summary(root)
    files, _, tranches = preflight(roots)
    assert len(files)==3 and sum(t['rows'] for t in tranches)==3
