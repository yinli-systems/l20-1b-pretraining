"""Verify shared Parquet read-only, with official sizes and hashes."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq


def check(args):
    root,item=args
    path=root/item['rfilename']
    before=path.stat()
    expected=item['lfs']['sha256']
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    assert before.st_size==item['size'],str(path)
    assert h.hexdigest()==expected,str(path)
    parquet=pq.ParquetFile(path)
    assert 'text' in parquet.schema.names
    samples=[]
    # Read from first/middle/last row groups, retaining only aggregate properties.
    for index in sorted(set([0,parquet.num_row_groups//2,parquet.num_row_groups-1])):
        batch=parquet.read_row_group(index)
        rows=batch.slice(0,32).to_pylist()
        assert rows and all(isinstance(x['text'],str) and x['text'] for x in rows)
        samples.extend(rows)
    after=path.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    record={'path':str(path),'bytes':before.st_size,'sha256':expected,'rows':parquet.metadata.num_rows,
            'row_groups':parquet.num_row_groups,'schema':str(parquet.schema_arrow),
            'sampled_rows':len(samples),'sample_mean_chars':sum(len(x['text']) for x in samples)/len(samples),
            'sample_languages':sorted({str(x.get('language')) for x in samples}),
            'sample_scores':sorted({str(x.get('int_score')) for x in samples}),
            'unchanged_during_read':True}
    print(json.dumps({k:v for k,v in record.items() if k!='schema'}),flush=True)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();m=json.loads(a.manifest.read_text());start=time.monotonic()
    with ThreadPoolExecutor(2) as pool:records=list(pool.map(check,[(a.root,x) for x in m['files']]))
    receipt={'status':'PASS_UPSTREAM_HASH_FOOTER_SCHEMA_AND_STRATIFIED_ROW_READ',
             'revision':m['revision'],'job_id':os.getenv('SLURM_JOB_ID'),'files':records,
             'total_bytes':sum(x['bytes'] for x in records),'total_rows':sum(x['rows'] for x in records),
             'seconds':time.monotonic()-start,
             'limitations':['Hash match establishes artifact identity; complete row scan and curation remain separate gates.',
                            'No training data quality, privacy or benchmark decontamination claim follows from this receipt.']}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(receipt,f,indent=2)
