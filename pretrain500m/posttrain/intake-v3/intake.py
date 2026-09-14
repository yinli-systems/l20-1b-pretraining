"""Bounded revision-pinned raw corpus intake. Outputs remain NOT_ADMITTED."""
import ast
import concurrent.futures
import datetime
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time

import pyarrow.parquet as pq
import requests
import http_ranges

ROOT=Path('/ssd/scxi253/pretrain500m-20260912-v1')
OUTPUT=ROOT/'data/diverse-intake-v3'
SOURCE=Path(__file__).resolve().parent
http_ranges.CAP=128*1024**2
MAX_TEXT_BYTES=192*1024**2
MAX_ROWS=100000
MAX_CODE_ATTEMPTS=25000
MIN_FREE=18*1024**3
LICENSES={'MIT','Apache-2.0','BSD-2-Clause','BSD-3-Clause','ISC','CC0-1.0','Unlicense'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def atomic_json(path,data):
    tmp=path.with_suffix('.next')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,path)

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def code_text(row):
    licenses=set(row.get('detected_licenses') or [])
    if row.get('license_type')!='permissive' or not licenses or not licenses<=LICENSES or row.get('int_score',0)<3:
        return None,'LICENSE_OR_SCORE_EXCLUDED'
    blob=row.get('blob_id','')
    if len(blob)!=40 or any(c not in '0123456789abcdef' for c in blob):return None,'INVALID_BLOB_ID'
    try:
        with requests.get('https://softwareheritage.s3.amazonaws.com/content/'+blob,stream=True,timeout=(5,15)) as r:
            r.raise_for_status();compressed=r.raw.read(1024**2+1)
        if len(compressed)>1024**2:return None,'SIZE_EXCLUDED'
        content=gzip.GzipFile(fileobj=io.BytesIO(compressed)).read(1024**2+1)
        if len(content)>1024**2:return None,'SIZE_EXCLUDED'
        if hashlib.sha1(content).hexdigest()!=blob:return None,'CONTENT_HASH_MISMATCH'
        text=content.decode('utf-8',errors='strict')
        if row.get('language')=='Python':
            try:ast.parse(text)
            except (SyntaxError,ValueError):return None,'PYTHON_SYNTAX_EXCLUDED'
        return dict(row,text=text,content_sha256=hashlib.sha256(content).hexdigest()),'CONTENT_VERIFIED'
    except Exception as exc:return None,type(exc).__name__

def acquire(segment):
    sid=segment['id'];final=OUTPUT/(sid+'.jsonl.gz');receipt_path=OUTPUT/(sid+'.receipt.json')
    if receipt_path.exists():
        old=json.loads(receipt_path.read_text())
        if old.get('status')=='RAW_SEGMENT_READY' and final.exists() and sha(final)==old.get('output_sha256'):
            return old
        raise RuntimeError('existing incomplete source requires inspection, not silent overwrite: '+sid)
    if final.exists() or final.with_suffix('.part').exists():raise RuntimeError('output already exists: '+sid)
    receipt=dict(segment,status='READING',started_utc=utc(),training_admitted=False,
                 full_shard_sha256_verified=False,verification='revision-pinned HTTPS and range hashes; full-file LFS digest is expected metadata only',
                 rows_written=0,uncompressed_output_bytes=0,code_rehydration_counts={},groups=[])
    atomic_json(receipt_path,receipt)
    remote=None;rows_written=0;bytes_written=0;code_attempts=0
    temp=final.with_suffix('.part')
    try:
        url=f'https://hf-mirror.com/datasets/{segment["repo_id"]}/resolve/{segment["revision"]}/{segment["path"]}?download=true'
        remote=http_ranges.BoundedHTTPFile(url)
        if remote.total!=segment['bytes']:raise ValueError('pinned file size mismatch')
        parquet=pq.ParquetFile(remote)
        # Retain parent, repository, URL, quality and license metadata when present.
        columns=[x for x in parquet.schema_arrow.names if x in {
            'text','content','url','id','language','language_score','score','int_score',
            'blob_id','repo_name','path','detected_licenses','license_type','prompt',
            'seed_data','parent_document_id','seed_document_url','metadata','audience','format','dump','date'}]
        receipt['schema']=parquet.schema_arrow.names
        prior_path=ROOT/'data/diverse-intake-v2'/(sid+'.receipt.json')
        if sha(prior_path)!=segment['prior_receipt_sha256']:raise ValueError('prior receipt changed')
        prior=json.loads(prior_path.read_text())
        if prior.get('status')!='RAW_SEGMENT_READY':raise ValueError('prior source is not complete')
        if any(prior[k]!=segment[k] for k in ['id','repo_id','revision','path']):raise ValueError('prior source identity mismatch')
        excluded=set(segment['exclude_row_groups'])
        if excluded!={g['row_group'] for g in prior['groups']}:raise ValueError('prior row-group exclusion mismatch')
        # The receipt retains immediate-prior exclusions for the existing auditor.
        # Sampling also excludes the full transitive history, including capped groups.
        excluded |= set(prior.get('exclude_row_groups', [])) | set(prior.get('all_excluded_row_groups', []))
        if excluded != set(segment['all_excluded_row_groups']):raise ValueError('cumulative row-group exclusion mismatch')
        if not 1<=segment['additional_row_groups']<=256:raise ValueError('unbounded additional group request')
        groups=sorted((i for i in range(parquet.num_row_groups) if i not in excluded),key=lambda i:hashlib.sha256((sid+str(i)+'p529m-intake-v1').encode()).hexdigest())
        with gzip.open(temp,'wt',encoding='utf-8',compresslevel=3) as writer:
            for group in groups[:segment['additional_row_groups']]:
                if shutil.disk_usage(OUTPUT).free<MIN_FREE:
                    receipt['stop_reason']='DISK_HEADROOM';break
                metadata=parquet.metadata.row_group(group)
                compressed=sum(metadata.column(i).total_compressed_size for i in range(metadata.num_columns)
                               if metadata.column(i).path_in_schema.split('.')[0] in columns)
                if remote.transferred+compressed+65536>http_ranges.CAP:
                    receipt['stop_reason']='PER_SOURCE_NETWORK_CAP';break
                table=parquet.read_row_group(group,columns=columns,use_threads=False)
                raw_rows=table.to_pylist()
                row_offset=sum(parquet.metadata.row_group(i).num_rows for i in range(group))
                decorated=[]
                for i,row in enumerate(raw_rows):
                    row=dict(row)
                    row['_provenance']={'source_id':sid,'repo_id':segment['repo_id'],'revision':segment['revision'],
                                        'shard':segment['path'],'row_group':group,'row_in_group':i,'physical_row':row_offset+i}
                    decorated.append(row)
                receipt['groups'].append({'row_group':group,'physical_row_offset':row_offset,'rows_read':len(decorated)})
                if sid.startswith('code_'):
                    eligible=[row for row in decorated if row.get('license_type')=='permissive' and
                              set(row.get('detected_licenses') or []) and
                              set(row['detected_licenses'])<=LICENSES and row.get('int_score',0)>=3]
                    eligible=eligible[:max(0,MAX_CODE_ATTEMPTS-code_attempts)]
                    code_attempts+=len(eligible)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                        outputs=list(pool.map(code_text,eligible))
                    accepted=[]
                    for row,status in outputs:
                        receipt['code_rehydration_counts'][status]=receipt['code_rehydration_counts'].get(status,0)+1
                        if row is not None:accepted.append(row)
                else:accepted=decorated
                for row in accepted:
                    text=row.get('text') or row.get('content')
                    if not isinstance(text,str) or not text.strip():continue
                    payload=json.dumps(row,ensure_ascii=False,default=str)+'\n'
                    size=len(payload.encode())
                    if rows_written>=MAX_ROWS or bytes_written+size>MAX_TEXT_BYTES:
                        receipt['stop_reason']='OUTPUT_CAP';break
                    writer.write(payload);rows_written+=1;bytes_written+=size
                writer.flush()
                receipt.update(rows_written=rows_written,uncompressed_output_bytes=bytes_written,network_bytes=remote.transferred,updated_utc=utc())
                atomic_json(receipt_path,receipt)
                print(json.dumps({'source':sid,'rows':rows_written,'network_bytes':remote.transferred,'code_attempts':code_attempts}),flush=True)
                if receipt.get('stop_reason') or code_attempts>=MAX_CODE_ATTEMPTS:break
        if rows_written==0:raise ValueError('no complete usable text rows acquired within bounds')
        os.replace(temp,final)
        receipt.update(status='RAW_SEGMENT_READY',rows_written=rows_written,uncompressed_output_bytes=bytes_written,
                       output_sha256=sha(final),output_bytes=final.stat().st_size,completed_utc=utc())
    except Exception as exc:
        # Do not expose signed redirect URLs or arbitrary source text in logs.
        receipt.update(status='BLOCKED',error_type=type(exc).__name__,error=str(exc).split('?')[0][:200],updated_utc=utc())
    finally:
        if remote:receipt.update(network_bytes=remote.transferred,range_receipts=remote.ranges)
        atomic_json(receipt_path,receipt)
    print(json.dumps({k:receipt.get(k) for k in ('id','status','rows_written','output_bytes','error_type','stop_reason')}),flush=True)
    return receipt

def main():
    OUTPUT.mkdir(exist_ok=True)
    lock=OUTPUT/'writer.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        segments=json.loads((SOURCE/'segments.json').read_text())['segments']
        # Start the slower code rehydration first; all sources have measured deficits.
        segments.sort(key=lambda x:(not x['id'].startswith('code_'),-x['observed_token_shortage_lower_bound']))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:receipts=list(pool.map(acquire,segments))
        atomic_json(OUTPUT/'intake-summary.json',{'status':'RAW_INTAKE_FINISHED_NOT_ADMITTED','checked_utc':utc(),
                    'sources_ready':sum(x['status']=='RAW_SEGMENT_READY' for x in receipts),
                    'sources_blocked':sum(x['status']!='RAW_SEGMENT_READY' for x in receipts),
                    'rows':sum(x.get('rows_written',0) for x in receipts),'training_admitted':False})
    finally:lock.unlink()

if __name__=='__main__':main()
