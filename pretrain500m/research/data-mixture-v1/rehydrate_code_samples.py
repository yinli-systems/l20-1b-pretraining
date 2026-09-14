"""Retrieve a bounded set of public Stack-Edu code files with identity checks.

SWH blob_id is the raw content SHA-1 (not Git's header-prefixed blob SHA-1).
Keep source license metadata; never run downloaded code.
"""
import concurrent.futures
import datetime as dt
import gzip
import hashlib
import io
import json
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent
LIMIT=1024*1024
LICENSES={'MIT','Apache-2.0','BSD-2-Clause','BSD-3-Clause','ISC','CC0-1.0','Unlicense'}

def fetch(row):
    url='https://softwareheritage.s3.amazonaws.com/content/'+row['blob_id']
    receipt={'blob_id':row['blob_id'],'url':url,'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        with requests.get(url,stream=True,timeout=20) as res:
            receipt['http_status']=res.status_code
            res.raise_for_status()
            payload=res.raw.read(LIMIT+1)
            if len(payload)>LIMIT:raise ValueError('compressed size cap exceeded')
        data=gzip.GzipFile(fileobj=io.BytesIO(payload)).read(LIMIT+1)
        if len(data)>LIMIT:raise ValueError('uncompressed size cap exceeded')
        sha1=hashlib.sha1(data).hexdigest()
        if sha1!=row['blob_id']:raise ValueError('raw content SHA-1 mismatch')
        text=data.decode('utf-8',errors='strict')
        receipt.update(status='CONTENT_IDENTITY_VERIFIED',bytes=len(data),sha1=sha1,
                       sha256=hashlib.sha256(data).hexdigest())
        return dict(row,text=text),receipt
    except Exception as exc:
        receipt.update(status='BLOCKED',error=str(exc).split('?')[0][:200]);return None,receipt

def main():
    rows=[json.loads(x) for x in (ROOT/'samples/stackedu_python_ids.jsonl').read_text().splitlines()]
    eligible=[r for r in rows if r.get('license_type')=='permissive' and
              set(r.get('detected_licenses') or []) and
              set(r['detected_licenses']) <= LICENSES and r.get('int_score',0)>=3]
    selected=eligible[:32]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(fetch,selected))
    accepted=[r for r,e in results if r]
    payload=('\n'.join(json.dumps(r,ensure_ascii=False) for r in accepted)+'\n').encode()
    (ROOT/'samples/stackedu_python_text.jsonl').write_bytes(payload)
    receipt={'method':'first up to 32 eligible rows within the pinned inspection sample',
             'eligible_license_and_score_rows':len(eligible),'attempted':len(selected),
             'verified_code_documents':len(accepted),'sample_sha256':hashlib.sha256(payload).hexdigest(),
             'sample_bytes':len(payload),'training_admitted':False,
             'still_required':['cross-source dedup','benchmark decontamination','repository split','full corpus audit'],
             'content_hash_note':'blob_id is raw-content SHA-1; do not compare it with sha1_git',
             'files':[e for r,e in results]}
    (ROOT/'code-rehydration-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['attempted','verified_code_documents','sample_bytes','training_admitted']}))

if __name__=='__main__':main()
