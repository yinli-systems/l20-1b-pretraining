"""Fetch bounded samples from pinned, public Parquet files, with receipts.

One hash-selected shard and row group per named stratum: engineering inspection
only, NOT a representative quality estimate or an admitted training corpus.
No authentication, remote code, or gated downloads. Network cap 48 MiB/source.
"""
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import io
import json
import statistics
from pathlib import Path
import re
import requests
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
CAP = 48 * 1024**2
STRATA = [
 ('fineweb_edu','HuggingFaceFW/fineweb-edu','data/CC-MAIN-2024-'),
 ('dclm','mlfoundations/dclm-baseline-1.0-parquet','filtered/'),
 ('finepdfs_en','HuggingFaceFW/finepdfs-edu','data/eng_Latn/train/'),
 ('finemath4','HuggingFaceTB/finemath','finemath-4plus/'),
 ('infiwebmath4','HuggingFaceTB/finemath','infiwebmath-4plus/'),
 ('stackedu_python_ids','HuggingFaceTB/stack-edu','Python/'),
 ('cosmopedia2','HuggingFaceTB/cosmopedia-v2','cosmopedia-v2/'),
 ('fineweb2hq_zh','epfml/FineWeb2-HQ','cmn_Hani/'),
 ('fineweb2hq_es','epfml/FineWeb2-HQ','spa_Latn/'),
 ('python_edu','HuggingFaceTB/smollm-corpus','python-edu/'),
 ('smol_smoltalk','HuggingFaceTB/smol-smoltalk','data/train-'),
]
SUPPLEMENTAL = [
 ('dclm_100bt','HuggingFaceFW/dclm_100BT','data/'),
 ('nemotron_mathtextbooks','nvidia/Nemotron-Pretraining-Specialized-v1','Nemotron-Pretraining-Math-Textbooks/'),
 ('nemotron_wikirewrite','nvidia/Nemotron-Pretraining-Specialized-v1','Nemotron-Pretraining-Wiki-Rewrite/'),
]

class BoundedHTTPFile(io.RawIOBase):
    def __init__(self,url):
        super().__init__()
        self.url=url; self.pos=0; self.transferred=0; self.ranges=[]
        self.total=None
        self.fetch(0,1)
    def fetch(self,start,n):
        if n < 0 or self.transferred+n>CAP:
            raise ValueError('network budget exceeded; choose a smaller row group or shard')
        if n==0: return b''
        with requests.get(self.url, headers={'Range':f'bytes={start}-{start+n-1}',
                          'Accept-Encoding':'identity'},stream=True,timeout=30) as res:
            if res.status_code!=206:
                raise ValueError(f'Range request rejected with HTTP {res.status_code}')
            cr=res.headers.get('Content-Range','')
            match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',cr)
            if not match or int(match[1])!=start or int(match[2])+1!=start+n:
                raise ValueError('invalid Content-Range')
            total=int(match[3])
            if self.total is not None and self.total!=total: raise ValueError('file size drift')
            self.total=total
            chunks=[]; count=0
            for part in res.iter_content(1024*1024):
                count+=len(part); self.transferred+=len(part)
                if count>n or self.transferred>CAP: raise ValueError('range body exceeded bound')
                chunks.append(part)
            payload=b''.join(chunks)
            if len(payload)!=n: raise ValueError('short range response')
            self.ranges.append({'offset':start,'length':n,'sha256':hashlib.sha256(payload).hexdigest()})
            return payload
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self,offset,whence=0):
        pos = offset if whence==0 else self.pos+offset if whence==1 else self.total+offset
        if not 0<=pos<=self.total: raise ValueError('invalid seek')
        self.pos=pos; return pos
    def read(self,n=-1):
        if n<0: n=self.total-self.pos
        n=min(n,self.total-self.pos)
        data=self.fetch(self.pos,n); self.pos+=len(data); return data
    def readinto(self,buf):
        data=self.read(len(buf));buf[:len(data)]=data;return len(data)

def selected(items,salt):
    return min(items,key=lambda x:hashlib.sha256((salt+str(x)).encode()).hexdigest())

def sample(target,inventory):
    key,repo,prefix=target; info=inventory[repo]
    result={'id':key,'repo_id':repo,'revision':info['sha'], 'prefix':prefix,
            'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'method':'hash-selected single shard and row group; up to 128 rows',
            'training_admitted':False,'representative_quality_estimate':False}
    remote=None
    try:
        if info.get('gated'): raise ValueError('gated; excluded')
        candidates=[p for p in info['files'] if p.startswith(prefix) and p.endswith('.parquet')]
        if not candidates: raise ValueError('no matching Parquet files')
        path=selected(candidates,'p529m-research-20260913')
        url=f'https://huggingface.co/datasets/{repo}/resolve/{info["sha"]}/{path}'
        result.update(path=path,url=url,candidate_shards=len(candidates))
        remote=BoundedHTTPFile(url)
        pf=pq.ParquetFile(remote)
        fields=pf.schema_arrow.names
        group=selected(list(range(pf.num_row_groups)),key)
        allowed=['text','content','messages','conversations','problem','generated_solution',
                 'url','id','blob_id','repo_name','path','language','score','int_score',
                 'detected_licenses','license_type','source','prompt','completion']
        cols=[x for x in allowed if x in fields]
        result.update(schema=fields,row_groups=pf.num_row_groups,selected_row_group=group,
                      file_bytes=remote.total,row_group_rows=pf.metadata.row_group(group).num_rows)
        table=pf.read_row_group(group,columns=cols,use_threads=False)
        n=min(128,table.num_rows)
        # Spread inspection rows through this row group, instead of only its head.
        indices=sorted({i*table.num_rows//n for i in range(n)}) if n else []
        rows=table.take(indices).to_pylist()
        payload=('\n'.join(json.dumps(r,ensure_ascii=False,default=str) for r in rows)+'\n').encode()
        folder=ROOT/'samples';folder.mkdir(exist_ok=True)
        (folder/f'{key}.jsonl').write_bytes(payload)
        texts=[r.get('text') or r.get('content') or '' for r in rows]
        lengths=[len(t) for t in texts if isinstance(t,str)]
        hashes=[hashlib.sha256(re.sub(r'\s+',' ',t).strip().encode()).hexdigest() for t in texts if isinstance(t,str) and t.strip()]
        result.update(status='SAMPLE_DOWNLOADED',rows=len(rows),sample_bytes=len(payload),
                      sample_sha256=hashlib.sha256(payload).hexdigest(),
                      text_rows=sum(bool(t) for t in texts),
                      local_sample=f'samples/{key}.jsonl',
                      median_chars=statistics.median(lengths) if lengths else None,
                      within_sample_exact_duplicates=len(hashes)-len(set(hashes)))
    except Exception as exc:
        result.update(status='BLOCKED',error=str(exc).split('?')[0][:350])
    finally:
        if remote: result.update(network_bytes=remote.transferred,range_receipts=remote.ranges)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--supplemental',action='store_true')
    args=parser.parse_args()
    inventory={x['repo_id']:x for x in json.loads((ROOT/'source-inventory.json').read_text())}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(lambda t:sample(t,inventory),SUPPLEMENTAL if args.supplemental else STRATA))
    filename='supplemental-sample-audit.json' if args.supplemental else 'sample-audit.json'
    (ROOT/filename).write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
    for result in results:
        print(json.dumps({k:result.get(k) for k in ['id','status','rows','text_rows','sample_bytes','network_bytes','error']}),flush=True)

if __name__=='__main__': main()
