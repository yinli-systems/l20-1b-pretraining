"""Resumable immutable downloader with exact size/hash checks and one writer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()

def fetch(root,item):
    dest=root/Path(item['path']).name
    partial=dest.with_suffix(dest.suffix+'.partial')
    if dest.exists():
        assert dest.stat().st_size==item['bytes'] and digest(dest)==item['sha256']
        return {'path':str(dest),'bytes':dest.stat().st_size,'sha256':item['sha256'],'reused':True}
    offset=partial.stat().st_size if partial.exists() else 0
    request=urllib.request.Request(item['url'],headers={'Range':f'bytes={offset}-'} if offset else {})
    with urllib.request.urlopen(request,timeout=120) as response:
        if offset and response.status!=206:
            raise RuntimeError(f'server refused resume for {partial}')
        with partial.open('ab') as f:
            while True:
                block=response.read(8*1024**2)
                if not block:break
                f.write(block)
    assert partial.stat().st_size==item['bytes'],(partial,partial.stat().st_size,item['bytes'])
    assert digest(partial)==item['sha256'],f'hash mismatch retained: {partial}'
    partial.rename(dest)
    print(json.dumps({'verified':str(dest),'bytes':dest.stat().st_size}),flush=True)
    return {'path':str(dest),'bytes':dest.stat().st_size,'sha256':item['sha256'],'reused':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True);m=json.loads(a.manifest.read_text());start=time.time()
    with ThreadPoolExecutor(a.workers) as pool:files=list(pool.map(lambda x:fetch(a.output_dir,x),m['files']))
    receipt={'status':'PASS_ALL_SIZE_SHA256','source_manifest_sha256':digest(a.manifest),
             'started_unix':start,'completed_unix':time.time(),'files':files}
    with a.receipt.open('x') as f:json.dump(receipt,f,indent=2)
