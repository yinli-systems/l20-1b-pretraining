"""Inspect one revision-pinned filtered book shard before enabling pilot B."""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

from continuation_common import RUN, atomic_json, lock, sha256
from source_docs import _download_parquet

REPO = 'common-pile/project_gutenberg_filtered'
REVISION = '3cdf6879c807f4e4e063f2ceb23bc268d8c29ab7'
FIRST_FILE = {'rfilename':'project_gutenberg-dolma-0000.json.gz','size':570083371,
              'lfs':{'sha256':'f7e3716e1e2be607a044920ffec69d1395598cca178d4ccbbd343f206b31effb'}}


def main():
    handle = lock(RUN/'narrative-audit.lock')
    path = _download_parquet(REPO,REVISION,FIRST_FILE)
    if sha256(path) != FIRST_FILE['lfs']['sha256']:
        raise RuntimeError('book shard hash mismatch')
    samples = []
    with gzip.open(path,'rt') as stream:
        for line in stream:
            row = json.loads(line)
            text = row.get('text','')
            samples.append({'id':row.get('id'),'keys':sorted(row),'metadata':row.get('metadata'),
                            'source':row.get('source'),'characters':len(text),
                            'head':text[:1800],'middle':text[len(text)//2:len(text)//2+1200]})
            if len(samples) >= 30:
                break
    record = {'repo':REPO,'revision':REVISION,'file':FIRST_FILE,'path':str(path),
              'sha256_verified':True,'samples':samples,'time':time.time(),
              'status':'pending_human_source_review','training_enabled':False}
    atomic_json(RUN/'narrative-audit.local.json',record)
    print(json.dumps({'status':record['status'],'sampled_documents':len(samples),
                      'receipt':str(RUN/'narrative-audit.local.json')}),flush=True)
    handle.close()


if __name__ == '__main__':
    main()
