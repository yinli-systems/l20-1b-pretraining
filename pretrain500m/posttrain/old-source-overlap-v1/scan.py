"""Bounded parallel comparison with the immutable old FineWeb source snapshot."""
import argparse
from collections import Counter, defaultdict
import datetime
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time
import unicodedata
from urllib.parse import urlsplit

import pyarrow.parquet as pq

LOOKUP = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def normalized_hash(text):
    return hashlib.sha256(' '.join(unicodedata.normalize('NFKC', text).lower().split()).encode()).hexdigest()


def write_json(path, value):
    tmp=path.with_suffix(path.suffix+'.next')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def scan_shard(item):
    spec, output = item
    path=Path(spec['path']);out=Path(output);started=time.monotonic()
    before=path.stat()
    if sha(path) != spec['sha256']:
        raise ValueError('old source shard identity mismatch')
    parquet=pq.ParquetFile(path)
    required={'text','url','language','language_score','int_score'}
    if not required <= set(parquet.schema_arrow.names):
        raise ValueError('old source metadata fields missing')
    hits=out/(path.name+'.matches.jsonl.gz');hosts=Counter();rows=matched=eligible_matched=0
    with gzip.open(hits,'xt',encoding='utf-8',compresslevel=1) as target:
        for batch in parquet.iter_batches(batch_size=512,columns=sorted(required),use_threads=False):
            for row in batch.to_pylist():
                row_number=rows;rows+=1
                text=row['text']
                eligible=(isinstance(text,str) and bool(text.strip()) and row['language']=='en'
                          and float(row['language_score'] or 0)>=0.90 and int(row['int_score'] or 0)>=3)
                if eligible:
                    try:host=(urlsplit(row['url'] or '').hostname or '').lower().rstrip('.')
                    except ValueError:host=''
                    if host:hosts[host]+=1
                if not isinstance(text,str) or not text.strip():continue
                h=normalized_hash(text)
                if h not in LOOKUP:continue
                matched+=1;eligible_matched+=int(eligible)
                target.write(json.dumps({'old_source_path':str(path),'old_physical_row':row_number,
                    'old_url':row['url'],'passes_original_language_and_score_filters':eligible,
                    'old_text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                    'legacy_normalized_sha256':h,'new_documents':LOOKUP[h]},ensure_ascii=False)+'\n')
            if rows%32768==0:
                progress={'status':'RUNNING','rows':rows,'matched_old_rows':matched,
                          'elapsed_seconds':time.monotonic()-started}
                write_json(out/(path.name+'.progress.json'),progress)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
        raise ValueError('old source changed during scan')
    if rows!=parquet.metadata.num_rows:raise ValueError('old row count mismatch')
    host_path=out/(path.name+'.eligible-hosts.json')
    host_path.write_text(json.dumps(dict(sorted(hosts.items())))+'\n')
    result={'status':'COMPLETED','path':str(path),'sha256':spec['sha256'],'rows':rows,
        'matched_old_rows':matched,'matched_old_rows_passing_initial_filters':eligible_matched,
        'matches':str(hits),'matches_sha256':sha(hits),'host_counts':str(host_path),
        'host_counts_sha256':sha(host_path),'elapsed_seconds':time.monotonic()-started}
    write_json(out/(path.name+'.progress.json'),result)
    return result


def run(args):
    global LOOKUP
    started=time.monotonic()
    if not 1<=args.workers<=4:raise ValueError('worker count must be 1..4')
    if sha(args.manifest)!=args.expected_manifest_sha256:raise ValueError('manifest identity mismatch')
    if sha(args.index_report)!=args.expected_index_report_sha256:raise ValueError('new index report identity mismatch')
    manifest=json.loads(args.manifest.read_text());report=json.loads(args.index_report.read_text())
    if report['status']!='LEGACY_HASH_SCAN_COMPLETE_NOT_ADMITTED':raise ValueError('new index is incomplete')
    if len(manifest['files'])!=14 or len({x['path'] for x in manifest['files']})!=14:
        raise ValueError('expected fourteen distinct old source files')
    root=Path(manifest['source_root'])
    if {str(p) for p in root.glob('*.parquet')}!={x['path'] for x in manifest['files']}:
        raise ValueError('old source file set mismatch')
    lookup=defaultdict(list);rows=0
    for f in report['files']:
        index=Path(f['index'])
        if sha(index)!=f['index_sha256']:raise ValueError('new normalized index identity mismatch')
        with gzip.open(index,'rt') as inp:
            current=0
            for line in inp:
                row=json.loads(line);h=row.pop('legacy_normalized_sha256');lookup[h].append(row);rows+=1;current+=1
        if current!=f['rows']:raise ValueError('new index row count mismatch')
        if sha(index)!=f['index_sha256']:raise ValueError('new normalized index changed during read')
    if rows!=report['rows']:raise ValueError('new total row count mismatch')
    LOOKUP=dict(lookup)
    if shutil.disk_usage(args.output.parent).free<18*1024**3:raise ValueError('insufficient output headroom')
    args.output.mkdir(exist_ok=False)
    (args.output/'launch.json').write_text(json.dumps({'status':'RUNNING','workers':args.workers,
        'new_rows':rows,'new_normalized_hashes':len(LOOKUP),'old_shards':14,'training_admitted':False})+'\n')
    with mp.get_context('fork').Pool(args.workers) as pool:
        results=list(pool.imap_unordered(scan_shard,[(f,str(args.output)) for f in manifest['files']],chunksize=1))
    if sha(args.manifest)!=args.expected_manifest_sha256 or sha(args.index_report)!=args.expected_index_report_sha256:
        raise ValueError('identity metadata changed')
    if {str(p) for p in root.glob('*.parquet')}!={x['path'] for x in manifest['files']}:
        raise ValueError('old source file set changed')
    final={'status':'OLD_SOURCE_NORMALIZED_OVERLAP_COMPLETE_NOT_ADMITTED','training_admitted':False,
        'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-started,
        'manifest_sha256':args.expected_manifest_sha256,'new_index_report_sha256':args.expected_index_report_sha256,
        'new_rows':rows,'old_rows':sum(x['rows'] for x in results),
        'matched_old_rows':sum(x['matched_old_rows'] for x in results),
        'files':sorted(results,key=lambda x:x['path']),
        'scope':'NFKC/lower/whitespace full-document SHA256 equality against all fourteen original source shards.',
        'limits':['Source-snapshot matches do not prove exact formal-pack membership; initial language/score eligibility is recorded separately.',
                  'Case/whitespace normalization does not establish code equivalence; matches may support conservative exclusions.',
                  'This does not detect paraphrases, translated overlaps or arbitrary near duplicates.',
                  'Host counts are metadata for later family checks, not certified independent family counts.',
                  'No shared corpus files were written, deleted, copied or repacked.']}
    tmp=args.output/'report.next';tmp.write_text(json.dumps(final,indent=2)+'\n');tmp.replace(args.output/'report.json')
    print(json.dumps({k:final[k] for k in ['status','elapsed_seconds','new_rows','old_rows','matched_old_rows']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--expected-manifest-sha256',required=True)
    p.add_argument('--index-report',type=Path,required=True);p.add_argument('--expected-index-report-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    run(p.parse_args())
