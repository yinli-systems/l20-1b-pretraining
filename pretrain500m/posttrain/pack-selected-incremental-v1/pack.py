"""Materialize bound selected rows; does not grant corpus admission."""
import argparse
from collections import defaultdict
import datetime
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

import numpy as np
from tokenizers import Tokenizer

ROWS={};ROOTS=[];RAW_FILES={};BINDINGS={};TOKENIZER=None;EOS=None;OUTPUT=None;RETENTION={}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.next');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def initialize(path):
    global TOKENIZER,EOS
    TOKENIZER=Tokenizer.from_file(path);EOS=TOKENIZER.token_to_id('<|endoftext|>')
    if EOS!=50279:raise ValueError('tokenizer EOS differs from frozen model')


def write_span(array,offset,values,tail):
    """Retain every encoded token, including the final incomplete train block."""
    if any(type(v) is not int or not 0<=v<65536 for v in values):raise ValueError('token does not fit uint16')
    end=offset+len(values);keep=max(0,min(len(values),len(array)-offset))
    if keep:array[offset:offset+keep]=values[:keep]
    tail.extend(values[keep:])
    return end


def validate_reused_source(sid,result):
    """Verify prior artifacts, then return whether their exact row assignment is reusable."""
    if result.get('source_id')!=sid or result.get('status')!='SELECTED_SOURCE_PACKED_NOT_ADMITTED':
        raise ValueError('reused source receipt identity mismatch')
    index_path=Path(result['document_index'])
    if sha(index_path)!=result['document_index_sha256']:
        raise ValueError('reused document index hash mismatch')
    for part,output in result['outputs'].items():
        path=Path(output['path'])
        if sha(path)!=output['sha256']:raise ValueError('reused packed array hash mismatch')
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        try:
            if array.dtype!=np.dtype(np.uint16) or array.ndim!=1 or len(array)!=output['array_tokens']:
                raise ValueError('reused packed array shape/dtype mismatch')
        finally:del array
        if output['encoded_tokens']!=output['array_tokens']+output['tail_tokens']:
            raise ValueError('reused packed token accounting mismatch')
        if part=='train':
            if output['blocks']*2049!=output['array_tokens'] or output['prediction_tokens']!=output['blocks']*2048:
                raise ValueError('reused packed block accounting mismatch')
            tail_path=Path(output['tail_path'])
            if sha(tail_path)!=output['tail_sha256']:raise ValueError('reused tail hash mismatch')
            tail=json.loads(tail_path.read_text())
            if len(tail['token_ids'])!=output['tail_tokens'] or tail['virtual_offset']!=output['array_tokens']:
                raise ValueError('reused tail accounting mismatch')
        elif output['tail_tokens']:
            raise ValueError('reused reserved stream was truncated')
    cursors=defaultdict(int);counts=defaultdict(int);seen=set();matches=True
    with gzip.open(index_path,'rt',encoding='utf-8') as handle:
        for line in handle:
            record=json.loads(line);key=(sid,record['tranche'],record['row'])
            if key in seen:raise ValueError('duplicate row in reused document index')
            seen.add(key);selected=ROWS.get(key)
            if selected is None:
                matches=False;continue
            part=selected['partition'];start=cursors[part];end=start+selected['encoded_tokens_including_one_eos']
            if any(record.get(k)!=selected[k] for k in ['family_id','text_sha256']) or \
               record.get('partition')!=part or record.get('start')!=start or record.get('end')!=end or \
               record.get('first_document_target_offset')!=start+1 or record.get('eos_id')!=50279:
                matches=False
            cursors[part]=end;counts[part]+=1
    expected={key for key in ROWS if key[0]==sid}
    if seen!=expected:matches=False
    if matches:
        for part,output in result['outputs'].items():
            stats=RETENTION[part][sid]
            if counts[part]!=stats['documents'] or cursors[part]!=stats['encoded_tokens'] or \
               output['documents']!=counts[part] or output['encoded_tokens']!=cursors[part]:matches=False
    return matches


def process_source(sid):
    started=time.monotonic();arrays={};cursors={};tails={};paths={};counts=defaultdict(int)
    for part,sources in RETENTION.items():
        if sid not in sources:continue
        total=sources[sid]['encoded_tokens'];length=total//2049*2049 if part=='train' else total
        if length==0:raise ValueError('selected source lacks a full training block')
        path=OUTPUT/(part+'-'+sid+('.blocks.npy' if part=='train' else '.stream.npy'))
        if path.exists():raise ValueError('output already exists')
        arrays[part]=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint16,shape=(length,))
        cursors[part]=0;tails[part]=[];paths[part]=path
    index_path=OUTPUT/(sid+'.documents.jsonl.gz')
    with gzip.open(index_path,'xt',encoding='utf-8',compresslevel=1) as index:
        for tranche,path in sorted(RAW_FILES.get(sid,[])):
            if sha(path)!=BINDINGS[str(path)]:raise ValueError('raw input changed')
            with gzip.open(path,'rt',encoding='utf-8') as f:
                for rowno,line in enumerate(f):
                    selected=ROWS.get((sid,tranche,rowno))
                    if selected is None:continue
                    raw=json.loads(line);text=raw.get('text') or raw.get('content')
                    if hashlib.sha256(text.encode()).hexdigest()!=selected['text_sha256']:raise ValueError('selected raw text identity changed')
                    ids=TOKENIZER.encode(text,add_special_tokens=False).ids+[EOS]
                    if len(ids)!=selected['encoded_tokens_including_one_eos']:raise ValueError('tokenizer result differs from bound count')
                    part=selected['partition'];start=cursors[part]
                    cursors[part]=write_span(arrays[part],start,ids,tails[part]);counts[part]+=1
                    index.write(json.dumps(dict(source_id=sid,tranche=tranche,row=rowno,partition=part,
                        family_id=selected['family_id'],text_sha256=selected['text_sha256'],start=start,
                        end=cursors[part],first_document_target_offset=start+1,eos_id=EOS))+'\n')
                    if sum(counts.values())%1000==0:
                        if shutil.disk_usage(OUTPUT).free<18*1024**3:raise ValueError('disk headroom floor')
                        write(OUTPUT/(sid+'.progress.json'),dict(status='RUNNING',source_id=sid,documents=sum(counts.values()),elapsed_seconds=time.monotonic()-started))
            if sha(path)!=BINDINGS[str(path)]:raise ValueError('raw input changed during packing')
    outputs={}
    for part,array in arrays.items():
        expected=RETENTION[part][sid]
        if cursors[part]!=expected['encoded_tokens'] or counts[part]!=expected['documents']:raise ValueError('incomplete selected source')
        if part!='train' and tails[part]:raise ValueError('reserved token stream truncated')
        if len(array)+len(tails[part])!=cursors[part] or len(tails[part])>=2049:raise ValueError('tail accounting mismatch')
        array.flush()
        with paths[part].open('rb') as f:os.fsync(f.fileno())
        output=dict(path=str(paths[part]),sha256=sha(paths[part]),documents=counts[part],encoded_tokens=cursors[part],
                    array_tokens=len(array),tail_tokens=len(tails[part]),dtype='uint16')
        if part=='train':
            tail_path=OUTPUT/(sid+'.train-tail.json');write(tail_path,dict(token_ids=tails[part],virtual_offset=len(array),training_admitted=False))
            output.update(blocks=len(array)//2049,prediction_tokens=len(array)//2049*2048,
                          tail_path=str(tail_path),tail_sha256=sha(tail_path))
        outputs[part]=output
    result=dict(status='SELECTED_SOURCE_PACKED_NOT_ADMITTED',source_id=sid,elapsed_seconds=time.monotonic()-started,
                outputs=outputs,document_index=str(index_path),document_index_sha256=sha(index_path))
    write(OUTPUT/(sid+'.progress.json'),result);return result


def run(args):
    global ROWS,ROOTS,RAW_FILES,BINDINGS,OUTPUT,RETENTION
    started=time.monotonic()
    if not 1<=args.workers<=4:raise ValueError('workers must be 1..4')
    if sha(args.family_report)!=args.expected_family_sha256 or sha(args.tokenizer)!=args.expected_tokenizer_sha256:raise ValueError('input identity mismatch')
    report=json.loads(args.family_report.read_text())
    if report['status']!='FAMILY_CLOSURE_AND_RESERVED_ASSIGNMENTS_COMPLETE_NOT_ADMITTED':raise ValueError('family stage incomplete')
    BINDINGS=report['input_bindings'];RETENTION=report['retained_sources'];OUTPUT=args.output
    assignment=args.family_report.parent/'row-assignments.jsonl.gz'
    if sha(assignment)!=report['output_files'][str(assignment)]:raise ValueError('assignment identity mismatch')
    families={};texts=set();observed=defaultdict(lambda:dict(documents=0,encoded_tokens=0))
    with gzip.open(assignment,'rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if row['partition']=='excluded':continue
            if not row['selected_eligible_representative'] or not row['passes_filters_and_bound_exclusions'] or row['family_exclusion_reasons']:raise ValueError('excluded row selected')
            family=row['family_id'];part=row['partition'];key=(row['source_id'],row['tranche'],row['row'])
            if key in ROWS or row['text_sha256'] in texts:raise ValueError('duplicate selected record')
            if family in families and families[family]!=part:raise ValueError('family crosses partition')
            families[family]=part;texts.add(row['text_sha256']);ROWS[key]=row
            observed[part,row['source_id']]['documents']+=1
            observed[part,row['source_id']]['encoded_tokens']+=row['encoded_tokens_including_one_eos']
    for part,sources in RETENTION.items():
        for sid,stats in sources.items():
            if any(observed[part,sid][k]!=stats[k] for k in ['documents','encoded_tokens']):raise ValueError('retention totals mismatch')
    raw_path=Path(report['raw_report_path']) if report.get('raw_report_path') else next(
        Path(p) for p in BINDINGS if p.endswith('/diverse-audit-v2-three-tranche/report.json'))
    if sha(raw_path)!=BINDINGS[str(raw_path)]:raise ValueError('raw audit changed')
    raw_report=json.loads(raw_path.read_text());ROOTS=[Path(t['path']) for t in raw_report['tranches']]
    if raw_report.get('files'):
        for item in raw_report['files']:
            RAW_FILES.setdefault(item['source_id'],[]).append((item['tranche'],Path(item['path'])))
    else:
        for tranche,root in enumerate(ROOTS):
            for sid in RETENTION.get('train',{}):
                path=root/(sid+'.jsonl.gz')
                if str(path) in BINDINGS:RAW_FILES.setdefault(sid,[]).append((tranche,path))
    sources=sorted({key[1] for key in observed})
    reused=[];pending=sources
    if args.reuse_report:
        if not args.expected_reuse_report_sha256 or sha(args.reuse_report)!=args.expected_reuse_report_sha256:
            raise ValueError('reused pack report identity mismatch')
        prior=json.loads(args.reuse_report.read_text())
        if prior.get('status')!='SELECTED_SOURCE_PACKS_COMPLETE_NOT_ADMITTED' or prior.get('tokenizer_sha256')!=args.expected_tokenizer_sha256:
            raise ValueError('reused pack report status/tokenizer mismatch')
        by_source={x['source_id']:x for x in prior['sources']}
        if set(by_source)!=set(sources):raise ValueError('reused pack source set mismatch')
        reused=[by_source[sid] for sid in sources if validate_reused_source(sid,by_source[sid])]
        reused_ids={x['source_id'] for x in reused};pending=[sid for sid in sources if sid not in reused_ids]
    total_bytes=sum(v['encoded_tokens']*2 for (part,sid),v in observed.items() if sid in pending)
    if shutil.disk_usage(args.output.parent).free<18*1024**3+total_bytes+128*1024**2:raise ValueError('aggregate packing headroom insufficient')
    args.output.mkdir(exist_ok=False)
    write(args.output/'launch.json',dict(status='PACKING_SELECTED_CONTENT_NOT_ADMITTED',workers=args.workers,
        selected_documents=len(ROWS),reserved_bytes=total_bytes+128*1024**2,tokenizer_sha256=args.expected_tokenizer_sha256,
        family_report_sha256=args.expected_family_sha256,reuse_report_sha256=args.expected_reuse_report_sha256,
        reused_sources=sorted(x['source_id'] for x in reused),pending_sources=pending,training_admitted=False))
    if pending:
        with mp.get_context('fork').Pool(min(args.workers,len(pending)),initializer=initialize,initargs=(str(args.tokenizer),)) as pool:
            results=reused+list(pool.imap_unordered(process_source,pending))
    else:results=reused
    for p,h in BINDINGS.items():
        if sha(p)!=h:raise ValueError('bound corpus input changed')
    if sha(assignment)!=report['output_files'][str(assignment)] or sha(args.family_report)!=args.expected_family_sha256 or sha(args.tokenizer)!=args.expected_tokenizer_sha256:raise ValueError('packing input changed')
    result=dict(status='SELECTED_SOURCE_PACKS_COMPLETE_NOT_ADMITTED',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-started,workers=args.workers,selected_documents=len(ROWS),
        family_report_sha256=args.expected_family_sha256,tokenizer_sha256=args.expected_tokenizer_sha256,
        reuse_report_sha256=args.expected_reuse_report_sha256,reused_sources=sorted(x['source_id'] for x in reused),
        sources=sorted(results,key=lambda r:r['source_id']),training_admitted=False,training_launched=False,
        limitations=['These are leaf-source artifacts, not recipe manifests or training admission.',
            'Reserved streams retain full documents and family/offset indexes; evaluator must mask document-first tokens and padding.',
            'Training block remainder is preserved separately, accounted explicitly and not used as a complete block.'])
    write(args.output/'report.json',result)
    print(json.dumps({k:result[k] for k in ['status','elapsed_seconds','selected_documents']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ['family-report','tokenizer','output']:p.add_argument('--'+field,type=Path,required=True)
    for field in ['expected-family-sha256','expected-tokenizer-sha256']:p.add_argument('--'+field,required=True)
    p.add_argument('--reuse-report',type=Path);p.add_argument('--expected-reuse-report-sha256')
    p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    try:run(args)
    except Exception as error:
        if args.output.exists():write(args.output/'failure.json',dict(status='FAILED_NOT_ADMITTED',error_type=type(error).__name__,error=str(error),training_admitted=False))
        raise
