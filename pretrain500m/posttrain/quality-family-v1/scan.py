"""Apply declared filters and emit family/MinHash features from bound raw rows."""
import argparse
from collections import Counter
import datetime
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time

from features import POLICY, PublicSuffix, features, predict_language

INDEX={};EXCLUSIONS={};PSL=None;MODEL=None


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write_json(path,value):
    tmp=path.with_suffix(path.suffix+'.next');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def initialize(model_path):
    global MODEL
    import fasttext
    MODEL=fasttext.load_model(model_path)


def process_one(item):
    sid,tranche,path,expected_rows,output=item;started=time.monotonic();rows=accepted=token_upper=0;flags=Counter()
    feature_path=Path(output)/f'{tranche}-{sid}.features.jsonl.gz'
    progress=Path(output)/f'{tranche}-{sid}.progress.json'
    with gzip.open(path,'rt',encoding='utf-8') as inp,gzip.open(feature_path,'xt',encoding='utf-8',compresslevel=1) as out:
        for row_number,line in enumerate(inp):
            row=json.loads(line);text=row.get('text') or row.get('content');key=(sid,tranche,row_number)
            expected=INDEX[key];raw_hash=hashlib.sha256(text.encode()).hexdigest()
            if raw_hash!=expected['text_sha256']:raise ValueError('raw text does not match bound tokenizer index')
            if row['_provenance']['source_id']!=sid:raise ValueError('source identity mismatch')
            record=features(row,sid,expected['tokens'],PSL,lambda t:predict_language(MODEL,t))
            exclusion_reasons=EXCLUSIONS.get(raw_hash,[])
            record.update(source_id=sid,tranche=tranche,row=row_number,text_sha256=raw_hash,
                encoded_tokens_including_one_eos=expected['tokens'],provenance=row['_provenance'],
                exclusion_reasons=exclusion_reasons,
                passes_filters_and_bound_exclusions=record['passes_declared_filters'] and not exclusion_reasons)
            out.write(json.dumps(record,ensure_ascii=False)+'\n');rows+=1
            flags.update(record['flags']);flags.update(exclusion_reasons)
            if record['passes_filters_and_bound_exclusions']:
                accepted+=1;token_upper+=expected['tokens']
            if rows%1000==0:
                if shutil.disk_usage(output).free<18*1024**3:raise ValueError('disk headroom fell below floor')
                write_json(progress,dict(status='RUNNING',source_id=sid,tranche=tranche,rows=rows,
                    passed_rows=accepted,elapsed_seconds=time.monotonic()-started))
    if rows!=expected_rows:raise ValueError('raw row count mismatch')
    result=dict(status='COMPLETED',source_id=sid,tranche=tranche,rows=rows,passed_rows=accepted,
        passed_encoded_token_upper_bound_before_eligible_dedup_and_reserves=token_upper,
        flags=dict(flags),features=str(feature_path),features_sha256=sha(feature_path),elapsed_seconds=time.monotonic()-started)
    write_json(progress,result);return result


def run(args):
    global INDEX,EXCLUSIONS,PSL
    started=time.monotonic()
    if not 1<=args.workers<=4:raise ValueError('workers must be 1..4')
    dependencies=json.loads((Path(__file__).parent/'dependency-receipts.json').read_text())
    for artifact in dependencies['files']:
        path=Path(__file__).parent/'vendor'/artifact['name']
        if sha(path)!=artifact['sha256']:raise ValueError('dependency artifact identity mismatch')
    if sha(args.raw_report)!=args.expected_raw_report_sha256:raise ValueError('raw audit identity mismatch')
    if sha(args.exclusions)!=args.expected_exclusions_sha256:raise ValueError('exclusion policy identity mismatch')
    report=json.loads(args.raw_report.read_text());exclusions=json.loads(args.exclusions.read_text())
    if report['status']!='COMBINED_RAW_INTAKE_MEASURED_NOT_ADMITTED':raise ValueError('raw audit incomplete')
    if exclusions['status']!='POLICY_EXCLUSIONS_BOUND_NOT_APPLIED_TO_PACK':raise ValueError('exclusions not bound')
    EXCLUSIONS=exclusions['exclude_text_sha256_reasons']
    sys.path.insert(0,str(args.audit_source));from audit_raw_intake_v2 import preflight
    inputs=[Path(t['path']) for t in report['tranches']];files,bindings,tranches=preflight(inputs)
    for p,h in bindings.items():
        if report['input_bindings'].get(str(p))!=h:raise ValueError('raw audit input binding changed')
    index=args.raw_report.parent/'row-index.jsonl'
    if sha(index)!=report['row_index_sha256']:raise ValueError('tokenizer index identity mismatch')
    with index.open() as handle:
        for line in handle:
            row=json.loads(line);key=(row['source_id'],row['tranche'],row['row'])
            if key in INDEX:raise ValueError('duplicate physical index key')
            INDEX[key]={'text_sha256':row['text_sha256'],'tokens':row['encoded_tokens_including_one_eos']}
    if len(INDEX)!=report['rows']:raise ValueError('tokenizer index row count mismatch')
    if sha(index)!=report['row_index_sha256']:raise ValueError('tokenizer index changed during read')
    PSL=PublicSuffix((Path(__file__).parent/'vendor/public_suffix_list.dat').read_text())
    if shutil.disk_usage(args.output.parent).free<18*1024**3+256*1024**2:raise ValueError('insufficient disk headroom')
    args.output.mkdir(exist_ok=False)
    write_json(args.output/'launch.json',dict(status='RUNNING',workers=args.workers,rows=report['rows'],
        raw_report_sha256=args.expected_raw_report_sha256,exclusion_plan_sha256=args.expected_exclusions_sha256,
        policy=POLICY,dependencies=dependencies,training_admitted=False))
    work=[(f['source_id'],f['tranche'],str(f['path']),f['receipt']['rows_written'],str(args.output)) for f in files]
    work.sort(key=lambda v:-Path(v[2]).stat().st_size)
    with mp.get_context('fork').Pool(args.workers,initializer=initialize,
            initargs=(str(Path(__file__).parent/'vendor/lid.176.bin'),)) as pool:
        results=list(pool.imap_unordered(process_one,work,chunksize=1))
    for p,h in bindings.items():
        if sha(p)!=h:raise ValueError('bound raw input changed')
    for root in inputs:
        if (root/'writer.lock').exists():raise ValueError('input writer appeared')
        for pattern in ['*.receipt.json','*.jsonl.gz']:
            if set(root.glob(pattern))!={p for p in bindings if p.parent==root and p.match(pattern)}:
                raise ValueError('raw input file set changed')
    if sha(index)!=report['row_index_sha256'] or sha(args.raw_report)!=args.expected_raw_report_sha256 or sha(args.exclusions)!=args.expected_exclusions_sha256:
        raise ValueError('audit/index/exclusion identity changed during scan')
    total=sum(v['rows'] for v in results)
    if total!=report['rows']:raise ValueError('total row count mismatch')
    final=dict(status='QUALITY_AND_FAMILY_FEATURES_COMPLETE_NOT_ADMITTED',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rows=total,passed_rows_before_eligible_dedup_and_reserves=sum(v['passed_rows'] for v in results),
        elapsed_seconds=time.monotonic()-started,workers=args.workers,policy=POLICY,
        raw_report_sha256=args.expected_raw_report_sha256,exclusion_plan_sha256=args.expected_exclusions_sha256,
        input_bindings={str(k):v for k,v in bindings.items()},files=sorted(results,key=lambda v:(v['tranche'],v['source_id'])),
        training_admitted=False,training_text_modified=False,
        limits=['Quality means the declared structural, source-score, license and sampled-language filters only; factual truth and educational value are not independently certified.',
                'MinHash is candidate generation over full-text Unicode-aware token shingles; no verified near-duplicate removal or family closure is completed here.',
                'Canonical URLs, repositories, numeric templates and registrable domains are features; repository aliases/forks and independent splits still need resolution.',
                'Token counts are reused from the identity-bound formal tokenizer index and remain upper bounds before eligible deduplication, family exclusion and reserves.',
                'Raw data is preserved. Bound exclusions are applied to eligibility records, not to a training pack.'])
    write_json(args.output/'report.json',final)
    print(json.dumps({k:final[k] for k in ['status','rows','passed_rows_before_eligible_dedup_and_reserves','elapsed_seconds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw-report',type=Path,required=True)
    p.add_argument('--expected-raw-report-sha256',required=True);p.add_argument('--exclusions',type=Path,required=True)
    p.add_argument('--expected-exclusions-sha256',required=True);p.add_argument('--audit-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    run(p.parse_args())
