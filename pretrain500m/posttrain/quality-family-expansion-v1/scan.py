"""Apply declared filters and emit family/MinHash features from bound raw rows."""
import argparse
import atexit
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


def completed_result(item):
    """Return a fully identity-checked completed shard, or None for pending work."""
    sid,tranche,_path,expected_rows,output=item
    feature_path=Path(output)/f'{tranche}-{sid}.features.jsonl.gz'
    progress=Path(output)/f'{tranche}-{sid}.progress.json'
    if not progress.exists():return None
    try:result=json.loads(progress.read_text())
    except (OSError,json.JSONDecodeError):return None
    if result.get('status')!='COMPLETED':return None
    if result.get('source_id')!=sid or result.get('tranche')!=tranche or result.get('rows')!=expected_rows:
        raise ValueError('completed shard identity mismatch')
    if Path(result.get('features',''))!=feature_path or not feature_path.is_file():
        raise ValueError('completed shard feature path mismatch')
    if result.get('features_sha256')!=sha(feature_path):
        raise ValueError('completed shard feature hash mismatch')
    for key in ['passed_rows','passed_encoded_token_upper_bound_before_eligible_dedup_and_reserves','elapsed_seconds']:
        if not isinstance(result.get(key),(int,float)):raise ValueError(f'completed shard missing {key}')
    if not isinstance(result.get('flags'),dict):raise ValueError('completed shard flags missing')
    return result


def reusable_results(work, reuse_output, expected_report_sha256, report, exclusions):
    prior_path=Path(reuse_output)/'report.json'
    if not expected_report_sha256 or sha(prior_path)!=expected_report_sha256:
        raise ValueError('reused quality report identity mismatch')
    prior=json.loads(prior_path.read_text())
    if prior.get('status')!='QUALITY_AND_FAMILY_FEATURES_COMPLETE_NOT_ADMITTED' or prior.get('policy')!=POLICY:
        raise ValueError('reused quality report status/policy mismatch')
    if report.get('incremental_from_report_sha256')!=prior.get('raw_report_sha256'):
        raise ValueError('raw audit is not an append of the reused quality prefix')
    if exclusions.get('prior_union_sha256')!=prior.get('exclusion_plan_sha256'):
        raise ValueError('exclusions are not an extension of the reused quality prefix')
    prior_files={(x['source_id'],x['tranche']):x for x in prior['files']}
    current_keys={(x[0],x[1]) for x in work}
    if not set(prior_files)<=current_keys or sum(x['rows'] for x in prior_files.values())!=prior['rows']:
        raise ValueError('reused quality file set is not a complete current prefix')
    results={}
    for item in work:
        sid,tranche,path,expected_rows,_output=item;key=(sid,tranche)
        if key not in prior_files:continue
        if prior['input_bindings'].get(path)!=report['input_bindings'].get(path):
            raise ValueError('reused raw file binding changed')
        reused_item=(sid,tranche,path,expected_rows,str(reuse_output))
        result=completed_result(reused_item)
        if result is None or result!=prior_files[key]:
            raise ValueError('reused quality shard/report mismatch')
        results[key]=result
    if set(results)!=set(prior_files):raise ValueError('not all reused quality shards verified')
    return results


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
    if not 1<=args.workers<=16:raise ValueError('workers must be 1..16')
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
    fresh=not args.output.exists()
    if args.reuse_output and not fresh:raise ValueError('--reuse-output requires a fresh output directory')
    if fresh:
        args.output.mkdir(exist_ok=False)
        launch=dict(status='RUNNING',workers=args.workers,rows=report['rows'],
            raw_report_sha256=args.expected_raw_report_sha256,exclusion_plan_sha256=args.expected_exclusions_sha256,
            policy=POLICY,dependencies=dependencies,training_admitted=False,resume_history=[],
            reuse_report_sha256=args.expected_reuse_report_sha256)
    else:
        if not args.resume:raise ValueError('output exists; explicit --resume is required')
        launch_path=args.output/'launch.json'
        if not launch_path.is_file():raise ValueError('resume launch receipt missing')
        launch=json.loads(launch_path.read_text())
        expected={'rows':report['rows'],'raw_report_sha256':args.expected_raw_report_sha256,
            'exclusion_plan_sha256':args.expected_exclusions_sha256,'policy':POLICY,'dependencies':dependencies,
            'training_admitted':False}
        for key,value in expected.items():
            if launch.get(key)!=value:raise ValueError(f'resume launch binding mismatch: {key}')
        launch.setdefault('resume_history',[]).append(dict(
            resumed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            prior_workers=launch.get('workers'),workers=args.workers))
        launch.update(status='RUNNING',workers=args.workers)
    lock=args.output/'writer.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    except FileExistsError as exc:raise ValueError('quality output writer already active') from exc
    atexit.register(lock.unlink,missing_ok=True)
    write_json(args.output/'launch.json',launch)
    work=[(f['source_id'],f['tranche'],str(f['path']),f['receipt']['rows_written'],str(args.output)) for f in files]
    work.sort(key=lambda v:-Path(v[2]).stat().st_size)
    reused_by_key=reusable_results(work,args.reuse_output,args.expected_reuse_report_sha256,report,exclusions) if args.reuse_output else {}
    reused=[];pending=[]
    for item in work:
        result=reused_by_key.get((item[0],item[1])) if args.reuse_output else completed_result(item)
        if result is not None:reused.append(result)
        else:
            sid,tranche,_path,_expected_rows,output=item
            for suffix in ['features.jsonl.gz','progress.json']:
                (Path(output)/f'{tranche}-{sid}.{suffix}').unlink(missing_ok=True)
            pending.append(item)
    if pending:
        with mp.get_context('fork').Pool(args.workers,initializer=initialize,
                initargs=(str(Path(__file__).parent/'vendor/lid.176.bin'),)) as pool:
            results=reused+list(pool.imap_unordered(process_one,pending,chunksize=1))
    else:results=reused
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
        elapsed_seconds=time.monotonic()-started,workers=args.workers,resumed_completed_files=len(reused),policy=POLICY,
        raw_report_sha256=args.expected_raw_report_sha256,exclusion_plan_sha256=args.expected_exclusions_sha256,
        input_bindings={str(k):v for k,v in bindings.items()},files=sorted(results,key=lambda v:(v['tranche'],v['source_id'])),
        training_admitted=False,training_text_modified=False,
        limits=['Quality means the declared structural, source-score, license and sampled-language filters only; factual truth and educational value are not independently certified.',
                'MinHash is candidate generation over full-text Unicode-aware token shingles; no verified near-duplicate removal or family closure is completed here.',
                'Canonical URLs, repositories, numeric templates and registrable domains are features; repository aliases/forks and independent splits still need resolution.',
                'Token counts are reused from the identity-bound formal tokenizer index and remain upper bounds before eligible deduplication, family exclusion and reserves.',
                'Raw data is preserved. Bound exclusions are applied to eligibility records, not to a training pack.'])
    write_json(args.output/'report.json',final)
    lock.unlink()
    print(json.dumps({k:final[k] for k in ['status','rows','passed_rows_before_eligible_dedup_and_reserves','elapsed_seconds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw-report',type=Path,required=True)
    p.add_argument('--expected-raw-report-sha256',required=True);p.add_argument('--exclusions',type=Path,required=True)
    p.add_argument('--expected-exclusions-sha256',required=True);p.add_argument('--audit-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=16)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--reuse-output',type=Path)
    p.add_argument('--expected-reuse-report-sha256')
    run(p.parse_args())
