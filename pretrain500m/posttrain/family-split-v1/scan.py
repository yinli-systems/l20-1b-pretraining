"""Single invocation: verified near pairs -> family closure -> reserved splits."""
import argparse
from collections import Counter, defaultdict
import datetime
from functools import lru_cache
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import time
import unicodedata

from families import DSU, POLICY, candidates, domain, join_keys, normalized, partition, select, shingles, verify

RECORDS = []; LOOKUP = {}; NEEDED = set(); TEXTS = {}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    p=path.with_suffix(path.suffix+'.next');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(path)


def load_raw(item):
    sid, tranche, path, expected_hash = item
    if sha(path) != expected_hash: raise ValueError('raw file hash mismatch')
    output=[]
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for rowno, line in enumerate(f):
            row=json.loads(line);i=LOOKUP[(sid,tranche,rowno)];text=row.get('text') or row.get('content')
            if hashlib.sha256(text.encode()).hexdigest()!=RECORDS[i]['text_sha256']: raise ValueError('raw text identity changed')
            if row['_provenance']['source_id']!=sid: raise ValueError('raw source identity changed')
            norm=hashlib.sha256(normalized(text,sid.startswith('code_')).encode()).hexdigest()
            legacy=hashlib.sha256(re.sub(r'\s+',' ',unicodedata.normalize('NFKC',text).lower()).strip().encode()).hexdigest()
            output.append((i,norm,legacy,text if i in NEEDED else None))
    if sha(path)!=expected_hash: raise ValueError('raw file changed during read')
    if len(output)!=sum(1 for r in RECORDS if r['source_id']==sid and r['tranche']==tranche): raise ValueError('raw row count changed')
    return output


@lru_cache(maxsize=96)
def cached_shingles(i):
    return shingles(TEXTS[i],RECORDS[i]['source_id'].startswith('code_'))


def verify_node(item):
    a, bs = item; result=[]; sa=cached_shingles(a)
    for b in bs:
        ok, intersection, union = verify(sa,cached_shingles(b))
        if ok: result.append((a,b,intersection,union))
    return len(bs),result


def run(args):
    global RECORDS,LOOKUP,NEEDED,TEXTS
    started=time.monotonic(); bindings={}
    def bound(path,expected):
        path=Path(path)
        if sha(path)!=expected: raise ValueError('input identity mismatch: '+str(path))
        bindings[str(path)]=expected
        return json.loads(path.read_text())
    report=bound(args.features_report,args.expected_features_sha256)
    raw=bound(args.raw_report,report['raw_report_sha256'])
    exclusions=bound(args.exclusions,report['exclusion_plan_sha256'])
    design=bound(args.design,args.expected_design_sha256)
    if report['status']!='QUALITY_AND_FAMILY_FEATURES_COMPLETE_NOT_ADMITTED': raise ValueError('incomplete feature report')
    if design['protocol_id']!='p529m-fast-start-nosynthetic-v1': raise ValueError('wrong screen protocol')
    if design['development']['minimum_distinct_document_families_per_domain']!=1000: raise ValueError('changed family minimum')
    if design['development']['minimum_prediction_tokens_per_domain']!=1048576: raise ValueError('changed token minimum')
    if not 1<=args.workers<=4: raise ValueError('workers must be 1..4')
    if shutil.disk_usage(args.output.parent).free<18*1024**3+256*1024**2: raise ValueError('disk headroom floor')
    args.output.mkdir(exist_ok=False)
    def progress(stage,**kw):
        write(args.output/'progress.json',dict(status='RUNNING',stage=stage,elapsed_seconds=time.monotonic()-started,**kw))
    write(args.output/'policy.json',POLICY);progress('load_bound_features')
    fields=['source_id','tranche','row','text_sha256','encoded_tokens_including_one_eos','canonical_family',
            'numeric_template_sha256','passes_filters_and_bound_exclusions','exclusion_reasons','minhash64_u32_le_hex']
    for item in sorted(report['files'],key=lambda f:(f['tranche'],f['source_id'])):
        path=Path(item['features']);expected=item['features_sha256']
        if sha(path)!=expected: raise ValueError('feature hash changed')
        bindings[str(path)]=expected;count=0
        with gzip.open(path,'rt',encoding='utf-8') as f:
            for line in f:
                record=json.loads(line);r={k:record[k] for k in fields};key=(r['source_id'],r['tranche'],r['row'])
                if key in LOOKUP: raise ValueError('duplicate physical row')
                if r['exclusion_reasons']!=exclusions['exclude_text_sha256_reasons'].get(r['text_sha256'],[]): raise ValueError('exclusions not applied')
                LOOKUP[key]=len(RECORDS);RECORDS.append(r);count+=1
        if count!=item['rows']: raise ValueError('feature row count changed')
    if len(RECORDS)!=report['rows']: raise ValueError('incomplete corpus')
    pairs,hot,hot_counts=candidates(RECORDS,lambda band,pairs,hot:progress('lsh_candidates',bands_completed=band,candidate_pairs=pairs,hot_rows=hot))
    n=len(RECORDS);NEEDED={i for p in pairs for i in (p//n,p%n)}
    progress('read_bound_raw_once',candidate_pairs=len(pairs),candidate_rows=len(NEEDED))
    inputs=[Path(t['path']) for t in raw['tranches']]
    raw_work=[]
    for item in report['files']:
        path=inputs[item['tranche']]/(item['source_id']+'.jsonl.gz')
        raw_work.append((item['source_id'],item['tranche'],str(path),report['input_bindings'][str(path)]))
    loaded=0
    with mp.get_context('fork').Pool(args.workers) as pool:
        for result in pool.imap_unordered(load_raw,raw_work):
            for i,norm,legacy,text in result:
                RECORDS[i]['normalized_text_sha256']=norm;RECORDS[i]['legacy_normalized_family_sha256']=legacy
                if text is not None: TEXTS[i]=text
            loaded+=len(result);progress('read_bound_raw_once',rows=loaded,candidate_pairs=len(pairs))
    if loaded!=n or set(TEXTS)!=NEEDED: raise ValueError('raw loading incomplete')
    families=DSU(n);duplicates=DSU(n);joins=join_keys(RECORDS,families,duplicates)
    by_a=defaultdict(list)
    for pair in pairs: by_a[pair//n].append(pair%n)
    checked=verified=0
    edge_path=args.output/'verified-near-edges.jsonl.gz'
    with gzip.open(edge_path,'xt',encoding='utf-8',compresslevel=1) as out,mp.get_context('fork').Pool(args.workers) as pool:
        for count,edges in pool.imap_unordered(verify_node,sorted(by_a.items()),chunksize=8):
            checked+=count
            for a,b,intersection,union in edges:
                families.union(a,b);duplicates.union(a,b);verified+=1
                out.write(json.dumps(dict(a=a,b=b,intersection=intersection,union=union))+'\n')
            if checked%1000<count: progress('verify_full_shingles',checked_pairs=checked,verified_edges=verified,total_pairs=len(pairs))
    if checked!=len(pairs): raise ValueError('not all candidate pairs verified')
    selected,quarantined=select(RECORDS,families,duplicates,hot)
    assignments,names,reserves=partition(RECORDS,families,selected)
    progress('write_family_partitions',selected_documents=len(selected),quarantined_families=len(quarantined))
    inventory_path=args.output/'row-assignments.jsonl.gz';counts=Counter();tokens=Counter();rows=Counter();excluded_rows=0
    all_family_names={};duplicate_names={}
    for i in range(n):
        root=families.find(i)
        all_family_names.setdefault(root,hashlib.sha256(f"family-min-physical-row:{i}:{RECORDS[i]['text_sha256']}".encode()).hexdigest())
        duplicate_names.setdefault(duplicates.find(i),i)
    with gzip.open(inventory_path,'xt',encoding='utf-8',compresslevel=1) as out:
        for i,r in enumerate(RECORDS):
            root=families.find(i);selected_row=i in selected
            part=assignments[root] if selected_row else 'excluded'
            reasons=sorted(quarantined.get(root,set()))
            if root in quarantined: excluded_rows+=1
            record={k:r[k] for k in ['source_id','tranche','row','text_sha256','encoded_tokens_including_one_eos']}
            record.update(family_id=all_family_names[root],duplicate_cluster=duplicate_names[duplicates.find(i)],partition=part,
                          selected_eligible_representative=selected_row,family_exclusion_reasons=reasons,
                          passes_filters_and_bound_exclusions=r['passes_filters_and_bound_exclusions'])
            out.write(json.dumps(record)+'\n');counts[part]+=1
            if selected_row:
                key=(part,r['source_id']);rows[key]+=1;tokens[key]+=r['encoded_tokens_including_one_eos']
    # Verify the same exact input identities after all stages, including raw receipts.
    bindings.update(report['input_bindings'])
    for p,h in bindings.items():
        if sha(p)!=h: raise ValueError('bound input changed before completion: '+p)
    for root in inputs:
        if (root/'writer.lock').exists(): raise ValueError('raw writer appeared')
    retained={part:{sid:dict(documents=rows[part,sid],encoded_tokens=tokens[part,sid],
                    packed_prediction_tokens_upper_bound=(tokens[part,sid]//2049)*2048) for part2,sid in sorted(tokens) if part2==part} for part in ['train','development','confirmation']}
    final=dict(status='FAMILY_CLOSURE_AND_RESERVED_ASSIGNMENTS_COMPLETE_NOT_ADMITTED',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-started,rows=n,workers=args.workers,policy=POLICY,input_bindings=bindings,
        exact_and_metadata_join_counts=joins,lsh_candidate_pairs=checked,verified_near_edges=verified,
        hot_rows=len(hot),hot_buckets_per_band=hot_counts,quarantined_families=len(quarantined),quarantined_rows=excluded_rows,
        selected_unique_eligible_documents=len(selected),partition_counts=dict(counts),retained_sources=retained,reserved_partitions=reserves,
        independent_under_observed_family_edges=True,training_admitted=False,training_launched=False,
        output_files={str(inventory_path):sha(inventory_path),str(edge_path):sha(edge_path),str(args.output/'policy.json'):sha(args.output/'policy.json')},
        limits=['Probabilistic LSH recall is not exhaustive; only full-shingle-verified candidate edges are called near duplicates.',
                'Family joins cover observed URLs, repositories, normalized documents and numeric templates; unobserved forks, paraphrases and translations can remain.',
                'Old FineWeb comparison is normalized exact overlap, not exhaustive old-corpus near-duplicate checking.',
                'Eligibility is declared structural/language/license filtering, not proof of factual or educational correctness.',
                'Reserved token counts are document prediction tokens. Training pack counts remain upper bounds; no pack or training admission is created.'])
    write(args.output/'report.json',final);write(args.output/'progress.json',{k:v for k,v in final.items() if k not in ['input_bindings','retained_sources','output_files']})
    print(json.dumps({k:final[k] for k in ['status','elapsed_seconds','rows','selected_unique_eligible_documents','partition_counts','lsh_candidate_pairs','verified_near_edges','quarantined_families']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for option in ['features-report','raw-report','exclusions','design','output']: parser.add_argument('--'+option,type=Path,required=True)
    for option in ['expected-features-sha256','expected-design-sha256']: parser.add_argument('--'+option,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    try: run(args)
    except Exception as error:
        if args.output.exists():
            write(args.output/'failure.json',dict(status='FAILED_NOT_ADMITTED',error_type=type(error).__name__,error=str(error),training_admitted=False))
        raise
