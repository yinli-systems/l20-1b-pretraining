"""Replace rejected reserved families without crossing train/development data."""
from collections import Counter, defaultdict
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path


DOMAINS=['general_web','knowledge_reading','math','code','multilingual']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def domain(source):
    if source=='dclm':return 'general_web'
    if source=='pdf_en':return 'knowledge_reading'
    if source in ('finemath4','infiwebmath4'):return 'math'
    if source.startswith('code_'):return 'code'
    if source.startswith('multilingual_'):return 'multilingual'
    raise ValueError(f'unknown source: {source}')


def choose_replacements(families, rejected, minimum_families=1000, minimum_tokens=1048576):
    reserved={p:{} for p in ('development','confirmation')}
    for partition in reserved:
        for d in DOMAINS:
            members={fid for fid,f in families.items() if f['partition']==partition and fid not in rejected and f['tokens'][d]>0}
            reserved[partition][d]={'families':members,'prediction_tokens':sum(families[f]['tokens'][d] for f in members)}
    order=sorted((fid for fid,f in families.items() if f['partition']=='train' and fid not in rejected),
                 key=lambda fid:hashlib.sha256(('quality-reserve-replacement-v1:'+fid).encode()).digest())
    used=set();moves={}
    for partition in ('development','confirmation'):
        for d in DOMAINS:
            while (len(reserved[partition][d]['families'])<minimum_families or
                   reserved[partition][d]['prediction_tokens']<minimum_tokens):
                fid=next((f for f in order if f not in used and families[f]['tokens'][d]>0),None)
                if fid is None:raise ValueError(f'no replacement family for {partition}/{d}')
                used.add(fid);moves[fid]=partition
                for covered in DOMAINS:
                    if families[fid]['tokens'][covered]>0:
                        reserved[partition][covered]['families'].add(fid)
                        reserved[partition][covered]['prediction_tokens']+=families[fid]['tokens'][covered]
    stats={p:{d:{'families':len(v['families']),'prediction_tokens':v['prediction_tokens']}
              for d,v in domains.items()} for p,domains in reserved.items()}
    return moves,stats


def run(args):
    if sha(args.family_report)!=args.expected_family_sha256:raise ValueError('family report mismatch')
    if sha(args.train_decisions)!=args.expected_train_decisions_sha256:raise ValueError('train decisions mismatch')
    if sha(args.reserved_decisions)!=args.expected_reserved_decisions_sha256:raise ValueError('reserved decisions mismatch')
    family=json.loads(args.family_report.read_text());train=json.loads(args.train_decisions.read_text());reserved=json.loads(args.reserved_decisions.read_text())
    if family['status']!='FAMILY_CLOSURE_AND_RESERVED_ASSIGNMENTS_COMPLETE_NOT_ADMITTED':raise ValueError('family stage incomplete')
    assignment=args.family_report.parent/'row-assignments.jsonl.gz'
    if sha(assignment)!=family['output_files'][str(assignment)]:raise ValueError('assignment mismatch')
    families={}
    with gzip.open(assignment,'rt',encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line)
            if not row['selected_eligible_representative'] or row['partition']=='excluded':continue
            f=families.setdefault(row['family_id'],{'partition':row['partition'],'tokens':Counter(),'documents':0})
            if f['partition']!=row['partition']:raise ValueError('family crosses original partitions')
            f['tokens'][domain(row['source_id'])]+=row['encoded_tokens_including_one_eos']-1;f['documents']+=1
    train_bad=set(train['exclude_family_id_reasons']);reserved_bad=set(reserved['exclude_family_id_reasons'])
    if train_bad & reserved_bad:raise ValueError('train and reserved rejected families overlap')
    if any(families[f]['partition']!='train' for f in train_bad):raise ValueError('train rejection outside train')
    if any(families[f]['partition']=='train' for f in reserved_bad):raise ValueError('reserved rejection inside train')
    moves,stats=choose_replacements(families,train_bad|reserved_bad)
    final_train={k:list(v) for k,v in train['exclude_family_id_reasons'].items()}
    for fid,partition in moves.items():final_train[fid]=[f'moved_to_{partition}_as_quality_replacement']
    output={
      'schema':'p529m-quality-reserved-split-patch-v1','status':'RESERVED_QUALITY_REPLACEMENTS_COMPLETE_NOT_APPLIED',
      'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'family_report_sha256':args.expected_family_sha256,'train_decisions_sha256':args.expected_train_decisions_sha256,
      'reserved_decisions_sha256':args.expected_reserved_decisions_sha256,
      'reserved_exclude_family_id_reasons':reserved['exclude_family_id_reasons'],
      'move_train_family_to_partition':dict(sorted(moves.items())),
      'final_train_exclude_family_id_reasons':dict(sorted(final_train.items())),
      'rejected_reserved_families':len(reserved_bad),'replacement_families':len(moves),
      'final_reserved_stats':stats,'training_admitted':False,
      'limits':['Replacements are deterministic observed families and preserve original raw/token identities.',
                'The patch must be consumed by both the final train pack and masked reserved pack before admission.']}
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'status':output['status'],'rejected_reserved_families':len(reserved_bad),'replacement_families':len(moves),'final_reserved_stats':stats,'sha256':sha(args.output)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family-report',type=Path,required=True);p.add_argument('--expected-family-sha256',required=True)
    p.add_argument('--train-decisions',type=Path,required=True);p.add_argument('--expected-train-decisions-sha256',required=True)
    p.add_argument('--reserved-decisions',type=Path,required=True);p.add_argument('--expected-reserved-decisions-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
