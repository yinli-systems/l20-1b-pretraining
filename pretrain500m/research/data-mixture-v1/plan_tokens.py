"""Materialize exact block/token quotas for the research hypotheses.

Creates no GPU jobs. Training integration and corpus admission remain separate.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def allocate(percent,total_blocks):
    if total_blocks<=0 or sum(percent.values())!=100 or any(v<0 for v in percent.values()):
        raise ValueError('invalid block budget or mixture percentages')
    counts={k:total_blocks*v//100 for k,v in percent.items()}
    remaining=total_blocks-sum(counts.values())
    order=sorted(percent,key=lambda k:(-(total_blocks*percent[k]%100),k))
    for key in order[:remaining]:counts[key]+=1
    assert sum(counts.values())==total_blocks
    return counts

def main():
    p=argparse.ArgumentParser();p.add_argument('--tokens',type=int,default=536870912)
    p.add_argument('--output',type=Path,default=ROOT/'token-budget-plan.json');a=p.parse_args()
    source=ROOT/'experiment-design.json';design=json.loads(source.read_text())
    seq=design['screen']['sequence_length'];gbs=design['screen']['global_prediction_tokens_per_step']
    if a.tokens%gbs:raise ValueError('budget must be divisible by global prediction tokens per optimizer step')
    recipes={}
    for key,percent in design['recipes_percent'].items():
        counts=allocate(percent,a.tokens//seq)
        recipes[key]={'blocks':counts,'prediction_tokens':{k:v*seq for k,v in counts.items()},
                      'realized_percent':{k:v*seq/a.tokens*100 for k,v in counts.items()},
                      'raw_uint16_packed_bytes':sum(counts.values())*(seq+1)*2}
    out={'design_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
         'status':'OFFLINE_QUOTA_PLAN_NOT_A_LAUNCH_RECEIPT',
         'prediction_tokens_per_run':a.tokens,'optimizer_steps':a.tokens//gbs,
         'total_screen_prediction_tokens':a.tokens*len(recipes)*len(design['screen']['seeds']),
         'recipes':recipes}
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'output':str(a.output),'steps':out['optimizer_steps'],'total_tokens':out['total_screen_prediction_tokens']}))

if __name__=='__main__':main()
