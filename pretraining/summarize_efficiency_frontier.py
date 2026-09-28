#!/usr/bin/env python3
"""Publish only hash-verified matched-protocol points, with explicit scope limits."""
import argparse
import html
import json
import math
from pathlib import Path
import time

from run_efficiency_evaluation import OUT, ROOT, atomic_json, read_plan, sha256


def apparent_frontier(points):
    eligible=[p for p in points if p['category']=='natural_corpus_base']
    return [p['id'] for p in eligible if not any(
        q['compute']<=p['compute'] and q['score']>=p['score'] and
        (q['compute']<p['compute'] or q['score']>p['score']) for q in eligible)]


def render_svg(points):
    left,right,top,bottom=85,1000,65,460
    lo=math.floor(math.log10(min(p['compute'] for p in points)))
    hi=math.ceil(math.log10(max(p['compute'] for p in points)))
    if lo==hi:hi+=1
    ymin=5*math.floor(min(p['score'] for p in points)/5)-5
    ymax=5*math.ceil(max(p['score'] for p in points)/5)+5
    x=lambda v:left+(math.log10(v)-lo)/(hi-lo)*(right-left)
    y=lambda v:bottom-(v-ymin)/(ymax-ymin)*(bottom-top)
    lines=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="565" viewBox="0 0 1100 565">',
           '<rect width="1100" height="565" fill="white"/>',
           '<g font-family="sans-serif" fill="#172536">',
           '<text x="85" y="30" font-size="20">Matched-protocol accuracy vs approximate training compute</text>']
    for exponent in range(lo,hi+1):
        xx=x(10**exponent)
        lines += [f'<path d="M {xx} {top} V {bottom}" stroke="#e1e5ea"/>',
                  f'<text x="{xx}" y="485" text-anchor="middle" font-size="13">10^{exponent}</text>']
    for score in range(int(ymin),int(ymax)+1,5):
        yy=y(score)
        lines += [f'<path d="M {left} {yy} H {right}" stroke="#e1e5ea"/>',
                  f'<text x="73" y="{yy+4}" text-anchor="end" font-size="13">{score}%</text>']
    frontier=apparent_frontier(points)
    linepoints=sorted([p for p in points if p['id'] in frontier],key=lambda p:p['compute'])
    if len(linepoints)>1:
        coords=' '.join(f'{x(p["compute"])},{y(p["score"])}' for p in linepoints)
        lines.append(f'<polyline points="{coords}" fill="none" stroke="#acb6c2" stroke-dasharray="5 5"/>')
    for p in points:
        color='#d47416' if p['category']=='teacher_synthetic_reference' else ('#008170' if p['id']=='ours_20b' else '#3469ac')
        label=html.escape(p['id'])
        lines += [f'<circle cx="{x(p["compute"])}" cy="{y(p["score"])}" r="5" fill="{color}"><title>{label}: {p["score"]:.3f}%</title></circle>',
                  f'<text x="{x(p["compute"])+7}" y="{y(p["score"])-7}" font-size="10">{label}</text>']
    lines += ['<text x="540" y="515" text-anchor="middle" font-size="14">6 × total parameters × training tokens (proxy FLOPs; log scale)</text>',
              '<text x="85" y="539" font-size="12">Dashed line: point-estimate frontier within evaluated natural-corpus candidates only; not global SOTA.</text>',
              '<text x="85" y="555" font-size="12">Orange: teacher/synthetic reference. Data construction and experiment-search compute excluded. See paired intervals in JSON.</text>',
              '</g></svg>']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    plan=read_plan(args.plan)
    receipt=json.loads((OUT/'receipt.json').read_text())
    if receipt['plan_sha256']!=sha256(args.plan):raise ValueError('Receipt/plan mismatch')
    points=[]
    for p in plan['reference_points']:
        for path,digest in p['evidence_sha256'].items():
            if sha256(ROOT/path)!=digest:raise ValueError('Reference evidence changed')
        points.append(dict(p))
    for job in plan['jobs']:
        r=receipt['jobs'].get(job['id'],{})
        if r.get('status')!='complete':continue
        path=OUT/'results'/f'{job["id"]}-summary.json'
        if sha256(path)!=r['summary_sha256']:raise ValueError('Analysis changed')
        s=json.loads(path.read_text())
        if sha256(OUT/'results'/f'{job["id"]}.json')!=s['result_sha256']:raise ValueError('Result changed')
        points.append({'id':job['id'],'compute':s['training_flops_proxy_6ND'],
                       'score':s['seven_task_macro']['baseline']*100,'category':job['category'],
                       'tokens':job['training_tokens'],'parameters':s['total_parameters'],
                       'paired_ours_minus_baseline':s['seven_task_macro'],
                       'six_without_boolq':s['six_task_without_boolq'],'summary_sha256':r['summary_sha256']})
    payload={'created_unix':time.time(),'plan_sha256':sha256(args.plan),'points':points,
             'apparent_candidate_frontier':apparent_frontier(points),'global_ranking_claim':False,
             'status':'partial' if receipt['status']!='complete' else 'declared_subset_complete',
             'limitations':['Do not combine upstream OLMES/old-harness results with these points.',
                            'Point-estimate frontier is not uncertainty-adjusted dominance.',
                            'Compute is approximate; token counts can be rounded; source provenance is in the plan.',
                            'Missing/gated/unreleased models mean this cannot establish a global frontier.']}
    atomic_json(OUT/'frontier-summary.json',payload)
    if len(points)>=3:
        (OUT/'frontier.svg').write_text(render_svg(points))
    print(json.dumps({'status':payload['status'],'verified_points':len(points)}),flush=True)


if __name__=='__main__':main()
