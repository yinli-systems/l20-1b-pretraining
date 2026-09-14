"""Read verified upstream 530M research evidence, separate from frozen evaluation."""
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pyarrow.compute as pc
import pyarrow.parquet as pq

source=Path('/Users/alice/Documents/ChatGPT/Pretraining 2')
cache=source/'cache/datadecide-upstream-20260912'
previous=json.loads((source/'reports/metrics/datadecide-upstream-scan-20260912.json').read_text())
tasks={'hellaswag','arc_easy','arc_challenge','openbookqa','piqa','winogrande','boolq'}
groups=defaultdict(dict)
for item in previous['source_files']:
    p=cache/Path(item['rfilename']).name
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    assert h.hexdigest()==item['lfs']['sha256']
    for batch in pq.ParquetFile(p).iter_batches(batch_size=8192):
        filtered=batch.filter(pc.equal(batch.column('params'),'530M'))
        for r in filtered.to_pylist():
            if r['task'] in tasks:
                key=(r['data'],r['step'],r['seed'],r['tokens'])
                assert r['task'] not in groups[key]
                groups[key][r['task']]=float(ast.literal_eval(r['metrics'])['primary_metric'])
rows=[]
for k,v in groups.items():
    if set(v)==tasks:
        rows.append(dict(recipe=k[0],step=k[1],seed=k[2],tokens=k[3],scores=v,
                         seven=sum(v.values())/7,six=sum(x for n,x in v.items() if n!='boolq')/6))
rows.sort(key=lambda x:(x['tokens'],x['recipe'],x['seed']))
out=Path(__file__).resolve().parent/'reports/datadecide-530m-upstream.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'revision':previous['revision'],'source_files':previous['source_files'],
    'scope':'Upstream OLMES research evidence; not scores under this project evaluation protocol.',
    'rows':rows},indent=2)+'\n')
for target in [2_000_000_000,5_000_000_000,10_000_000_000,20_000_000_000,53_000_000_000]:
    budgets=sorted({x['tokens'] for x in rows});chosen=min(budgets,key=lambda x:abs(x-target))
    ranked=defaultdict(list)
    for r in rows:
        if r['tokens']==chosen:ranked[r['recipe']].append(r)
    values=[{'recipe':k,'seeds':len(v),'seven':sum(x['seven'] for x in v)/len(v),'six':sum(x['six'] for x in v)/len(v)} for k,v in ranked.items()]
    values.sort(key=lambda x:-x['seven'])
    print(json.dumps({'tokens':chosen,'top10':values[:10],
                      'anchors':[x for x in values if x['recipe'] in ['FineWeb-Edu','DCLM-Baseline','DCLM-Baseline (QC 20%)']]}),flush=True)
