"""Reproduce a bounded evaluation-only qualification on the existing L20 assets.
No optimizer steps, training loops, scorer changes or checkpoint writes.
"""
from __future__ import annotations
import argparse,fcntl,gc,hashlib,json,os,random,sys,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
from prepared_eval import PreparedEvaluation

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
def fingerprint(lab):
    h=hashlib.sha256()
    for name,p in lab.named:h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def select(rows,n):
    chosen=[]
    for domain in ('plotqa','docmatix'):
        pool=[r for r in rows if r['split']=='train' and r['domain']==domain]
        random.Random(2026092906).shuffle(pool);chosen+=pool[:n//2]
    if len(chosen)!=n:raise ValueError('insufficient TRAIN data')
    return chosen

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--images',type=int,default=128)
    ap.add_argument('--timing-images',type=int,default=64)
    ap.add_argument('--abba-blocks',type=int,default=2)
    args=ap.parse_args();w=args.workspace.resolve();out=args.output.resolve()
    if args.images%16 or not 16<=args.images<=128:raise ValueError('16..128 images in blocks of16')
    if args.timing_images%16 or not 16<=args.timing_images<=args.images:raise ValueError('invalid timing subset')
    if not 1<=args.abba_blocks<=4:raise ValueError('bounded ABBA count required')
    if out.exists():raise FileExistsError('output already exists; do not overwrite evidence')
    lock=(w.parent/'general-vision-gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=w/'optimization-20260929T0502Z/prepared-v2/rows.json'
    if sha(source)!='55918147e585d17fde0f20b922d33d3d8de6013a041559946da39f1ad96318d2':raise ValueError('derived data changed')
    sys.path.insert(0,str(w));import spatial_route as spatial
    out.mkdir(parents=True)
    write(out/'policy.json',{'images':args.images,'timing_images':args.timing_images,'max_new_tokens':24,'batch':8,
       'abba_blocks':args.abba_blocks,'precision':'unchanged','pixels_and_features_not_recomputed_between_three_checks':True,
       'generated_token_ids_must_be_identical':True,'diagnostic_values_must_be_identical':True,'optimizer_steps':0,
       'source_hashes':{p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('prepared_eval.py'))}})
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    lab,opt,_=spatial.load_parent('T');before=fingerprint(lab)
    raw=select(json.loads(source.read_text()),args.images)
    rows=[dict(spatial.core.encode(lab,r,448),_route='T',_resolution=448) for r in raw]
    write(out/'selection.json',{'ids':[r['selection_sha256'] for r in rows]})
    original=lab.language.generate
    def reference(part,capture=False):
        ids=[]
        def record(*a,**kw):
            value=original(*a,**kw);ids.append(value.detach().cpu().tolist());return value
        if capture:lab.language.generate=record
        try:
            true=lab.generate(part,max_new=24);wrong=lab.generate(part,max_new=24,mode='shuffled')
            diag=lab.diagnostics(lab.batch(part))
        finally:
            if capture:lab.language.generate=original
        return true,wrong,diag,ids
    def prepared(part,capture=False):
        with PreparedEvaluation(lab,part,spatial.core) as bundle:
            t,ts=bundle.generate(24);f,fs=bundle.generate(24,mode='shuffled');diag=bundle.diagnostics()
            ids=[t.detach().cpu().tolist(),f.detach().cpu().tolist()] if capture else []
        return ts,fs,diag,ids
    reference(rows[:8]);prepared(rows[:8]);qualification=[]
    for i in range(0,len(rows),8):
        a=reference(rows[i:i+8],True);b=prepared(rows[i:i+8],True)
        row={'batch':i//8,'true_tokens_equal':a[3][0]==b[3][0],'wrong_tokens_equal':a[3][1]==b[3][1],
             'strings_equal':a[:2]==b[:2],'diagnostics_max_abs':max(abs(a[2][k]-b[2][k]) for k in a[2])}
        qualification.append(row)
    passed=all(r['true_tokens_equal'] and r['wrong_tokens_equal'] and r['diagnostics_max_abs']==0 for r in qualification)
    write(out/'qualification.json',{'pass':passed,'rows':qualification})
    if not passed:raise RuntimeError('numerical gate failed; no performance promotion')
    # Regression on the original low-resolution path: AMP scopes differ by route.
    low=[];seen=set()
    for row in lab.rows['train']:
        identity=row.get('image_sha256',row.get('image_cluster'))
        if identity and identity not in seen and row['source']=='open_images_localized_narratives':
            low.append(dict(row,_resolution=224,_route='G'));seen.add(identity)
        if len(low)==8:break
    a=reference(low,True);b=prepared(low,True)
    low_report={'images':8,'true_tokens_equal':a[3][0]==b[3][0],'wrong_tokens_equal':a[3][1]==b[3][1],
                'diagnostics_max_abs':max(abs(a[2][k]-b[2][k]) for k in a[2])}
    write(out/'original224-regression.json',low_report)
    if not (low_report['true_tokens_equal'] and low_report['wrong_tokens_equal'] and low_report['diagnostics_max_abs']==0):
        raise RuntimeError('Original224 regression failed')
    bench=rows[:args.timing_images//2]+rows[args.images//2:args.images//2+args.timing_images//2]
    timing=[]
    for i,mode in enumerate(['reference','prepared','prepared','reference']*args.abba_blocks):
        gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        for j in range(0,len(bench),8):(reference if mode=='reference' else prepared)(bench[j:j+8])
        torch.cuda.synchronize()
        row={'sequence':i,'block':i//4,'mode':mode,'seconds':time.perf_counter()-started,
             'images':len(bench),'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        timing.append(row);print(json.dumps(row),flush=True)
    a=sum(r['seconds'] for r in timing if r['mode']=='reference');b=sum(r['seconds'] for r in timing if r['mode']=='prepared')
    summary={'status':'complete','qualification_pass':passed,'original224_regression':low_report,'timing':timing,'speed_ratio':a/b,
        'time_reduction_percent':100*(1-b/a),'weights_unchanged':fingerprint(lab)==before,
        'checkpoint_manifest_sha256':sha(spatial.PARENT/'manifest.json'),'optimizer_steps':0,
        'scope':'true/wrong/NLL evaluation bundle only; not full training or benchmark single-pass throughput',
        'environment':{'torch':torch.__version__,'gpu':torch.cuda.get_device_name()},'created_unix':time.time()}
    write(out/'summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
