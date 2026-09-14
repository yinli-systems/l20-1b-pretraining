"""Compare actual trained GPU checkpoints after continuous and split runs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

def equal(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        if a.shape!=b.shape or a.dtype!=b.dtype or not torch.equal(a,b):raise ValueError('tensor mismatch: '+path)
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b):raise ValueError('array mismatch: '+path)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('key mismatch: '+path)
        for k in a:equal(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise ValueError('length mismatch: '+path)
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
    elif a!=b:raise ValueError('value mismatch: '+path)

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    report={'status':'FAIL','formal_quality_gain':False,'scope':'same-allocation 4-GPU recovery with frozen FineWeb data'}
    try:
        a=torch.load(args.root/'continuous/resume.pt',map_location='cpu',weights_only=False,mmap=True)
        b=torch.load(args.root/'split/resume.pt',map_location='cpu',weights_only=False,mmap=True)
        equal(a,b)
        for name in ('continuous','split'):
            status=json.loads((args.root/name/'training-status.json').read_text())
            if status['status']!='QUALIFICATION_COMPLETED' or not status['mfu_window_passed']:
                raise ValueError('MFU qualification did not pass: '+name)
        report.update(status='PASS_GPU_RECOVERY_EXACT',steps=a['step'],run_fingerprint=a['run_fingerprint'],
                      compared='model, optimizer, reader cursor, origin, fingerprint and all-rank RNG state')
    except Exception as exc:report['error']=str(exc)
    (args.root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if report['status']!='PASS_GPU_RECOVERY_EXACT':raise SystemExit(1)

if __name__=='__main__':main()
