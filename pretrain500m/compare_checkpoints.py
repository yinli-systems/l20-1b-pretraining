#!/usr/bin/env python3
"""Bitwise compare two complete training checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def equal(a,b,path='root'):
    if type(a) is not type(b):raise AssertionError(f'{path}: {type(a)} != {type(b)}')
    if isinstance(a,torch.Tensor):
        if not torch.equal(a,b):raise AssertionError(f'{path}: tensor mismatch')
        return (1,a.numel())
    if isinstance(a,np.ndarray):
        if not np.array_equal(a,b):raise AssertionError(f'{path}: ndarray mismatch')
        return (0,0)
    if isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError(f'{path}: keys mismatch')
        counts=[equal(a[k],b[k],f'{path}.{k}') for k in a]
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise AssertionError(f'{path}: length mismatch')
        counts=[equal(x,y,f'{path}[{i}]') for i,(x,y) in enumerate(zip(a,b))]
    else:
        if a!=b:raise AssertionError(f'{path}: {a!r} != {b!r}')
        counts=[]
    return (sum(x for x,_ in counts),sum(y for _,y in counts))


def main():
    p=argparse.ArgumentParser();p.add_argument('left',type=Path);p.add_argument('right',type=Path);a=p.parse_args()
    left=torch.load(a.left,map_location='cpu',weights_only=False)
    right=torch.load(a.right,map_location='cpu',weights_only=False)
    tensors,elements=equal(left,right)
    print(json.dumps({'status':'PASS_BITWISE_COMPLETE_STATE_RESUME','step':left['step'],
                      'tensors_compared':tensors,'tensor_elements_compared':elements,
                      'left':str(a.left),'right':str(a.right)}))


if __name__=='__main__':main()
