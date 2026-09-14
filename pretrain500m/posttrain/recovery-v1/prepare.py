"""Reference existing frozen packed data without duplicating it."""
import hashlib
import json
from pathlib import Path
from mixture import MixtureReader, digest

ROOT=Path('/ssd/scxi253/pretrain500m-20260912-v1')
FROZEN=ROOT/'data/fineweb-edu-formal-v1/manifest.json'
EXPECTED='29f75eca421f26653dd520e650d50e6a7802e9a499944b3b8382688fa3806e1a'

def main():
    assert digest(FROZEN)==EXPECTED
    frozen=json.loads(FROZEN.read_text())
    out=ROOT/'posttrain/recovery-v1-inputs'
    out.mkdir(exist_ok=True)
    tokenizer=ROOT/'formal/hf-v5/tokenizer.json'
    assert tokenizer.is_file()
    tokenizer_sha=digest(tokenizer)
    for name,key,tokens in [('train','train_shards',67108864),('validation','validation_shards',131072)]:
        shards=[{'path':s['path'],'sha256':s['sha256'],'blocks':s['tokens']//2049} for s in frozen[key]]
        doc={'schema':'p529m-packed-mixture-v2','sequence_length':2048,
             'tokenizer_sha256':tokenizer_sha,'weights_percent':{'fineweb_edu':100},
             'sources':[{'id':'fineweb_edu','revision':frozen['revision'],
                         'prior_prediction_tokens':15999172608 if name=='train' else 0,
                         'max_cumulative_epochs':2.2 if name=='train' else 1,
                         'shards':shards}]}
        p=out/(name+'.json')
        payload=json.dumps(doc,indent=2)+'\n'
        if p.exists():assert p.read_text()==payload
        else:p.write_text(payload)
        reader=MixtureReader(p,20260914,tokens,digest(p))
        print(json.dumps({'manifest':name,'sha256':digest(p),'blocks':reader.total_sequences}),flush=True)
    protocol={'protocol_id':'p529m-gpu-recovery-v1','status':'ENGINEERING_ONLY',
              'base_sha256':'13aa21721e15c48cdfdafe30d8fdd41d9c95c90be766327661d96af1e90dd6cf',
              'total_steps':32,'split_at_step':16,'prediction_tokens_per_branch':67108864,
              'world_size':4,'microbatch':4,'accumulation':64,'lr':0.0001,
              'minimum_rolling_mfu_strict':0.5,'claim':'GPU resume equivalence; no model-quality or diverse-corpus admission claim'}
    p=out/'protocol.json';p.write_text(json.dumps(protocol,indent=2)+'\n')
    receipt={'schema':'p529m-frozen-data-engineering-v1','status':'CONTENT_VERIFIED_FOR_ENGINEERING_ONLY',
             'frozen_manifest':str(FROZEN),'frozen_manifest_sha256':EXPECTED,
             'training_manifest_sha256':digest(out/'train.json'),'validation_manifest_sha256':digest(out/'validation.json'),
             'protocol_sha256':digest(p),'corpus_admission_for_new_mixture':False}
    (out/'engineering-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
