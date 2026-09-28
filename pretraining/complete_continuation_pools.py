"""Refill v2 shortfalls with newly filtered documents, then publish manifests.

No training or quality approval is triggered by this CPU-only preparation job.
"""
import json
import math
from pathlib import Path
import time

from continuation_common import RUN, BLOCK, GLOBAL, MIXES, PILOT_STEPS, atomic_json, lock, quotas, sha256
from continuation_quality import VERSION, reject_reason
from prepare_continuation import build_source


def main():
    handle = lock(RUN/'prepare.lock')
    counts = [quotas(mix,PILOT_STEPS*GLOBAL) for mix in MIXES.values()]
    for source in ('math','web','dclm','code'):
        path = RUN/f'curated-v2/{source}/manifest.json'
        m = json.loads(path.read_text())
        if m['quality_version'] != VERSION or m['quality_policy_sha256'] != sha256(Path(__file__).with_name('continuation_quality.py')):
            raise RuntimeError('curated pool policy mismatch')
        required = max(q.get('fresh_'+source,0) for q in counts)*BLOCK
        inputs = [path]
        records = list(m['shards'])
        if m['tokens'] < required:
            deficit = required-m['tokens']
            directory = RUN/f'refill-v2/{source}'
            target = math.ceil((deficit*1.05+BLOCK)/BLOCK)*BLOCK
            print(json.dumps({'stage':'refilling','source':source,'required':required,
                              'curated_tokens':m['tokens'],'additional_target':target}),flush=True)
            build_source(source,RUN,target,directory=directory,quality_filter=reject_reason)
            extra_path = directory/'manifest.json'
            extra = json.loads(extra_path.read_text())
            for key in ('quality_version','quality_policy_sha256','tokenizer_sha256','base_inputs_sha256','benchmark_sha256'):
                if m[key] != extra[key]:
                    raise RuntimeError(f'refill binding changed: {key}')
            inputs.append(extra_path)
            records.extend(extra['shards'])
        for record in records:
            if sha256(Path(record['path'])) != record['sha256']:
                raise RuntimeError('pool changed before composition')
        result = {k:m[k] for k in ('source','source_revisions','tokenizer_sha256',
                                   'base_inputs_sha256','benchmark_sha256','quality_version','quality_policy_sha256')}
        result.update(shards=records,tokens=sum(x['tokens'] for x in records),required_input_tokens=required,
                      input_manifests=[{'path':str(p),'sha256':sha256(p)} for p in inputs],
                      stage='composed_pending_review',completed_unix=time.time())
        output = RUN/f'review-candidate-v2/{source}/manifest.json'
        if output.exists():
            old = json.loads(output.read_text())
            if {k:v for k,v in old.items() if k!='completed_unix'} != {k:v for k,v in result.items() if k!='completed_unix'}:
                raise RuntimeError('composed pool is frozen')
        else:
            atomic_json(output,result)
        print(json.dumps({'stage':'candidate_pool_ready','source':source,'tokens':result['tokens']}),flush=True)
    handle.close()


if __name__ == '__main__':
    main()
