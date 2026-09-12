#!/usr/bin/env python3
"""Verify frozen exclusion inputs before and after the isolated B builder.

Kept separate so the already completed B refill builder and its hash bindings
remain unchanged. Initial four pools additionally receive a post-build audit.
"""
import json
from pathlib import Path
import sys
import time

import prepare_continuation_b as builder
from continuation_common import ROOT, RUN, atomic_json, sha256


def verify_inputs():
    base_path = RUN/'base-inputs.json'
    base = json.loads(base_path.read_text())
    result = {'base_inputs_sha256':sha256(base_path),'indices':{}}
    for split in ('train','val'):
        item = base['indices'][split]
        path = RUN/f'indices/old-{split}.npy'
        if Path(item['path']).resolve()!=path.resolve() or sha256(path)!=item['sha256']:
            raise RuntimeError(f'frozen {split} exclusion index mismatch')
        result['indices'][split] = {'path':str(path),'sha256':item['sha256']}
    old = ROOT/'manifests/full-dedup.sqlite'
    if sha256(old)!=base['original_dedup_sha256']:
        raise RuntimeError('original document exclusion database changed')
    result['old_dedup_sha256'] = base['original_dedup_sha256']
    result['tokenizer_sha256'] = sha256(ROOT/'tokenizer/tokenizer.json')
    result['benchmark_sha256'] = sha256(RUN/'benchmark.sqlite')
    if result['tokenizer_sha256']!=base['tokenizer_sha256']:
        raise RuntimeError('frozen tokenizer changed')
    review = json.loads((RUN/'quality-review-A.json').read_text())
    for source,path in builder.BASES.items():
        if sha256(path)!=review['manifests'][source]:
            raise RuntimeError('A source pool changed')
        m = json.loads(path.read_text())
        for key in ('base_inputs_sha256','tokenizer_sha256','benchmark_sha256'):
            if m[key]!=result[key]:
                raise RuntimeError(f'A source exclusion binding changed: {key}')
    # Existing B outputs must bind these same actual exclusion inputs and shards.
    result['existing_B_manifests'] = {}
    for path in sorted(builder.BROOT.glob('*/manifest.json')):
        m = json.loads(path.read_text())
        for key in ('base_inputs_sha256','tokenizer_sha256','benchmark_sha256'):
            if m[key]!=result[key]:
                raise RuntimeError(f'B source exclusion binding changed: {path}/{key}')
        for item in m['shards']:
            if sha256(Path(item['path']))!=item['sha256']:
                raise RuntimeError('existing B token shard changed')
        result['existing_B_manifests'][str(path)] = sha256(path)
    return result


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        builder.main()
        return
    before = verify_inputs()
    stamp = time.time_ns()
    output = builder.BROOT/f'input-verification-{stamp}.json'
    receipt = {'stage':'precheck_passed','before':before,'time':time.time(),
               'builder_sha256':sha256(Path(builder.__file__)),
               'verifier_sha256':sha256(Path(__file__)),
               'note':'Existing B pools are audited post-build; new source uses pre/post checks.'}
    atomic_json(output,receipt)
    print(json.dumps({'stage':'B_exclusion_inputs_verified','receipt':str(output)}),flush=True)
    try:
        builder.main()
        after = verify_inputs()
        for key in before:
            if key!='existing_B_manifests' and before[key]!=after[key]:
                raise RuntimeError('frozen exclusion input changed during build')
        receipt.update(stage='pre_and_post_checks_passed',after=after,completed_unix=time.time())
    except (Exception,KeyboardInterrupt) as error:
        receipt.update(stage='failed',error=repr(error),failed_unix=time.time())
        atomic_json(output,receipt)
        raise
    atomic_json(output,receipt)


if __name__=='__main__':
    main()
