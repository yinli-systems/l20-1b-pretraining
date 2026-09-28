"""Raise only this child process's file limit, validate mappings, then exec training."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

from continuation_common import RUN, atomic_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--branch',choices=('A','B'),required=True)
    parser.add_argument('--check-only',action='store_true')
    args = parser.parse_args()
    from run_continuation import PlannedData
    # This is a resource probe, not data admission. The exec'd trainer performs
    # the full program/review/shard hash verification before any backward.
    plan = json.loads((RUN/f'pilot-{args.branch}/data-plan.json').read_text())
    count = sum(len(c['records']) for c in plan['components'])
    before,hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    required = max(4096,count+512)
    if hard != resource.RLIM_INFINITY and required > hard:
        raise RuntimeError('file descriptor hard limit cannot cover frozen plan')
    resource.setrlimit(resource.RLIMIT_NOFILE,(max(before,required),hard))
    data = PlannedData(plan,RUN/f'pilot-{args.branch}/order.npy')
    batch = data.batch(0)
    if batch.shape != (6,2049) or batch.min() < 0 or batch.max() >= 32000:
        raise RuntimeError('first real batch failed shape/token checks')
    record = {'scope':'child_process_only','branch':args.branch,'time':time.time(),
              'nofile_before':before,'nofile_after':resource.getrlimit(resource.RLIMIT_NOFILE)[0],
              'nofile_hard':hard,'mapped_shards':count,'first_batch_shape':list(batch.shape),
              'first_batch_min_token':int(batch.min()),'first_batch_max_token':int(batch.max()),
              'launcher_sha256':sha256(Path(__file__)),'new_training_tokens':0,
              'data_plan_sha256':sha256(RUN/f'pilot-{args.branch}/data-plan.json')}
    atomic_json(RUN/f'pilot-{args.branch}/resource-preflight.json',record)
    print(json.dumps(record),flush=True)
    if args.check_only:
        return
    # exec drops the validation-only mappings; the frozen trainer recreates them.
    os.execv(sys.executable,[sys.executable,'-u',str(Path(__file__).with_name('run_continuation.py')),
                            '--phase','train','--branch',args.branch])


if __name__ == '__main__':
    main()
