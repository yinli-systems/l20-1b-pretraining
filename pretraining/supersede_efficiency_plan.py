"""Supersede only our empty, waiting v1 queue before any GPU evaluation starts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from run_efficiency_evaluation import OUT, CONTINUATION, sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid',type=int,required=True)
    parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args()
    receipt_path=OUT/'receipt.json'
    archive=OUT/'receipt-plan-v1-superseded.json'
    history=OUT/'plan-v1-supersession.json'
    if archive.exists() or history.exists():
        raise FileExistsError('Supersession already exists')
    r=json.loads(receipt_path.read_text())
    if r.get('pid') != args.pid or r.get('plan_sha256') != args.plan_sha256 or r.get('status') != 'waiting_for_B_gpu_lock' or r.get('jobs') != {}:
        raise ValueError('Not the expected empty waiting queue')
    command=(Path('/proc')/str(args.pid)/'cmdline').read_bytes().split(b'\0')
    if str(OUT/'run_efficiency_evaluation.py').encode() not in command or b'--queue' not in command or b'--worker' in command:
        raise ValueError('PID no longer identifies our queue')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).split()
    if str(args.pid) in active or not active:
        raise ValueError('Unexpected CUDA ownership')
    with (CONTINUATION/'gpu.lock').open('r+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError('GPU is no longer occupied; inspect before superseding')
    prior_sha=sha256(receipt_path)
    os.kill(args.pid,signal.SIGTERM)
    for _ in range(50):
        if not (Path('/proc')/str(args.pid)).exists():
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Queue did not exit; do not archive')
    if sha256(receipt_path) != prior_sha:
        raise ValueError('Receipt changed during supersession')
    receipt_path.rename(archive)
    event={'time':time.time(),'terminated_queue_pid':args.pid,'previous_plan_sha256':args.plan_sha256,
           'previous_receipt_sha256':prior_sha,'archive':str(archive),'evaluations_started':0,
           'reason':'Add predeclared lower-compute DataDecide 14.42B and TinyLlama 10B controls before any new benchmark scores.',
           'training_processes_untouched':active}
    with history.open('x') as stream:
        json.dump(event,stream,indent=2)
    print(json.dumps(event),flush=True)


if __name__=='__main__':main()
