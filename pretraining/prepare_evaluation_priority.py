"""Record the user-requested safe pause and archive the empty old queue."""
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time
from run_efficiency_evaluation import OUT, CONTINUATION, sha256


def main():
    destination=OUT/'evaluation-priority-pause.json'
    archive=OUT/'receipt-plan-v2-paused.json'
    if destination.exists() or archive.exists():
        raise FileExistsError('Priority transition already recorded')
    with (OUT/'queue.lock').open('a') as q, (CONTINUATION/'gpu.lock').open('r+') as g:
        fcntl.flock(q,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=json.loads((CONTINUATION/'pilot-B/status.json').read_text())
        checkpoint=json.loads((CONTINUATION/'pilot-B/resume.json').read_text())
        assert status['stage']=='checkpointed_stop'
        assert checkpoint['step']==status['branch_step']
        path=CONTINUATION/'pilot-B/resume.pth'
        assert path.stat().st_size==checkpoint['bytes']
        assert sha256(path)==checkpoint['sha256']
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        assert not gpu, gpu
        receipt=json.loads((OUT/'receipt.json').read_text())
        assert receipt['jobs']=={} and receipt['status']=='failed'
        assert receipt['plan_sha256']=='cb12b567149ef30aa23bde14a05c56e201d6e8bd4461f7c7a8a73fe3d1c3eb78'
        assert not (Path('/proc')/str(receipt['pid'])).exists()
        old_runner=OUT/'run_efficiency_evaluation-plan-v2.py'
        assert not old_runner.exists()
        shutil.copyfile(OUT/'run_efficiency_evaluation.py',old_runner)
        (OUT/'receipt.json').rename(archive)
        record={'reason':'user_requested_evaluation_priority',
                'user_request':'先完成评测吧，训练可以等等', 'time':time.time(),
                'status':status,'checkpoint':checkpoint,'cuda_processes':[],
                'old_queue_archive':str(archive),'old_queue_sha256':sha256(archive),
                'old_runner_sha256':sha256(old_runner),'automatic_training_resume':False}
        with destination.open('x') as stream: json.dump(record,stream,indent=2,ensure_ascii=False)
        print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
