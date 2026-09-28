"""CPU-only real-checkpoint transport, exact-loading and harness smoke.

One invented likelihood example, not a benchmark and not score selection.
"""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    available = next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    if available < 8*1024**3:
        raise RuntimeError('Not enough host RAM to smoke alongside training')
    import torch
    from lm_eval.api.instance import Instance
    from strict_efficiency_hf import StrictHFLM
    torch.set_num_threads(2)
    started = time.time()
    model = StrictHFLM(pretrained=str(args.model), device='cpu', dtype='bfloat16',
                       batch_size=1, max_length=2048, trust_remote_code=False)
    with torch.inference_mode():
        logits = model.model(input_ids=torch.tensor([[2,3,4,5]]), use_cache=False).logits
    assert torch.isfinite(logits).all()
    instance = Instance(request_type='loglikelihood', doc={},
                        arguments=('The cup is on the', ' table.'), idx=0,
                        metadata=('nonbenchmark_smoke', 0, 1))
    score = model.loglikelihood([instance])
    assert len(score) == 1 and torch.isfinite(torch.tensor(score[0][0]))
    assert not torch.cuda.is_initialized()
    record = {'status':'actual_checkpoint_cpu_pass', 'device':'cpu',
              'model':str(args.model), 'total_parameters':sum(p.numel() for p in model.model.parameters()),
              'loading_info':model.efficiency_loading_info, 'finite_logits':True,
              'harness_single_synthetic_example':score, 'seconds':time.time()-started,
              'time':time.time(), 'torch':torch.__version__,
              'claim_boundary':'Not a benchmark result; does not validate GPU execution.'}
    with args.output.open('x') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
