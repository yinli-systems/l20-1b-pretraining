"""Content hashes at the recovery boundary; no large diagnostic tensor files."""
import hashlib
import json
import time
import torch


def fingerprint(value):
    if isinstance(value, torch.Tensor):
        cpu = value.detach().cpu().contiguous()
        raw = cpu.reshape(-1).view(torch.uint8).numpy()
        return {'type': 'tensor', 'dtype': str(cpu.dtype), 'shape': list(cpu.shape),
                'sha256': hashlib.sha256(memoryview(raw)).hexdigest()}
    if isinstance(value, dict):
        return {'type': 'dict', 'entries': [[str(k), fingerprint(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {'type': type(value).__name__, 'items': [fingerprint(v) for v in value]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError('unsupported diagnostic value: ' + str(type(value)))


def append_trace(directory, step, phase, state, ddp):
    begin = time.monotonic()
    info = fingerprint(state)
    record = {'step': step, 'phase': phase, 'unix': time.time(),
              'fingerprint': info, 'seconds': time.monotonic() - begin,
              'buckets_rebuilt': bool(ddp._has_rebuilt_buckets),
              'bucket_sizes': ddp._get_ddp_logging_data().get('bucket_sizes')}
    # Each gradient rank writes its own file; only rank zero writes state traces.
    filename = phase + '.jsonl' if phase.startswith('gradient_rank_') else 'recovery-trace.jsonl'
    with (directory / filename).open('a') as handle:
        handle.write(json.dumps(record, sort_keys=True) + '\n')
