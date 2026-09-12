"""Convert one hash-pinned official PyTorch state dict to safetensors.

This program deliberately permits ``weights_only=False`` only after an exact
source hash and a narrow static pickle audit. Run it in a networkless,
read-only container with only the output directory writable.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import pickletools
import tempfile
import time
import zipfile

ALLOWED_OPCODES = {
    'PROTO', 'FRAME', 'EMPTY_DICT', 'MEMOIZE', 'MARK', 'SHORT_BINUNICODE',
    'STACK_GLOBAL', 'BINGET', 'BININT1', 'BININT2', 'BININT', 'TUPLE',
    'REDUCE', 'TUPLE2', 'BINPERSID', 'NEWFALSE', 'EMPTY_TUPLE', 'NEWTRUE',
    'SETITEM', 'TUPLE1', 'SETITEMS', 'STOP',
}
ALLOWED_GLOBALS = {
    ('torch._tensor', '_rebuild_from_type_v2'),
    ('torch._utils', '_rebuild_tensor_v2'),
    ('torch', 'Tensor'),
    ('collections', 'OrderedDict'),
    ('torch', 'FloatStorage'),
}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def inspect_archive(path: Path) -> dict:
    if not zipfile.is_zipfile(path):
        raise ValueError('Only a zip-format torch checkpoint is admitted')
    with zipfile.ZipFile(path) as archive:
        pickle_members = [name for name in archive.namelist() if name.endswith('/data.pkl')]
        if len(pickle_members) != 1:
            raise ValueError('Expected exactly one data.pkl')
        data = archive.read(pickle_members[0])
    operations = list(pickletools.genops(data))
    counts = Counter(op.name for op, _, _ in operations)
    unknown = set(counts) - ALLOWED_OPCODES
    if unknown:
        raise ValueError(f'Unadmitted pickle opcodes: {sorted(unknown)}')
    if not operations or operations[0][0].name != 'PROTO' or operations[0][1] != 5:
        raise ValueError('Expected pickle protocol 5')
    memo_strings = {}
    memo_index = 0
    for index, (operation, _, _) in enumerate(operations):
        if operation.name == 'MEMOIZE':
            if index and operations[index - 1][0].name == 'SHORT_BINUNICODE':
                memo_strings[memo_index] = operations[index - 1][1]
            memo_index += 1
    globals_seen = []
    for index, (operation, _, _) in enumerate(operations):
        if operation.name != 'STACK_GLOBAL':
            continue
        producers = []
        cursor = index - 1
        while cursor >= 0 and len(producers) < 2:
            if operations[cursor][0].name != 'MEMOIZE':
                producers.append(operations[cursor])
            cursor -= 1
        if len(producers) != 2 or producers[0][0].name != 'SHORT_BINUNICODE':
            raise ValueError('Unrecognized STACK_GLOBAL name encoding')
        name = producers[0][1]
        module_op, module_arg, _ = producers[1]
        if module_op.name == 'SHORT_BINUNICODE':
            module = module_arg
        elif module_op.name == 'BINGET' and module_arg in memo_strings:
            module = memo_strings[module_arg]
        else:
            raise ValueError('Unadmitted indirect STACK_GLOBAL module encoding')
        target = (module, name)
        if target not in ALLOWED_GLOBALS:
            raise ValueError(f'Unadmitted pickle global: {target}')
        globals_seen.append('.'.join(target))
    if set(tuple(x.rsplit('.', 1)) for x in globals_seen) != ALLOWED_GLOBALS:
        raise ValueError('Expected torch state-dict globals were not all present')
    return {'pickle_member': pickle_members[0], 'protocol': 5,
            'opcode_counts': dict(sorted(counts.items())),
            'globals': sorted(globals_seen)}


def write_json(path: Path, value: dict) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--expected-tensors', type=int, required=True)
    parser.add_argument('--expected-numel', type=int, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists() or args.output.parent != args.receipt.parent:
        raise ValueError('Output and receipt must be new files in one directory')
    if sha256(args.source) != args.source_sha256:
        raise ValueError('Source SHA-256 mismatch')
    pickle_audit = inspect_archive(args.source)

    import torch
    from safetensors.torch import load_file, save_file

    # The exact source is an immutable official artifact and this process must
    # be externally sandboxed. No other code path in the evaluator permits it.
    state = torch.load(args.source, map_location='cpu', mmap=True, weights_only=False)
    if type(state) is not dict or len(state) != args.expected_tensors:
        raise ValueError('Unexpected checkpoint container or tensor count')
    if any(type(value) is not torch.Tensor for value in state.values()):
        raise ValueError('Checkpoint contains non-tensor values or subclasses')
    if any(value.device.type != 'cpu' or value.layout != torch.strided or value.is_sparse or value.is_quantized
           for value in state.values()):
        raise ValueError('Only dense CPU tensors are admitted')
    if any(value.dtype != torch.float32 or not value.is_contiguous() for value in state.values()):
        raise ValueError('Expected contiguous FP32 model weights')
    numel = sum(value.numel() for value in state.values())
    if numel != args.expected_numel:
        raise ValueError(f'Parameter count mismatch: {numel}')

    save_file(state, args.output, metadata={'format': 'pt',
        'source_sha256': args.source_sha256,
        'conversion': 'hash-pinned static-audited official state_dict'})
    converted = load_file(args.output, device='cpu')
    if set(converted) != set(state):
        raise ValueError('Converted key set mismatch')
    for name, original in state.items():
        candidate = converted[name]
        if candidate.shape != original.shape or candidate.dtype != original.dtype or not torch.equal(candidate, original):
            raise ValueError(f'Converted tensor mismatch: {name}')

    result = {'schema': 1, 'status': 'exact_tensor_equality_verified',
              'source': str(args.source), 'source_bytes': args.source.stat().st_size,
              'source_sha256': args.source_sha256,
              'output': str(args.output), 'output_bytes': args.output.stat().st_size,
              'output_sha256': sha256(args.output), 'tensor_count': len(state),
              'numel': numel, 'dtype': 'float32', 'pickle_audit': pickle_audit,
              'torch_version': torch.__version__, 'safetensors_version': __import__('safetensors').__version__,
              'completed_unix': time.time(),
              'required_launcher_boundary': 'network=none, read-only root, cap-drop=all, no-new-privileges, pids<=64'}
    write_json(args.receipt, result)
    print(json.dumps(result, separators=(',', ':')))


if __name__ == '__main__':
    main()
