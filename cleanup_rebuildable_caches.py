"""Explicit, receipted cleanup of three rebuildable caches on the L20 host.

Never touches pretraining data/checkpoints, installed packages, or evaluation data.
Audit first; apply requires the exact inventory hash and idle cooperative locks.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time

ROOT = Path('/home/hhai/pretrain')
CACHE = Path('/home/hhai/.cache')
PIP = CACHE/'pip/http-v2'
MODELS = {
    'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T': 'tinyllama_3t',
    'TinyLlama/TinyLlama_v1.1': 'tinyllama_v1_1',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def safe_file(path):
    s = path.lstat()
    if path.resolve() != path or not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or s.st_nlink != 1:
        raise ValueError(f'Unsafe owner/path/type/link count: {path}')
    return s


def item(path, reason, **extra):
    s = safe_file(path)
    return {'path': str(path), 'bytes': s.st_size, 'allocated_bytes': s.st_blocks*512,
            'inode': s.st_ino, 'mtime_ns': s.st_mtime_ns, 'sha256': digest(path),
            'reason': reason, **extra}


def inventory():
    receipt_path = ROOT/'evaluations/comparison-20260911/receipt.json'
    receipt = json.loads(receipt_path.read_text())
    preserved = {str(receipt_path): digest(receipt_path)}
    records = []
    if PIP.exists():
        if PIP.resolve() != PIP or PIP.stat().st_uid != os.getuid():
            raise ValueError('Unexpected pip cache root')
        for path in sorted(PIP.rglob('*')):
            if path.is_symlink():
                raise ValueError('Unexpected symlink in pip cache')
            if path.is_file():
                records.append(item(path, 'pip HTTP download cache; installed environment retained'))
    for repo, prefix in MODELS.items():
        jobs = [receipt['jobs'][prefix+suffix] for suffix in ('_core', '_mmlu_5shot', '_gsm8k_5shot')]
        revisions = {job['job']['revision'] for job in jobs}
        if len(revisions) != 1 or any(j['status'] != 'complete' or j['job']['model'] != repo for j in jobs):
            raise ValueError('Official comparison evaluations incomplete or identity changed')
        revision = revisions.pop()
        for job in jobs:
            p = Path(job['result'])
            if digest(p) != job['sha256']:
                raise ValueError('Completed evaluation result changed')
            preserved[str(p)] = job['sha256']
        directory = CACHE/'huggingface/hub'/('models--'+repo.replace('/', '--'))
        for path in sorted((directory/'blobs').iterdir()):
            if path.stat().st_size < 1024**3:
                continue
            record = item(path, 'Official baseline weights; completed raw evaluations retained',
                          repo=repo, revision=revision)
            if path.name != record['sha256']:
                raise ValueError('Hub LFS content hash does not match blob name')
            links = [p for p in (directory/'snapshots'/revision).iterdir()
                     if p.is_symlink() and p.resolve() == path]
            if not links:
                raise ValueError('Blob not referenced by evaluated pinned revision')
            record['source_files'] = [p.name for p in links]
            records.append(record)
    return {'schema': 1, 'files': records, 'preserved_result_sha256': preserved,
            'bytes': sum(x['bytes'] for x in records),
            'allocated_bytes': sum(x['allocated_bytes'] for x in records)}


def assert_unused(paths):
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('CUDA work active; inspect before cleanup')
    ancestors = set()
    parent = os.getppid()
    while parent > 1 and parent not in ancestors:
        ancestors.add(parent)
        try:
            fields = (Path('/proc')/str(parent)/'stat').read_text().split()
            parent = int(fields[3])
        except (FileNotFoundError, IndexError, ValueError):
            break
    for proc in Path('/proc').iterdir():
        try:
            if not proc.name.isdigit() or proc.stat().st_uid != os.getuid() or int(proc.name) == os.getpid():
                continue
            status = (proc/'status').read_text()
            if '\nState:\tZ' in status:
                continue
            command = (proc/'cmdline').read_bytes().split(b'\0')
            if any(b'run_efficiency_evaluation' in a or b'run_continuation' in a or
                   a in (b'pip', b'pip3') or a.endswith(b'/pip') for a in command):
                raise RuntimeError(f'Relevant job active: pid {proc.name}')
            for fd in (proc/'fd').iterdir():
                try:
                    target = os.readlink(fd)
                except FileNotFoundError:
                    continue
                if target in paths:
                    raise RuntimeError(f'Cache open by pid {proc.name}')
            maps = (proc/'maps').read_text()
            if any(path in maps for path in paths):
                raise RuntimeError(f'Cache mapped by pid {proc.name}')
        except FileNotFoundError:
            continue
        except PermissionError:
            # hidepid can protect per-user session infrastructure even from the
            # same account. Admit only a fixed set after its cmdline was already
            # checked above; unknown inaccessible processes remain fail-closed.
            try:
                after = (proc/'status').read_text()
                command = (proc/'cmdline').read_bytes().split(b'\0')
            except FileNotFoundError:
                continue
            if '\nState:\tZ' in after:
                continue
            executable = Path(os.fsdecode(command[0])).name if command and command[0] else ''
            if int(proc.name) in ancestors or executable in {'systemd', '(sd-pam)', 'pipewire', 'wireplumber',
                              'pipewire-pulse', 'dbus-daemon',
                              'xdg-document-portal', 'xdg-permission-store'}:
                continue
            raise RuntimeError(f'Cannot inspect live process {proc.name}: {after.splitlines()[:4]}') from None


def save_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path)
    p.add_argument('--apply', type=Path)
    p.add_argument('--expected-sha256')
    p.add_argument('--receipt', type=Path)
    args = p.parse_args()
    if bool(args.audit) == bool(args.apply):
        p.error('Choose audit or apply')
    with contextlib.ExitStack() as stack:
        for path in [ROOT/'continuation/20260911-v1/gpu.lock',
                     ROOT/'evaluations/efficiency-20260912-v1/queue.lock',
                     ROOT/'evaluations/comparison-20260911/evaluation.lock']:
            lock = stack.enter_context(path.open('r+'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = inventory()
        assert_unused({r['path'] for r in current['files']})
        if args.audit:
            save_new(args.audit, current)
            print(json.dumps({'audit': str(args.audit), 'sha256': digest(args.audit),
                              'files': len(current['files']), 'bytes': current['bytes']}))
            return
        if not args.receipt or args.receipt.exists() or digest(args.apply) != args.expected_sha256:
            raise ValueError('Missing/new receipt or exact audit hash required')
        expected = json.loads(args.apply.read_text())
        if current != expected:
            raise ValueError('Inventory changed; audit again before deleting')
        before = shutil.disk_usage(ROOT).free
        result = {'audit_sha256': args.expected_sha256, 'started_unix': time.time(),
                  'deleted': [], 'free_bytes_before': before,
                  'recovery': 'Redownload pip packages or pinned official Hub files; no local undo copy.'}
        try:
            for r in current['files']:
                path = Path(r['path'])
                s = safe_file(path)
                if (s.st_ino, s.st_size, s.st_mtime_ns) != (r['inode'], r['bytes'], r['mtime_ns']):
                    raise ValueError('File changed after audit')
                path.unlink()
                if path.exists() or path.is_symlink():
                    raise RuntimeError('Deletion not verified')
                result['deleted'].append(r['path'])
            for name, expected_hash in current['preserved_result_sha256'].items():
                if digest(Path(name)) != expected_hash:
                    raise ValueError('Preserved evaluation changed')
            result.update(status='verified_complete', freed_allocated_bytes=current['allocated_bytes'])
        finally:
            result.update(free_bytes_after=shutil.disk_usage(ROOT).free, completed_unix=time.time())
            save_new(args.receipt, result)
        print(json.dumps({k: v for k, v in result.items() if k != 'deleted'}))


if __name__ == '__main__':
    main()
