"""Launch the bound F2 continuation CPU pack stage on the owned ParaCloud root."""

from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess


ROOT = Path('/ssd/scxi253/pretrain500m-20260912-v1')
SOURCE = ROOT / 'source/pack-selected-incremental-v1'
FAMILY = ROOT / 'data/family-split-f2-continuation-v1/report.json'
PRIOR = ROOT / 'data/selected-packs-expansion-v1/report.json'
TOKENIZER = ROOT / 'formal/hf-v5/tokenizer.json'
OUTPUT = ROOT / 'data/selected-packs-f2-continuation-v1'
LOG = ROOT / 'logs/f2-continuation-v1/pack.log'
RECEIPT = ROOT / 'receipts/pack-f2-continuation-v1-launch.json'
TESTS = ROOT / 'receipts/pack-f2-continuation-v1-tests.txt'
RUNTIME = Path('/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


assert os.getuid() == 1256 and ROOT.stat().st_uid == 1256 and not ROOT.is_symlink()
assert (ROOT / 'OWNER.txt').read_text() == (
    'pretrain500m 2026-09-12 isolated run; owner task '
    '01a09290-b43f-7431-be8a-412ea5d37954\n'
)
assert all(not path.exists() and not path.is_symlink() for path in (OUTPUT, LOG, RECEIPT, TESTS))
assert RUNTIME.is_file() and SOURCE.is_dir() and not SOURCE.is_symlink()
assert sha(SOURCE / 'pack.py') == '67a9bf7c3240576c9fef5a05302c5891c8e5bebca8c89a4ec04afa5752850f19'
assert sha(FAMILY) == 'd02fe6a46088f19b85aa85061eabb878c9252c76751ef4ccf75006bb002ddda0'
assert sha(PRIOR) == '9c03634cf2ac12adf0198d63f2024f2236afb358fc904a350baf8a3f7e68b700'
assert sha(TOKENIZER) == '30d71356c5ba154006df5bbb4a0583fc434525ceeb2f27a7d8a237ce5db26dc6'

family = json.loads(FAMILY.read_text())
prior = json.loads(PRIOR.read_text())
assert family['status'] == 'FAMILY_CLOSURE_AND_RESERVED_ASSIGNMENTS_COMPLETE_NOT_ADMITTED'
assert family['training_admitted'] is False and family['training_launched'] is False
assert prior['status'] == 'SELECTED_SOURCE_PACKS_COMPLETE_NOT_ADMITTED'
assert prior['tokenizer_sha256'] == sha(TOKENIZER)
assert set(item['source_id'] for item in prior['sources']) == set(family['retained_sources']['train'])
for path, expected in family['output_files'].items():
    assert sha(Path(path)) == expected
total_bytes = sum(
    stats['encoded_tokens'] * 2
    for sources in family['retained_sources'].values()
    for stats in sources.values()
)
free_bytes = shutil.disk_usage(ROOT).free
assert free_bytes > 18 * 1024**3 + total_bytes + 128 * 1024**2

env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
test = subprocess.run(
    [str(RUNTIME), '-m', 'unittest', 'discover', '-s', str(SOURCE), '-p', 'test_*.py', '-v'],
    env=env, capture_output=True, text=True, timeout=60,
)
with TESTS.open('x') as handle:
    handle.write(test.stdout + test.stderr)
assert test.returncode == 0 and 'Ran 4 tests' in test.stderr and 'skipped' not in test.stderr

command = [str(RUNTIME), str(SOURCE / 'pack.py'), '--family-report', str(FAMILY),
           '--expected-family-sha256', sha(FAMILY), '--tokenizer', str(TOKENIZER),
           '--expected-tokenizer-sha256', sha(TOKENIZER), '--output', str(OUTPUT),
           '--reuse-report', str(PRIOR), '--expected-reuse-report-sha256', sha(PRIOR),
           '--workers', '4']
with LOG.open('xb') as handle:
    worker = subprocess.Popen(command, cwd=SOURCE, env=env, stdin=subprocess.DEVNULL,
                              stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
receipt = dict(
    status='LAUNCHED_F2_CONTINUATION_SELECTED_PACK_NOT_ADMITTED',
    checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    pid=worker.pid, command=command, source=str(SOURCE), output=str(OUTPUT), log=str(LOG),
    family_report_sha256=sha(FAMILY), prior_pack_report_sha256=sha(PRIOR),
    tokenizer_sha256=sha(TOKENIZER), packer_sha256=sha(SOURCE / 'pack.py'),
    remote_tests_passed=4, workers=4, free_bytes_before_launch=free_bytes,
    training_admitted=False, training_launched=False,
)
with RECEIPT.open('x') as handle:
    json.dump(receipt, handle, indent=2)
    handle.write('\n')
print(json.dumps(receipt))
