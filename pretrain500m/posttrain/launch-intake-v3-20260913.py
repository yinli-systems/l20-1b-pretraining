"""Deploy the frozen, bounded two-source extension once; preserve launch evidence."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
ARCHIVE = PROJECT / 'posttrain/intake-v3-20260913.tar.gz'
DIGEST = 'eebdb4296180d57040be2d8ff6d31584e8d6ceba8220b2f2f91181697cf67ec8'
REMOTE = r'''
import datetime, hashlib, io, json, os, shutil, subprocess, sys, tarfile
from pathlib import Path
r = Path('/ssd/scxi253/pretrain500m-20260912-v1')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert os.getuid() == 1256 and r.stat().st_uid == 1256 and not r.is_symlink()
assert (r/'OWNER.txt').read_text() == 'pretrain500m 2026-09-12 isolated run; owner task 01a09290-b43f-7431-be8a-412ea5d37954\n'
for d in ['source','vendor','data','logs','receipts']:
    p=r/d
    assert p.is_dir() and not p.is_symlink() and p.stat().st_uid == 1256
source=r/'source/intake-v3'; archive=r/'vendor/intake-v3-20260913.tar.gz'
output=r/'data/diverse-intake-v3'; log=r/'logs/intake-v3.log'
receipt=r/'receipts/intake-v3-launch.json'; tests=r/'receipts/intake-v3-tests.xml'
assert all(not p.exists() and not p.is_symlink() for p in [source,archive,output,log,receipt,tests])
assert shutil.disk_usage(r).free > 18*1024**3 + 512*1024**2
body=sys.stdin.buffer.read(2*1024**2)
digest='eebdb4296180d57040be2d8ff6d31584e8d6ceba8220b2f2f91181697cf67ec8'
assert len(body)==10864 and hashlib.sha256(body).hexdigest()==digest
expected={'README.md','SHA256SUMS','http_ranges.py','intake.py','prior-group-manifest.json','segments.json','test_intake_v3.py'}
with tarfile.open(fileobj=io.BytesIO(body),mode='r:gz') as tf:
    members=tf.getmembers()
    assert len(members)==len(expected) and {m.name for m in members}==expected
    assert all(m.isfile() and m.size<1024**2 and '/' not in m.name for m in members)
    files={m.name:tf.extractfile(m).read() for m in members}
checks=dict((line.split()[1],line.split()[0]) for line in files['SHA256SUMS'].decode().splitlines())
assert set(checks)==expected-{'SHA256SUMS'}
assert all(hashlib.sha256(files[n]).hexdigest()==h for n,h in checks.items())
plan=json.loads(files['segments.json'])
assert sha(r/'data/diverse-audit-v2-combined/report.json')==plan['combined_audit_sha256']
assert {s['id'] for s in plan['segments']}=={'pdf_en','dclm'}
for segment in plan['segments']:
    p=r/'data/diverse-intake-v2'/(segment['id']+'.receipt.json')
    assert sha(p)==segment['prior_receipt_sha256']
    old=json.loads(p.read_text())
    assert old['status']=='RAW_SEGMENT_READY'
    assert all(old[k]==segment[k] for k in ['id','repo_id','revision','path'])
    assert {g['row_group'] for g in old['groups']}==set(segment['exclude_row_groups'])
    assert set(segment['all_excluded_row_groups'])==set(segment['exclude_row_groups'])|set(old.get('exclude_row_groups',[]))|set(old.get('all_excluded_row_groups',[]))
with archive.open('xb') as f:f.write(body)
archive.chmod(0o444); source.mkdir()
for n,b in files.items():
    p=source/n
    with p.open('xb') as f:f.write(b)
    p.chmod(0o444)
assert all(sha(source/n)==h for n,h in checks.items())
runtime='/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python'
env=dict(os.environ,PYTHONPATH=str(r/'overlay'),PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
test=subprocess.run([runtime,'-m','pytest','-q','-p','no:cacheprovider',str(source/'test_intake_v3.py'),'--junitxml='+str(tests)],cwd=source,env=env,capture_output=True,text=True,timeout=60)
assert test.returncode==0, test.stdout[-6000:]+test.stderr[-2000:]
import xml.etree.ElementTree as ET
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(s.attrib.get('tests',0)) for s in suites)==7
assert all(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0))==0 for s in suites)
assert not output.exists() and not log.exists() and shutil.disk_usage(r).free>18*1024**3+512*1024**2
command=[runtime,str(source/'intake.py')]
with log.open('xb') as handle:
    worker=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='LAUNCHED_RAW_INTAKE_NOT_ADMITTED',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=worker.pid,command=command,source=str(source),output=str(output),log=str(log),archive_sha256=digest,tests_passed=7,test_receipt=str(tests),free_bytes=shutil.disk_usage(r).free,planned_sources=2,source_workers=2,max_network_bytes_per_source=128*1024**2,max_uncompressed_output_bytes_per_source=192*1024**2,min_free_bytes=18*1024**3,training_admitted=False,formal_training_started=False)
with receipt.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
print(json.dumps(result))
'''

if __name__ == '__main__':
    body = ARCHIVE.read_bytes()
    assert hashlib.sha256(body).hexdigest() == DIGEST
    bind = subprocess.check_output(['ipconfig', 'getifaddr', 'en0'], text=True).strip()
    result = subprocess.run(['ssh', '-b', bind, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
        '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2',
        'scxi253@BSCC-N56R5@ssh.cn-zhongwei-1.paracloud.com', 'python3 -c '+shlex.quote(REMOTE)],
        input=body, capture_output=True, timeout=100)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-6000:])
    receipt = json.loads(result.stdout)
    path = PROJECT/'reports/intake-v3-launch-20260913.json'
    with path.open('x') as handle:
        json.dump(receipt, handle, indent=2); handle.write('\n')
    print(json.dumps(receipt, indent=2))
