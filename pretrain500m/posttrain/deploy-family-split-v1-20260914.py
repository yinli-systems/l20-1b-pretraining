from pathlib import Path
import datetime,hashlib,io,json,os,shutil,subprocess,sys,tarfile
r=Path('/ssd/scxi253/pretrain500m-20260912-v1');source=r/'source/family-split-v1';output=r/'data/family-split-v1';log=r/'logs/family-split-v1.log';receipt=r/'receipts/family-split-v1-launch.json';archive=r/'vendor/family-split-v1-20260914.tar.gz'
assert os.getuid()==1256 and r.stat().st_uid==1256 and not r.is_symlink()
assert (r/'OWNER.txt').read_text()=='pretrain500m 2026-09-12 isolated run; owner task 01a09290-b43f-7431-be8a-412ea5d37954\n'
assert all(not q.exists() and not q.is_symlink() for q in [source,output,log,receipt,archive])
assert shutil.disk_usage(r).free>18*1024**3+256*1024**2
b=sys.stdin.buffer.read(1024**2);assert hashlib.sha256(b).hexdigest()=='45575131223dc790c6682939176720ddcc5f5bc52d9ac7d4da4cb0255cee08e2'
with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:
 members=t.getmembers();assert {m.name for m in members}=={'SHA256SUMS','README.md','families.py','scan.py','test_families.py','test_scan.py'}
 assert all(m.isfile() and 0<m.size<100000 for m in members)
 files={m.name:t.extractfile(m).read() for m in members}
checks={l.split()[1]:l.split()[0] for l in files['SHA256SUMS'].decode().splitlines()}
assert set(checks)==set(files)-{'SHA256SUMS'} and all(hashlib.sha256(files[n]).hexdigest()==h for n,h in checks.items())
with archive.open('xb') as f:f.write(b)
source.mkdir()
for name,data in files.items():
 with (source/name).open('xb') as f:f.write(data)
 (source/name).chmod(0o444)
runtime='/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python'
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
test=subprocess.run([runtime,'-m','unittest','discover','-s',str(source),'-p','test_*.py','-v'],env=env,capture_output=True,text=True,timeout=60)
with (r/'receipts/family-split-v1-tests.txt').open('x') as f:f.write(test.stdout+test.stderr)
assert test.returncode==0 and 'Ran 11 tests' in test.stderr and 'skipped' not in test.stderr,test.stderr
cmd=[runtime,str(source/'scan.py'),'--features-report',str(r/'data/quality-family-v1/report.json'),'--expected-features-sha256','8e4d1446084cc9893f0d08250f1b79579dfd4630a0bb148ac84098d44ea055ba','--raw-report',str(r/'data/diverse-audit-v2-three-tranche/report.json'),'--exclusions',str(r/'receipts/exclusion-union-v1.json'),'--design',str(r/'source/fast-start-nosynthetic-v1/experiment-design.json'),'--expected-design-sha256','3d58055a1d2398b90c963c9ed4be7e99e1d1644281b95599c4ad5339d805d105','--output',str(output),'--workers','4']
with log.open('xb') as f:worker=subprocess.Popen(cmd,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
x=dict(status='LAUNCHED_FAMILY_CLOSURE_AND_RESERVED_SPLITS_NOT_ADMITTED',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=worker.pid,command=cmd,source=str(source),output=str(output),log=str(log),workers=4,archive_sha256='45575131223dc790c6682939176720ddcc5f5bc52d9ac7d4da4cb0255cee08e2',local_tests_passed=11,remote_tests_passed=11,training_admitted=False,training_launched=False,free_bytes=shutil.disk_usage(r).free)
with receipt.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
print(json.dumps(x))
