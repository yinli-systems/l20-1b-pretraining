from pathlib import Path
import datetime,hashlib,io,json,os,shutil,subprocess,sys,tarfile
r=Path('/ssd/scxi253/pretrain500m-20260912-v1');source=r/'source/pack-selected-v1';output=r/'data/selected-packs-v1';log=r/'logs/pack-selected-v1.log';receipt=r/'receipts/pack-selected-v1-launch.json';archive=r/'vendor/pack-selected-v1-20260914.tar.gz'
assert os.getuid()==1256 and r.stat().st_uid==1256 and not r.is_symlink()
assert (r/'OWNER.txt').read_text()=='pretrain500m 2026-09-12 isolated run; owner task 01a09290-b43f-7431-be8a-412ea5d37954\n'
assert all(not q.exists() and not q.is_symlink() for q in [source,output,log,receipt,archive])
assert shutil.disk_usage(r).free>19*1024**3
b=sys.stdin.buffer.read(1024**2);assert hashlib.sha256(b).hexdigest()=='61b556b801592323eb321f891d65ac7c8917f44cf9e91bd80be5895f7ea3f3a9'
with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:
 members=t.getmembers();assert {m.name for m in members}=={'SHA256SUMS','pack.py','test_pack.py'}
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
with (r/'receipts/pack-selected-v1-tests.txt').open('x') as f:f.write(test.stdout+test.stderr)
assert test.returncode==0 and 'Ran 3 tests' in test.stderr and 'skipped' not in test.stderr,test.stderr
cmd=[runtime,str(source/'pack.py'),'--family-report',str(r/'data/family-split-v1/report.json'),'--expected-family-sha256','aa6f8d733135f410f318e9bf4944033670103eaf118a1cdb0c43663609680a4c','--tokenizer',str(r/'formal/hf-v5/tokenizer.json'),'--expected-tokenizer-sha256','30d71356c5ba154006df5bbb4a0583fc434525ceeb2f27a7d8a237ce5db26dc6','--output',str(output),'--workers','4']
with log.open('xb') as f:worker=subprocess.Popen(cmd,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
x=dict(status='LAUNCHED_SELECTED_SOURCE_PACKING_NOT_ADMITTED',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=worker.pid,command=cmd,source=str(source),output=str(output),log=str(log),workers=4,archive_sha256='61b556b801592323eb321f891d65ac7c8917f44cf9e91bd80be5895f7ea3f3a9',local_tests_passed=3,remote_tests_passed=3,training_admitted=False,training_launched=False,free_bytes=shutil.disk_usage(r).free)
with receipt.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
print(json.dumps(x))
