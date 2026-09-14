
from pathlib import Path
import datetime,hashlib,io,json,os,shutil,subprocess,sys,tarfile
r=Path('/ssd/scxi253/pretrain500m-20260912-v1');source=r/'source/quality-family-v1';archive=r/'vendor/quality-family-v1-20260913.tar.gz';output=r/'data/quality-family-v1';overlay=r/'overlay-quality-family-v1';log=r/'logs/quality-family-v1.log';receipt=r/'receipts/quality-family-v1-launch.json'
assert os.getuid()==1256 and r.stat().st_uid==1256 and not r.is_symlink()
assert (r/'OWNER.txt').read_text()=='pretrain500m 2026-09-12 isolated run; owner task 01a09290-b43f-7431-be8a-412ea5d37954\n'
assert all(not q.exists() and not q.is_symlink() for q in [source,archive,output,overlay,log,receipt])
assert shutil.disk_usage(r).free>18*1024**3+768*1024**2
b=sys.stdin.buffer.read(140*1024**2);archive_sha=hashlib.sha256(b).hexdigest();assert len(b)==127817767 and archive_sha=='1442fc0c86d2d74f2b2d7f73f92bbe1f5b647fc21d92199f63f263f125594fd6'
expected=['README.md', 'SHA256SUMS', 'dependency-receipts.json', 'features.py', 'policy.json', 'scan.py', 'test_features.py', 'test_scan.py', 'vendor/fasttext_wheel-0.9.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl', 'vendor/lid.176.bin', 'vendor/public_suffix_list.dat']
with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:
 members=t.getmembers();assert len(members)==len(expected) and {m.name for m in members}==set(expected)
 assert all(m.isfile() and 0<m.size<140*1024**2 and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
 files={m.name:t.extractfile(m).read() for m in members}
checks={l.split()[1]:l.split()[0] for l in files['SHA256SUMS'].decode().splitlines()};assert set(checks)==set(expected)-{'SHA256SUMS'}
assert all(hashlib.sha256(files[n]).hexdigest()==h for n,h in checks.items())
raw=r/'data/diverse-audit-v2-three-tranche/report.json';exclusions=r/'receipts/exclusion-union-v1.json';sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
assert sha(raw)=='71240c265d14bc4ddb234503ca2a50724de4cbb9874637e8c9609a8e961319c8'
assert sha(exclusions)=='c622932c94ae2c56dfd0e14691e72975214381288c03979c4be759211acb00df'
with archive.open('xb') as f:f.write(b)
archive.chmod(0o444);source.mkdir();(source/'vendor').mkdir()
for n,b in files.items():
 with (source/n).open('xb') as f:f.write(b)
 (source/n).chmod(0o444)
runtime='/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python';wheel=next((source/'vendor').glob('*.whl'))
env=dict(os.environ,PYTHONPATH=str(overlay),PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
install=subprocess.run([runtime,'-m','pip','install','--no-deps','--no-index','--no-cache-dir','--target',str(overlay),str(wheel)],env=env,capture_output=True,text=True,timeout=60)
assert install.returncode==0,install.stderr[-4000:]
with (r/'receipts/quality-family-v1-overlay.txt').open('x') as f:f.write(install.stdout+install.stderr)
test=subprocess.run([runtime,'-m','unittest','discover','-s',str(source),'-p','test_*.py','-v'],cwd=source,env=env,capture_output=True,text=True,timeout=60)
with (r/'receipts/quality-family-v1-tests.txt').open('x') as f:f.write(test.stdout+test.stderr)
assert test.returncode==0 and 'Ran 12 tests' in test.stderr and 'skipped' not in test.stderr,test.stderr[-5000:]
cmd=[runtime,str(source/'scan.py'),'--raw-report',str(raw),'--expected-raw-report-sha256',sha(raw),'--exclusions',str(exclusions),'--expected-exclusions-sha256',sha(exclusions),'--audit-source',str(r/'source/raw-audit-v2'),'--output',str(output),'--workers','4']
assert not output.exists() and not log.exists() and shutil.disk_usage(r).free>18*1024**3+256*1024**2
with log.open('xb') as f:worker=subprocess.Popen(cmd,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
x={'status':'LAUNCHED_QUALITY_FAMILY_FEATURES_NOT_ADMITTED','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':worker.pid,'command':cmd,'source':str(source),'output':str(output),'log':str(log),'overlay':str(overlay),'workers':4,'archive_sha256':archive_sha,'local_tests_passed':11,'local_native_test_skipped':1,'remote_tests_passed':12,'native_language_inference_passed':True,'training_admitted':False,'free_bytes':shutil.disk_usage(r).free}
with receipt.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
print(json.dumps(x))
