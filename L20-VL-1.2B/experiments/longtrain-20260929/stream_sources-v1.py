"""Pinned, bounded, single-pass source stream for one existing-L20 long run.
Each training update owns an immutable metadata catalog and source cursor.
Whole-source hashes, earlier image exclusions and document splits are mandatory.
"""
from __future__ import annotations
import collections,copy,gc,gzip,hashlib,io,json,math,os,re,shutil,sys,time
from decimal import Decimal
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import requests
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
WORK=PROJECT/'vision-scale50-20260928T1954Z'
OLD=PROJECT/'scale-next-20260927T2058Z'
CONFIG=json.loads((ROOT/'config.json').read_text())
PLAN_SHA=hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest()
sys.path.insert(0,str(OLD))
from stream_data import Near,fingerprint
from fast_exclusions import build as build_old_exclusions
sys.path.insert(0,str(WORK))
from input_runtime import ImageLedger
DATA=ROOT/'data'
NUMERIC=re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?')
IDENTIFIER=re.compile(r'\b(social security|passport number|credit card|bank account|phone number|telephone number|e-?mail address)\b',re.I)
SPAM=re.compile(r'(?:free (?:pdf|ebook) download|download (?:the )?(?:book|ebook|pdf) for free|less latency time to download|digital library saves the book|unlimited access to our library)',re.I)
EMPTY_Q=re.compile(r'Q\s*\d+\s*[:.?!]?|question\s*\d+\s*[:.?!]?',re.I)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.writing')
    with tmp.open('w') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def emit(**kw):print(json.dumps(kw,ensure_ascii=False),flush=True)
def available_ram():return next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
def normalized_training_answer(a):
    candidate=a[:-1] if a.endswith('.') else a
    if a.endswith('.') and NUMERIC.fullmatch(candidate) and Decimal(candidate.rstrip('%')).is_finite():return candidate
    return a

def choices(domain,turns,image_sha,tok,counts):
    answers=collections.defaultdict(set);out=[]
    if domain=='docmatix' and any(SPAM.search(str(t.get('user',''))+' '+str(t.get('assistant',''))) for t in turns):
        counts['document_download_spam']+=1;return []
    for i,t in enumerate(turns):
        q=t.get('user','');a=t.get('assistant','')
        if not isinstance(q,str) or not isinstance(a,str):counts['nontext_QA']+=1;continue
        q=q.strip();a=a.strip()
        if not q or not a or EMPTY_Q.fullmatch(q):counts['empty_or_placeholder_QA']+=1;continue
        if len(q)>6000 or len(a)>8000:counts['oversized_text']+=1;continue
        if IDENTIFIER.search(q):counts['identifier_question']+=1;continue
        clean=normalized_training_answer(a)
        pi=tok.encode('User: '+q+'\nAssistant:',add_special_tokens=False)
        ai=tok.encode(' '+clean,add_special_tokens=False)
        if len(pi)+len(ai)+1>512 or len(ai)+1>128:counts['complete_target_overlength']+=1;continue
        key=' '.join(q.casefold().split());answers[key].add(clean.casefold())
        out.append((hashlib.sha256((image_sha+'|'+q).encode()).hexdigest(),i,q,clean,a,t.get('source'),key))
    return sorted(x for x in out if len(answers[x[-1]])==1)

def exclusions():
    DATA.mkdir(exist_ok=True)
    cache=DATA/'prior-identities.json.gz';receipt=DATA/'prior-identities-receipt.json'
    if cache.exists() and receipt.exists():
        report=json.loads(receipt.read_text())
        if report['config_sha256']!=PLAN_SHA or sha(cache)!=report['cache_sha256']:raise RuntimeError('Exclusion cache mismatch')
        if any(sha(p)!=h for p,h in report['sources'].items()):raise RuntimeError('Frozen historical exclusion source changed')
        with gzip.open(cache,'rt') as f:v=json.load(f)
        near=Near()
        for n in v['dhash']:near.add(int(n,16))
        emit(stage='EXCLUSION_CACHE_VERIFIED',raw_and_pixel_hashes=len(v['hashes']),document_keys=len(v['documents']))
        return set(v['hashes']),set(v['ids']),near,set(v['documents'])
    hashes,ids,near=build_old_exclusions(DATA)
    prior=json.loads((DATA/'prior-exclusion-receipt.json').read_text());sources=dict(prior['sources']);docs=set()
    for path in sorted((OLD/'memory-data/catalog').glob('*.json')):
        sources[str(path)]=sha(path);v=json.loads(path.read_text())
        for r in v['seen_records']:hashes.add(r['raw']);hashes.add(r['pixel']);near.add(int(r['dhash'],16))
    pilot=WORK/'optimization-20260929T0502Z/prepared-v2/rows.json'
    sources[str(pilot)]=sha(pilot)
    for r in json.loads(pilot.read_text()):
        hashes.update((r['image_sha256'],r['pixel224_sha256']));near.add(int(r['dhash64'],16));docs.add(r['document_key'])
    dh={v for values in near.buckets.values() for v in values}
    value={'hashes':sorted(hashes),'ids':sorted(ids),'documents':sorted(docs),'dhash':[f'{v:016x}' for v in sorted(dh)]}
    tmp=cache.with_suffix('.writing')
    with gzip.open(tmp,'wt') as f:json.dump(value,f,separators=(',',':'))
    os.replace(tmp,cache)
    write(receipt,{'config_sha256':PLAN_SHA,'sources':sources,'cache_sha256':sha(cache),
        'raw_and_pixel_hashes':len(hashes),'known_document_keys':len(docs),'all_previous_stream_and_pilot_splits_excluded':True,
        'semantic_document_overlap_exhaustively_excluded':False,'created_unix':time.time()})
    emit(stage='EXCLUSIONS_FROZEN',hashes=len(hashes),known_documents=len(docs))
    return hashes,ids,near,docs

def download(meta,cache_first,deadline):
    name=meta['domain']+'-'+meta['lfs']['sha256']+'.parquet'
    seed=DATA/'seed-source-cache'/name
    if seed.exists():
        if seed.stat().st_size!=meta['size'] or sha(seed)!=meta['lfs']['sha256']:raise RuntimeError('Cached source corrupted')
        return io.BytesIO(seed.read_bytes()),0.,False
    if available_ram()<meta['size']*2+2*2**30:raise MemoryError('Source buffering would exceed memory reserve')
    buf=io.BytesIO();start=time.monotonic()
    url=f"https://hf-mirror.com/datasets/{meta['repo']}/resolve/{meta['revision']}/{meta['rfilename']}"
    for attempt in range(3):
        if time.monotonic()>deadline:raise TimeoutError('Long-run wall budget before download')
        offset=buf.tell()
        try:
            with requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(15,60)) as response:
                response.raise_for_status()
                if offset and (response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):raise RuntimeError('Unexpected source resume range')
                if not offset and response.status_code==206 and response.headers.get('Content-Range')!=f"bytes 0-{meta['size']-1}/{meta['size']}":raise RuntimeError('Partial cached response is not the source')
                for block in response.iter_content(2<<20):
                    if time.monotonic()>deadline:raise TimeoutError('Long-run download wall budget')
                    buf.write(block)
                    if buf.tell()>meta['size']:raise RuntimeError('Oversized source response')
            break
        except (requests.Timeout,requests.ConnectionError) as e:
            emit(stage='SOURCE_NETWORK_RETRY',file=meta['rfilename'],attempt=attempt+1,error=str(e)[:150])
            if attempt==2:raise
            time.sleep(2)
    if buf.tell()!=meta['size'] or hashlib.sha256(buf.getbuffer()).hexdigest()!=meta['lfs']['sha256']:raise RuntimeError('Whole source size/SHA256 failed')
    if cache_first:
        seed.parent.mkdir(exist_ok=True)
        used=sum(p.stat().st_size for p in seed.parent.glob('*.parquet'))
        if used+meta['size']>2*2**30 or shutil.disk_usage(ROOT).free<meta['size']+CONFIG['minimum_disk_free_bytes']:raise RuntimeError('Seed cache storage cap')
        with seed.open('xb') as f:f.write(buf.getbuffer());f.flush();os.fsync(f.fileno())
    return buf,time.monotonic()-start,True

class Lane:
    def __init__(self,domain,state,owner):
        self.domain=domain;self.state=state;self.owner=owner;self.reader=None;self.buffer=None;self.iterator=None
    def close(self):
        if self.reader is not None:self.reader.close()
        self.reader=None;self.iterator=None;self.buffer=None;gc.collect()
    def load(self):
        meta=CONFIG['source_plan'][self.domain][self.state['file']]
        prior_bytes=self.owner.download_bytes
        cache=(DATA/'seed-source-cache'/(self.domain+'-'+meta['lfs']['sha256']+'.parquet'))
        if not cache.exists() and prior_bytes+meta['size']>CONFIG['maximum_source_download_bytes']:raise RuntimeError('Download byte budget reached')
        self.buffer,secs,fetched=download(meta,self.state['file']==0,self.owner.deadline)
        if fetched:self.owner.download_bytes+=meta['size']
        write(DATA/'download-budget.json',{'bytes':self.owner.download_bytes,'cap':CONFIG['maximum_source_download_bytes']})
        self.reader=pq.ParquetFile(pa.BufferReader(self.buffer.getbuffer()))
        start_row=self.state['row']
        def iterate():
            offset=0
            for batch in self.reader.iter_batches(batch_size=8):
                n=batch.num_rows
                if offset+n<=start_row:offset+=n;continue
                for i,row in enumerate(batch.to_pylist()):
                    if offset+i>=start_row:yield offset+i,row
                offset+=n
        self.iterator=iter(iterate())
        receipt={'meta':meta,'whole_SHA256_verified':True,'rows':self.reader.metadata.num_rows,'download_seconds':secs,'disk_cache_only_first_source':True}
        path=DATA/'source-receipts'/f"{self.domain}-{self.state['file']:04d}.json"
        if not path.exists():write(path,receipt)
        emit(stage='SOURCE_VERIFIED',domain=self.domain,file=meta['rfilename'],rows=self.reader.metadata.num_rows,seconds=secs)
    def next(self):
        while self.state['file']<len(CONFIG['source_plan'][self.domain]):
            if self.owner.stop.is_set() or time.monotonic()>self.owner.deadline:raise TimeoutError('Source stream stop/budget')
            if self.iterator is None:self.load()
            try:pos,row=next(self.iterator)
            except StopIteration:
                self.close();self.state['file']+=1;self.state['row']=0;continue
            self.state['row']=pos+1
            meta=CONFIG['source_plan'][self.domain][self.state['file']]
            images=row.get('images',[])
            if len(images)!=1:self.owner.counts[self.domain+'/multi_image_excluded']+=1;continue
            raw=images[0].get('bytes')
            if not isinstance(raw,bytes) or not raw:self.owner.counts['empty_image']+=1;continue
            h=hashlib.sha256(raw).hexdigest()
            if h in self.owner.hashes:self.owner.counts['historical_or_duplicate_raw']+=1;continue
            text=row.get('texts',[])
            candidates=choices(self.domain,text,h,self.owner.tok,self.owner.counts)
            if not candidates:continue
            _,turn,q,a,original,docid,_=candidates[0]
            source_docs={str(t.get('source')) for t in text if t.get('source')}
            if self.domain=='docmatix' and len(source_docs)>1:self.owner.counts['conflicting_document_identity']+=1;continue
            doc=hashlib.sha256(('docmatix|'+str(docid)).encode()).hexdigest() if self.domain=='docmatix' and docid else h
            if doc in self.owner.docs:self.owner.counts['historical_or_duplicate_document']+=1;continue
            try:pixel,dh,size=fingerprint(raw)
            except (OSError,ValueError) as e:self.owner.counts['invalid_image']+=1;continue
            if min(size)<32 or size[0]*size[1]>40_000_000:self.owner.counts['image_size_rejected']+=1;continue
            if pixel in self.owner.hashes or self.owner.near.contains(dh):self.owner.counts['historical_or_stream_pixel_near']+=1;continue
            split=ImageLedger.split(doc)
            sid=hashlib.sha256((meta['repo']+'|'+meta['revision']+'|'+meta['rfilename']+'|'+str(pos)+'|'+str(turn)+'|'+h).encode()).hexdigest()
            self.owner.hashes.update((h,pixel));self.owner.docs.add(doc);self.owner.near.add(dh)
            seen={'raw':h,'pixel':pixel,'doc':doc,'dhash':f'{dh:016x}','split':split}
            self.owner.seen_this.append(seen)
            response=a if split=='train' else original
            value={'selection_sha256':sid,'image_sha256':h,'pixel224_sha256':pixel,'dhash64':f'{dh:016x}',
                'document_key':doc,'image_cluster':doc,'domain':self.domain,'source':self.domain,'split':split,
                'prompt':q,'response':response,'references':[response],'original_response':original,'source_repo':meta['repo'],
                'source_revision':meta['revision'],'source_shard':meta['rfilename'],'source_shard_sha256':meta['lfs']['sha256'],
                'source_row':pos,'turn_index':turn,'original_size':size,'complete_answer_not_truncated':True,'_bytes':raw}
            if split!='train':
                key=self.domain+'/'+split;n=self.owner.state['held_counts'].get(key,0)
                if n<64:
                    path=DATA/'holdout'/(h+'.image');path.parent.mkdir(exist_ok=True)
                    if path.exists():
                        if sha(path)!=h:raise RuntimeError('Reserved image changed')
                    else:
                        if shutil.disk_usage(ROOT).free<len(raw)+CONFIG['minimum_disk_free_bytes']:raise RuntimeError('Holdout storage reserve')
                        with path.open('xb') as f:f.write(raw)
                    value.pop('_bytes');value['image_path']=str(path);self.owner.held_this.append(value)
                    self.owner.state['held_counts'][key]=n+1
                self.owner.counts['reserved_not_trained']+=1;continue
            return value
        return None

class SourceStream:
    def __init__(self,tok,stop,deadline,cursor=-1):
        self.tok=tok;self.stop=stop;self.deadline=deadline;self.counts=collections.Counter()
        self.hashes,self.ids,self.near,self.docs=exclusions()
        for folder in ('catalog','source-receipts','holdout'):(DATA/folder).mkdir(exist_ok=True)
        self.state={'lanes':{'plotqa':{'file':0,'row':0},'docmatix':{'file':0,'row':0}},'next_catalog':0,'train':0,'held_counts':{}}
        for i in range(cursor+1):
            path=DATA/'catalog'/f'{i:06d}.json';record=json.loads(path.read_text())
            if record['config_sha256']!=PLAN_SHA or record['catalog']!=i:raise RuntimeError('Restored catalog mismatch')
            for v in record['seen_records']:
                self.hashes.update((v['raw'],v['pixel']));self.docs.add(v['doc']);self.near.add(int(v['dhash'],16))
            self.state=record['state_after']
        self.download_bytes=json.loads((DATA/'download-budget.json').read_text())['bytes'] if (DATA/'download-budget.json').exists() else 0
        self.lanes={d:Lane(d,self.state['lanes'][d],self) for d in ('plotqa','docmatix')}
        self.seen_this=[];self.held_this=[]
    def __iter__(self):return self
    def __next__(self):
        if self.stop.is_set() or self.state['train']>=CONFIG['max_new_unique_images']:raise StopIteration
        self.seen_this=[];self.held_this=[];rows=[];cid=self.state['next_catalog']
        for domain in ('plotqa','docmatix'):
            for _ in range(CONFIG['new_per_step'][domain]):
                row=self.lanes[domain].next()
                if row is None:
                    write(DATA/'complete.json',{'reason':'source_exhausted','domain':domain,'complete_prepared_train_images':self.state['train'],
                        'unconsumed_partial_new_images':len(rows),'caps_are_not_completed_counts':True,'remaining_source_fallback_or_repetition':False})
                    raise StopIteration
                rows.append(row)
        if len(rows)!=32 or len({r['image_sha256'] for r in rows})!=32:raise RuntimeError('Training-image plan mismatch')
        self.state['train']+=32;self.state['next_catalog']+=1
        payload={'catalog':cid,'config_sha256':PLAN_SHA,'train_rows':[{k:v for k,v in r.items() if k!='_bytes'} for r in rows],
            'heldout_rows':self.held_this,'seen_records':self.seen_this,'state_after':copy.deepcopy(self.state)}
        path=DATA/'catalog'/f'{cid:06d}.json'
        if path.exists():
            if json.loads(path.read_text())!=payload:raise RuntimeError('Regenerated source catalog changed')
        else:write(path,payload)
        write(DATA/'live.json',{'prepared_unique_train_images':self.state['train'],'next_catalog':self.state['next_catalog'],
            'rejections_this_process':dict(self.counts),'held_counts':self.state['held_counts'],'unix':time.time()})
        return {'catalog':cid,'catalog_sha256':sha(path),'rows':rows,'source_state_after':copy.deepcopy(self.state)}
    def close(self):
        for lane in self.lanes.values():lane.close()
