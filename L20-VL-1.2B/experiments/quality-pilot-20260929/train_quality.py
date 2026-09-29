"""User-requested bounded VLM quality experiment on existing local assets.
128 optimizer updates per arm; no downloads/deletion/base-model substitution.
Only explicitly TRAIN rows update parameters; original retention anchors remain.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import argparse, collections, fcntl, gc, hashlib, json, math, random, re, shutil, signal, sqlite3, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file, load_file
from peft import set_peft_model_state_dict
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
WORK = PROJECT/'vision-scale50-20260928T1954Z'
DATA = WORK/'optimization-20260929T0502Z/prepared-v2'
DATA_SHA = '55918147e585d17fde0f20b922d33d3d8de6013a041559946da39f1ad96318d2'
PARENT_SHA = '3751ac488fa45c20d29b2cf231ee9593ec59d7ee52cf24e15724e3e75489efa7'
SEED = 2026092911
STEPS = 128
sys.path.insert(0, str(WORK))
import spatial_route as spatial
from input_runtime import OrderedPrefetch, fd_count
core = spatial.core
sys.path.insert(0, str(PROJECT/'data-scale-20260927T1756Z'))
import train_scale as legacy
STOP = False

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4<<20), b''): h.update(b)
    return h.hexdigest()

def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.writing')
    with tmp.open('w') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def emit(**kw):print(json.dumps(kw,ensure_ascii=False,allow_nan=False),flush=True)
def request_stop(*_):
    global STOP
    STOP=True

def guard():
    if shutil.disk_usage(ROOT).free < 3*2**30:raise RuntimeError('Persistent reserve below 3GiB')
    mem=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    if mem < 1500*2**20:raise RuntimeError('Host memory reserve below 1.5GiB')

def weight_hash(lab):
    h=hashlib.sha256()
    for n,p in lab.named:h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class RecipeCollator:
    def __init__(self,proc,pad,prompt_only=False):
        self.proc=proc;self.pad=pad;self.prompt_only=prompt_only;self.old=spatial.Collator(proc,pad,prompt_only)
    def __call__(self,rows):
        if {r.get('_route','T') for r in rows}!={'A'}:return self.old(rows)
        ids,mask,labels=core.pack_text(rows,self.pad,self.prompt_only);core.check_context(ids.shape[1],448)
        images=[]
        for r in rows:
            with Image.open(r['image_path']) as im:im,_=spatial.canvas(im,448)
            images.append(im)
        pixels=self.proc(images=images,do_resize=False,return_tensors='pt')['pixel_values']
        return {'pixels':pixels,'ids':ids,'mask':mask,'labels':labels,'resolution':448,'route':'T',
                'task':rows[0].get('_task','evaluate'),'image_events':len(rows),
                'answer_tokens':sum(len(r['_ai']) for r in rows),'visual_tokens':len(rows)*784,
                'input_tokens':int(mask.sum())+len(rows)*786}

class QualityLab(spatial.ResolutionLab):
    def batch(self,rows,prompt_only=False):return core.cuda(RecipeCollator(self.proc,self.tok.pad_token_id,prompt_only)(rows))

def load_lab(arm):
    if sha(spatial.PARENT/'manifest.json')!=PARENT_SHA:raise ValueError('Healthy parent changed')
    m=spatial.verify_checkpoint(spatial.PARENT)
    lab=QualityLab();opt=lab.optimizer()
    core.previous.restore(lab,opt,spatial.PARENT,m['data_plan_sha256']);lab.set_gc(False)
    if arm=='H896':lab.install(opt)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    return lab,opt

def encode(lab,row,res=448,route='T',task='new'):
    r=core.encode(lab,row,res)
    if r['_n']>512 or len(r['_ai'])>128:raise ValueError('Complete answer exceeds declared bounds')
    return dict(r,_route=route,_resolution=res,_task=task)

def make_data(lab,arm):
    if sha(DATA/'rows.json')!=DATA_SHA:raise ValueError('Pilot rows changed')
    raw=json.loads((DATA/'rows.json').read_text());admission=json.loads((DATA/'admission.json').read_text())
    for name,h in admission['files'].items():
        if sha(DATA/name)!=h:raise ValueError('Data artifact hash: '+name)
    with sqlite3.connect('file:'+str(DATA/'identity.sqlite')+'?mode=ro',uri=True) as db:
        ledger={h.hex():(s,d.hex()) for h,s,d in db.execute('SELECT raw,split,document FROM images')}
    for r in raw:
        if ledger[r['image_sha256']]!=(r['split'],r['document_key']):raise ValueError('Ledger split mismatch')
        if r['split']=='train' and (r['original_pilot_split']!='train' or r['previous_ledger_split']!='train'):raise ValueError('Heldout became train')
        if sha(r['image_path'])!=r['image_sha256']:raise ValueError('Image changed')
    route={'C448':'T','A448':'A','H896':'H'}[arm]
    pools={}
    for d in ('plotqa','docmatix'):
        rr=[r for r in raw if r['domain']==d and r['split']=='train'];random.Random(SEED).shuffle(rr);pools[d]=rr
    selected=[]
    for step in range(STEPS):selected.append(pools['plotqa'][step*24:(step+1)*24]+pools['docmatix'][step*8:(step+1)*8])
    if any(len(x)!=32 for x in selected):raise ValueError('Insufficient distinct TRAIN images')
    if len({r['image_sha256'] for x in selected for r in x})!=STEPS*32:raise ValueError('Repeated new image in plan')
    oldroot=PROJECT/'data-scale-20260927T1756Z'
    frozen=json.loads((oldroot/'run/evaluation-freeze.json').read_text())
    general=json.loads((PROJECT/'general-vision-20260925T1758Z/admitted-rows.json').read_text())
    byid={r['selection_sha256']:r for r in general}
    oldqa=[encode(lab,byid[i],224,'G','qa') for i in frozen['old_qa_ids']]
    captions={r['selection_sha256']:r for r in lab.rows['development'] if r['source']=='open_images_localized_narratives'}
    oldcaption=[dict(captions[i],_resolution=224,_route='G',_task='retain') for i in frozen['old_caption_ids']]
    qapool=[r for r in general if r['split']=='train' and r['source']=='vqa_official_consensus'];random.Random(SEED+1).shuffle(qapool)
    cappool=[r for r in lab.rows['train'] if r['source']=='open_images_localized_narratives'];random.Random(SEED+2).shuffle(cappool)
    multi=json.loads((oldroot/'prepared/admitted-rows.json').read_text());domains={}
    for i,d in enumerate(('textvqa','docvqa','chartqa','ai2d')):
        rr=[dict(r) for r in multi if r['split']=='train' and r['domain']==d];random.Random(SEED+3+i).shuffle(rr)
        domains[d]=[]
        for r in rr:
            if d=='ai2d':
                m=re.fullmatch(r'(?:Answer:\s*)?([A-D])[.!]?',r['response'],re.I)
                if m:r['response']=m.group(1).upper();r['references']=[r['response']]
            e=encode(lab,r,448,'T','replay')
            if e['_n']<=256:domains[d].append(e)
            if len(domains[d])>=STEPS*2:break
        if len(domains[d])<STEPS*2:raise ValueError('Insufficient old domain replay')
    plans=[]
    for step,new in enumerate(selected):
        gs=[]
        for d in ('plotqa','docmatix'):
            group=[encode(lab,r,448,route,'new') for r in new if r['domain']==d]
            for start in range(0,len(group),4):gs.append(group[start:start+4])
        gs.append([encode(lab,qapool[(step*16+j)%len(qapool)],224,'G','qa') for j in range(16)])
        gs.append([domains[d][step*2+j] for d in domains for j in range(2)])
        gs.append([dict(cappool[(step*8+j)%len(cappool)],_resolution=224,_route='G',_task='retain') for j in range(8)])
        if sum(map(len,gs))!=64 or any(r['split']!='train' for g in gs for r in g):raise ValueError('Unsafe training plan')
        plans.append(gs)
    dev={}
    for d in ('plotqa','docmatix'):
        rr=sorted([r for r in raw if r['split']=='development' and r['domain']==d],key=lambda r:r['selection_sha256'])[:32]
        dev[d]=[encode(lab,r,448,route,'evaluate') for r in rr]
    ids=[[[r['selection_sha256'] for r in g] for g in step] for step in plans]
    data={'new_unique_images':4096,'new_domains':{'plotqa':3072,'docmatix':1024},'global_image_events':8192,
          'plans':ids,'development_ids':{d:[r['selection_sha256'] for r in rr] for d,rr in dev.items()},
          'old_qa_ids':frozen['old_qa_ids'],'old_caption_ids':frozen['old_caption_ids'],'no_confirmation_access':True}
    return plans,{'oldqa':oldqa,'oldcaption':oldcaption,'development':dev},data

def update(lab,opt,teacher,batches,replay,step):
    lab.train();lab.zero();losses={};answer_tokens=0;images=0;visual=0
    for cpu in batches:
        b=core.cuda(cpu);task=b['task'];n=b['image_events']
        if task in ('qa','retain'):
            ce,kl=core.preserved_task(lab,teacher,b);loss=(.30 if task=='qa' else .15)*ce+.5*kl
            losses[task+'_ce']=float(ce.detach());losses[task+'_kl']=float(kl.detach())
        else:
            ce=lab.losses(b).sum();loss=(.40/32 if task=='new' else .15/8)*ce
            losses[task]=losses.get(task,0.)+float(loss.detach())
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
        loss.backward();answer_tokens+=b['answer_tokens'];images+=n;visual+=b['visual_tokens']
    text_tokens=0
    if step%4==0:
        tx=core.previous.text_loss(lab,replay,(1600+step)%(len(replay)//2));(.1*tx).backward()
        losses['text_ce']=float(tx.detach());text_tokens=4096
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in lab.params):raise FloatingPointError('Missing/nonfinite trainable gradient')
    norm=torch.nn.utils.clip_grad_norm_(lab.params,1.)
    if not bool(torch.isfinite(norm)):raise FloatingPointError('Nonfinite gradient norm')
    scale=min(1.,(step+1)/16)*(.25+.75*.5*(1+math.cos(math.pi*max(0,step-16)/(STEPS-17))))
    for i,g in enumerate(opt.param_groups):g['lr']=(5e-5 if i==2 else 1e-5)*scale
    opt.step()
    return {'losses':losses,'weighted_loss_sum':sum(v for k,v in losses.items() if k in ('new','replay')),
            'gradient_norm':float(norm),'image_events':images,'new_image_events':32,'answer_tokens':answer_tokens,
            'visual_tokens':visual,'text_tokens':text_tokens,'learning_rates':[g['lr'] for g in opt.param_groups]}

def save_checkpoint(lab,opt,out,step,ph,bad,stats):
    guard();cp=out/'checkpoints'/f'step-{step:06d}';tmp=cp.with_name(cp.name+'.partial')
    if cp.exists() or tmp.exists():raise FileExistsError('Refuse checkpoint overwrite')
    tmp.mkdir(parents=True)
    save_file({n:t.detach().cpu().contiguous() for n,t in lab.bridge.state_dict().items()},tmp/'bridge.safetensors')
    lab.language.save_pretrained(tmp/'language_adapter',safe_serialization=True)
    torch.save({'step':step,'optimizer':opt.state_dict(),'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),
        'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'plan_sha256':ph,
        'consecutive_bad':bad,'stats':stats},tmp/'resume.pt')
    files={str(p.relative_to(tmp)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in tmp.rglob('*') if p.is_file()}
    write(tmp/'manifest.json',{'files':files,'step':step,'plan_sha256':ph,'parent_manifest_sha256':PARENT_SHA,
        'protocol_sha256':sha(out/'protocol.json'),'arm':out.name,'warmstart_not_base_replacement':True,
        'optimizer_RNG_and_next_plan_cursor_saved':True,'automatic_promotion':False})
    for p in tmp.rglob('*'):
        if p.is_file():
            with p.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,cp);fd=os.open(str(cp.parent),os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    spatial.verify_checkpoint(cp)
    write(out/'latest.json',{'step':step,'checkpoint':str(cp),'manifest_sha256':sha(cp/'manifest.json'),'unix':time.time()})
    emit(stage='CHECKPOINT_SAVED',arm=out.name,step=step,checkpoint=str(cp))
    return cp

def restore(lab,opt,cp,ph):
    m=spatial.verify_checkpoint(cp)
    if m['plan_sha256']!=ph:raise ValueError('Resume plan mismatch')
    lab.bridge.load_state_dict(load_file(cp/'bridge.safetensors'),strict=True)
    result=set_peft_model_state_dict(lab.language,load_file(cp/'language_adapter/adapter_model.safetensors'),adapter_name='default')
    if result.unexpected_keys:raise ValueError('Adapter key mismatch')
    state=torch.load(cp/'resume.pt',map_location='cpu',weights_only=False)
    opt.load_state_dict(state['optimizer']);random.setstate(state['rng_python']);np.random.set_state(state['rng_numpy'])
    torch.set_rng_state(state['rng_torch']);torch.cuda.set_rng_state_all(state['rng_cuda']);lab.zero()
    return state

def health(lab,sets):
    ret=legacy.retention(lab,sets)
    anchors={'web':2.387526661157608,'dclm':2.713881492614746,'math':2.39383202791214,'code':1.020752675831318}
    checks={'old_QA':ret['old_QA']>=.5125,'caption':ret['old_caption_nll']<=1.6197589449584484+.08,
            'text':max(ret['text_nll'][k]-v for k,v in anchors.items())<=.05}
    return {'scores':ret,'checks':checks,'absolute_old_QA_anchor':.5625,'absolute_old_QA_floor':.5125}

def development(lab,sets):
    scores={}
    for domain,rows in sets['development'].items():
        vals=[];pred=[]
        for i in range(0,len(rows),4):
            part=rows[i:i+4]
            with torch.no_grad():
                lab.language.eval();lab.bridge.eval();vals.extend(lab.losses(lab.batch(part)).detach().float().cpu().tolist())
            answers=lab.generate(part,max_new=64)
            for r,a in zip(part,answers):pred.append({'id':r['selection_sha256'],'prediction':a,'reference':r['response']})
        scores[domain]={'images':len(rows),'mean_answer_including_EOS_NLL':float(np.mean(vals)),
                        'normalized_exact_match':float(np.mean([core.normalize(p['prediction'])==core.normalize(p['reference']) for p in pred])),
                        'predictions':pred,'not_official_benchmark':True,'max_new_tokens':64}
    lab.train();return scores

def main(arm,resume=None):
    global STOP
    out=ROOT/arm
    if resume is None and out.exists():raise FileExistsError('Existing arm requires explicit resume')
    lock=(PROJECT/'general-vision-gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    guard();out.mkdir(exist_ok=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    write(out/'live.json',{'phase':'loading_and_verifying','pid':os.getpid(),'arm':arm,'unix':time.time()})
    lab,opt=load_lab(arm);plans,sets,data=make_data(lab,arm)
    ph=hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    teacher=core.Teacher(lab);replay=np.load(core.previous.REPLAY,mmap_mode='r')
    if sha(core.previous.REPLAY)!=core.previous.REPLAY_SHA:raise ValueError('Text replay changed')
    source_paths=[Path(__file__),WORK/'spatial_route.py',WORK/'input_runtime.py',Path(core.__file__),Path(core.previous.__file__),Path(legacy.__file__),PROJECT/'vlm-opt-20260925T1537Z/vlm_lab.py']
    source_hashes={str(p):sha(p) for p in source_paths}
    protocol={'arm':arm,'steps':STEPS,'global_image_batch':64,'new_images_per_step':32,
       'new_mixture':{'plotqa':24,'single_page_docmatix':8},'old_replay':{'natural_QA':16,'balanced_four_domains':8,'caption':8},
       'loss_weights':{'new_CE':.40,'QA_CE':.30,'domain_replay_CE':.15,'caption_CE':.15,'each_old_task_KL':.50,'text_every4steps':.10},
       'learning_rates':{'bridge':1e-5,'LoRA':1e-5,'H896_packing':5e-5},'warmup':16,'same_parent_optimizer_moments_inherited':True,
       'parent':str(spatial.PARENT),'parent_manifest_sha256':PARENT_SHA,'plan_sha256':ph,'pilot_rows_sha256':DATA_SHA,
       'full_LM_and_vision_frozen':True,'retention_original_anchor':.5625,'retention_floor':.5125,'stop_after_consecutive_failures':2,
       'wall_seconds_cap':2400,'checkpoint_steps':[1,16,32,64,96,128],'development_not_confirmation_for_selection':True,
       'not_8_4M_bulk_training':True,'automatic_promotion':False,'no_download_or_deletion':True,'source_sha256':source_hashes,
       'torch':torch.__version__,'trainable_parameters':sum(p.numel() for p in lab.params),'seed':SEED}
    totals=collections.Counter();bad=0;start_step=0;last_cp=None;discarded=0
    if resume is None:
        write(out/'protocol.json',protocol);write(out/'plan.json',data)
        base_hash=weight_hash(lab);write(out/'initial.json',{'weight_hash':base_hash,'parent_manifest_sha256':PARENT_SHA})
        base_health=health(lab,sets);write(out/'baseline-health.json',base_health)
        if not all(base_health['checks'].values()):raise RuntimeError('Parent failed original retention gate')
        write(out/'baseline-development.json',development(lab,sets))
    else:
        old=json.loads((out/'protocol.json').read_text())
        if old!=protocol:raise ValueError('Protocol changed; not a resume')
        state=restore(lab,opt,resume,ph);start_step=state['step'];bad=state['consecutive_bad'];totals.update(state['stats']['totals']);last_cp=resume
        base_hash=json.loads((out/'initial.json').read_text())['weight_hash']
    before=time.monotonic();reason='planned_endpoint';collator=RecipeCollator(lab.proc,lab.tok.pad_token_id)
    def collate_step(gs):return [collator(g) for g in gs]
    with OrderedPrefetch(plans[start_step:],collate_step,workers=2,depth=2,timeout=120) as prefetch:
        for step,bs in enumerate(prefetch,start_step):
            if STOP or (ROOT/'STOP').exists() or time.monotonic()-before>2400:reason='user_or_wall_stop';break
            guard();torch.cuda.synchronize();t=time.perf_counter();event=update(lab,opt,teacher,bs,replay,step);torch.cuda.synchronize()
            event.update(step=step+1,seconds=time.perf_counter()-t,arm=arm,fd_count=fd_count(),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            totals.update({k:event[k] for k in ('new_image_events','image_events','answer_tokens','visual_tokens','text_tokens')})
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(event,allow_nan=False)+'\n');f.flush()
            write(out/'live.json',{'phase':'training','arm':arm,'step':step+1,'target':STEPS,'pid':os.getpid(),
                'new_image_events':totals['new_image_events'],'global_image_events':totals['image_events'],
                'last_step_seconds':event['seconds'],'losses':event['losses'],'unix':time.time()})
            if step<4 or (step+1)%8==0:emit(stage='OPTIMIZER_UPDATE',**event)
            if step==0:
                last_cp=save_checkpoint(lab,opt,out,1,ph,bad,{'totals':dict(totals)})
                snapshot={n:p.detach().cpu().clone() for n,p in lab.named};restore(lab,opt,last_cp,ph)
                delta=max(float((p.detach().cpu()-snapshot[n]).abs().max()) for n,p in lab.named)
                changed=weight_hash(lab)!=base_hash
                write(out/'first-update-verification.json',{'optimizer_updates':1,'weights_changed_from_parent':changed,
                   'reload_parameter_max_abs':delta,'saved_optimizer_and_RNG_restored':True,'fresh_process_resume_tested':False,
                   'checkpoint_manifest_sha256':sha(last_cp/'manifest.json')})
                if not changed or delta!=0:raise RuntimeError('No real update or checkpoint reload mismatch')
                del snapshot
            if (step+1)%32==0:
                h=health(lab,sets);bad=0 if all(h['checks'].values()) else bad+1;h['consecutive_bad']=bad
                write(out/f'step-{step+1:06d}-health.json',h);write(out/f'step-{step+1:06d}-development.json',development(lab,sets))
                emit(stage='HEALTH',arm=arm,step=step+1,checks=h['checks'],old_QA=h['scores']['old_QA'])
            if step+1 in (16,32,64,96,128):last_cp=save_checkpoint(lab,opt,out,step+1,ph,bad,{'totals':dict(totals)})
            if bad>=2:reason='retention_guard';break
    final_step=totals['new_image_events']//32
    if final_step and (last_cp is None or json.loads((last_cp/'manifest.json').read_text())['step']!=final_step):
        last_cp=save_checkpoint(lab,opt,out,final_step,ph,bad,{'totals':dict(totals)})
    if any(sha(p)!=v for p,v in source_hashes.items()):raise RuntimeError('Source changed during run')
    if teacher.small_fingerprint()!=teacher.fingerprint:raise RuntimeError('Teacher changed')
    if sha(spatial.PARENT/'manifest.json')!=PARENT_SHA:raise RuntimeError('Immutable parent changed')
    write(out/'endpoint-freeze.json',{'step':final_step,'checkpoint':str(last_cp),'stop_reason':reason,'confirmation_not_accessed':True})
    summary={'status':'complete' if final_step==STEPS else 'bounded_stop','arm':arm,'steps':final_step,'totals':dict(totals),
       'checkpoint':str(last_cp),'stop_reason':reason,'weights_changed':weight_hash(lab)!=base_hash,
       'parent_unchanged':True,'consecutive_retention_failures':bad,'confirmation_not_accessed':True,
       'automatic_promotion':False,'wall_seconds_this_process':time.monotonic()-before,'unix':time.time()}
    write(out/'summary.json',summary);write(out/'live.json',summary);emit(stage='ARM_FINISHED',**summary)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True,choices=['C448','A448','H896']);ap.add_argument('--resume',type=Path);args=ap.parse_args()
    try:main(args.arm,args.resume)
    except BaseException as e:
        write(ROOT/(args.arm+'-failure.json'),{'type':type(e).__name__,'message':str(e),'automatic_retry':False,'unix':time.time()});raise
