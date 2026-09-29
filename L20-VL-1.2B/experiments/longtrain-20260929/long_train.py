"""One quality-gated, resumable long training job on the user's existing L20.
No multi-arm scheduler, model substitution, automatic retry or old-file deletion.
The wall budget and unique-image ceiling are limits, not promised final results.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse,collections,fcntl,gc,hashlib,io,json,math,queue,random,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from peft import set_peft_model_state_dict
import stream_sources as stream
ROOT=Path(__file__).resolve().parent
C=stream.CONFIG
RUN=ROOT/'run'
PROJECT=ROOT.parent
PILOT=PROJECT/'quality-pilot-20260929T1105Z'
sys.path.insert(0,str(PILOT))
import train_quality as pilot
core=pilot.core
sha=stream.sha
write=stream.write
emit=stream.emit
STOP=threading.Event()

def stop_requested(*_):STOP.set()
def guard():
    if shutil.disk_usage(ROOT).free<C['minimum_disk_free_bytes']:raise RuntimeError('Disk reserve below fixed minimum')
    if stream.available_ram()<1500*2**20:raise RuntimeError('Host memory reserve below 1.5GiB')

def checkpoint(lab,opt,step,cursor,chain,totals,bad,elapsed,protocol_sha):
    guard();cp=RUN/'checkpoints'/f'step-{step:06d}';tmp=cp.with_name(cp.name+'.partial')
    if cp.exists() or tmp.exists():raise FileExistsError('Checkpoint overwrite prohibited')
    tmp.mkdir(parents=True)
    save_file({n:v.detach().cpu().contiguous() for n,v in lab.bridge.state_dict().items()},tmp/'bridge.safetensors')
    lab.language.save_pretrained(tmp/'language_adapter',safe_serialization=True)
    state={'step':step,'last_consumed_catalog':cursor,'catalog_chain_sha256':chain,'totals':dict(totals),'consecutive_bad':bad,
        'elapsed_wall_seconds':elapsed,'config_sha256':stream.PLAN_SHA,'optimizer':opt.state_dict(),
        'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()}
    torch.save(state,tmp/'resume.pt')
    files={str(p.relative_to(tmp)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in tmp.rglob('*') if p.is_file()}
    write(tmp/'manifest.json',{'files':files,'step':step,'config_sha256':stream.PLAN_SHA,'protocol_sha256':protocol_sha,
       'warmstart_manifest_sha256':C['warmstart_manifest_sha256'],'teacher_manifest_sha256':C['teacher_manifest_sha256'],
       'last_consumed_catalog':cursor,'catalog_chain_sha256':chain,'checkpoint_includes_optimizer_RNG_and_data_cursor':True,
       'requires_unchanged_original_language_and_vision_models':True,'automatic_promotion':False})
    for path in tmp.rglob('*'):
        if path.is_file():
            with path.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,cp);fd=os.open(str(cp.parent),os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    pilot.spatial.verify_checkpoint(cp)
    write(RUN/'latest.json',{'step':step,'checkpoint':str(cp),'manifest_sha256':sha(cp/'manifest.json'),'unix':time.time()})
    emit(stage='LONG_CHECKPOINT',step=step,checkpoint=str(cp))
    return cp

def restore(lab,opt,cp,protocol_sha):
    m=pilot.spatial.verify_checkpoint(cp)
    if m['config_sha256']!=stream.PLAN_SHA or m['protocol_sha256']!=protocol_sha:raise RuntimeError('Resume contract mismatch')
    lab.bridge.load_state_dict(load_file(cp/'bridge.safetensors'),strict=True)
    result=set_peft_model_state_dict(lab.language,load_file(cp/'language_adapter/adapter_model.safetensors'),adapter_name='default')
    if result.unexpected_keys:raise RuntimeError('Adapter keys do not match')
    state=torch.load(cp/'resume.pt',map_location='cpu',weights_only=False)
    opt.load_state_dict(state['optimizer']);random.setstate(state['rng_python']);np.random.set_state(state['rng_numpy'])
    torch.set_rng_state(state['rng_torch']);torch.cuda.set_rng_state_all(state['rng_cuda']);lab.zero()
    return state

def producer(lab,old_groups,cursor,out_queue,deadline):
    source=None
    try:
        source=stream.SourceStream(lab.tok,STOP,deadline,cursor)
        collate=pilot.RecipeCollator(lab.proc,lab.tok.pad_token_id)
        for packet in source:
            if STOP.is_set():break
            groups=[]
            for domain in ('plotqa','docmatix'):
                rr=[pilot.encode(lab,dict(r,image_path=io.BytesIO(r['_bytes'])),448,'T','new') for r in packet['rows'] if r['domain']==domain]
                groups.extend(rr[j:j+4] for j in range(0,len(rr),4))
            groups.extend(old_groups[packet['catalog']%len(old_groups)])
            if sum(map(len,groups))!=64 or any(r['split']!='train' for g in groups for r in g):raise RuntimeError('Unsafe batch composition')
            batches=[collate(g) for g in groups]
            message={'kind':'data','catalog':packet['catalog'],'catalog_sha256':packet['catalog_sha256'],'batches':batches}
            while not STOP.is_set():
                try:out_queue.put(message,timeout=2);break
                except queue.Full:pass
        if not STOP.is_set():
            while not STOP.is_set():
                try:out_queue.put({'kind':'end'},timeout=2);break
                except queue.Full:pass
    except BaseException as e:
        if STOP.is_set():return
        if isinstance(e,TimeoutError) and time.monotonic()>=deadline:
            while not STOP.is_set():
                try:out_queue.put({'kind':'wall_budget'},timeout=2);break
                except queue.Full:pass
            return
        write(ROOT/'data-producer-failure.json',{'type':type(e).__name__,'message':str(e),'unix':time.time(),'automatic_retry':False})
        while not STOP.is_set():
            try:out_queue.put({'kind':'error','type':type(e).__name__,'message':str(e)},timeout=2);break
            except queue.Full:pass
    finally:
        if source is not None:source.close()

def update(lab,opt,teacher,batches,replay,step):
    lab.train();lab.zero();losses={};weighted=0.;answers=0;visual=0;image_events=0
    for cpu in batches:
        b=core.cuda(cpu);task=b['task']
        if task in ('qa','retain'):
            ce,kl=core.preserved_task(lab,teacher,b)
            ce_weight=C['old_QA_CE_weight'] if task=='qa' else C['old_caption_CE_weight']
            kl_weight=C['old_QA_teacher_KL_weight'] if task=='qa' else C['old_caption_teacher_KL_weight']
            loss=ce_weight*ce+kl_weight*kl;losses[task+'_ce']=float(ce.detach());losses[task+'_kl']=float(kl.detach())
        else:
            ce=lab.losses(b).sum()
            loss=(C['new_loss_weight']/32 if task=='new' else C['old_domain_replay_CE_weight']/8)*ce
            losses[task+'_weighted']=losses.get(task+'_weighted',0.)+float(loss.detach())
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
        weighted+=float(loss.detach());loss.backward();answers+=b['answer_tokens'];visual+=b['visual_tokens'];image_events+=b['image_events']
    text_tokens=0
    if step%4==0:
        tx=core.previous.text_loss(lab,replay,(1800+step)%(len(replay)//2))
        if not bool(torch.isfinite(tx)):raise FloatingPointError('Nonfinite replay loss')
        (C['text_replay_weight_every4updates']*tx).backward();weighted+=C['text_replay_weight_every4updates']*float(tx.detach())
        losses['text_CE']=float(tx.detach());text_tokens=4096
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in lab.params):raise FloatingPointError('Missing/nonfinite trainable gradient')
    norm=torch.nn.utils.clip_grad_norm_(lab.params,1.)
    if not bool(torch.isfinite(norm)):raise FloatingPointError('Nonfinite gradient norm')
    warm=C['warmup_steps'];scale=min(1.,(step+1)/warm)
    if step>=warm:scale=.25+.75*.5*(1+math.cos(math.pi*(step-warm)/(C['max_updates']-warm)))
    opt.param_groups[0]['lr']=C['bridge_lr']*scale;opt.param_groups[1]['lr']=C['lora_lr']*scale;opt.step()
    return {'losses':losses,'full_weighted_objective':weighted,'gradient_norm':float(norm),'image_events':image_events,
        'new_image_events':32,'answer_tokens':answers,'visual_tokens':visual,'text_tokens':text_tokens,
        'bridge_lr':opt.param_groups[0]['lr'],'LoRA_lr':opt.param_groups[1]['lr']}

def main(resume=None):
    process_start=time.monotonic()
    if (resume is None and RUN.exists()) or (resume is not None and not RUN.exists()):raise RuntimeError('Explicit new-run/resume required')
    lock=(PROJECT/'general-vision-gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied; do not compete')
    guard();RUN.mkdir(exist_ok=True);signal.signal(signal.SIGTERM,stop_requested);signal.signal(signal.SIGINT,stop_requested)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    qualification=json.loads((ROOT/'qualification.json').read_text())
    if not qualification['pass'] or qualification['config_sha256']!=stream.PLAN_SHA:raise RuntimeError('Preflight not passed for this configuration')
    write(RUN/'live.json',{'phase':'loading_verified_model','pid':os.getpid(),'new_optimizer_updates':0,'unix':time.time()})
    if sha(Path(C['teacher_checkpoint'])/'manifest.json')!=C['teacher_manifest_sha256']:raise RuntimeError('Teacher checkpoint changed')
    lab,opt=pilot.load_lab('C448');teacher=core.Teacher(lab)
    teacher_hash=teacher.fingerprint
    warm=Path(C['warmstart_checkpoint']);warm_manifest=pilot.spatial.verify_checkpoint(warm)
    if sha(warm/'manifest.json')!=C['warmstart_manifest_sha256']:raise RuntimeError('Warmstart checkpoint changed')
    pilot.restore(lab,opt,warm,warm_manifest['plan_sha256'])
    warm_hash=pilot.weight_hash(lab)
    plans,sets,old_data=pilot.make_data(lab,'C448');old_groups=[p[8:] for p in plans];del plans,old_data;gc.collect()
    replay=np.load(core.previous.REPLAY,mmap_mode='r')
    if sha(core.previous.REPLAY)!=core.previous.REPLAY_SHA:raise RuntimeError('Text replay identity changed')
    paths=[Path(__file__),ROOT/'stream_sources.py',ROOT/'config.json',Path(pilot.__file__),Path(core.__file__),Path(core.previous.__file__),
        pilot.WORK/'spatial_route.py',pilot.WORK/'input_runtime.py',PROJECT/'vlm-opt-20260925T1537Z/vlm_lab.py']
    source_hashes={str(p):sha(p) for p in paths}
    if source_hashes[str(ROOT/'stream_sources.py')]!=qualification['stream_source_sha256'] or source_hashes[str(Path(__file__))]!=qualification['trainer_sha256']:raise RuntimeError('Qualified source changed')
    protocol={'config_sha256':stream.PLAN_SHA,'source_sha256':source_hashes,'warmstart':C['warmstart_checkpoint'],
        'warmstart_manifest_sha256':C['warmstart_manifest_sha256'],'teacher':C['teacher_checkpoint'],'teacher_fingerprint':teacher_hash,
        'actual_teacher_is_healthier_pre_pilot_checkpoint':True,'frozen_language_and_vision':True,'trainable_parameters':sum(p.numel() for p in lab.params),'trainable_tensors':len(lab.params),
        'maximum_updates':C['max_updates'],'maximum_wall_seconds':C['maximum_wall_seconds'],
        'new_source_images_without_replacement':True,'old_replay_plan_repeats_are_reported_separately':True,
        'old_task_replay_cycles_over_128_frozen_pilot_groups':True,'final_confirmation_not_used_for_decisions':True,
        'checkpoint_policy':[1,16,64,128,256],'then_every_updates':256,'no_deletion_or_new_compute':True,
        'optimizer_inherited_from_successful_128step_pilot':True,'old_QA_anchor':C['old_QA_anchor'],'old_QA_floor':C['old_QA_floor'],
        'quality_claim':'hypothesis_under_test_not_guaranteed_best_model','torch':torch.__version__}
    elapsed_before=0.;step=0;cursor=-1;chain='0'*64;totals=collections.Counter();bad=0;last_cp=None
    if resume is None:
        write(RUN/'protocol.json',protocol);random.seed(C['seed']);np.random.seed(C['seed']);torch.manual_seed(C['seed']);torch.cuda.manual_seed_all(C['seed'])
        write(RUN/'initial.json',{'warm_model_hash':warm_hash,'teacher_fingerprint':teacher_hash,'starting_checkpoint':str(warm),'new_updates':0})
        health=pilot.health(lab,sets);write(RUN/'baseline-health.json',health)
        if not all(health['checks'].values()):raise RuntimeError('Starting checkpoint failed original retention floor')
        write(RUN/'baseline-development.json',pilot.development(lab,sets))
    else:
        if json.loads((RUN/'protocol.json').read_text())!=protocol:raise RuntimeError('Cannot resume changed protocol')
        state=restore(lab,opt,resume,sha(RUN/'protocol.json'));step=state['step'];cursor=state['last_consumed_catalog'];chain=state['catalog_chain_sha256']
        totals.update(state['totals']);bad=state['consecutive_bad'];elapsed_before=state['elapsed_wall_seconds'];last_cp=resume
        if bad>=C['stop_after_consecutive_retention_failures']:raise RuntimeError('Retention-stopped run is not auto-resumable')
    protocol_sha=sha(RUN/'protocol.json');deadline=process_start+C['maximum_wall_seconds']-elapsed_before
    if time.monotonic()>=deadline:raise TimeoutError('No wall budget remains')
    batches=queue.Queue(maxsize=2);thread=threading.Thread(target=producer,args=(lab,old_groups,cursor,batches,deadline),daemon=True,name='bounded-training-input');thread.start()
    last_input=time.monotonic();reason='planned_update_cap';train_seconds=0.;input_wait=0.;last_saved_step=step
    try:
        while step<C['max_updates']:
            if STOP.is_set() or (ROOT/'STOP').exists():reason='user_stop';break
            if time.monotonic()>=deadline:reason='wall_budget';break
            guard();wait_start=time.monotonic()
            try:packet=batches.get(timeout=10)
            except queue.Empty:
                if not thread.is_alive():raise RuntimeError('Input producer exited unexpectedly')
                if time.monotonic()-last_input>900:raise TimeoutError('No verified input for 15 minutes')
                write(RUN/'live.json',{'phase':'waiting_for_verified_source','step':step,'pid':os.getpid(),'unix':time.time()});continue
            input_wait+=time.monotonic()-wait_start;last_input=time.monotonic()
            if packet['kind']=='wall_budget':reason='wall_budget';break
            if packet['kind']=='error':raise RuntimeError('Input producer '+packet['type']+': '+packet['message'])
            if packet['kind']=='end':reason='source_exhausted';break
            if packet['catalog']!=cursor+1 or packet['catalog']!=step:raise RuntimeError('Input sequence changed')
            cat=stream.DATA/'catalog'/f"{packet['catalog']:06d}.json"
            if sha(cat)!=packet['catalog_sha256']:raise RuntimeError('Training catalog changed')
            torch.cuda.synchronize();start=time.perf_counter();event=update(lab,opt,teacher,packet['batches'],replay,step);torch.cuda.synchronize()
            duration=time.perf_counter()-start;train_seconds+=duration;step+=1;cursor=packet['catalog'];chain=hashlib.sha256((chain+packet['catalog_sha256']).encode()).hexdigest()
            totals.update({k:event[k] for k in ('image_events','new_image_events','answer_tokens','visual_tokens','text_tokens')})
            event.update(step=step,catalog=cursor,seconds=duration,fd_count=len(list(Path('/proc/self/fd').iterdir())),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            with (RUN/'train.jsonl').open('a') as f:f.write(json.dumps(event,allow_nan=False)+'\n');f.flush()
            write(RUN/'live.json',{'phase':'training','step':step,'maximum_updates':C['max_updates'],'new_unique_image_events':totals['new_image_events'],
                'global_image_events':totals['image_events'],'last_step_seconds':duration,'last_full_weighted_objective':event['full_weighted_objective'],
                'last_consumed_catalog':cursor,'pid':os.getpid(),'remaining_wall_seconds':max(0,deadline-time.monotonic()),'unix':time.time()})
            if step<=4 or step%32==0:emit(stage='LONG_OPTIMIZER_UPDATE',**event)
            if step in (64,128) or step%256==0:
                health=pilot.health(lab,sets);bad=0 if all(health['checks'].values()) else bad+1;health['consecutive_bad']=bad
                write(RUN/f'step-{step:06d}-health.json',health)
                write(RUN/f'step-{step:06d}-development.json',pilot.development(lab,sets))
                emit(stage='LONG_RETENTION',step=step,old_QA=health['scores']['old_QA'],checks=health['checks'],consecutive_bad=bad)
            if step in (1,16,64,128) or step%256==0:
                if any(sha(p)!=v for p,v in source_hashes.items()):raise RuntimeError('Source code changed during run')
                last_cp=checkpoint(lab,opt,step,cursor,chain,totals,bad,elapsed_before+time.monotonic()-process_start,protocol_sha);last_saved_step=step
                if step==1:
                    current=pilot.weight_hash(lab);restore(lab,opt,last_cp,protocol_sha);restored=pilot.weight_hash(lab)
                    receipt={'optimizer_updates':1,'parameters_changed_from_warmstart':current!=warm_hash,'same_process_reload_hash_equal':current==restored,
                             'optimizer_rng_and_consumed_data_cursor_saved':True,'fresh_process_GPU_resume_equivalence_tested':False,'checkpoint_manifest_sha256':sha(last_cp/'manifest.json')}
                    write(RUN/'first-update-verification.json',receipt)
                    if not receipt['parameters_changed_from_warmstart'] or not receipt['same_process_reload_hash_equal']:raise RuntimeError('First update/save verification failed')
            del packet;gc.collect()
            if bad>=C['stop_after_consecutive_retention_failures']:reason='retention_guard';break
        if step>last_saved_step:last_cp=checkpoint(lab,opt,step,cursor,chain,totals,bad,elapsed_before+time.monotonic()-process_start,protocol_sha)
        if teacher.small_fingerprint()!=teacher_hash:raise RuntimeError('Frozen teacher changed')
        if sha(Path(C['warmstart_checkpoint'])/'manifest.json')!=C['warmstart_manifest_sha256']:raise RuntimeError('Original warmstart modified')
        write(RUN/'endpoint-freeze.json',{'step':step,'checkpoint':str(last_cp),'stop_reason':reason,'confirmation_not_accessed':True})
        summary={'status':'complete' if reason in ('planned_update_cap','source_exhausted') else 'bounded_stop','steps':step,'checkpoint':str(last_cp),
            'totals':dict(totals),'stop_reason':reason,'consecutive_retention_failures':bad,'training_seconds_this_process':train_seconds,
            'input_wait_seconds_this_process':input_wait,'elapsed_wall_seconds':elapsed_before+time.monotonic()-process_start,
            'new_full_public_benchmark_scores':False,'automatic_promotion':False,'teacher_and_parent_unchanged':True,'unix':time.time()}
        write(RUN/'summary.json',summary);write(RUN/'live.json',summary);emit(stage='LONG_RUN_FINISHED',**summary)
    finally:STOP.set();thread.join(timeout=5)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--resume',type=Path);args=ap.parse_args()
    try:main(args.resume)
    except BaseException as e:
        STOP.set();write(ROOT/'training-failure.json',{'type':type(e).__name__,'message':str(e),'unix':time.time(),'automatic_retry':False,'last_saved_checkpoint_retained':True});raise
