"""Phase 6 HotSpot execution, communication accounting, and policy replay."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import tempfile
import threading
import time

from .allocation import POLICIES,Config,initial_state,transition,balance_trace
from .capabilities import collect,command
from .concurrency import analyze_profile
from .experiment import save_new,distribution
from .hotspot_reference import model,generated,write_input,read_grid,trajectory,communication,close,ATOL,RTOL


def observe(row,run_id):
    return dict(run_id=run_id,epoch=0,iteration=row['index'],cpu_items=row['cpu_rows'],
        t_cpu=(row['cpu_end_ns']-row['release_ns'])/1e9 if row['cpu_end_ns'] is not None else None,
        t_gpu=(row['gpu_end_ns']-row['release_ns'])/1e9 if row['gpu_end_ns'] is not None else None,
        valid=row['status']=='completed' and row['errors']==0,common_origin=True)


def execute(binary,dataset,output,case,prefix=()):
    cfg=Config(case['policy'],case['rows']) if case['policy']!='scripted' else None
    state=initial_state(case['run_id']);decisions=[];lines=[]
    argv=[*prefix,str(Path(binary).resolve()),str(Path(dataset).resolve()),str(case['counts'][0]),
          str(case['steps']),str(case['workers']),case['mode'],str(Path(output).resolve()),'feedback']
    started=time.monotonic_ns();expired=threading.Event();failure=None
    with tempfile.TemporaryFile(mode='w+t') as errors:
        proc=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors,text=True,encoding='utf-8',bufsize=1)
        def kill():
            expired.set()
            try:proc.kill()
            except ProcessLookupError:pass
        timer=threading.Timer(case.get('timeout_s',120),kill);timer.daemon=True;timer.start()
        try:
            for line in proc.stdout:
                lines.append(line)
                if not line.startswith('{'):continue
                row=json.loads(line)
                if row.get('kind')!='iteration' or row['warmup']:continue
                if row['status']!='completed':raise ValueError('Native step failed')
                if row['index']+1<case['steps']:
                    begin=time.monotonic_ns()
                    if cfg:
                        decision=transition(cfg,state,observe(row,case['run_id']));state=decision['post_state'];cut=decision['cpu_items']
                    else:
                        cut=case['counts'][row['index']+1] if len(case['counts'])>1 else case['counts'][0]
                        decision=dict(reason='scripted_schedule',iteration=row['index'],cpu_items=cut)
                    decisions.append(dict(decision=decision,controller_ns=time.monotonic_ns()-begin))
                    proc.stdin.write(str(cut)+'\n');proc.stdin.flush()
            proc.wait(timeout=5)
        except (ValueError,OSError,subprocess.TimeoutExpired) as error:failure=str(error)
        finally:
            timer.cancel()
            if proc.poll() is None:proc.kill()
            proc.wait();proc.stdin.close();proc.stdout.close()
        errors.seek(0);stderr=errors.read()
    return dict(argv=argv,status='ok' if proc.returncode==0 and not failure and not expired.is_set() else 'failed',
        returncode=proc.returncode,stdout=''.join(lines),stderr=stderr,failure=failure,timed_out=expired.is_set(),
        elapsed_ns=time.monotonic_ns()-started,decisions=decisions,configuration=asdict(cfg) if cfg else None)


def assess(case,meta,record,output,golden=None):
    if record['status']!='ok' or record['returncode']!=0:raise ValueError('HotSpot command failed: '+record.get('stderr',''))
    raw=[json.loads(line) for line in record['stdout'].splitlines() if line.startswith('{')]
    if len(raw)!=case['steps']+3 or raw[0]['kind']!='configuration' or raw[-1]['kind']!='summary':raise ValueError('Missing configuration/steps/summary')
    cfg,end=raw[0],raw[-1];steps=raw[1:-1];rows,cols=meta['rows'],meta['cols']
    if cfg['schema_version']!='6.0' or cfg['clock']!='CLOCK_MONOTONIC' or cfg['feedback'] is not True:raise ValueError('Invalid clock/schema/protocol')
    if any(cfg[k]!=case[k] for k in ('rows','cols','steps','workers','mode')) or (cfg['rows'],cfg['cols'])!=(rows,cols):raise ValueError('Work budget mismatch')
    if cfg['atol']!=ATOL or cfg['rtol']!=RTOL:raise ValueError('Tolerance changed')
    if any(not math.isclose(cfg[k],v,rel_tol=2e-15,abs_tol=0) for k,v in meta['model'].items() if k not in ('ambient','limited')):raise ValueError('Thermal model mismatch')
    policy=Config(case['policy'],rows) if case['policy']!='scripted' else None
    if record['configuration']!=(asdict(policy) if policy else None):raise ValueError('Policy configuration changed')
    old=rows;power_begin=rows;last_validation=0;measured=[]
    for index,r in enumerate(steps):
        warmup=index==0;i=index-1;cut=r['cpu_rows']
        if r['kind']!='iteration' or r['index']!=i or r['warmup'] is not warmup or type(cut) is not int or not 0<=cut<=rows:raise ValueError('Invalid step sequence')
        if r['old_cpu_rows']!=old or r['input_epoch']!=max(0,i) or r['output_epoch']!=max(0,i)+1:raise ValueError('Stale timestep or ownership metadata')
        expected=communication(rows,cols,old,cut,max(0,i),power_begin)
        if any(r[k]!=v for k,v in expected.items()):raise ValueError('Migration/halo/power accounting mismatch')
        checked=case['mode']=='verify' or warmup
        if r['status']!='completed' or r['errors']!=0 or r['validation']!=('checked' if checked else 'deferred'):raise ValueError('Invalid correctness observation')
        if not math.isfinite(r['max_error']) or r['max_error']<0:raise ValueError('Invalid error metric')
        if r['validation_d2h_bytes']!=((rows-cut)*cols*8 if checked else 0):raise ValueError('Validation traffic mixed with required traffic')
        stamps=[r[k] for k in ('release_ns','communication_end_ns','end_ns','validation_end_ns')]
        if any(type(t) is not int or t<0 for t in stamps) or stamps!=sorted(stamps) or stamps[0]<last_validation:raise ValueError('Invalid common-clock timing')
        for active,names in ((cut,('cpu_start_ns','cpu_end_ns')),(rows-cut,('gpu_end_ns',))):
            values=[r[k] for k in names]
            if active:
                if any(type(v) is not int for v in values) or values!=sorted(values) or not stamps[1]<=values[0]<=values[-1]<=stamps[2]:raise ValueError('Invalid branch timing')
            elif any(v is not None for v in values):raise ValueError('Empty branch has a fake timing')
        if type(r['control_ns']) is not int or r['control_ns']<0 or (i<=0 and r['control_ns']!=0):raise ValueError('Invalid control time')
        if i>0 and r['control_ns']>r['release_ns']-last_validation:raise ValueError('Control exceeds inter-step gap')
        if checked and golden is not None:
            actual=r.get('temperatures');reference=golden[max(0,i)]
            if actual is None or len(actual)!=rows*cols or not all(close(a,b) for a,b in zip(actual,reference)):raise ValueError('Independent Python trajectory mismatch')
        last_validation=r['validation_end_ns']
        if warmup:old=rows;power_begin=rows
        else:old=cut;power_begin=expected['power_begin_after'];measured.append(r)
    if measured[0]['cpu_rows']!=case['counts'][0]:raise ValueError('Initial partition mismatch')
    if len(record['decisions'])!=len(measured)-1:raise ValueError('Missing feedback decisions')
    state=initial_state(case['run_id']);observations=[observe(r,case['run_id']) for r in measured]
    for i,item in enumerate(record['decisions']):
        if policy:
            decision=transition(policy,state,observations[i]);state=decision['post_state']
        else:decision=dict(reason='scripted_schedule',iteration=i,cpu_items=case['counts'][i+1] if len(case['counts'])>1 else case['counts'][0])
        if item['decision']!=decision or measured[i+1]['cpu_rows']!=decision['cpu_items']:raise ValueError('Feedback decision replay mismatch')
        if type(item['controller_ns']) is not int or item['controller_ns']<0:raise ValueError('Invalid controller duration')
    active=sum(r['end_ns']-r['release_ns'] for r in measured);control=sum(r['control_ns'] for r in measured)
    expected_gather=(rows-measured[-1]['cpu_rows'])*cols*8 if case['mode']=='timing' else 0
    validation_bytes=sum(r['validation_d2h_bytes'] for r in measured)
    if (end['status']!='passed' or end['errors']!=0 or end['steps']!=case['steps'] or end['active_ns']!=active or end['control_ns']!=control
            or end['gather_d2h_bytes']!=expected_gather or end['validation_d2h_bytes']!=validation_bytes
            or type(end['gather_ns']) is not int or end['gather_ns']<0 or end['work_and_control_and_gather_ns']!=active+control+end['gather_ns']
            or end['loop_elapsed_ns']!=measured[-1]['end_ns']-measured[0]['release_ns'] or active+control>end['loop_elapsed_ns']
            or not math.isclose(end['simulated_seconds'],case['steps']*meta['model']['dt'],rel_tol=2e-15)):
        raise ValueError('Summary timing/work/transfer mismatch')
    h,w,values=read_grid(output)
    if (h,w)!=(rows,cols) or (golden is not None and not all(close(a,b) for a,b in zip(values,golden[-1]))):raise ValueError('Final temperature field mismatch')
    required_keys=('initial_h2d_bytes','migration_h2d_bytes','migration_d2h_bytes','halo_h2d_bytes','halo_d2h_bytes','power_h2d_bytes')
    reasons=Counter(d['decision']['reason'] for d in record['decisions'])
    return dict(case=case,status='passed',measured_steps=len(measured),output_sha256=hashlib.sha256(Path(output).read_bytes()).hexdigest(),
        max_error=end['max_error'],active_s=active/1e9,control_s=control/1e9,gather_s=end['gather_ns']/1e9,
        work_and_control_and_gather_s=end['work_and_control_and_gather_ns']/1e9,loop_elapsed_s=end['loop_elapsed_ns']/1e9,
        command_elapsed_s=record['elapsed_ns']/1e9,controller_compute_s=sum(d['controller_ns'] for d in record['decisions'])/1e9,
        communication_s=sum(r['communication_end_ns']-r['release_ns'] for r in measured)/1e9,
        traffic={k:sum(r[k] for r in measured) for k in required_keys},validation_d2h_bytes=validation_bytes,gather_d2h_bytes=expected_gather,
        shares=[r['cpu_rows']/rows for r in measured],diagnostic=balance_trace(observations),reasons=dict(reasons),energy_j=None)


def summarize(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());cases=json.loads((root/'plan.json').read_text());results=[]
    first={};equivalent=True
    for case in cases:
        golden=json.loads((root/case['golden']).read_text()) if case.get('golden') else None
        result=assess(case,manifest['datasets'][case['dataset']],json.loads((root/case['record']).read_text()),root/case['output'],golden)
        key=(case['dataset'],case['steps'])
        if key in first:
            previous=first[key]
            if result['output_sha256']!=previous['output_sha256']:
                a=read_grid(root/case['output'])[2];b=read_grid(root/previous['case']['output'])[2]
                if not all(close(x,y) for x,y in zip(a,b)):raise ValueError('Same-input final fields differ across schedules/policies')
        else:first[key]=result
        results.append(result)
    comparisons=[]
    for stage in ('coarse','refine','heldout'):
        for name in sorted({r['case']['name'] for r in results if r['case']['stage']==stage}):
            group=[r for r in results if r['case']['stage']==stage and r['case']['name']==name]
            comparisons.append(dict(stage=stage,name=name,trials=len(group),runtime_s=distribution([r['work_and_control_and_gather_s'] for r in group]),
                communication_s=distribution([r['communication_s'] for r in group]),balanced_at_end=sum(r['diagnostic']['balanced_at_end'] for r in group),energy_j=None))
    overlap=analyze_profile(root/'profile.sqlite',3,'hotspot_step','hotspot_iteration','hotspot_cpu_compute')
    return dict(schema_version='6.0',implementation_status='passed' if overlap['status']=='demonstrated' else 'incomplete',
        results=results,comparisons=comparisons,same_input_field_equivalence=equivalent,concurrency=overlap,
        measured_steps=sum(r['measured_steps'] for r in results),energy_qualification='blocked_phase2',hardware_settings_changed=False,
        clock_policy='unchanged automatic platform policy; no fixed-capacity claim',
        timing_scope='sum of timestep work plus feedback/export/IPC and final gather; validation copies explicitly separate')


def best(results,stage):
    names=sorted({r['case']['counts'][0] for r in results if r['case']['stage']==stage})
    return min(names,key=lambda cut:(sum(r['work_and_control_and_gather_s'] for r in results if r['case']['stage']==stage and r['case']['counts'][0]==cut)
        /sum(r['case']['stage']==stage and r['case']['counts'][0]==cut for r in results),cut))


def run(output,binary,workers=4,blocks=3):
    if platform.system()!='Linux':raise ValueError('Run in WSL/Linux')
    if not 1<=workers<=min(8,max(1,(os.cpu_count() or 1)-2)) or not 2<=blocks<=10:raise ValueError('Invalid worker/block budget')
    output=Path(output);binary=Path(binary).resolve(strict=True);output.mkdir(parents=True,exist_ok=False)
    cases=[];results=[]
    try:
        collect(output/'environment','/usr/local/cuda-13.1/bin/nvcc',None)
        (output/'environment/raw/CMakeCache.txt').write_bytes((binary.parent/'CMakeCache.txt').read_bytes())
        shapes=dict(single=(1,1,1,'random'),row=(1,17,2,'edge'),column=(17,1,3,'impulse'),
            irregular=(19,23,4,'random'),equilibrium=(7,9,5,'equilibrium'),edge=(13,11,6,'edge'),
            tuning=(257,259,6601,'random'),heldout=(257,259,6602,'random'),profile=(2048,2048,6603,'random'))
        (output/'datasets').mkdir();(output/'golden').mkdir();(output/'fields').mkdir()
        metadata={};small={}
        for name,(rows,cols,seed,kind) in shapes.items():
            temp,power,provenance=generated(rows,cols,seed,kind)
            metadata[name]=write_input(output/'datasets'/(name+'.hsdat'),rows,cols,temp,power,provenance)
            if rows*cols<=1024:small[name]=(temp,power,rows,cols)
        save_new(output/'manifest.json',dict(schema_version='6.0',datasets=metadata,workers=workers,blocks=blocks,
            binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),source_hashes=json.loads((output/'environment/source_hashes.json').read_text()),
            reference_files={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('references/rodinia').glob('*')) if p.is_file()},
            policy_configs={p:asdict(Config(p,257)) for p in (*POLICIES,'static')},energy_qualification='blocked_phase2',
            protocol='verify every step on small inputs; timing runs validate final full field against independent native reference',
            clock_policy='unchanged automatic policy',simulated_time='steps * shape-dependent stable dt'))
        def trial(dataset,counts,steps,mode,policy,stage,name=None,block=None,golden=None,prefix=()):
            i=len(cases);m=metadata[dataset]
            case=dict(run_id=f'hotspot-{i:03d}',record=f'trial-{i:03d}.json',output=f'fields/trial-{i:03d}.hstemp',dataset=dataset,
                rows=m['rows'],cols=m['cols'],counts=counts,steps=steps,workers=workers,mode=mode,policy=policy,stage=stage,
                name=name or f'{policy}@{counts[0]}',block=block,golden=golden,timeout_s=600 if stage=='long' else 120)
            save_new(output/f'plan-{i:03d}.json',case);cases.append(case)
            record=execute(binary,output/'datasets'/(dataset+'.hsdat'),output/case['output'],case,prefix)
            save_new(output/case['record'],record)
            expected=json.loads((output/golden).read_text()) if golden else None
            result=assess(case,m,record,output/case['output'],expected);results.append(result)
            print(json.dumps(dict(trial=i,stage=stage,dataset=dataset,status=result['status'])),flush=True)
        for dataset,(temp,power,rows,cols) in small.items():
            steps=12;golden=f'golden/{dataset}.json';save_new(output/golden,trajectory(temp,power,rows,cols,steps))
            for cut in sorted({0,rows,1,rows-1,rows//2}):trial(dataset,[cut],steps,'verify','static','correctness',golden=golden)
            schedule=([rows,rows//2,0,rows,1,max(0,rows-1)]*2)
            trial(dataset,schedule,steps,'verify','scripted','migration',golden=golden)
            for policy in POLICIES:trial(dataset,[0],steps,'verify',policy,'policy_correctness',golden=golden)
            # Same changing-boundary sequence without validation transfers; final Python oracle still applies.
            trial(dataset,schedule,steps,'timing','scripted','deferred_correctness',golden=golden)
        # Long small-grid trajectory checks accumulated error at every timestep.
        temp,power,rows,cols=small['irregular'];golden='golden/irregular-long.json'
        save_new(output/golden,trajectory(temp,power,rows,cols,600))
        trial('irregular',([rows,1,rows//2,rows-1,0]*120),600,'verify','scripted','long',golden=golden)
        def stage(stage_name,dataset,counts,seed):
            rng=random.Random(seed);order=[]
            for block in range(blocks):
                candidates=sorted(set(counts));rng.shuffle(candidates);order.extend((block,c) for c in candidates)
            save_new(output/(stage_name+'-plan.json'),dict(order=order,dataset=dataset,steps=24,seed=seed))
            for block,cut in order:trial(dataset,[cut],24,'timing','static',stage_name,name=f'static@{cut}',block=block)
        rows=metadata['tuning']['rows'];coarse=[0,rows//4,rows//2,3*rows//4,rows]
        stage('coarse','tuning',coarse,6610);winner=best(results,'coarse')
        refine=sorted({max(0,winner-rows//8),winner,min(rows,winner+rows//8)})
        stage('refine','tuning',refine,6611);selected=best(results,'refine')
        save_new(output/'selection.json',dict(cpu_rows=selected,coarse_best=winner,refined=refine,selected_on='refine tuning mean; ties lower CPU rows'))
        candidates=[('static',c,f'static@{c}') for c in sorted(set(coarse+refine))]
        candidates += [(p,Config(p,rows).nearest(s),f'{p}@{Config(p,rows).nearest(s)}') for p in POLICIES for s in (.15,.5,.9)]
        rng=random.Random(6612);order=[]
        for block in range(blocks):
            shuffled=candidates.copy();rng.shuffle(shuffled);order.extend((block,*c) for c in shuffled)
        save_new(output/'heldout-plan.json',dict(order=order,dataset='heldout',steps=24,seed=6612))
        for block,p,cut,name in order:trial('heldout',[cut],24,'timing',p,'heldout',name,block)
        # Match the paper's grid/timestep counts only; generated input and stable FP64 model differ.
        trial('profile',sum(([cut]*100 for cut in (2048,1024,0,1536,1,2047)),[]),600,'timing','scripted','long','paper_shape_migration')
        save_new(output/'profiler-version.json',command(['nsys','--version']))
        trial('profile',[1536],3,'timing','static','profile',prefix=('nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--export=sqlite',f'--output={output/"profile"}'))
        save_new(output/'plan.json',cases)
        summary=summarize(output);save_new(output/'summary.json',summary);save_new(output/'timeline.json',summary['concurrency']['timeline'])
        save_new(output/'checksums.json',{p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.rglob('*')) if p.is_file()})
        return summary
    except BaseException as error:
        if not (output/'plan.json').exists():save_new(output/'plan.json',cases)
        save_new(output/'failure.json',dict(status='failed',reason=f'{type(error).__name__}: {error}'))
        raise


def replay(root):
    root=Path(root);expected=json.loads((root/'checksums.json').read_text())
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p!=root/'checksums.json'}
    if actual!=set(expected):raise ValueError('Archive inventory changed')
    for name,value in expected.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=value:raise ValueError('Evidence checksum mismatch: '+name)
    return summarize(root)
