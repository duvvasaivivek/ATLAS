"""Frozen, randomized matched evaluation with independent tuning and raw replay."""
from dataclasses import asdict
import json
from pathlib import Path
import random
import statistics

from .experiment import save_new,distribution
from .frequency_experiment import sha
from .integration import Config
from .integration_runner import execute
from .phase9 import assess
from . import kmeans_reference as kr,hotspot_reference as hr


def interval(values):
    if not values:return None
    rng=random.Random(20261102)
    means=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(2000))
    return [means[49],means[1949]]


def comparisons_for(results):
    comparisons=[]
    for workload,size in sorted({(r['workload'],r['size']) for r in results if r['stage']=='evaluation'}):
        group=[r for r in results if (r['workload'],r['size'],r['stage'])==(workload,size,'evaluation')]
        baseline={r['block']:r['primary_s'] for r in group if r['name']=='gpu_only' and r['status']=='passed'}
        for name in sorted({r['name'] for r in group}):
            requested=[r for r in group if r['name']==name]
            rows=[r for r in requested if r['status']=='passed' and r['block'] in baseline]
            deltas=[100*(r['primary_s']/baseline[r['block']]-1) for r in rows]
            comparisons.append(dict(workload=workload,size=size,name=name,trials=len(rows),requested_trials=len(requested),
                failed_trials=sum(r['status']!='passed' for r in requested),unpaired_trials=len(requested)-len(rows),
                primary_s=distribution([r['primary_s'] for r in rows]),command_s=distribution([r['command_s'] for r in rows]),
                feedback_s=distribution([r['feedback_s'] for r in rows]),
                paired_runtime_change_percent=statistics.mean(deltas) if deltas else None,paired_bootstrap_95_percent=interval(deltas)))
    return comparisons


def summarize(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text())
    results=[]
    for case in plan['cases']:
        record=json.loads((root/case['record']).read_text())
        identity=dict(name=case['name'],workload=case['workload'],size=case['size'],block=case['block'],stage=case['stage'],run_id=case['integration']['run_id'])
        try:checked=assess(root,case,plan['datasets'][case['dataset']])
        except (ValueError,KeyError,TypeError,IndexError) as error:
            results.append(dict(**identity,status='failed',failure=record.get('failure') or str(error),stderr=record.get('stderr'),command_s=record['elapsed_ns']/1e9))
            continue
        native=[json.loads(line) for line in record['stdout'].splitlines() if line.startswith('{')]
        end=native[-1]
        # Contiguous workload envelope includes IPC and in-loop checks, plus required final gathering.
        elapsed=(end['loop_elapsed_ns']+end.get('gather_ns',0))/1e9
        results.append(dict(**identity,status='passed',primary_s=elapsed,command_s=checked['command_s'],
                            active_s=checked['active_s'],feedback_s=checked['control_gap_s'],
                            ownership_changes=checked['ownership_changes']))
    failed=sum(r['status']!='passed' for r in results)
    return dict(schema_version='performance-1.0',status='incomplete_failed_trials' if failed else 'passed',failed_trials=failed,
                comparisons=comparisons_for(results),trials=len(results),
                primary_scope='contiguous first release to last completion plus final gather; final independent validation excluded and command time retained',
                baseline='matched optimized GPU-only under unchanged automatic clocks',
                uncertainty='paired block bootstrap, 2000 resamples, fixed seed; independent trials are the sampling unit',
                telemetry='disabled equally in all arms; independent qualification is separate',energy_savings=None,
                setting_changes=False,results=results)


def replay(root):
    root=Path(root);expected=json.loads((root/'checksums.json').read_text())
    files={p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file() and p!=root/'checksums.json'}
    if set(files)!=set(expected) or any(p.is_symlink() or sha(p)!=expected[n] for n,p in files.items()):raise ValueError('Performance archive integrity failure')
    plan=json.loads((root/'plan.json').read_text())
    for case in plan['cases']:
        if any(not (root/case[k]).resolve().is_relative_to(root.resolve()) for k in ('record','dataset','field') if case.get(k)):
            raise ValueError('Performance archive path escape')
    if any(not (root/p).resolve().is_relative_to(root.resolve()) or sha(root/p)!=meta['sha256'] for p,meta in plan['datasets'].items()):raise ValueError('Dataset identity changed')
    result=summarize(root)
    if result!=json.loads((root/'summary.json').read_text()):raise ValueError('Performance summary replay mismatch')
    return result


def run(output,build,baseline_build=None,blocks=10):
    if type(blocks) is not int or not 2<=blocks<=30:raise ValueError('Require 2..30 independent blocks')
    root=Path(output);build=Path(build).resolve();baseline=Path(baseline_build).resolve() if baseline_build else None
    root.mkdir(parents=True,exist_ok=False)
    for folder in ('datasets','fields'): (root/folder).mkdir()
    datasets={};cases=[];selections={};rng=random.Random(20261102)
    for workload in ('kmeans','hotspot'):
        for size in (0,1):
            for stage,seed in (('tuning',7100),('evaluation',7200)):
                name=f'datasets/{workload}-{size}-{stage}.data'
                if workload=='kmeans':datasets[name]=kr.write_dataset(root/name,*kr.clustered((32771,65539)[size],8,8,seed+size))
                else:
                    h,w=((257,259),(513,515))[size];t,p,prov=hr.generated(h,w,seed+size)
                    datasets[name]=hr.write_input(root/name,h,w,t,p,prov)
    def make(workload,size,stage,block,name,share=.5,policy='static',batch=16,validation='timing',reduction='hierarchical',transfer='streamed',communication='serial',legacy=False):
        dataset=f'datasets/{workload}-{size}-{stage}.data';meta=datasets[dataset];n=meta['n'] if workload=='kmeans' else meta['rows']
        run_id=f'eval-{len(cases):04d}'
        cfg=Config(run_id,n,round(n*share),mode='allocation_only',allocation_policy=policy,
                   allocation_iterations=4,allocation_interval_ns=0,cadence_profile='batched_epochs_v1' if not legacy else 'observed_cadence_v1')
        case=dict(name=name,workload=workload,size=size,stage=stage,block=block,dataset=dataset,record=run_id+'.json',
                  field=f'fields/{run_id}.hstemp' if workload=='hotspot' else None,steps=100 if workload=='kmeans' else 128,
                  workers=4,native_mode='fixed' if workload=='kmeans' else 'timing',integration=asdict(cfg),golden=None,
                  feedback_batch=1 if legacy else batch,telemetry=False,binary=str((baseline if legacy else build)/('atlas_'+workload)))
        if not legacy:
            case['runtime_version']='runtime-2.0'
            if workload=='kmeans':case.update(validation_mode=validation,reduction=reduction,transfer_mode=transfer)
            else:case['communication_mode']=communication
        cases.append(case);return case
    def collect(case):
        result=execute(case['binary'],root/case['dataset'],root/case['field'] if case['field'] else None,case,None)
        save_new(root/case['record'],result)
        try:checked=assess(root,case,datasets[case['dataset']])
        except (ValueError,KeyError,TypeError,IndexError) as error:
            print(json.dumps(dict(run=case['integration']['run_id'],name=case['name'],status='failed',failure=result.get('failure') or str(error))),flush=True)
            if case['stage']=='tuning':raise ValueError('Tuning failed; no selection frozen') from error
            return None
        end=json.loads(result['stdout'].splitlines()[-1])
        print(json.dumps(dict(run=case['integration']['run_id'],name=case['name'],workload=case['workload'],block=case['block'],status=checked['status'])),flush=True)
        return (end['loop_elapsed_ns']+end.get('gather_ns',0))/1e9
    # Pilot inputs are separate. All pilot costs and choices are retained.
    pilot=[]
    for workload in ('kmeans','hotspot'):
        for size in (0,1):
            for block in range(2):
                shares=[0.,.25,.5,.75,1.];rng.shuffle(shares)
                for share in shares:pilot.append(make(workload,size,'tuning',block,f'pilot-{share}',share=share))
    save_new(root/'pilot-plan.json',dict(cases=pilot,datasets=datasets,seed=20261102))
    scores={}
    for case in pilot:
        key=(case['workload'],case['size'],case['integration']['initial_cpu_items'])
        scores.setdefault(key,[]).append(collect(case))
    for workload,size in {(w,s) for w,s,c in scores}:
        counts=[c for w,s,c in scores if (w,s)==(workload,size)]
        selections[f'{workload}-{size}']=min(counts,key=lambda c:(statistics.mean(scores[workload,size,c]),c))
    for workload in ('kmeans','hotspot'):
        for size in (0,1):
            meta=datasets[f'datasets/{workload}-{size}-evaluation.data'];n=meta['n'] if workload=='kmeans' else meta['rows']
            variants=[('gpu_only',dict(share=0)),('cpu_only',dict(share=1)),('static50',{}),
                      ('tuned_static',dict(share=selections[f'{workload}-{size}']/n)),
                      ('cost_aware',dict(policy='cost_aware_v1',batch=4)),
                      ('feedback_every_step',dict(batch=1))]
            if workload=='kmeans':variants.extend([('atomic_verified',dict(validation='verify',reduction='atomic',transfer='diagnostic',batch=1)),
                                                   ('atomic_timing',dict(reduction='atomic',transfer='diagnostic',batch=1)),
                                                   ('hierarchical_diagnostic',dict(transfer='diagnostic',batch=1)),
                                                   ('batched_atomic',dict(reduction='atomic',transfer='diagnostic')),
                                                   ('streamed_atomic',dict(reduction='atomic'))])
            else:variants.append(('overlap50',dict(communication='overlap')))
            if baseline:variants.append(('legacy_static50',dict(legacy=True)))
            for block in range(blocks):
                order=list(variants);rng.shuffle(order)
                for name,options in order:make(workload,size,'evaluation',block,name,**options)
    save_new(root/'plan.json',dict(cases=cases,datasets=datasets,selections=selections,blocks=blocks,seed=20261102,
                                 binary_hashes={c['binary']:sha(Path(c['binary'])) for c in cases},settings='unchanged automatic clocks; no thermal isolation claim'))
    sources=[p for folder in ('src','native','tests') for p in Path(folder).rglob('*') if p.suffix in ('.py','.cu','.cpp','.hpp')]
    save_new(root/'source_hashes.json',{p.as_posix():sha(p) for p in sorted(sources)})
    for case in cases[len(pilot):]:collect(case)
    result=summarize(root);save_new(root/'summary.json',result)
    save_new(root/'checksums.json',{p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file()})
    replay(root)
    return {k:v for k,v in result.items() if k!='results'}
