"""Persistent native feedback, deterministic decision replay, and Phase 5 evidence."""

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import tempfile
import threading
import time

from .allocation import Config, POLICIES, initial_state, transition, balance_trace
from .allocation_scenarios import suite
from .capabilities import collect
from .experiment import distribution, save_new
from .kmeans import assess, replay as replay_kmeans
from .kmeans_reference import fixtures, clustered, write_dataset, trajectory, close
from .supervision import Supervisor


def observation(row, run_id):
    release = row['release_ns']
    return dict(run_id=run_id, epoch=0, iteration=row['index'], cpu_items=row['cpu_items'],
                t_cpu=(row['cpu_end_ns']-release)/1e9 if row['cpu_items'] else None,
                t_gpu=(row['gpu_end_ns']-release)/1e9 if row['gpu_items'] else None,
                valid=row['status']=='passed' and all(row[k]==0 for k in ('label_errors','count_errors','numeric_errors')),
                common_origin=True)


def feedback(binary, dataset, case, timeout=120):
    """One process per complete trial; watchdog bounds broken protocols and EOF."""
    config = Config(case['policy'],case['n'])
    state = initial_state(case['run_id'])
    argv = [str(Path(binary).resolve()),str(Path(dataset).resolve()),str(case['initial_cpu_items']),
            str(case['max_iterations']),str(case['workers_requested']),case['mode'],'feedback']
    lines, decisions = [], []
    started = time.monotonic_ns()
    with tempfile.TemporaryFile(mode='w+t') as errors:
        supervisor = Supervisor(argv, errors, timeout)
        process = supervisor.process
        expired = supervisor.expired
        failure = None
        try:
            for line in supervisor.lines():
                lines.append(line)
                row = json.loads(line)
                if row.get('kind') != 'iteration' or row['warmup']:
                    continue
                if row['status'] != 'passed':
                    raise ValueError('Native correctness failure during feedback')
                last = row['index']+1 >= case['max_iterations'] or (case['mode']=='converge' and row['converged'])
                if not last:
                    begin = time.monotonic_ns()
                    decision = transition(config,state,observation(row,case['run_id']))
                    duration = time.monotonic_ns()-begin
                    state = decision['post_state']
                    decisions.append(dict(decision=decision,controller_ns=duration))
                    process.stdin.write(str(decision['cpu_items'])+'\n');process.stdin.flush()
            process.wait(timeout=5)
        except (ValueError,OSError,subprocess.TimeoutExpired) as error:
            failure = str(error)
        finally:
            supervisor.close()
        errors.seek(0);stderr=errors.read()
    return dict(argv=argv,returncode=process.returncode,
                status='ok' if process.returncode==0 and not failure and not expired.is_set() else 'failed',
                stdout=''.join(lines),stderr=stderr,decisions=decisions,
                timed_out=expired.is_set(),failure=failure,elapsed_ns=time.monotonic_ns()-started,
                configuration=asdict(config))


def assess_feedback(case, metadata, process, golden=None):
    if process['status'] != 'ok':
        raise ValueError(f"Feedback command failed: {process.get('failure')} {process.get('stderr')}")
    rows = [json.loads(line) for line in process['stdout'].splitlines()]
    cfg, end = rows[0],rows[-1]
    measured = [r for r in rows if r.get('kind')=='iteration' and not r['warmup']]
    if not measured or cfg.get('feedback') is not True or measured[0]['cpu_items'] != case['initial_cpu_items']:
        raise ValueError('Missing feedback configuration or wrong initial split')
    counts = [r['cpu_items'] for r in measured]
    schedule = counts + [counts[-1]]*(case['max_iterations']-len(counts))
    native_case = dict(case,cpu_counts=schedule,stage=case['stage'])
    result = assess(native_case,metadata,process,golden)
    if result['status'] != 'passed':
        raise ValueError(result.get('reason'))
    policy = Config(case['policy'],case['n'])
    if process['configuration'] != json.loads(json.dumps(asdict(policy))):
        raise ValueError('Controller configuration differs from frozen defaults')
    state = initial_state(case['run_id'])
    if len(process['decisions']) != len(measured)-1:
        raise ValueError('Missing/extra feedback decisions')
    observations = [observation(r,case['run_id']) for r in measured]
    for i, retained in enumerate(process['decisions']):
        expected = transition(policy,state,observations[i])
        if retained['decision'] != expected or measured[i+1]['cpu_items'] != expected['cpu_items']:
            raise ValueError('Decision replay or realized next split mismatch')
        if type(retained['controller_ns']) is not int or retained['controller_ns'] < 0:
            raise ValueError('Invalid controller duration')
        state = expected['post_state']
    control = 0
    for i,row in enumerate(measured):
        value = row['control_ns']
        if type(value) is not int or value < 0 or (i==0 and value != 0):
            raise ValueError('Invalid feedback interval')
        if i and value > row['release_ns']-measured[i-1]['validation_end_ns']:
            raise ValueError('Control interval exceeds enclosing gap')
        control += value
    active = sum(r['end_ns']-r['release_ns'] for r in measured)
    if end['control_ns'] != control or end['active_and_control_ns'] != active+control or active+control > end['loop_elapsed_ns']:
        raise ValueError('Feedback timing total mismatch')
    reasons = [d['decision']['reason'] for d in process['decisions']]
    diagnostic = balance_trace(observations)
    signs = [1 if o['t_cpu']>o['t_gpu'] else -1 if o['t_cpu']<o['t_gpu'] else 0
             for o in observations if o['t_cpu'] is not None and o['t_gpu'] is not None]
    reversals = sum(a*b<0 for a,b in zip(signs,signs[1:]))
    return dict(case=case,status='passed',iterations=len(measured),active_s=active/1e9,
                control_s=control/1e9,active_and_control_s=(active+control)/1e9,
                loop_elapsed_s=end['loop_elapsed_ns']/1e9,command_elapsed_s=process['elapsed_ns']/1e9,
                controller_compute_s=sum(d['controller_ns'] for d in process['decisions'])/1e9,
                # controller_compute_s is nested in control_s, never added a second time.
                points_uploaded=sum(r['h2d_point_bytes'] for r in measured)//(metadata['d']*8),
                migration_points=sum(r['ownership_changed_points'] for r in measured),
                h2d_point_bytes=sum(r['h2d_point_bytes'] for r in measured),
                shares=[c/case['n'] for c in counts],observations=observations,
                diagnostic=diagnostic,reasons=dict(Counter(reasons)),sign_reversals=reversals,
                allocation_outcome=('static_baseline' if case['policy']=='static' else
                    'sustained_balance_at_end' if diagnostic['balanced_at_end'] else
                    (reasons[-1] if reasons and reasons[-1].startswith(('stop_','no_','invalid_','unsupported','budget_'))
                     else 'iteration_budget_without_sustained_balance')),
                oscillation_observed=reversals>=4 and not diagnostic['balanced_at_end'],
                final_label_hash=result['final_label_hash'],final_centroids=result['final_centroids'],
                final_objective=result['final_objective'],workload_stop_reason=result['stop_reason'],
                trajectory_hashes=[r['label_hash'] for r in measured],energy_j=None)


def summarize(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    cases=json.loads((root/'plan.json').read_text())
    results=[]
    for case in cases:
        process=json.loads((root/case['record']).read_text())
        golden=json.loads((root/case['golden']).read_text()) if case.get('golden') else None
        results.append(assess_feedback(case,manifest['datasets'][case['dataset']],process,golden))
    for dataset,mode in sorted({(r['case']['dataset'],r['case']['mode']) for r in results}):
        group=[r for r in results if (r['case']['dataset'],r['case']['mode'])==(dataset,mode)]
        first=group[0]
        if any(r['trajectory_hashes'] != first['trajectory_hashes'] or not close(r['final_objective'],first['final_objective'])
               or not all(close(a,b) for a,b in zip(r['final_centroids'],first['final_centroids'])) for r in group):
            raise ValueError('Changing allocations altered the clustering trajectory/output')
    comparisons=[]
    bench=[r for r in results if r['case']['stage']=='heldout']
    for name in sorted({r['case']['name'] for r in bench}):
        group=[r for r in bench if r['case']['name']==name]
        comparisons.append(dict(name=name,trials=len(group),
            active_and_control_s=distribution([r['active_and_control_s'] for r in group]),
            loop_elapsed_s=distribution([r['loop_elapsed_s'] for r in group]),
            controller_compute_s=distribution([r['controller_compute_s'] for r in group]),
            balanced_at_end=sum(r['diagnostic']['balanced_at_end'] for r in group),
            outcomes=dict(Counter(r['allocation_outcome'] for r in group)),energy_j=None))
    return dict(schema_version='5.0',implementation_status='passed',
        full_phase_gate='blocked_energy_and_fixed_capacity_qualification',results=results,comparisons=comparisons,
        measured_iterations=sum(r['iterations'] for r in results),
        same_input_trajectory_equivalence=True,energy_j=None,hardware_settings_changed=False,
        timing_scope='Complete iteration work plus feedback/export/IPC intervals; controller compute is nested, not additive. Loop elapsed includes validation.',
        clock_policy='unchanged automatic platform policy; fixed effective capacity not established',
        statistics='three randomized blocks by default; exploratory trial-level distributions, no significance claim')


def run(output,binary,workers=4,blocks=3):
    if platform.system()!='Linux':raise ValueError('Run in WSL/Linux')
    if not 1<=workers<=min(8,max(1,(os.cpu_count() or 1)-2)) or not 2<=blocks<=10:
        raise ValueError('Invalid worker/block budget')
    binary=Path(binary).resolve(strict=True);output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    try:
        prior=Path('results/phase4/2026-10-31-final-v2')
        p4=replay_kmeans(prior)
        if p4['implementation_status']!='passed':raise ValueError('Phase 4 prerequisite failed')
        # Estimate a near-balance starting point only from earlier tuning observations.
        tuning=[r for r in p4['results'] if r['case']['stage'] in ('coarse_tuning','refine_tuning')]
        interior=[r for r in tuning if 0<r['case']['cpu_counts'][0]<r['case']['n']]
        def error(r):
            row=r['iterations'][-1];a=row['cpu_end_ns']-row['release_ns'];b=row['gpu_end_ns']-row['release_ns']
            return abs(a-b)/max(a,b)
        near=min(interior,key=error)['case']['cpu_counts'][0]/interior[0]['case']['n']
        starts=sorted(set((.15,.5,.9,near)))
        collect(output/'environment','/usr/local/cuda-13.1/bin/nvcc',None)
        (output/'environment/raw/CMakeCache.txt').write_bytes((binary.parent/'CMakeCache.txt').read_bytes())
        datasets={name:fixtures()[name] for name in ('single','ties','empty','awkward','multidimensional')}
        datasets['heldout']=clustered(32771,8,8,6505)
        (output/'datasets').mkdir();(output/'golden').mkdir()
        metadata={name:write_dataset(output/'datasets'/(name+'.kmdat'),*data) for name,data in datasets.items()}
        save_new(output/'manifest.json',dict(schema_version='5.0',datasets=metadata,workers=workers,blocks=blocks,
            binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            source_hashes=json.loads((output/'environment/source_hashes.json').read_text()),
            policies={p:asdict(Config(p,32771)) for p in (*POLICIES,'static')},
            phase4_summary_sha256=hashlib.sha256((prior/'summary.json').read_bytes()).hexdigest(),
            starts=starts,near_balance_source='minimum final-iteration relative imbalance among Phase 4 tuning trials',
            static_selected_cpu_items=p4['tuning_selected_cpu_items'],fixed_iterations=24,energy_qualification='blocked_phase2',
            clock_policy='unchanged automatic policy; no fixed-capacity guarantee',wall_timeout_s=120))
        save_new(output/'synthetic.json',suite())
        cases=[]
        def add(dataset,policy,count,mode,budget,stage,name=None,block=None,golden=None):
            i=len(cases)
            cases.append(dict(run_id=f'phase5-{i:03d}',record=f'trial-{i:03d}.json',dataset=dataset,
                policy=policy,n=metadata[dataset]['n'],initial_cpu_items=count,mode=mode,max_iterations=budget,
                workers_requested=workers,stage=stage,name=name or policy,block=block,golden=golden))
        for name,(points,centers,_) in datasets.items():
            if name=='heldout':continue
            golden=f'golden/{name}-fixed.json';save_new(output/golden,trajectory(points,centers,8))
            n=len(points)
            for policy in POLICIES:
                for cut in sorted({0,n//2,n}):add(name,policy,cut,'fixed',8,'correctness',golden=golden)
        points,centers,_=datasets['awkward'];golden='golden/awkward-converge.json'
        save_new(output/golden,trajectory(points,centers,30,'converge'))
        for policy in POLICIES:
            for share in (.15,.5,.9):add('awkward',policy,Config(policy,len(points)).nearest(share),'converge',30,'convergence',golden=golden)
        n=metadata['heldout']['n'];candidates=[]
        for policy in POLICIES:
            for share in starts:
                count=Config(policy,n).nearest(share)
                candidates.append((policy,count,f'{policy}@{count}/{n}'))
        candidates.extend(('static',cut,name) for cut,name in ((0,'gpu_only'),(n,'cpu_only'),(n//2,'static_half'),
            (p4['tuning_selected_cpu_items'],'static_tuned')))
        rng=random.Random(5505)
        for block in range(blocks):
            order=list(candidates);rng.shuffle(order)
            for policy,cut,name in order:add('heldout',policy,cut,'fixed',24,'heldout',name,block)
        save_new(output/'plan.json',cases)
        for case in cases:
            process=feedback(binary,output/'datasets'/(case['dataset']+'.kmdat'),case)
            save_new(output/case['record'],process)
            golden=json.loads((output/case['golden']).read_text()) if case['golden'] else None
            result=assess_feedback(case,metadata[case['dataset']],process,golden)
            print(json.dumps(dict(run=case['run_id'],stage=case['stage'],policy=case['policy'],status=result['status'])),flush=True)
        summary=summarize(output);save_new(output/'summary.json',summary)
        save_new(output/'checksums.json',{p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.rglob('*')) if p.is_file()})
        return summary
    except BaseException as error:
        save_new(output/'failure.json',dict(status='failed',reason=f'{type(error).__name__}: {error}'))
        raise


def replay(root):
    root=Path(root)
    expected=json.loads((root/'checksums.json').read_text())
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p!=root/'checksums.json'}
    if actual!=set(expected):raise ValueError('Archive inventory changed')
    for name,value in expected.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=value:
            raise ValueError('Archive checksum mismatch: '+name)
    if json.loads((root/'synthetic.json').read_text())!=suite():
        raise ValueError('Synthetic transition replay changed')
    return summarize(root)
