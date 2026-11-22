"""Counterbalanced same-input dispatch profiles; no clock changes or relaxed gates."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from greengpu import kmeans_reference as kr, hotspot_reference as hr
from greengpu.actuation import NvmlClockBackend
from greengpu.concurrency import analyze_profile
from greengpu.experiment import save_new
from greengpu.frequency_experiment import sha
from greengpu.integration import Config
from greengpu.integration_runner import execute
from greengpu.phase9 import assess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--baseline-build',type=Path,help='Compare the same dispatch order on this build in every block')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--blocks',type=int,default=3)
    parser.add_argument('--orders',nargs='+',choices=('cpu_first','gpu_first'),
                        default=('cpu_first','gpu_first'))
    parser.add_argument('--fixed',action='store_true')
    parser.add_argument('--optimized-kmeans',action='store_true')
    args=parser.parse_args()
    if not 1<=args.blocks<=10:parser.error('Require 1..10 blocks')
    if len(set(args.orders))!=len(args.orders):parser.error('Select each dispatch order once')
    builds={'candidate':args.build, 'baseline':args.baseline_build} if args.baseline_build else {'candidate':args.build}
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    datasets={}
    datasets['kmeans.data']=kr.write_dataset(root/'kmeans.data',*kr.clustered(32771,8,8,9003))
    t,p,prov=hr.generated(513,515,9013)
    datasets['hotspot.data']=hr.write_input(root/'hotspot.data',513,515,t,p,prov)
    cases=[];rng=random.Random(20261110)
    for block in range(args.blocks):
        for workload in ('kmeans','hotspot'):
            order=[(build,dispatch) for build in builds for dispatch in args.orders];rng.shuffle(order)
            for build,dispatch in order:
                name=f'{workload}-{block}-{dispatch}'+(f'-{build}' if args.baseline_build else '')
                n=datasets[workload+'.data']['n' if workload=='kmeans' else 'rows']
                cfg=Config(name,n,n//2,mode='allocation_only',allocation_policy='static' if args.fixed else 'robust_proportional_epoch_v1',
                           allocation_interval_ns=50_000_000,allocation_iterations=4,cadence_profile='batched_epochs_v1')
                case=dict(workload=workload,dataset=workload+'.data',record=name+'.json',
                          field=name+'.hstemp' if workload=='hotspot' else None,workers=4,steps=256,
                          stage='profile',native_mode='fixed' if workload=='kmeans' else 'timing',
                          integration=asdict(cfg),runtime_version='runtime-2.0',feedback_batch=16,
                          validation_mode='timing',dispatch_order=dispatch,block=block,golden=None,
                          profile=name+'.sqlite')
                if workload=='kmeans' and args.optimized_kmeans:
                    case.update(reduction='hierarchical',transfer_mode='streamed')
                if args.baseline_build:
                    case['build']=build
                cases.append(case)
    binaries={(build,w):(path/('atlas_'+w)).resolve() for build,path in builds.items() for w in ('kmeans','hotspot')}
    hashes={f'{build}:{w}':sha(path) for (build,w),path in binaries.items()}
    save_new(root/'plan.json',dict(cases=cases,datasets=datasets,seed=20261110,
                                 binary_hashes=hashes,clock_policy='unchanged automatic'))
    backend=NvmlClockBackend();results=[]
    try:
        save_new(root/'capabilities.json',backend.discover())
        for case in cases:
            name=Path(case['record']).stem;w=case['workload']
            prefix=('nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--export=sqlite',f'--output={root/name}')
            record=execute(binaries[case.get('build','candidate'),w],root/case['dataset'],root/case['field'] if case['field'] else None,case,backend,prefix)
            save_new(root/case['record'],record)
            checked=assess(root,case,datasets[case['dataset']])
            kernel=('kmeans_assign_reduced' if case.get('reduction')=='hierarchical' else 'kmeans_assign') if w=='kmeans' else 'hotspot_step'
            profile=analyze_profile(root/case['profile'],256,kernel,w+'_iteration',w+'_cpu_compute')
            save_new(root/(name+'.profile.json'),profile)
            row=dict(workload=w,block=case['block'],dispatch_order=case['dispatch_order'],correct=checked['status'],
                     overlap=sum(r['demonstrated'] for r in profile['iterations']),iterations=256,
                     profile_status=profile['status'],active_s=checked['active_s'],command_s=checked['command_s'],
                     ownership_changes=checked['ownership_changes'])
            if 'build' in case:row['build']=case['build']
            results.append(row);print(json.dumps(row),flush=True)
        save_new(root/'summary.json',dict(results=results,hardware_setter_calls=0,
                                         strict_gate_passed=all(r['overlap']==256 for r in results if r.get('build')!='baseline')))
        save_new(root/'checksums.json',{p.name:sha(p) for p in sorted(root.iterdir()) if p.is_file()})
    finally:backend.close()


if __name__=='__main__':main()
