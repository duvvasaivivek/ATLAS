"""Verify and re-evaluate a retained dispatch experiment without launching work."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from greengpu.concurrency import analyze_profile
from greengpu.frequency_experiment import sha
from greengpu.phase9 import assess


def replay(root):
    root=Path(root)
    hashes=json.loads((root/'checksums.json').read_text())
    files={p.name:p for p in root.iterdir() if p.is_file() and p.name!='checksums.json'}
    if files.keys()!=hashes.keys() or any(p.is_symlink() or sha(p)!=hashes[n] for n,p in files.items()):
        raise ValueError('Dispatch archive integrity failure')
    plan=json.loads((root/'plan.json').read_text())
    results=[]
    for case in plan['cases']:
        workload=case['workload'];name=Path(case['record']).stem
        checked=assess(root,case,plan['datasets'][case['dataset']])
        kernel=('kmeans_assign_reduced' if case.get('reduction')=='hierarchical' else 'kmeans_assign') if workload=='kmeans' else 'hotspot_step'
        profile=analyze_profile(root/case['profile'],256,kernel,
                                workload+'_iteration',workload+'_cpu_compute')
        if profile!=json.loads((root/(name+'.profile.json')).read_text()):
            raise ValueError('Dispatch profile replay mismatch')
        results.append(dict(workload=workload,block=case['block'],dispatch_order=case['dispatch_order'],
                            correct=checked['status'],overlap=sum(r['demonstrated'] for r in profile['iterations']),
                            iterations=256,profile_status=profile['status'],active_s=checked['active_s'],
                            command_s=checked['command_s'],ownership_changes=checked['ownership_changes']))
        if 'build' in case:results[-1]['build']=case['build']
    summary=dict(results=results,hardware_setter_calls=0,strict_gate_passed=all(r['overlap']==256 for r in results if r.get('build')!='baseline'))
    if summary!=json.loads((root/'summary.json').read_text()):raise ValueError('Dispatch summary replay mismatch')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    print(json.dumps(dict(replay='passed',**replay(parser.parse_args().archive)),indent=2))
