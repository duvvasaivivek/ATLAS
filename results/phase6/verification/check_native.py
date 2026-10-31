"""Retained verification driver; run once from the repository root in WSL."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,'src')
from greengpu.capabilities import command
from greengpu.experiment import save_new
from greengpu.hotspot import execute,assess
from greengpu.hotspot_reference import generated,write_input,trajectory
from greengpu.kmeans import assess as assess_kmeans
from greengpu.kmeans_reference import clustered,write_dataset,trajectory as kmeans_trajectory

root=Path('results/phase6/verification')
binary=Path('build/phase6/atlas_hotspot').resolve()
fixture=Path('results/phase6/2026-10-31-final/datasets/irregular.hsdat').resolve()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for tool in ('memcheck','racecheck'):
    argv=['/usr/local/cuda-13.1/bin/compute-sanitizer','--tool',tool,'--error-exitcode','86',str(binary),str(fixture),
          '19','6','4','verify',str((root/(tool+'.hstemp')).resolve()),'feedback']
    input_text='9\n0\n19\n1\n18\n'
    proc=subprocess.run(argv,input=input_text,text=True,capture_output=True,timeout=120)
    save_new(root/(tool+'.json'),dict(argv=argv,stdin=input_text,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,binary_sha256=sha(binary)))
    assert proc.returncode==0,proc.stderr
    print(tool,'\n'.join(line for line in (proc.stdout+proc.stderr).splitlines() if 'SUMMARY' in line),flush=True)

for name,text in (('eof',''),('illegal','20\n'),('malformed','x\n')):
    argv=[str(binary),str(fixture),'19','2','4','verify',str((root/(name+'.hstemp')).resolve()),'feedback']
    proc=subprocess.run(argv,input=text,text=True,capture_output=True,timeout=20)
    save_new(root/(name+'.json'),dict(argv=argv,stdin=text,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr))
    assert proc.returncode!=0

case=dict(run_id='watchdog',rows=19,cols=23,counts=[9],steps=8,workers=4,mode='verify',policy='static',timeout_s=.001)
record=execute(binary,fixture,root/'watchdog.hstemp',case)
save_new(root/'watchdog.json',record)
assert record['timed_out'] and record['status']=='failed' and record['returncode']!=0

# Matched ownership path, different workloads: compare traffic, never raw speeds.
folder=root/'cache-comparison';folder.mkdir()
rows,cols=19,23;counts=[19,9,0,19,0];steps=len(counts)
temp,power,origin=generated(rows,cols,772)
meta=write_input(folder/'thermal.hsdat',rows,cols,temp,power,origin)
case=dict(run_id='cache-comparison',rows=rows,cols=cols,counts=counts,steps=steps,workers=4,mode='timing',policy='scripted',stage='comparison')
raw=execute(binary,folder/'thermal.hsdat',folder/'thermal.hstemp',case)
save_new(folder/'hotspot-command.json',raw)
hot=assess(case,meta,raw,folder/'thermal.hstemp',trajectory(temp,power,rows,cols,steps))
save_new(folder/'hotspot-summary.json',hot)
points,centers,origin=clustered(rows*cols,3,4,772)
meta=write_dataset(folder/'kmeans.kmdat',points,centers,origin)
cuts=[r*cols for r in counts]
kbinary=Path('build/phase6/atlas_kmeans').resolve()
raw=command([str(kbinary),str((folder/'kmeans.kmdat').resolve()),','.join(map(str,cuts)),str(steps),'4','fixed'],timeout=60)
save_new(folder/'kmeans-command.json',raw)
kcase=dict(dataset='comparison',n=rows*cols,cpu_counts=cuts,max_iterations=steps,workers_requested=4,mode='fixed',stage='comparison')
k=assess_kmeans(kcase,meta,raw,kmeans_trajectory(points,centers,steps))
assert k['status']=='passed',k
save_new(folder/'kmeans-summary.json',k)
grid=rows*cols
point_h2d=sum(r['h2d_point_bytes'] for r in k['iterations'])
temp_h2d=hot['traffic']['initial_h2d_bytes']+hot['traffic']['migration_h2d_bytes']
assert point_h2d==grid*3*8
assert temp_h2d==2*grid*8
assert hot['traffic']['migration_d2h_bytes']==grid*8
result=dict(cpu_row_schedule=counts,grid_cells=grid,kmeans_points=grid,
    hotspot_temperature_h2d_bytes=temp_h2d,hotspot_temperature_d2h_migration_bytes=hot['traffic']['migration_d2h_bytes'],
    hotspot_halo_h2d_bytes=hot['traffic']['halo_h2d_bytes'],hotspot_halo_d2h_bytes=hot['traffic']['halo_d2h_bytes'],
    kmeans_point_h2d_bytes=point_h2d,kmeans_point_d2h_bytes=0,
    temperature_field_h2d_equivalents=temp_h2d/(grid*8),immutable_point_set_h2d_equivalents=point_h2d/(grid*3*8),
    note='Kmeans also downloads memberships/aggregates; different work and datum sizes prohibit a speed or total-byte comparison from these counters alone.',
    hotspot_binary_sha256=sha(binary),kmeans_binary_sha256=sha(kbinary))
save_new(folder/'comparison.json',result)
print(json.dumps(result,indent=2),flush=True)
