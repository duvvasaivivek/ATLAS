"""Independent kmeans oracle and verify/deferred-validation equivalence."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from greengpu.kmeans_reference import clustered,write_dataset,trajectory,close,fixtures

binary=sys.argv[1]
with tempfile.TemporaryDirectory() as folder:
    inputs=dict(fixtures())
    inputs.update({f'random-{n}':clustered(n,d,k,20261102) for n,d,k in ((17,3,4),(513,8,8),(257,32,32),(2053,8,8),(1025,32,32))})
    for name,(points,centers,provenance) in inputs.items():
        n=len(points)
        path=Path(folder)/f"{name}.kmdat"
        write_dataset(path,points,centers,provenance)
        schedule=[n,n//2,0,n,1,n-1,0]
        gold=trajectory(points,centers,len(schedule),"fixed")[-1]
        for mode,options in (("verify",[]),("timing",[]),("verify",["reduced"]),("timing",["reduced","streamed"])):
            process=subprocess.run([binary,str(path),','.join(map(str,schedule)),str(len(schedule)),"4","fixed",mode,*options],
                                   text=True,capture_output=True,timeout=60)
            assert process.returncode==0,process.stderr
            records=[json.loads(line) for line in process.stdout.splitlines()]
            end=records[-1]
            assert end['independent_validation']=='passed'
            assert end['final_reference_workers_requested']==(4 if mode=='timing' and n>=1024 else 1)
            assert end['final_labels']==gold['labels']
            assert all(close(a,b) for a,b in zip(end['final_centroids'],gold['centroids']))
            assert all(r['validation']==('checked' if r['warmup'] or mode=='verify' else 'deferred') for r in records[1:-1])
print('Kmeans verification and timing modes match independent trajectories')
