"""CUDA sanitizer checks across endpoint migration, partial blocks and maximum k/d."""
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from greengpu import kmeans_reference as kr,hotspot_reference as hr


def main():
    build=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve()
    gpu_first=len(sys.argv)>3 and sys.argv[3]=='gpu-first'
    if len(sys.argv)>3 and not gpu_first:raise ValueError('Unknown dispatch option')
    output.mkdir(parents=True,exist_ok=False)
    kr.write_dataset(output/'input.kmdat',*kr.clustered(1025,32,32,20261102))
    temp,power,provenance=hr.generated(33,35,20261102)
    hr.write_input(output/'input.hsdat',33,35,temp,power,provenance)
    results=[]
    for tool in ('memcheck','racecheck','synccheck','initcheck'):
        for workload in ('kmeans','hotspot'):
            arguments=([str(output/'input.kmdat'),'1025,512,0,1,1024,0','6','4','fixed','timing','reduced','streamed']
                       if workload=='kmeans' else [str(output/'input.hsdat'),'33,16,0,1,32,0','6','4','timing',str(output/f'{tool}.hstemp'),'overlap'])
            if gpu_first:
                if workload=='hotspot':arguments.remove('overlap')
                arguments.append('gpu-first')
            argv=['/usr/local/cuda-13.1/bin/compute-sanitizer','--tool',tool,'--error-exitcode','1',str(build/f'atlas_{workload}'),*arguments]
            with (output/f'{workload}-{tool}.txt').open('x') as stream:
                process=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
            row=dict(workload=workload,tool=tool,returncode=process.returncode,argv=argv)
            results.append(row);print(json.dumps(row),flush=True)
    (output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    if any(row['returncode'] for row in results):raise SystemExit(1)


if __name__=='__main__':main()
