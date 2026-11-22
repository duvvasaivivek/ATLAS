"""CTest integration: scripted/feedback equivalence and independent Python field."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from greengpu.hotspot_reference import trajectory,read_grid,close,generated,write_input

binary,dataset,mode=sys.argv[1:]
def check(dataset,folder):
    data=Path(dataset).read_bytes();magic,rows,cols=struct.unpack_from('<8sII',data)
    assert magic==b'ATLSHS01'
    n=rows*cols;values=struct.unpack_from(f'<{2*n}d',data,16)
    counts=[rows,1,0,rows,rows-1,0]*5
    gold=trajectory(list(values[:n]),list(values[n:]),rows,cols,len(counts))[-1]
    fields=[]
    for feedback,overlap in ((False,False),(True,False),(False,True),(True,True)):
        out=Path(folder)/(('feedback' if feedback else 'scripted')+str(overlap))
        args=[binary,dataset,str(counts[0]) if feedback else ','.join(map(str,counts)),str(len(counts)),'4',mode,str(out)]
        if feedback:args.append('feedback')
        if overlap:args.append('overlap')
        p=subprocess.run(args,input=''.join(f'{c}\n' for c in counts[1:]) if feedback else '',text=True,capture_output=True,timeout=60)
        assert p.returncode==0,p.stderr+p.stdout
        records=[json.loads(line) for line in p.stdout.splitlines()]
        assert records[-1]['status']=='passed'
        assert records[-1]['final_reference_workers_requested']==(4 if mode=='timing' and n>=1024 else 1)
        actual=read_grid(out)[2];assert all(close(a,b) for a,b in zip(actual,gold))
        fields.append(actual)
    assert all(field==fields[0] for field in fields)
with tempfile.TemporaryDirectory() as folder:
    root=Path(folder);small=root/'small';large=root/'large';small.mkdir();large.mkdir()
    check(dataset,small)
    temp,power,provenance=generated(33,35,20261102)
    path=root/'parallel-reference.hsdat';write_input(path,33,35,temp,power,provenance)
    check(str(path),large)
print('HotSpot scripted/feedback outputs match independent Python reference:',mode)
