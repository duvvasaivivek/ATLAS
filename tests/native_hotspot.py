"""CTest integration: scripted/feedback equivalence and independent Python field."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from greengpu.hotspot_reference import trajectory,read_grid,close

binary,dataset,mode=sys.argv[1:]
data=Path(dataset).read_bytes();magic,rows,cols=struct.unpack_from('<8sII',data)
assert magic==b'ATLSHS01'
n=rows*cols;values=struct.unpack_from(f'<{2*n}d',data,16)
counts=[rows,1,0,rows,rows-1,0]*5
gold=trajectory(list(values[:n]),list(values[n:]),rows,cols,len(counts))[-1]
with tempfile.TemporaryDirectory() as folder:
    fields=[]
    for feedback in (False,True):
        out=Path(folder)/('feedback' if feedback else 'scripted')
        args=[binary,dataset,str(counts[0]) if feedback else ','.join(map(str,counts)),str(len(counts)),'4',mode,str(out)]
        if feedback:args.append('feedback')
        p=subprocess.run(args,input=''.join(f'{c}\n' for c in counts[1:]) if feedback else '',text=True,capture_output=True,timeout=60)
        assert p.returncode==0,p.stderr+p.stdout
        records=[json.loads(line) for line in p.stdout.splitlines()]
        assert records[-1]['status']=='passed'
        actual=read_grid(out)[2];assert all(close(a,b) for a,b in zip(actual,gold))
        fields.append(actual)
    assert fields[0]==fields[1]
print('HotSpot scripted/feedback outputs match independent Python reference:',mode)
