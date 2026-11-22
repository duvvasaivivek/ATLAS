"""Independent references and endpoints for the explicit GPU-first dispatch."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from greengpu import kmeans_reference as kr,hotspot_reference as hr

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    data=kr.clustered(1025,8,8,811)
    kr.write_dataset(root/'input.kmdat',*data)
    t,p,provenance=hr.generated(33,35,811)
    hr.write_input(root/'input.hsdat',33,35,t,p,provenance)
    for order in ('cpu_first','gpu_first'):
        flags=['gpu-first'] if order=='gpu_first' else []
        for streamed in (False,True):
            argv=[sys.argv[1],str(root/'input.kmdat'),'1025,0,1,1024,512,0','6','4','fixed','verify',*flags]
            if streamed:argv+=['reduced','streamed']
            result=subprocess.run(argv,capture_output=True,text=True,timeout=45,check=True)
            rows=[json.loads(s) for s in result.stdout.splitlines()]
            assert rows[0]['dispatch_order']==order
            assert rows[-1]['independent_validation']=='passed'
            golden=kr.trajectory(*data[:2],6,'fixed')[-1]
            assert rows[-1]['final_labels']==golden['labels']
        field=root/(order+'.hstemp')
        result=subprocess.run([sys.argv[2],str(root/'input.hsdat'),'33,0,1,32,16,0','6','4','verify',str(field),*flags],
                              capture_output=True,text=True,timeout=45,check=True)
        rows=[json.loads(s) for s in result.stdout.splitlines()]
        assert rows[0]['dispatch_order']==order and rows[-1]['status']=='passed'
        expected=hr.trajectory(t,p,33,35,6)[-1]
        actual=hr.read_grid(field)[2]
        assert all(hr.close(a,b) for a,b in zip(actual,expected))
print('Dispatch endpoints, ownership migration, independent references, and both kmeans transfer modes pass')
