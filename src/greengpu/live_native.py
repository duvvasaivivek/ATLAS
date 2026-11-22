"""Armed native workload adapter for the separately qualified live controller."""
import json
from pathlib import Path
import shutil
import struct
import time

from . import kmeans_reference as kr, hotspot_reference as hr
from .actuation import ActuationBlocked
from .experiment import save_new
from .frequency_experiment import sha
from .integration_runner import runtime_observation
from .supervision import Supervisor


def pci_identity(value):
    try:
        domain,bus,tail=value.lower().split(':')
        slot,function=tail.split('.')
        return tuple(int(v,16) for v in (domain,bus,slot,function))
    except (ValueError,AttributeError):
        raise ValueError('Invalid PCI identity') from None


def validate(directory):
    from .kmeans import assess as check_kmeans
    from .hotspot import assess as check_hotspot
    root=Path(directory)
    plan=json.loads((root/'plan.json').read_text())
    record=json.loads((root/'record.json').read_text())
    if sha(root/'input.data')!=plan['metadata']['sha256']:raise ValueError('Live workload input changed')
    case=plan['case'];meta=plan['metadata'];rows=[json.loads(s) for s in record['stdout'].splitlines()]
    measured=[r for r in rows if r.get('kind')=='iteration' and not r['warmup']]
    counts=plan['counts']
    if case['workload']=='kmeans':
        checked=check_kmeans(dict(case,mode='fixed',max_iterations=case['steps'],workers_requested=case['workers'],
                                 cpu_counts=counts,validation_mode='timing'),meta,record)
        if checked['status']!='passed':raise ValueError(checked['reason'])
    else:
        view=dict(record,configuration=None,decisions=[dict(decision=dict(reason='scripted_schedule',iteration=i,
                                                                         cpu_items=counts[i+1]),controller_ns=0)
                                                     for i in range(len(counts)-1)])
        checked=check_hotspot(dict(case,rows=meta['rows'],cols=meta['cols'],mode='timing',policy='scripted',
                                  run_id=plan['run_id'],counts=counts),meta,view,root/'output.hstemp')
    if len(measured)!=len(counts) or len(counts)!=case['steps']:
        raise ValueError('Live work count mismatch')
    return dict(status='passed',iterations=len(counts),input_sha256=meta['sha256'])


class NativeWorkload:
    def __init__(self, binary, dataset, workload, steps, workers, directory, dispatch_order='cpu_first', timeout=120,
                 *, optimized_kmeans=False):
        if workload not in ('kmeans','hotspot') or type(steps) is not int or not 1<=steps<=100000:
            raise ValueError('Invalid workload/budget')
        if type(workers) is not int or not 1<=workers<=8 or dispatch_order not in ('cpu_first','gpu_first'):
            raise ValueError('Invalid workers/dispatch')
        self.binary=Path(binary).resolve(strict=True);self.dataset=Path(dataset).resolve(strict=True)
        self.directory=Path(directory);self.timeout=timeout
        self.case=dict(workload=workload,steps=steps,workers=workers,dispatch_order=dispatch_order,feedback_batch=1)
        if workload=='kmeans' and optimized_kmeans:
            self.case.update(reduction='hierarchical', transfer_mode='streamed')
        self.supervisor=None;self.errors=None;self.lines=[];self.counts=[];self.finished=False;self.released=False

    def start(self, config, qualification):
        self.config=config;self.started=time.monotonic_ns()
        root=self.directory;root.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(self.dataset,root/'input.data')
        if self.case['workload']=='kmeans':
            points,centers=kr.read_dataset(root/'input.data')
            self.metadata=dict(n=len(points),d=len(points[0]),k=len(centers),initial_centroids=[v for row in centers for v in row],sha256=sha(root/'input.data'))
            if config.n!=len(points):raise ValueError('Controller work size differs from input')
            args=[str(config.initial_cpu_items),str(self.case['steps']),str(self.case['workers']),'fixed','timing','feedback','armed']
            if self.case.get('reduction')=='hierarchical':args.extend(('reduced','streamed'))
        else:
            raw=(root/'input.data').read_bytes();magic,rows,cols=struct.unpack_from('<8sII',raw)
            if magic!=b'ATLSHS01' or len(raw)!=16+rows*cols*16 or config.n!=rows:raise ValueError('Thermal input/work size mismatch')
            self.metadata=dict(rows=rows,cols=cols,model=hr.model(rows,cols),sha256=sha(root/'input.data'))
            args=[str(config.initial_cpu_items),str(self.case['steps']),str(self.case['workers']),'timing',str((root/'output.hstemp').resolve()),'feedback','armed']
        if self.case['dispatch_order']=='gpu_first':args.append('gpu-first')
        self.errors=(root/'stderr.txt').open('x',encoding='utf-8')
        self.supervisor=Supervisor([str(self.binary),str((root/'input.data').resolve()),*args],self.errors,self.timeout)
        line=self.supervisor.receive(timeout=10)
        if line is None:raise RuntimeError('Missing native configuration')
        self.lines.append(line);cfg=json.loads(line)
        if (cfg.get('kind')!='configuration' or cfg.get('armed') is not True or cfg.get('feedback_batch')!=1 or
                pci_identity(cfg.get('pci_bus_id'))!=pci_identity(qualification['pci_bus_id'])):
            raise ActuationBlocked('Native device mismatch or workload did not arm')
        self.expected_cut=config.initial_cpu_items

    def release(self, cpu_items):
        if type(cpu_items) is not int or not 0<=cpu_items<=self.config.n or self.finished:
            raise ValueError('Invalid native release')
        if not self.released and cpu_items!=self.config.initial_cpu_items:raise ValueError('Initial partition changed')
        message=str(cpu_items) if self.released else 'start'
        self.supervisor.process.stdin.write(message+'\n');self.supervisor.process.stdin.flush()
        self.released=True;self.expected_cut=cpu_items

    def observe(self, run_id):
        for line in self.supervisor.lines():
            self.lines.append(line);row=json.loads(line)
            if row.get('kind')!='iteration':raise RuntimeError('Unexpected native record before final iteration')
            if row['warmup']:continue
            key='cpu_items' if self.case['workload']=='kmeans' else 'cpu_rows'
            if row['index']!=len(self.counts) or row[key]!=self.expected_cut:raise ValueError('Native sequence/partition mismatch')
            self.counts.append(row[key])
            obs=runtime_observation(row,self.config,self.case)
            return obs,len(self.counts)==self.case['steps'],row.get('phase',0)
        raise RuntimeError('Native execution ended before completing its budget')

    def finish(self):
        for line in self.supervisor.lines():self.lines.append(line)
        code=self.supervisor.process.wait(timeout=5)
        self.errors.flush()
        self.finished=True
        save_new(self.directory/'record.json',dict(status='ok' if code==0 else 'failed',returncode=code,
                                                  stdout=''.join(self.lines),stderr=(self.directory/'stderr.txt').read_text(),
                                                  elapsed_ns=time.monotonic_ns()-self.started))
        save_new(self.directory/'plan.json',dict(case=self.case,metadata=self.metadata,counts=self.counts,
                                                run_id=self.config.run_id,binary_sha256=sha(self.binary)))
        checked=validate(self.directory)
        checked['native_archive']='native'
        return checked

    def close(self):
        if self.supervisor is not None:self.supervisor.close()
        if self.errors is not None:self.errors.close()
        if not self.finished and self.directory.exists():
            save_new(self.directory/'partial.json',dict(stdout=''.join(self.lines),counts=self.counts))
