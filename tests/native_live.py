"""Real CUDA/CPU work with an explicitly fake actuator; never hardware-control evidence."""
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from greengpu import kmeans_reference as kr,hotspot_reference as hr
from greengpu.actuation import NvmlClockBackend
from greengpu.integration import Config
from greengpu.live import run_live,replay
from greengpu.live_native import NativeWorkload
from test_live import Backend,PAIRS


class TestProvider:
    def __init__(self,pci):
        self.backend=Backend();self.pci=pci;self.last=0
        self.created=time.monotonic_ns()
    def qualification(self):
        return dict(device_uuid=self.backend.device_uuid,pci_bus_id=self.pci,evidence_kind='software_test',
                    telemetry_clock='qualified',restoration_probe='qualified',device_timing='qualified',
                    evidence_id='native-software-test-not-hardware',qualified_at_ns=self.created,
                    valid_until_ns=self.created+180_000_000_000,verified_pairs=PAIRS)
    def samples(self):
        now=time.monotonic_ns()
        if now-self.last<200_000_000:return []
        self.last=now
        return [dict(device_uuid=self.backend.device_uuid,sample_timestamp_us=now//1000,sample_valid=True,
                     sample_monotonic_estimate_ns=now,sample_age_us=0,acquisition_start_ns=now,acquisition_end_ns=now,
                     core_utilization=.9,memory_utilization=.9,temperature_c=60,clock_event_reasons=0,
                     core_mhz=self.backend.policy['core'][0],memory_mhz=self.backend.policy['memory'][0])]


def main():
    # Read PCI identity only. All setters below belong to the dictionary test double.
    backend=NvmlClockBackend()
    try:pci=backend.discover()['pci_bus_id']
    finally:backend.close()
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        for workload,binary in zip(('kmeans','hotspot'),sys.argv[1:3]):
            dataset=root/(workload+'.data')
            if workload=='kmeans':kr.write_dataset(dataset,*kr.clustered(513,8,8,2201));n=513
            else:
                t,p,provenance=hr.generated(33,35,2201)
                hr.write_input(dataset,33,35,t,p,provenance);n=33
            output=root/workload;provider=TestProvider(pci)
            cfg=Config('native-'+workload,n,n//2,PAIRS,allocation_policy='paper_simple',allocation_interval_ns=50_000_000)
            work=NativeWorkload(binary,dataset,workload,1200,4,output/'native',timeout=120)
            result=run_live(provider,work,cfg,output)
            assert result['status'] in ('passed','incomplete_settling','incomplete_verification')
            assert result['correctness']['status']=='passed' and result['correctness']['iterations']==1200
            assert provider.backend.policy==provider.backend.original and not result['physical_control_evidence']
            assert replay(output)==result
            print(json.dumps(dict(workload=workload,status=result['status'],evidence_kind=result['evidence_kind'],
                                  iterations=1200,frequency_changes=sum(r['decision']['requested_pair'] is not None for r in result['records']),
                                  allocation_decisions=sum(r['decision']['allocation']['allocation'] is not None for r in result['records']),
                                  restoration=result['restoration'],physical_clock_setter_calls=0)),flush=True)
            if len(sys.argv)>3:
                import shutil
                destination=Path(sys.argv[3])/workload
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copytree(output,destination)


if __name__=='__main__':main()
