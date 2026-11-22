"""Exercise real batched feedback, phase changes and deterministic replay."""
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from greengpu.integration import Config
from greengpu.integration_runner import execute,replay_decisions
from greengpu import kmeans_reference as kr,hotspot_reference as hr


class NoTelemetry:
    def __init__(self,*args):self.error=None;self.count=0;self.wall_ns=0;self.cpu_ns=0
    def start(self):pass
    def stop(self):pass
    def drain(self):return []


with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    for workload,binary in zip(("kmeans","hotspot"),sys.argv[1:]):
        dataset=root/(workload+".data");field=root/"field"
        if workload=="kmeans":kr.write_dataset(dataset,*kr.clustered(513,8,8,919));n=513
        else:
            t,p,provenance=hr.generated(19,23,919);hr.write_input(dataset,19,23,t,p,provenance);n=19
        cfg=Config("batch-test",n,n//2,mode="allocation_only",allocation_policy="robust_proportional_epoch_v1",
                   allocation_iterations=1,allocation_interval_ns=0,cadence_profile="batched_epochs_v1")
        case=dict(workload=workload,steps=16,workers=4,native_mode="fixed" if workload=="kmeans" else "timing",
                  integration=asdict(cfg),feedback_batch=4,runtime_version="runtime-2.0",phase_at=8,phase_workers=1)
        if workload=="kmeans":case.update(validation_mode="timing",reduction="hierarchical",transfer_mode="streamed")
        with patch("greengpu.integration_runner.TelemetryPump",NoTelemetry):
            record=execute(binary,dataset,field,case,None)
        assert record['status']=='ok',record
        assert len(replay_decisions(case,record))==16
        actions=[e for e in record['barriers'] if e['decision']['allocation']]
        assert {e['input']['phase'] for e in actions}=={0,1}
        assert all(e['input']['action_boundary'] for e in actions)
        assert sum(e['input']['action_boundary'] for e in record['barriers'])==4
print('Both native workloads pass batched feedback, phase change and replay')
