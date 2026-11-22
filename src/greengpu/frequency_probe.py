"""Bounded native diagnostics and read-only loaded NVML sampling."""

import json
import queue
import subprocess
import threading
import time
from .supervision import Supervisor, terminate_group


class Probe:
    def __init__(self,binary,mode,elements,rounds,stderr_path):
        self.stderr=stderr_path.open("x",encoding="utf-8")
        self.process=None
        self.queue=queue.Queue()
        try:
            self.supervisor=Supervisor([str(binary),mode,str(elements),str(rounds),"1729"],self.stderr,3600)
            self.process=self.supervisor.process
            self.ready=self.receive("ready")
        except BaseException:
            self.close();raise

    def receive(self,kind,timeout=40):
        line=self.supervisor.receive(timeout)
        if line is None: raise RuntimeError("Diagnostic worker terminated unexpectedly")
        result=json.loads(line)
        if result.get("kind")!=kind: raise ValueError("Unexpected native protocol record")
        return result

    def start(self,repetitions):
        self.process.stdin.write(f"run {repetitions}\n");self.process.stdin.flush()

    def close(self):
        if self.process is not None:
            self.supervisor.close()
        self.stderr.close()

    def finish(self):
        self.process.stdin.write("quit\n");self.process.stdin.flush()
        code=self.process.wait(timeout=10)
        if code:raise RuntimeError("Diagnostic worker failed: "+str(code))
        return code


class LoadedSampler:
    def __init__(self,backend):
        self.backend=backend
        self.since={"core":time.time_ns()//1000,"memory":time.time_ns()//1000}

    def sample(self):
        b=self.backend;n=b.nv;h=b.handle
        start=time.monotonic_ns();wall=time.time_ns()//1000
        frame=dict(acquisition_start_ns=start,wall_us=wall,buffer_samples={},errors=[])
        for domain,kind in (("core",n.NVML_GPU_UTILIZATION_SAMPLES),("memory",n.NVML_MEMORY_UTILIZATION_SAMPLES)):
            try:
                value_type,samples=n.nvmlDeviceGetSamples(h,kind,self.since[domain])
                if value_type!=n.NVML_VALUE_TYPE_UNSIGNED_INT:raise ValueError("Unexpected NVML sample type")
                values=[dict(timestamp_us=s.timeStamp,percent=s.sampleValue.uiVal) for s in samples]
                frame["buffer_samples"][domain]=values
                if values:self.since[domain]=max(v["timestamp_us"] for v in values)
            except (n.NVMLError,ValueError) as error:
                frame["buffer_samples"][domain]=[];frame["errors"].append(dict(domain=domain,error=str(error)))
        fields=(("core_mhz",n.nvmlDeviceGetClockInfo,(n.NVML_CLOCK_GRAPHICS,)),
                ("memory_mhz",n.nvmlDeviceGetClockInfo,(n.NVML_CLOCK_MEM,)),
                ("temperature_c",n.nvmlDeviceGetTemperature,(n.NVML_TEMPERATURE_GPU,)),
                ("power_mw",n.nvmlDeviceGetPowerUsage,()),
                ("energy_mj",n.nvmlDeviceGetTotalEnergyConsumption,()),
                ("clock_event_reasons",n.nvmlDeviceGetCurrentClocksEventReasons,()))
        frame["readings"]={key:b.query(fn,*args) for key,fn,args in fields}
        frame["acquisition_end_ns"]=time.monotonic_ns()
        frame["wall_end_us"]=time.time_ns()//1000
        return frame


def run_batch(probe,repetitions,sampler,period=.1):
    from .sampling import IsolatedSampler
    isolated=IsolatedSampler()
    records=[];errors=[];stop=threading.Event()
    def sample():
        try:
            while not stop.is_set():
                before=time.monotonic()
                frame=isolated.sample();records.append(frame)
                temp=frame["readings"]["temperature_c"]
                if temp["status"]=="ok" and temp["value"]>=85:
                    errors.append("temperature_limit_85C");terminate_group(probe.process);return
                stop.wait(max(0,period-(time.monotonic()-before)))
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}");terminate_group(probe.process)
    host_start=time.monotonic_ns()
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    try:
        probe.start(repetitions)
        batch=probe.receive("batch")
        host_end=time.monotonic_ns()
    except BaseException as error:
        stop.set()
        thread.join(timeout=4)
        error.evidence=dict(host_start_ns=host_start,frames=list(records),errors=list(errors),complete=False)
        raise
    finally:
        stop.set();thread.join(timeout=4);isolated.close()
        if thread.is_alive():
            sampler.backend.worker_stuck=True
            raise RuntimeError("NVML sampling thread did not stop")
    if errors:raise RuntimeError("Sampler failed: "+"; ".join(errors))
    if batch["mismatches"] or batch["elements_verified"]!=probe.ready["elements"]:
        raise RuntimeError("Diagnostic correctness failed")
    return dict(native=batch,host_start_ns=host_start,host_end_ns=host_end,frames=records,
                errors=errors,requested_pair=None,acknowledged_pair=None,
                clock_policy="unchanged_platform_policy_not_fixed_clocks")
