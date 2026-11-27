"""Persistent native feedback with independent telemetry; no clock setter imports."""
from dataclasses import asdict
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time

from .allocation_experiment import observation as kmeans_observation
from .hotspot import observe as hotspot_observation
from .frequency_probe import LoadedSampler
from .integration import Config,Coordinator,Lifecycle
from .supervision import Supervisor, terminate_group
from .sampling import IsolatedSampler


def runtime_observation(row,cfg,case):
    observation=(kmeans_observation if case["workload"]=="kmeans" else hotspot_observation)(row,cfg.run_id)
    observation["release_ns"]=row["release_ns"]
    if cfg.allocation_policy=="cost_aware_v1":
        observation.update(work_s=(row["end_ns"]-row["release_ns"]+row.get("control_ns",0))/1e9,
            migration_s=(row.get("communication_end_ns",row["end_ns"])-row["release_ns"])/1e9,
            remaining_iterations=case["steps"]-row["index"]-1)
    return observation


class DisabledTelemetry:
    """Explicit matched no-monitoring arm; never eligible for shadow control."""
    def __init__(self):self.count=0;self.error=None;self.wall_ns=0;self.cpu_ns=0
    def start(self):pass
    def drain(self):return []
    def stop(self):pass


class TelemetryPump:
    def __init__(self,backend,period=.1):
        self.sampler=None;self.backend=backend;self.period=period
        self.stop_event=threading.Event();self.lock=threading.Lock();self.pending=[]
        self.count=0;self.error=None;self.wall_ns=0;self.cpu_ns=0
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.started=False
    def run(self):
        try:
            while not self.stop_event.is_set():
                start=time.monotonic_ns();cpu=time.thread_time_ns()
                frame=self.sampler.sample()
                frame.update(frame_id=self.count)
                self.count+=1;self.wall_ns+=time.monotonic_ns()-start;self.cpu_ns+=frame["poll_cpu_ns"]
                with self.lock:
                    if len(self.pending)>=10000:raise RuntimeError("Telemetry consumer stalled")
                    self.pending.append(frame)
                t=frame["readings"]["temperature_c"]
                if t["status"]=="ok" and t["value"]>=85:raise RuntimeError("temperature_limit_85C")
                self.stop_event.wait(max(0,self.period-(time.monotonic_ns()-start)/1e9))
        except Exception as error:self.error=f"{type(error).__name__}: {error}"
        finally:
            if self.sampler is not None:
                try:self.sampler.close()
                except Exception as error:self.error=self.error or str(error)
    def drain(self):
        with self.lock:result=self.pending;self.pending=[]
        return result
    def start(self):
        # Complete the bounded worker/NVML initialization before launching native
        # work or Nsight. Query deadlines then cover queries, not process startup.
        self.sampler=IsolatedSampler()
        self.thread.start();self.started=True
    def stop(self):
        if not self.started:
            if self.sampler is not None:self.sampler.close()
            return
        self.stop_event.set();self.thread.join(timeout=4)
        if self.thread.is_alive():
            self.backend.worker_stuck=True
            raise RuntimeError("Telemetry thread failed to stop")


def execute(binary,dataset,field,case,backend,prefix=(),timeout=120,on_iteration=None,on_process=None,cancelled=None,*,on_feedback=None):
    cfg=Config(**case["integration"]);engine=Coordinator(cfg);lifecycle=Lifecycle()
    if case["workload"]=="kmeans":
        args=[str(cfg.initial_cpu_items),str(case["steps"]),str(case["workers"]),case["native_mode"],"feedback"]
        if case.get("validation_mode"):args.append(case["validation_mode"])
        if case.get("reduction")=="hierarchical":args.append("reduced")
        if case.get("transfer_mode")=="streamed":args.append("streamed")
    else:
        args=[str(cfg.initial_cpu_items),str(case["steps"]),str(case["workers"]),case["native_mode"],str(Path(field).resolve()),"feedback"]
    if case["workload"]=="hotspot" and case.get("communication_mode")=="overlap":args.append("overlap")
    if case.get("dispatch_order","cpu_first") not in ("cpu_first","gpu_first"):
        raise ValueError("Unknown dispatch order")
    if case.get("dispatch_order")=="gpu_first":args.append("gpu-first")
    if case.get("feedback_batch",1)!=1:args.append(f"batch={case['feedback_batch']}")
    if case.get("phase_at") is not None:args.append(f"phase={case['phase_at']}:{case.get('phase_workers',1)}")
    argv=[*prefix,str(Path(binary).resolve()),str(Path(dataset).resolve()),*args]
    lines=[];barriers=[];trailing=[];failure=None;expired=threading.Event();process=None;timer=None
    if case.get("telemetry") is False and cfg.mode!="allocation_only":raise ValueError("Shadow mode requires telemetry")
    pump=DisabledTelemetry() if case.get("telemetry") is False else TelemetryPump(backend)
    started=time.monotonic_ns();returncode=None;supervisor=None
    with tempfile.TemporaryFile(mode="w+t") as stderr:
        try:
            pump.start()
            supervisor=Supervisor(argv,stderr,timeout,failure=lambda:pump.error or (
                "Stopped by user" if cancelled is not None and cancelled() else None))
            process=supervisor.process;expired=supervisor.expired
            if on_process is not None:
                on_process(process.pid)
            for line in supervisor.lines():
                lines.append(line)
                if not line.startswith("{"):continue
                row=json.loads(line)
                if pump.error:raise RuntimeError(pump.error)
                if row.get("kind")=="configuration":lifecycle.move("warmup")
                if row.get("kind")!="iteration":continue
                if row["warmup"]:lifecycle.move("execute");continue
                lifecycle.move("observe")
                observation=runtime_observation(row,cfg,case)
                final=row["index"]+1>=case["steps"] or (case["native_mode"]=="converge" and row["converged"])
                boundary=final or (row["index"]+1)%case.get("feedback_batch",1)==0
                frames=pump.drain() if boundary else [];now=time.monotonic_ns() if boundary else row["validation_end_ns"];begin=time.monotonic_ns()
                inputs=dict(observation=observation,now_ns=now,new_frames=frames,final=final,phase=row.get("phase",0))
                if case.get("runtime_version"):inputs["action_boundary"]=boundary
                decision=engine.barrier(**inputs)
                duration=time.monotonic_ns()-begin
                barriers.append(dict(input=inputs,decision=decision,controller_ns=duration))
                if final:lifecycle.move("finalize")
                else:
                    if boundary:process.stdin.write(str(decision["next_cpu_items"])+"\n");process.stdin.flush()
                    lifecycle.move("execute")
                if on_iteration is not None:on_iteration(deepcopy(row))
                if on_feedback is not None:
                    on_feedback(deepcopy(row), deepcopy(barriers[-1]))
            returncode=process.wait(timeout=5)
            if returncode or expired.is_set() or lifecycle.state!="finalize":raise RuntimeError("Incomplete/failed native workload")
        except BaseException as error:
            failure=f"{type(error).__name__}: {error}"
            lifecycle.move("failed")
        finally:
            if supervisor is not None:
                supervisor.close();returncode=process.returncode
            try:pump.stop();trailing=pump.drain()
            except Exception as error:
                failure=failure or str(error)
                if lifecycle.state!="failed":lifecycle.move("failed")
            if pump.error:
                failure=failure or pump.error
                if lifecycle.state!="failed":lifecycle.move("failed")
            lifecycle.move("restore");lifecycle.move("stopped" if failure else "complete")
        stderr.seek(0);errors=stderr.read()
    return dict(status="failed" if failure else "ok",returncode=returncode,stdout="".join(lines),stderr=errors,
                failure=failure,timed_out=expired.is_set(),argv=argv,configuration=json.loads(json.dumps(asdict(cfg))),barriers=barriers,
                trailing_frames=trailing,lifecycle=lifecycle.events,elapsed_ns=time.monotonic_ns()-started,
                polling=dict(frames=pump.count,query_wall_ns=pump.wall_ns,query_thread_cpu_ns=pump.cpu_ns,error=pump.error),
                hardware_setter_calls=0,restoration="not_needed_no_settings_changed")


def replay_decisions(case,record):
    cfg=Config(**case["integration"]);engine=Coordinator(cfg)
    if record["status"]!="ok" or record["returncode"]!=0 or record["failure"] or record["polling"]["error"]:
        raise ValueError("Failed integrated trial")
    if record["configuration"]!=json.loads(json.dumps(asdict(cfg))):raise ValueError("Runtime configuration changed")
    native=[json.loads(s) for s in record["stdout"].splitlines() if s.startswith("{")]
    rows=[r for r in native if r.get("kind")=="iteration" and not r["warmup"]]
    if len(rows)!=len(record["barriers"]) or not rows:raise ValueError("Incomplete barrier trace")
    lifecycle=Lifecycle();lifecycle.move("warmup");lifecycle.move("execute")
    for i,(row,entry) in enumerate(zip(rows,record["barriers"])):
        o=runtime_observation(row,cfg,case)
        inp=entry["input"]
        expected_phase=int(case.get("phase_at") is not None and row["index"]>=case["phase_at"])
        if inp["observation"]!=o or inp["phase"]!=expected_phase or inp["now_ns"]<row["validation_end_ns"]:raise ValueError("Barrier input not bound to native work")
        if case.get("runtime_version"):
            boundary=inp["final"] or (row["index"]+1)%case.get("feedback_batch",1)==0
            if inp.get("action_boundary") is not boundary or row.get("phase")!=expected_phase or row.get("feedback_due") is not (boundary and not inp["final"]):
                raise ValueError("Native batch/phase boundary mismatch")
        if i+1<len(rows) and inp["now_ns"]>rows[i+1]["release_ns"]:raise ValueError("Decision after next workload release")
        if inp["final"] is not (i+1==len(rows)):raise ValueError("Incorrect final barrier")
        expected=engine.barrier(**inp)
        if expected!=entry["decision"]:raise ValueError("Coordinator replay mismatch")
        if type(entry["controller_ns"]) is not int or entry["controller_ns"]<0:raise ValueError("Invalid controller duration")
        if i+1<len(rows):
            realized=rows[i+1]["cpu_items" if case["workload"]=="kmeans" else "cpu_rows"]
            if realized!=expected["next_cpu_items"]:raise ValueError("Requested/realized ownership mismatch")
        lifecycle.move("observe");lifecycle.move("finalize" if inp["final"] else "execute")
    lifecycle.move("restore");lifecycle.move("complete")
    if record["lifecycle"]!=lifecycle.events or record["hardware_setter_calls"]!=0 or record["restoration"]!="not_needed_no_settings_changed":
        raise ValueError("Lifecycle or shadow-actuation violation")
    all_frames=[f for e in record["barriers"] for f in e["input"]["new_frames"]]+record["trailing_frames"]
    if [f["frame_id"] for f in all_frames]!=list(range(record["polling"]["frames"])):raise ValueError("Telemetry inventory mismatch")
    return rows
