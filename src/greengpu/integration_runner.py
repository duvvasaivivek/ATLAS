"""Persistent native feedback with independent telemetry; no clock setter imports."""
from dataclasses import asdict
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


class TelemetryPump:
    def __init__(self,backend,period=.1):
        self.sampler=LoadedSampler(backend);self.backend=backend;self.period=period
        self.stop_event=threading.Event();self.lock=threading.Lock();self.pending=[]
        self.count=0;self.error=None;self.wall_ns=0;self.cpu_ns=0
        self.thread=threading.Thread(target=self.run,daemon=True)
    def run(self):
        try:
            while not self.stop_event.is_set():
                start=time.monotonic_ns();cpu=time.thread_time_ns()
                frame=self.sampler.sample()
                frame.update(frame_id=self.count,poll_cpu_ns=time.thread_time_ns()-cpu)
                self.count+=1;self.wall_ns+=time.monotonic_ns()-start;self.cpu_ns+=frame["poll_cpu_ns"]
                with self.lock:self.pending.append(frame)
                t=frame["readings"]["temperature_c"]
                if t["status"]=="ok" and t["value"]>=85:raise RuntimeError("temperature_limit_85C")
                self.stop_event.wait(max(0,self.period-(time.monotonic_ns()-start)/1e9))
        except Exception as error:self.error=f"{type(error).__name__}: {error}"
    def drain(self):
        with self.lock:result=self.pending;self.pending=[]
        return result
    def start(self):self.thread.start()
    def stop(self):
        self.stop_event.set();self.thread.join(timeout=30)
        if self.thread.is_alive():
            self.backend.worker_stuck=True
            raise RuntimeError("Telemetry thread failed to stop")


def terminate_group(process):
    if process is not None and process.poll() is None:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass


def execute(binary,dataset,field,case,backend,prefix=(),timeout=120):
    cfg=Config(**case["integration"]);engine=Coordinator(cfg);lifecycle=Lifecycle()
    if case["workload"]=="kmeans":
        args=[str(cfg.initial_cpu_items),str(case["steps"]),str(case["workers"]),case["native_mode"],"feedback"]
    else:
        args=[str(cfg.initial_cpu_items),str(case["steps"]),str(case["workers"]),case["native_mode"],str(Path(field).resolve()),"feedback"]
    argv=[*prefix,str(Path(binary).resolve()),str(Path(dataset).resolve()),*args]
    lines=[];barriers=[];trailing=[];failure=None;expired=threading.Event();process=None;timer=None
    pump=TelemetryPump(backend);started=time.monotonic_ns();returncode=None
    with tempfile.TemporaryFile(mode="w+t") as stderr:
        try:
            pump.start()
            process=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,
                                     text=True,encoding="utf-8",bufsize=1,start_new_session=True)
            def expire():expired.set();terminate_group(process)
            timer=threading.Timer(timeout,expire);timer.daemon=True;timer.start()
            for line in process.stdout:
                lines.append(line)
                if not line.startswith("{"):continue
                row=json.loads(line)
                if pump.error:raise RuntimeError(pump.error)
                if row.get("kind")=="configuration":lifecycle.move("warmup")
                if row.get("kind")!="iteration":continue
                if row["warmup"]:lifecycle.move("execute");continue
                lifecycle.move("observe")
                observation=(kmeans_observation if case["workload"]=="kmeans" else hotspot_observation)(row,cfg.run_id)
                observation["release_ns"]=row["release_ns"]
                final=row["index"]+1>=case["steps"] or (case["native_mode"]=="converge" and row["converged"])
                frames=pump.drain();now=time.monotonic_ns();begin=time.monotonic_ns()
                inputs=dict(observation=observation,now_ns=now,new_frames=frames,final=final,phase=0)
                decision=engine.barrier(**inputs)
                duration=time.monotonic_ns()-begin
                barriers.append(dict(input=inputs,decision=decision,controller_ns=duration))
                if final:lifecycle.move("finalize")
                else:
                    process.stdin.write(str(decision["next_cpu_items"])+"\n");process.stdin.flush()
                    lifecycle.move("execute")
            returncode=process.wait(timeout=5)
            if returncode or expired.is_set() or lifecycle.state!="finalize":raise RuntimeError("Incomplete/failed native workload")
        except BaseException as error:
            failure=f"{type(error).__name__}: {error}"
            lifecycle.move("failed")
        finally:
            if timer:timer.cancel()
            terminate_group(process)
            if process is not None:
                returncode=process.wait(timeout=5);process.stdin.close();process.stdout.close()
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
                failure=failure,timed_out=expired.is_set(),argv=argv,configuration=asdict(cfg),barriers=barriers,
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
        o=(kmeans_observation if case["workload"]=="kmeans" else hotspot_observation)(row,cfg.run_id)
        o["release_ns"]=row["release_ns"]
        inp=entry["input"]
        if inp["observation"]!=o or inp["phase"]!=0 or inp["now_ns"]<row["validation_end_ns"]:raise ValueError("Barrier input not bound to native work")
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
