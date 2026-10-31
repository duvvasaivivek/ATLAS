"""Phase 9 deterministic coordination; hardware modes have no actuator dependency."""
from copy import deepcopy
from dataclasses import asdict, dataclass

from . import allocation as a
from . import frequency as f
from .experiment import digest

SPEC_VERSION="integration-1.0"
MODES=("allocation_only","allocation_shadow","synthetic_joint")


@dataclass(frozen=True)
class Config:
    run_id: str
    n: int
    initial_cpu_items: int
    pairs: tuple=()
    mode: str="allocation_shadow"
    allocation_policy: str="paper_proportional"
    frequency_policy: str="paper_additive"
    allocation_iterations: int=4
    allocation_interval_ns: int=2_000_000_000
    frequency_interval_ns: int=200_000_000
    frequency_guard_ns: int=1_000_000_000
    settle_iterations: int=1
    cadence_profile: str="observed_cadence_v1"

    def __post_init__(self):
        if self.mode not in MODES:raise ValueError("Live joint control is blocked; choose an explicit supported mode")
        if not self.run_id or type(self.n) is not int or self.n<1 or type(self.initial_cpu_items) is not int or not 0<=self.initial_cpu_items<=self.n:
            raise ValueError("Invalid run identity or partition")
        for key in ("allocation_iterations","frequency_interval_ns"):
            if type(getattr(self,key)) is not int or getattr(self,key)<1:raise ValueError("Invalid cadence")
        for key in ("allocation_interval_ns","frequency_guard_ns","settle_iterations"):
            if type(getattr(self,key)) is not int or getattr(self,key)<0:raise ValueError("Invalid guard")
        if self.cadence_profile not in ("observed_cadence_v1","paper_ratio_40_v1"):raise ValueError("Unknown cadence profile")
        if self.mode!="synthetic_joint" and self.allocation_policy=="bracketed_bisection":
            raise ValueError("Hardware automatic clocks do not establish stationary capacity for a bracket")
        a.Config(self.allocation_policy,self.n)
        fc=f.Config(self.pairs,policy=self.frequency_policy)
        object.__setattr__(self,"pairs",fc.pairs)

    @property
    def identity(self):return digest(dict(spec_version=SPEC_VERSION,**asdict(self)))

    @property
    def allocation_period(self):
        return max(self.allocation_interval_ns,40*self.frequency_interval_ns) if self.cadence_profile=="paper_ratio_40_v1" else self.allocation_interval_ns


def sample_for_window(frames, since_ns, now_ns, last_timestamp):
    """Join exact driver stamps, with immutable source refs and a conservative guard.

    This is shadow eligibility only. Query clocks are context, not synchronized
    settings acknowledged by a controller. No live qualification is inferred.
    """
    indexes={d:{} for d in ("core","memory")};conflicts={d:set() for d in indexes}
    for frame in frames:
        if frame["acquisition_end_ns"]>now_ns:continue
        for domain in indexes:
            for sample in frame["buffer_samples"][domain]:
                stamp,value=sample["timestamp_us"],sample["percent"]
                age=frame["wall_end_us"]-stamp
                estimated=frame["acquisition_end_ns"]-age*1000
                current_age=age+(now_ns-frame["acquisition_end_ns"])/1000
                if (type(stamp) is not int or stamp<=last_timestamp or not f.fraction(value/100 if f.finite(value) else None)
                        or age<0 or not 0<=current_age<=500_000 or not since_ns<=estimated<=now_ns):continue
                previous=indexes[domain].get(stamp)
                if previous and previous[0]!=value:conflicts[domain].add(stamp)
                indexes[domain][stamp]=(value,frame["frame_id"],estimated)
    for d in indexes:
        for stamp in conflicts[d]:indexes[d].pop(stamp,None)
    shared=indexes["core"].keys()&indexes["memory"].keys()
    if not shared:return None
    stamp=max(shared);core=indexes["core"][stamp];memory=indexes["memory"][stamp]
    return dict(timestamp_us=stamp,core=core[0]/100,memory=memory[0]/100,
                frame_ids=sorted({core[1],memory[1]}),estimated_sample_ns=min(core[2],memory[2]),
                qualification="shadow_only_exact_timestamp_join_approximate_clock_mapping")


class Coordinator:
    def __init__(self,config):
        self.config=config;self.allocation=a.Config(config.allocation_policy,config.n)
        self.frequency=f.Config(config.pairs,policy=config.frequency_policy,
                                provenance="synthetic_assumed" if config.mode=="synthetic_joint" else "hardware_reported_unverified")
        self.astate=a.initial_state(config.run_id);self.fstate=f.initial_state(self.frequency,config.run_id)
        self.cut=config.initial_cpu_items;self.iteration=-1;self.phase=0;self.capacity_epoch=0;self.partition_epoch=0
        self.since_ns=None;self.last_allocation_ns=None;self.last_frequency_ns=None;self.settling=0
        self.last_barrier_ns=-1
        self.window=[];self.frames=[];self.frame_id=-1;self.logical_pair=config.pairs[0] if config.pairs else None

    def barrier(self,observation,now_ns,new_frames=(),final=False,phase=0,synthetic_utilization=None):
        c=self.config;o=deepcopy(observation)
        if (o.get("run_id")!=c.run_id or type(o.get("iteration")) is not int or o["iteration"]!=self.iteration+1 or
                o.get("cpu_items")!=self.cut or o.get("valid") is not True or o.get("common_origin") is not True or
                type(now_ns) is not int or now_ns<self.last_barrier_ns or type(o.get("release_ns")) is not int or not 0<=o["release_ns"]<=now_ns or
                type(phase) is not int or phase<self.phase):raise ValueError("Invalid barrier identity, ownership, or correctness")
        for count,key in ((self.cut,"t_cpu"),(c.n-self.cut,"t_gpu")):
            value=o.get(key)
            if (count and (not f.finite(value) or value<=0)) or (not count and value is not None):raise ValueError("Invalid branch time")
        if self.since_ns is None:self.since_ns=self.last_allocation_ns=self.last_frequency_ns=o["release_ns"]
        if now_ns<self.last_allocation_ns or now_ns<self.last_frequency_ns:raise ValueError("Nonmonotonic barrier time")
        if phase!=self.phase:
            self.phase=phase;self.capacity_epoch+=1;self.window=[];self.since_ns=o["release_ns"];self.settling=c.settle_iterations
        for frame in new_frames:
            if type(frame.get("frame_id")) is not int or frame["frame_id"]!=self.frame_id+1:raise ValueError("Telemetry sequence gap/duplicate")
            if frame["acquisition_end_ns"]>now_ns:raise ValueError("Future telemetry")
            self.frame_id=frame["frame_id"];self.frames.append(deepcopy(frame))
        # Bound controller memory; raw frames remain in the retained barrier input.
        self.frames=[frame for frame in self.frames if frame["acquisition_end_ns"]>=now_ns-2_000_000_000]
        self.iteration=o["iteration"]
        self.last_barrier_ns=now_ns
        result=dict(spec_version=SPEC_VERSION,config_hash=c.identity,iteration=self.iteration,mode=c.mode,
                    events=["observe","decide"],allocation=None,frequency=None,frequency_reason="cadence_not_due",
                    allocation_reason="collect_epoch",next_cpu_items=self.cut,partition_epoch=self.partition_epoch,
                    capacity_epoch=self.capacity_epoch,synthetic_applied_pair=None,hardware_requested_pair=None)
        was_settling=self.settling>0
        if was_settling:self.settling-=1;result["allocation_reason"]="settling_iteration_excluded"
        else:self.window.append(o)
        if final:
            result.update(allocation_reason="workload_complete",frequency_reason="workload_complete",events=["observe","finalize"])
            return result
        # Frequency observes the old partition, before any allocation recommendation is applied.
        if c.mode=="allocation_only":result["frequency_reason"]="disabled_allocation_only"
        elif now_ns-self.last_frequency_ns>=c.frequency_interval_ns:
            self.last_frequency_ns=now_ns
            sample=None
            if c.mode=="synthetic_joint":
                if synthetic_utilization is not None and now_ns>=self.since_ns+c.frequency_guard_ns:
                    window=dict(kind="synthetic_interval",start=o["release_ns"],end=now_ns)
                    sample=dict(core=synthetic_utilization[0],memory=synthetic_utilization[1],core_window=window,memory_window=dict(window),
                                semantics="synthetic_busy_fraction",source_ref=f"synthetic:{c.run_id}:{self.iteration}")
            else:
                joined=sample_for_window(self.frames,self.since_ns+c.frequency_guard_ns,now_ns,
                                         self.fstate["last_window"]["end"] if self.fstate["last_window"] else -1)
                if joined:
                    window=dict(kind="nvml_buffer_timestamp_us",start=None,end=joined["timestamp_us"])
                    sample=dict(core=joined["core"],memory=joined["memory"],core_window=window,memory_window=dict(window),
                                semantics="nvml_kernel_and_memory_busy_time_fraction",source_ref="telemetry_frames:"+",".join(map(str,joined["frame_ids"])),join=joined)
            if sample:
                sample.update(run_id=c.run_id,sample_id=self.iteration,epoch=self.partition_epoch+self.phase,valid=True)
                decision=f.transition(self.frequency,self.fstate,sample);self.fstate=decision["post_state"]
                result.update(frequency=decision,frequency_reason=decision["reason"])
                pair=tuple(decision["recommended_pair"]) if decision["recommended_pair"] else None
                if c.mode=="synthetic_joint" and pair!=self.logical_pair and decision["reason"]=="recommend_pair":
                    self.logical_pair=pair;self.capacity_epoch+=1;self.window=[];self.since_ns=now_ns;self.settling=c.settle_iterations
                    result["synthetic_applied_pair"]=list(pair);result["events"].extend(["apply_synthetic_clock","settle"])
            else:result["frequency_reason"]="hold_insufficient_fresh_window"
        if (not was_settling and len(self.window)>=c.allocation_iterations and now_ns-self.last_allocation_ns>=c.allocation_period):
            aggregated=dict(run_id=c.run_id,epoch=self.capacity_epoch,iteration=self.iteration,cpu_items=self.cut,
                            t_cpu=sum(x["t_cpu"] for x in self.window) if self.cut else None,
                            t_gpu=sum(x["t_gpu"] for x in self.window) if self.cut<c.n else None,
                            valid=True,common_origin=True,source_iterations=[x["iteration"] for x in self.window])
            decision=a.transition(self.allocation,self.astate,aggregated);self.astate=decision["post_state"]
            self.last_allocation_ns=now_ns;self.window=[]
            result.update(allocation=decision,allocation_reason=decision["reason"],next_cpu_items=decision["cpu_items"])
            if decision["cpu_items"]!=self.cut:
                self.cut=decision["cpu_items"];self.partition_epoch+=1;self.since_ns=now_ns;self.settling=c.settle_iterations
                result["events"].extend(["apply_allocation_at_barrier","settle"])
        result["events"].append("execute")
        return result


class Lifecycle:
    ALLOWED={"prepare":{"warmup","failed"},"warmup":{"execute","failed"},
             "execute":{"observe","finalize","failed"},"observe":{"execute","finalize","failed"},
             "finalize":{"restore","failed"},"failed":{"restore"},"restore":{"complete","stopped"}}
    def __init__(self):self.state="prepare";self.events=["prepare"]
    def move(self,state):
        if state not in self.ALLOWED.get(self.state,set()):raise ValueError("Illegal runtime lifecycle transition")
        self.state=state;self.events.append(state)
