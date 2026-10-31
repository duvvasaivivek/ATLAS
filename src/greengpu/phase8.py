"""Phase 8 capability gate and real diagnostic evidence; no inferred restoration."""

from collections import Counter
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import sys

from .actuation import ActuationBlocked, ClockTransaction, NvmlClockBackend
from .capabilities import cpu_interfaces
from .experiment import save_new
from .frequency_experiment import read_lines, sha
from .frequency_hardware import SPEC_VERSION
from .frequency_probe import LoadedSampler, Probe, run_batch


def candidate_subset(pairs):
    """Small explicit subset with core-only and memory-only contrasts where possible."""
    pairs=sorted(set(tuple(p) for p in pairs))
    if not pairs:return []
    memories=sorted({m for c,m in pairs})
    high_memory=memories[-1]
    cores=sorted(c for c,m in pairs if m==high_memory)
    chosen={(cores[i],high_memory) for i in {len(cores)//4,len(cores)//2,3*len(cores)//4}}
    anchor=cores[len(cores)//2]
    for m in memories[len(memories)//2:]:
        if (anchor,m) in pairs:chosen.add((anchor,m))
    return [list(p) for p in sorted(chosen)]


def characterize(trial):
    batch=trial["native"]
    if (batch["mismatches"]!=0 or batch["elements_verified"]!=batch["elements"] or
            batch["oracle"]!="masked_uint64_reference" or batch["tolerance"]!=0):
        raise ValueError("Incorrect or unqualified native output")
    if not (trial["host_start_ns"]<=batch["start_ns"]<batch["compute_end_ns"]<=batch["download_end_ns"]<=batch["validation_end_ns"]<=trial["host_end_ns"]):
        raise ValueError("Incompatible native/host timing order")
    seconds=(batch["compute_end_ns"]-batch["start_ns"])/1e9
    indexes={d:{} for d in ("core","memory")};conflicts={d:set() for d in indexes}
    readings={k:[] for k in ("core_mhz","memory_mhz","temperature_c","power_mw")}
    event_counts=Counter(); acquisition=[];errors=0
    for frame in trial["frames"]:
        if frame["acquisition_start_ns"]>frame["acquisition_end_ns"]:raise ValueError("Invalid acquisition bounds")
        acquisition.append((frame["acquisition_end_ns"]-frame["acquisition_start_ns"])/1e6)
        errors+=len(frame["errors"])
        # Only polls wholly inside the compute interval characterize loaded behavior.
        loaded=batch["start_ns"]<=frame["acquisition_start_ns"]<=frame["acquisition_end_ns"]<=batch["compute_end_ns"]
        if not loaded:continue
        for key in readings:
            q=frame["readings"][key]
            if q["status"]=="ok":readings[key].append(q["value"])
        q=frame["readings"]["clock_event_reasons"]
        if q["status"]=="ok":event_counts[str(q["value"])]+=1
        for domain in indexes:
            for sample in frame["buffer_samples"][domain]:
                stamp,value=sample["timestamp_us"],sample["percent"]
                age=frame["wall_end_us"]-stamp
                estimate=frame["acquisition_end_ns"]-age*1000
                # Conservative integration guard, with explicitly approximate wall/mono mapping.
                if (type(stamp) is not int or type(value) not in (int,float) or not math.isfinite(value) or
                        not 0<=value<=100 or not 0<=age<=500000 or estimate<batch["start_ns"]+1_000_000_000):continue
                if stamp in indexes[domain] and indexes[domain][stamp]!=value:conflicts[domain].add(stamp)
                indexes[domain][stamp]=value
    for d in indexes:
        for stamp in conflicts[d]:indexes[d].pop(stamp,None)
    stamps=sorted(indexes["core"].keys()&indexes["memory"].keys())
    med=lambda values:statistics.median(values) if values else None
    gaps=[(b-a)/1000 for a,b in zip(stamps,stamps[1:])]
    return dict(mode=batch["mode"],repetitions=batch["repetitions"],elements=batch["elements"],
                host_compute_s=seconds,download_s=(batch["download_end_ns"]-batch["compute_end_ns"])/1e9,
                validation_s=(batch["validation_end_ns"]-batch["download_end_ns"])/1e9,
                host_envelope_s=(trial["host_end_ns"]-trial["host_start_ns"])/1e9,
                logical_io_gb_per_s=batch["elements"]*8*batch["repetitions"]/(seconds*1e9),
                io_semantics="logical_read_plus_write_bytes_not_measured_DRAM_traffic",
                guarded_matched_samples=len(stamps),core_busy_percent_median=med([indexes["core"][s] for s in stamps]),
                memory_busy_percent_median=med([indexes["memory"][s] for s in stamps]),
                paired_sample_gap_ms_median=med(gaps),paired_sample_gap_ms_max=max(gaps) if gaps else None,
                loaded_readings={k:dict(n=len(v),median=med(v),min=min(v) if v else None,max=max(v) if v else None) for k,v in readings.items()},
                query_wall_ms_median=med(acquisition),query_wall_ms_max=max(acquisition) if acquisition else None,
                sample_query_errors=errors,clock_event_masks=dict(event_counts),correct=True,
                telemetry_scope="guarded_timestamp_join_with_approximate_wall_to_monotonic_mapping",
                live_feedback_qualified=False,energy_j=None,energy_status="blocked_phase2_qualification")


def evaluate(root):
    manifest=json.loads((root/"manifest.json").read_text())
    capability=json.loads((root/"capabilities.json").read_text())
    if manifest["spec_version"]!=SPEC_VERSION:raise ValueError("Unsupported Phase 8 schema")
    if manifest["native_exit_codes"]!={"compute":0,"stream":0}:raise ValueError("Native worker exit failure")
    if type(manifest["blocks"]) is not int or not 1<=manifest["blocks"]<=10:raise ValueError("Invalid block count")
    names=["compute.pilot.json","stream.pilot.json"]
    for block in range(manifest["blocks"]):
        names.extend(f"{mode}.block-{block}.json" for mode in (("compute","stream") if block%2==0 else ("stream","compute")))
    if manifest["trials"]!=names:raise ValueError("Diagnostic inventory/order mismatch")
    if capability["prior"]["restorable"] is not False:raise ValueError("This archive schema is the blocked-control diagnostic path")
    if capability["selected_pairs"]!=candidate_subset(capability["pairs"]):raise ValueError("Candidate subset mismatch")
    journal=read_lines(root/"actuation-gate/journal.jsonl")
    if [e["kind"] for e in journal]!=["snapshot","blocked"] or journal[0]["snapshot"]["restorable"] is not False:
        raise ValueError("Unknown policy must block before mutation")
    trials=[]
    checksums={};counts=Counter()
    for name in manifest["trials"]:
        trial=json.loads((root/name).read_text())
        batch=trial["native"]; mode=batch["mode"]
        if mode not in ("compute","stream"):raise ValueError("Unexpected diagnostic mode")
        ready=json.loads((root/(mode+".ready.json")).read_text())
        if any(batch[key]!=ready[key] for key in ("mode","elements","rounds","seed")):
            raise ValueError("Native configuration mismatch")
        if batch["batch"]!=counts[mode]:raise ValueError("Native sequence mismatch")
        counts[mode]+=1
        if mode in checksums and checksums[mode]!=batch["checksum"]:raise ValueError("Repeated fixed-input checksum changed")
        checksums[mode]=batch["checksum"]
        if trial["requested_pair"] is not None or trial["acknowledged_pair"] is not None:
            raise ValueError("Unexpected actuation in blocked-control diagnostic")
        result=characterize(trial)
        if trial["stage"] not in ("pilot","measured") or (trial["stage"]=="pilot" and (batch["repetitions"]!=manifest.get("pilot_repetitions",1) or trial["block"]!=-1)):
            raise ValueError("Invalid diagnostic stage")
        result.update(block=trial["block"],stage=trial["stage"],file=name)
        trials.append(result)
    measured=[t for t in trials if t["stage"]=="measured"]
    for mode in ("compute","stream"):
        group=[t for t in measured if t["mode"]==mode]
        if len(group)!=manifest["blocks"] or len({t["repetitions"] for t in group})!=1:
            raise ValueError("Incomplete fixed-work diagnostic matrix")
        if {t["block"] for t in group}!=set(range(manifest["blocks"])):raise ValueError("Missing diagnostic block")
    return dict(spec_version=SPEC_VERSION,diagnostic_status="passed",implementation_scope="transaction_engine_and_hardware_capability_gate",
                hardware_exit_gate="blocked",reason="prior_locked_clock_policy_not_queryable",
                clock_setter_calls=0,restoration="not_needed_no_settings_changed",reported_pairs=len(capability["pairs"]),
                selected_pairs=capability["selected_pairs"],verified_pairs=[],static_frequency_sweep="blocked",
                frequency_only_runs="blocked",cpu_policy_control="unavailable_or_unverified_no_changes",
                energy_savings=None,energy_qualification="blocked_phase2",device_event_timing="not_used_unqualified_phase2",
                diagnostic_trials=trials,measured_trials=len(measured),
                correct_batches=len(trials),matched_loaded_samples=sum(t["guarded_matched_samples"] for t in measured),
                physical_bottleneck_classification="not_established_by_kernel_name_or_busy_time_alone",
                source_verification="source_hashes.json",hardware_settings_changed=False)


def replay(root):
    root=Path(root);expected=json.loads((root/"checksums.json").read_text())
    actual={p.relative_to(root).as_posix():p for p in root.rglob("*") if p.is_file() and p!=root/"checksums.json"}
    if set(actual)!=set(expected):raise ValueError("Phase 8 archive inventory mismatch")
    for name,path in actual.items():
        if path.is_symlink() or sha(path)!=expected[name]:raise ValueError("Phase 8 checksum mismatch: "+name)
    result=evaluate(root)
    if result!=json.loads((root/"summary.json").read_text()):raise ValueError("Phase 8 summary mismatch")
    return result


def run(output,binary,blocks=3,target_seconds=2.):
    if type(blocks) is not int or not 1<=blocks<=10 or not math.isfinite(target_seconds) or not 1<=target_seconds<=5:
        raise ValueError("Require 1..10 blocks and 1..5 target seconds")
    root=Path(output);binary=Path(binary).resolve(strict=True)
    root.mkdir(parents=True,exist_ok=False)
    backend=None;probes={};prior_handler=signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,lambda signum,frame: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
    try:
        backend=NvmlClockBackend();capability=backend.discover()
        capability["selected_pairs"]=candidate_subset(capability["pairs"])
        capability["cpu_interfaces"]=cpu_interfaces()
        save_new(root/"capabilities.json",capability)
        try:
            with ClockTransaction(backend,root/"actuation-gate",capability["selected_pairs"]):
                raise RuntimeError("Unexpected restorable NVML policy; use the qualified control harness")
        except ActuationBlocked:pass
        save_new(root/"recovery.json",dict(required=False,reason="zero_setter_calls_unknown_policy_gate",
                                          device_uuid=backend.device_uuid,
                                          instruction="No reset is appropriate: ATLAS did not change the pre-existing policy."))
        sampler=LoadedSampler(backend);trials=[];counts={}
        pilot_repetitions=64
        for mode,elements,rounds in (("compute",65536,4096),("stream",16777216,1)):
            probe=Probe(binary,mode,elements,rounds,root/(mode+".stderr.txt"));probes[mode]=probe
            if probe.ready["pci_bus_id"].lower().lstrip("0")!=capability["pci_bus_id"].lower().lstrip("0"):
                raise RuntimeError("CUDA/NVML PCI identity mismatch")
            save_new(root/(mode+".ready.json"),probe.ready)
            trial=run_batch(probe,pilot_repetitions,sampler);trial.update(stage="pilot",block=-1)
            name=mode+".pilot.json";save_new(root/name,trial);trials.append(name)
            seconds=characterize(trial)["host_compute_s"]
            counts[mode]=max(1,min(10000,math.ceil(target_seconds*pilot_repetitions/seconds)))
        for block in range(blocks):
            for mode in (("compute","stream") if block%2==0 else ("stream","compute")):
                trial=run_batch(probes[mode],counts[mode],sampler);trial.update(stage="measured",block=block)
                name=f"{mode}.block-{block}.json";save_new(root/name,trial);trials.append(name)
        exit_codes={mode:probe.finish() for mode,probe in probes.items()}
        save_new(root/"manifest.json",dict(spec_version=SPEC_VERSION,blocks=blocks,target_seconds=target_seconds,trials=trials,pilot_repetitions=pilot_repetitions,
                                          python=sys.version,platform=platform.platform(),euid=os.geteuid(),
                                          binary=str(binary),binary_sha256=sha(binary),native_exit_codes=exit_codes,
                                          cpu_policy="recorded_unchanged_not_fixed_capacity",energy_qualified=False))
        sources=[p for d in ("src","native","tests") for p in Path(d).rglob("*") if p.is_file() and p.suffix in (".py",".cu",".cpp",".hpp",".h")]
        sources.extend(Path(p) for p in ("pyproject.toml","CMakeLists.txt"))
        save_new(root/"source_hashes.json",{p.as_posix():sha(p) for p in sorted(sources)})
        summary=evaluate(root);save_new(root/"summary.json",summary)
        save_new(root/"checksums.json",{p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob("*")) if p.is_file()})
        replay(root)
        return summary
    except BaseException as error:
        save_new(root/"failure.json",dict(error=f"{type(error).__name__}: {error}",complete=False,
                                         partial_trial=getattr(error,"evidence",None)))
        raise
    finally:
        for probe in probes.values():probe.close()
        if backend is not None and not getattr(backend,"worker_stuck",False):backend.close()
        signal.signal(signal.SIGTERM,prior_handler)
