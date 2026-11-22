"""Integrated real allocation/shadow runs, independent workload validation, replay."""
from copy import deepcopy
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import sys

from .actuation import NvmlClockBackend
from .capabilities import collect
from .concurrency import analyze_profile
from .experiment import save_new
from .frequency_experiment import sha
from .phase8 import candidate_subset
from .integration import Config,SPEC_VERSION
from .integration_runner import execute,replay_decisions
from .integration_scenarios import definitions,simulate
from . import kmeans_reference as kr
from . import hotspot_reference as hr
from .kmeans import assess as assess_kmeans
from .hotspot import assess as assess_hotspot
from .timing import breakdown


def assess(root,case,meta):
    record=json.loads((root/case["record"]).read_text());rows=replay_decisions(case,record)
    counts=[r["cpu_items" if case["workload"]=="kmeans" else "cpu_rows"] for r in rows]
    golden=json.loads((root/case["golden"]).read_text()) if case.get("golden") else None
    view=dict(record,stdout="\n".join(s for s in record["stdout"].splitlines() if s.startswith("{")))
    if case["workload"]=="kmeans":
        native_case=dict(case,n=meta["n"],d=meta["d"],k=meta["k"],mode=case["native_mode"],
                         max_iterations=case["steps"],workers_requested=case["workers"],
                         cpu_counts=counts+[counts[-1]]*(case["steps"]-len(counts)))
        result=assess_kmeans(native_case,meta,view,golden)
        if result["status"]!="passed":raise ValueError(result["reason"])
        final=dict(label_hash=result["final_label_hash"],centroids=result["final_centroids"])
        migration_bytes=sum(r["h2d_point_bytes"] for r in rows)
        communication_ns=(None if case.get("transfer_mode")=="streamed" else
            sum((r["h2d_end_ns"]-r["gpu_dispatch_ns"])+(r["gpu_end_ns"]-r["compute_end_ns"]) for r in rows if r["gpu_items"]))
    else:
        # Materialize the realized schedule only for the established workload verifier.
        # Coordinator decisions and next-step ownership were independently replayed above.
        native_case=dict(case,policy="scripted",run_id=case["integration"]["run_id"],rows=meta["rows"],cols=meta["cols"],
                         counts=counts,mode=case["native_mode"])
        view.update(configuration=None,decisions=[dict(decision=dict(reason="scripted_schedule",iteration=i,cpu_items=counts[i+1]),
                                                     controller_ns=record["barriers"][i]["controller_ns"]) for i in range(len(counts)-1)])
        result=assess_hotspot(native_case,meta,view,root/case["field"],golden)
        final=dict(field_sha256=result["output_sha256"])
        migration_bytes=sum(r["migration_h2d_bytes"]+r["migration_d2h_bytes"] for r in rows)
        communication_ns=sum(r["communication_end_ns"]-r["release_ns"] for r in rows)
    frequency=[e["decision"]["frequency"] for e in record["barriers"] if e["decision"]["frequency"]]
    allocation=[e["decision"]["allocation"] for e in record["barriers"] if e["decision"]["allocation"]]
    control=sum(r["control_ns"] for r in rows);active=sum(r["end_ns"]-r["release_ns"] for r in rows)
    frames=[f for e in record["barriers"] for f in e["input"]["new_frames"]]+record["trailing_frames"]
    masks=Counter(str(f["readings"]["clock_event_reasons"]["value"]) for f in frames if f["readings"]["clock_event_reasons"]["status"]=="ok")
    observed=[(f["readings"]["core_mhz"]["value"],f["readings"]["memory_mhz"]["value"]) for f in frames
              if all(f["readings"][key]["status"]=="ok" for key in ("core_mhz","memory_mhz"))]
    ages=[frame["wall_end_us"]-s["timestamp_us"] for frame in frames for domain in ("core","memory") for s in frame["buffer_samples"][domain]]
    extra={}
    if case.get("runtime_version"):
        parts=[breakdown(row,case["workload"]) for row in rows]
        extra=dict(timing_components={key:sum(p[key] for p in parts)/1e9 for key in
                    ("active_ns","shared_communication_ns","parallel_envelope_ns","join_tail_ns","merge_ns")},
                   phase_decisions=dict(Counter(str(e["input"]["phase"]) for e in record["barriers"] if e["decision"]["allocation"])),
                   feedback_boundaries=sum(e["input"]["action_boundary"] and not e["input"]["final"] for e in record["barriers"]),
                   validation_mode=case.get("validation_mode",case["native_mode"]))
    return dict(**extra,run_id=case["integration"]["run_id"],workload=case["workload"],dataset=case["dataset"],stage=case["stage"],
                mode=case["integration"]["mode"],status="passed",iterations=len(rows),
                allocation_decisions=len(allocation),ownership_changes=sum(x!=y for x,y in zip(counts,counts[1:])),
                frequency_decisions=len(frequency),frequency_updates=frequency[-1]["post_state"]["updates"] if frequency else 0,
                insufficient_frequency_windows=sum(e["decision"]["frequency_reason"]=="hold_insufficient_fresh_window" for e in record["barriers"]),
                active_s=active/1e9,control_gap_s=control/1e9,controller_compute_s=sum(e["controller_ns"] for e in record["barriers"] if not e["input"]["final"])/1e9,
                final_barrier_compute_s=sum(e["controller_ns"] for e in record["barriers"] if e["input"]["final"])/1e9,
                command_s=record["elapsed_ns"]/1e9,poll_query_wall_s=record["polling"]["query_wall_ns"]/1e9,
                poll_thread_cpu_s=record["polling"]["query_thread_cpu_ns"]/1e9,poll_frames=record["polling"]["frames"],
                migration_or_residency_bytes=migration_bytes,communication_s=communication_ns/1e9 if communication_ns is not None else None,
                migration_time_s=None,migration_timing_scope="not_isolated_from_other_transfer_work",
                settling_work_s=sum(row["end_ns"]-row["release_ns"] for row,e in zip(rows,record["barriers"]) if e["decision"]["allocation_reason"]=="settling_iteration_excluded")/1e9,
                actuation_overhead_s=0.,driver_clock_event_masks=dict(masks),observed_clock_changes=sum(x!=y for x,y in zip(observed,observed[1:])),
                future_driver_timestamp_records=sum(age<0 for age in ages),minimum_driver_sample_age_us=min(ages) if ages else None,
                final=final,energy_j=None,hardware_setter_calls=0,clock_policy="unchanged_automatic_not_fixed_capacity")


def evaluate(root):
    manifest=json.loads((root/"manifest.json").read_text())
    if manifest["spec_version"]!=SPEC_VERSION:raise ValueError("Unsupported integration version")
    for name,meta in manifest["datasets"].items():
        if not (root/name).resolve().is_relative_to(root.resolve()) or sha(root/name)!=meta["sha256"]:
            raise ValueError("Dataset identity mismatch")
    results=[]
    for case in manifest["cases"]:
        for key in ("record","dataset","field","golden"):
            if case.get(key) and (root/case[key]).resolve().is_relative_to(root.resolve()) is False:raise ValueError("Archive path escape")
        results.append(assess(root,case,manifest["datasets"][case["dataset"]]))
    # Workload equivalence across shadow/allocation modes, independent of changed schedules.
    for dataset in {r["dataset"] for r in results}:
        group=[r for r in results if r["dataset"]==dataset and r["stage"]!="convergence"]
        if len(group)<2:continue
        first=group[0]
        for row in group[1:]:
            if row["workload"]=="kmeans":
                if first["final"]["label_hash"]!=row["final"]["label_hash"] or not all(kr.close(a,b) for a,b in zip(first["final"]["centroids"],row["final"]["centroids"])):
                    raise ValueError("Shadow/allocation clustering mismatch")
            else:
                cases=[next(c for c in manifest["cases"] if c["integration"]["run_id"]==x["run_id"]) for x in (first,row)]
                fields=[hr.read_grid(root/c["field"])[2] for c in cases]
                if not all(hr.close(a,b) for a,b in zip(*fields)):raise ValueError("Shadow/allocation thermal field mismatch")
    synthetic=json.loads((root/"synthetic.json").read_text())
    expected=[simulate(case) for case in definitions()]
    if synthetic!=json.loads(json.dumps(expected)):raise ValueError("Synthetic joint-control replay mismatch")
    profiles={}
    for case in manifest["cases"]:
        if case["stage"]!="profile":continue
        k=case["workload"]
        profiles[k]=analyze_profile(root/f"profile-{k}.sqlite",case["steps"],
                                    ("kmeans_assign_reduced" if case.get("reduction")=="hierarchical" else "kmeans_assign") if k=="kmeans" else "hotspot_step",
                                    "kmeans_iteration" if k=="kmeans" else "hotspot_iteration",
                                    "kmeans_cpu_compute" if k=="kmeans" else "hotspot_cpu_compute")
    demonstrated=len(profiles)==2 and all(p["status"]=="demonstrated" for p in profiles.values())
    coverage=True
    if manifest.get("runtime_version"):
        integrated=[r for r in results if r["stage"]=="integration"]
        coverage=len(integrated)==6 and all(r["allocation_decisions"]>=2 and r["ownership_changes"]>=1 and all(r["phase_decisions"].get(str(p),0)>0 for p in (0,1))
                                            for r in integrated)
    return dict(spec_version=SPEC_VERSION,software_gate="incomplete_adaptation_coverage" if not coverage else "passed" if demonstrated else "incomplete_concurrency_evidence",
                live_joint_hardware_gate="blocked_phase8_prior_policy_and_measurement_qualification",
                trials=results,hardware_trials=len(results),iterations=sum(r["iterations"] for r in results),
                allocation_decisions=sum(r["allocation_decisions"] for r in results),
                shadow_frequency_updates=sum(r["frequency_updates"] for r in results),
                ownership_changes=sum(r["ownership_changes"] for r in results),
                synthetic_cases=len(synthetic),synthetic_barriers=sum(len(s["records"]) for s in synthetic),
                synthetic_expected_failures=sum(s["status"]=="expected_failure_restored" for s in synthetic),
                concurrency=profiles,hardware_setter_calls=0,energy_savings=None,
                overhead_scope=("batched controller processing can overlap native work; feedback gaps and polling are reported separately, never added twice" if manifest.get("runtime_version") else
                                "nonfinal controller compute nests in feedback gaps; final-barrier compute is separate; polling is concurrent, not added to job time"),
                hardware_effect_of_shadow_recommendations="none_applied_no_counterfactual_claim")


def replay(root):
    root=Path(root);hashes=json.loads((root/"checksums.json").read_text())
    actual={p.relative_to(root).as_posix():p for p in root.rglob("*") if p.is_file() and p!=root/"checksums.json"}
    if set(actual)!=set(hashes):raise ValueError("Integration inventory mismatch")
    for name,path in actual.items():
        if path.is_symlink() or sha(path)!=hashes[name]:raise ValueError("Integration checksum mismatch: "+name)
    result=evaluate(root)
    if result!=json.loads((root/"summary.json").read_text()):raise ValueError("Integration summary mismatch")
    return result


def run(output,kmeans_binary,hotspot_binary,workers=4):
    if type(workers) is not int or not 1<=workers<=8:raise ValueError("Workers must be 1..8")
    root=Path(output);bins={"kmeans":Path(kmeans_binary).resolve(strict=True),"hotspot":Path(hotspot_binary).resolve(strict=True)}
    root.mkdir(parents=True,exist_ok=False)
    for folder in ("datasets","golden","fields"): (root/folder).mkdir()
    backend=None;previous=signal.getsignal(signal.SIGTERM)
    def terminated(signum,frame):raise KeyboardInterrupt("SIGTERM")
    signal.signal(signal.SIGTERM,terminated)
    try:
        collect(root/"environment","/usr/local/cuda-13.1/bin/nvcc",None)
        backend=NvmlClockBackend();capability=backend.discover();save_new(root/"capabilities.json",capability)
        pairs=candidate_subset(capability["pairs"]);datasets={};small={}
        for name,n,d,k,seed in (("k-small",257,3,4,9001),("k-large",65539,8,8,9002),("k-profile",32771,8,8,9003)):
            data=kr.clustered(n,d,k,seed);path=f"datasets/{name}.kmdat"
            datasets[path]=kr.write_dataset(root/path,*data)
            if n<1024:small[path]=data
        for name,h,w,seed in (("h-small",19,23,9011),("h-large",257,259,9012),("h-profile",513,515,9013)):
            temp,power,prov=hr.generated(h,w,seed);path=f"datasets/{name}.hsdat"
            datasets[path]=hr.write_input(root/path,h,w,temp,power,prov)
            if h*w<1024:small[path]=(temp,power,h,w)
        cases=[]
        def trial(workload,dataset,mode,steps,stage,frequency_policy="paper_additive",native_mode=None):
            meta=datasets[dataset];n=meta["n"] if workload=="kmeans" else meta["rows"]
            i=len(cases);run_id=f"joint-{i:02d}"
            cfg=Config(run_id,n,n//2,tuple(map(tuple,pairs)),mode,frequency_policy=frequency_policy,
                       allocation_policy="robust_proportional_epoch_v1",
                       allocation_interval_ns=0 if stage in ("fixture","convergence") else 50_000_000,
                       allocation_iterations=4,cadence_profile="batched_epochs_v1")
            case=dict(workload=workload,dataset=dataset,record=f"trial-{i:02d}.json",field=f"fields/trial-{i:02d}.hstemp" if workload=="hotspot" else None,
                      workers=workers,steps=steps,stage=stage,integration=asdict(cfg),golden=None,
                      runtime_version="runtime-2.0",feedback_batch=1 if stage in ("fixture","convergence") else 16,
                      phase_at=steps//2 if stage=="integration" else None,phase_workers=1,
                      **(dict(validation_mode="verify" if stage in ("fixture","convergence") else "timing",
                              reduction="atomic",transfer_mode="diagnostic") if workload=="kmeans" else {}),
                      native_mode=native_mode or ("fixed" if workload=="kmeans" else "verify" if stage=="fixture" else "timing"))
            if dataset in small:
                golden=kr.trajectory(*small[dataset][:2],steps,case["native_mode"]) if workload=="kmeans" else hr.trajectory(*small[dataset],steps)
                case["golden"]=f"golden/{run_id}.json";save_new(root/case["golden"],golden)
            prefix=("nsys","profile","--trace=cuda,nvtx","--sample=none","--cpuctxsw=none","--export=sqlite",f"--output={root/f'profile-{workload}'}") if stage=="profile" else ()
            record=execute(bins[workload],root/dataset,root/case["field"] if case["field"] else None,case,backend,prefix)
            save_new(root/case["record"],record);cases.append(case)
            save_new(root/f"case-{i:02d}.json",case)
            assess(root,case,meta)
        for workload,prefix,suffix in (("kmeans","k","kmdat"),("hotspot","h","hsdat")):
            for mode in ("allocation_only","allocation_shadow"):
                trial(workload,f"datasets/{prefix}-small.{suffix}",mode,24,"fixture")
            for mode,policy in (("allocation_only","paper_additive"),("allocation_shadow","paper_additive"),("allocation_shadow","discounted_loss_v1")):
                trial(workload,f"datasets/{prefix}-large.{suffix}",mode,1000 if workload=="kmeans" else 6000,"integration",policy)
            trial(workload,f"datasets/{prefix}-profile.{suffix}","allocation_shadow",256,"profile")
        trial("kmeans","datasets/k-small.kmdat","allocation_shadow",24,"convergence",native_mode="converge")
        save_new(root/"synthetic.json",[simulate(case) for case in definitions()])
        save_new(root/"manifest.json",dict(spec_version=SPEC_VERSION,runtime_version="runtime-2.0",cases=cases,datasets=datasets,python=sys.version,
                                          binaries={k:dict(path=str(v),sha256=sha(v)) for k,v in bins.items()},hardware_setter_calls=0))
        summary=evaluate(root);save_new(root/"summary.json",summary)
        save_new(root/"checksums.json",{p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob("*")) if p.is_file()})
        replay(root);return summary
    except BaseException as error:
        save_new(root/"failure.json",dict(error=f"{type(error).__name__}: {error}"));raise
    finally:
        if backend is not None and not getattr(backend,"worker_stuck",False):backend.close()
        signal.signal(signal.SIGTERM,previous)
