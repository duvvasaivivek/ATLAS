"""Phase 4 validation, fixed-work split sweeps, and reproducible evidence replay."""

import hashlib
import json
import math
import os
import platform
import random
from pathlib import Path

from .capabilities import collect,command,utc_now
from .concurrency import analyze_profile
from .experiment import distribution,save_new
from .kmeans_reference import ATOL,RTOL,SHIFT_TOL,close,clustered,fixtures,label_hash,trajectory,write_dataset


def assess(case,metadata,process,golden=None,profiled=False):
    result=dict(status="failed",case=case,energy_j=None,energy_qualification="blocked_phase2",device_event_timing="not_used")
    try:
        if process["status"]!="ok" or process["returncode"]!=0:raise ValueError("Native/profiler command failed")
        records=[json.loads(line) for line in process["stdout"].splitlines() if not profiled or line.startswith("{")]
        if len(records)<4 or records[0].get("kind")!="configuration" or records[-1].get("kind")!="summary":
            raise ValueError("Missing configuration/iterations/summary")
        cfg=records[0];end=records[-1];rows=records[1:-1]
        if cfg["schema_version"]!="4.0" or cfg["clock"]!="CLOCK_MONOTONIC":raise ValueError("Unsupported schema/clock")
        if any(cfg[k]!=metadata[k] for k in ("n","d","k")) or cfg["initial_centroids"]!=metadata["initial_centroids"]:
            raise ValueError("Input/initialization mismatch")
        if any(cfg[k]!=case[k] for k in ("mode","max_iterations","workers_requested")):
            raise ValueError("Execution mode/work budget mismatch")
        if (cfg["atol"],cfg["rtol"],cfg["shift_tolerance"])!=(ATOL,RTOL,SHIFT_TOL):raise ValueError("Tolerance changed")
        n,d,k=metadata["n"],metadata["d"],metadata["k"]
        schedule=case["cpu_counts"]
        if len(schedule) not in (1,case["max_iterations"]) or any(type(v) is not int or not 0<=v<=n for v in schedule):
            raise ValueError("Invalid partition schedule")
        measured=[];resident=n;previous_cut=n;last_validation=cfg["preparation_start_ns"];last_quality=None
        for index,r in enumerate(rows):
            warmup=index==0
            if r["kind"]!="iteration" or r["index"]!=index-1 or r["warmup"] is not warmup:raise ValueError("Iteration sequence mismatch")
            cut=schedule[0 if warmup or len(schedule)==1 else index-1];gpu=n-cut
            if r["status"]!="passed" or any(r[key]!=0 for key in ("label_errors","count_errors","numeric_errors")):
                raise ValueError("Native correctness validation failed")
            if r["cpu_items"]!=cut or r["gpu_items"]!=gpu:raise ValueError("Incorrect realized split")
            expected=dict(resident_before=resident,resident_after=min(cut,resident),ownership_changed_points=abs(cut-previous_cut),
                          h2d_point_bytes=max(0,resident-cut)*d*8,h2d_centroid_bytes=k*d*8 if gpu else 0,
                          d2h_label_bytes=gpu*4,d2h_aggregate_bytes=k*d*8+k*4+8 if gpu else 0)
            if any(r[key]!=value for key,value in expected.items()):raise ValueError("Residency/migration accounting mismatch")
            shared=[r[key] for key in ("release_ns","barrier_end_ns","end_ns","validation_end_ns")]
            if any(type(t) is not int or t<0 for t in shared) or shared!=sorted(shared) or shared[0]<last_validation:
                raise ValueError("Invalid host timing order")
            last_validation=shared[-1]
            for active,keys in ((cut,("cpu_start_ns","cpu_end_ns")),(gpu,("gpu_dispatch_ns","h2d_end_ns","compute_end_ns","gpu_end_ns"))):
                stamps=[r[key] for key in keys]
                if active:
                    if any(type(t) is not int for t in stamps) or stamps!=sorted(stamps) or not shared[0]<=stamps[0]<=stamps[-1]<=shared[1]:
                        raise ValueError("Invalid branch timings")
                elif any(t is not None for t in stamps):raise ValueError("Empty branch has timing")
            if (cut and not 1<=r["workers_used"]<=case["workers_requested"]) or (not cut and r["workers_used"]!=0):
                raise ValueError("CPU worker contract mismatch")
            if len(r["counts"])!=k or any(type(c) is not int or c<0 for c in r["counts"]) or sum(r["counts"])!=n:
                raise ValueError("Cluster counts do not cover input")
            if len(r["sums"])!=k*d or len(r["centroids"])!=k*d:raise ValueError("Aggregate shape mismatch")
            numeric=r["sums"]+r["centroids"]+[r[key] for key in ("objective_before_update","objective_after_update","max_centroid_shift")]
            if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in numeric):raise ValueError("Nonfinite numeric observation")
            if min(r["objective_before_update"],r["objective_after_update"],r["max_centroid_shift"])<0:raise ValueError("Negative squared metric")
            if r["objective_after_update"]>r["objective_before_update"] and not close(r["objective_after_update"],r["objective_before_update"]):
                raise ValueError("Centroid update increased objective")
            if last_quality is not None and r["objective_before_update"]>last_quality and not close(r["objective_before_update"],last_quality):
                raise ValueError("Assignment increased objective")
            if r["converged"] is not (r["changed_labels"]==0 or r["max_centroid_shift"]<=SHIFT_TOL):raise ValueError("Stopping rule mismatch")
            if "labels" in r and (len(r["labels"])!=n or any(type(v) is not int or not 0<=v<k for v in r["labels"]) or label_hash(r["labels"])!=r["label_hash"]):
                raise ValueError("Invalid membership labels/hash")
            if golden is not None:
                ref=golden[0 if warmup else index-1]
                for key in ("labels","label_hash","counts","changed_labels","empty_clusters","converged"):
                    if r[key]!=ref[key]:raise ValueError(f"Python reference mismatch: {key}")
                for key in ("centroids","sums"):
                    if not all(close(a,b) for a,b in zip(r[key],ref[key])):raise ValueError(f"Python reference mismatch: {key}")
                for key in ("objective_before_update","objective_after_update","max_centroid_shift"):
                    if not close(r[key],ref[key]):raise ValueError(f"Python reference mismatch: {key}")
            if warmup:
                resident=n;previous_cut=n;last_quality=None
            else:
                measured.append(r);resident=expected["resident_after"];previous_cut=cut;last_quality=r["objective_after_update"]
        if not measured or len(measured)>case["max_iterations"]:raise ValueError("Invalid iteration count")
        if case["mode"]=="fixed":
            if len(measured)!=case["max_iterations"] or end["stop_reason"]!="fixed_budget":raise ValueError("Fixed work stopped early")
        else:
            if any(r["converged"] for r in measured[:-1]):raise ValueError("Convergence mode failed to stop")
            reason="converged" if measured[-1]["converged"] else "max_iterations"
            if end["stop_reason"]!=reason or (reason=="max_iterations" and len(measured)!=case["max_iterations"]):raise ValueError("Invalid convergence outcome")
        if golden is not None and len(measured)!=len(golden):raise ValueError("Python iteration budget mismatch")
        active=sum(r["end_ns"]-r["release_ns"] for r in measured)
        loop=measured[-1]["end_ns"]-measured[0]["release_ns"]
        if end["status"]!="passed" or end["iterations"]!=len(measured) or end["active_ns"]!=active or end["loop_elapsed_ns"]!=loop or not close(end["final_objective"],measured[-1]["objective_after_update"]):
            raise ValueError("Summary inconsistent with retained iterations")
        labels=end["final_labels"]
        if len(labels)!=n or any(type(v) is not int or not 0<=v<k for v in labels) or label_hash(labels)!=measured[-1]["label_hash"]:
            raise ValueError("Exported membership mismatch")
        if [labels.count(c) for c in range(k)]!=measured[-1]["counts"] or end["final_centroids"]!=measured[-1]["centroids"]:
            raise ValueError("Exported clustering result mismatch")
        result.update(status="passed",iterations=measured,active_s=active/1e9,loop_elapsed_s=loop/1e9,
                      final_objective=end["final_objective"],stop_reason=end["stop_reason"],
                      final_label_hash=measured[-1]["label_hash"],final_centroids=measured[-1]["centroids"],
                      verification="native_independent_trajectory_and_python" if golden is not None else "native_independent_trajectory")
    except (ValueError,KeyError,IndexError,TypeError,OverflowError) as error:
        result["reason"]=str(error)
    return result


def curves(results,stage):
    rows=[r for r in results if r["case"]["stage"]==stage and r["status"]=="passed"]
    output=[]
    for cut in sorted({r["case"]["cpu_counts"][0] for r in rows}):
        group=[r for r in rows if r["case"]["cpu_counts"][0]==cut]
        output.append(dict(cpu_items=cut,cpu_share=cut/group[0]["case"]["n"],active_s=distribution([r["active_s"] for r in group]),
                           loop_elapsed_s=distribution([r["loop_elapsed_s"] for r in group]),energy_j=None))
    return output


def best_point(points):
    if not points:raise ValueError("No valid timing candidates")
    return min(points,key=lambda p:(p["active_s"]["mean"],p["cpu_items"]))["cpu_items"]


def summarize(root):
    root=Path(root)
    manifest=json.loads((root/"manifest.json").read_text());index=json.loads((root/"trial-index.json").read_text())
    results=[]
    for case in index:
        golden=json.loads((root/case["golden"]).read_text()) if case.get("golden") else None
        results.append(assess(case,manifest["datasets"][case["dataset"]],json.loads((root/case["record"]).read_text()),golden,case["stage"]=="profile"))
    equiv=[]
    for dataset in sorted({r["case"]["dataset"] for r in results}):
        for mode in ("fixed","converge"):
            group=[r for r in results if r["case"]["dataset"]==dataset and r["case"]["mode"]==mode and r["case"]["stage"]!="profile"]
            if not group:continue
            okay=all(r["status"]=="passed" for r in group)
            if okay:
                first=group[0]
                okay=all(len(r["iterations"])==len(first["iterations"]) and all(a["label_hash"]==b["label_hash"] and a["counts"]==b["counts"] and all(close(x,y) for x,y in zip(a["centroids"],b["centroids"])) for a,b in zip(r["iterations"],first["iterations"])) for r in group)
            equiv.append(dict(dataset=dataset,mode=mode,trials=len(group),equivalent=okay))
    profile=next(r for r in results if r["case"]["stage"]=="profile")
    overlap=analyze_profile(root/"profile.sqlite",profile["case"]["max_iterations"],"kmeans_assign","kmeans_iteration","kmeans_cpu_compute")
    series={stage:curves(results,stage) for stage in ("coarse_tuning","refine_tuning","heldout_evaluation")}
    chosen=json.loads((root/"selection.json").read_text())
    passed=all(r["status"]=="passed" for r in results) and all(e["equivalent"] for e in equiv) and overlap["status"]=="demonstrated"
    return dict(schema_version="4.0",implementation_status="passed" if passed else "incomplete",full_phase_gate="blocked_energy_qualification",
                results=results,split_equivalence=equiv,concurrency=overlap,timing_curves=series,
                tuning_selected_cpu_items=chosen["cpu_items"],heldout_observed_timing_minimum_cpu_items=best_point(series["heldout_evaluation"]),
                energy_minimum=dict(status="blocked",joules=None,reason="Phase 2 counter/power inconsistency; no qualified energy domain"),
                clock_policy="unchanged platform automatic policy; no fixed-clock claim",hardware_settings_changed=False,
                timing_scope="sum of complete iteration intervals, including transfers/migration/merge; excludes inter-iteration validation and logging",
                statistics="exploratory trial-level mean/stdev/range; held-out generated input, no statistical significance or global optimum claim")


def run(output,binary,workers=4,blocks=3):
    if platform.system()!="Linux":raise ValueError("Use Linux/WSL")
    if not 1<=workers<=min(8,max(1,(os.cpu_count() or 1)-2)) or not 2<=blocks<=10:raise ValueError("Invalid worker/block budget")
    binary=Path(binary).resolve(strict=True);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    try:
        collect(output/"environment","/usr/local/cuda-13.1/bin/nvcc",None)
        cache=binary.parent/"CMakeCache.txt"
        if cache.exists():(output/"environment/raw/CMakeCache.txt").write_bytes(cache.read_bytes())
        datasets=fixtures()
        datasets["tuning"]=clustered(32771,8,8,2027)
        datasets["heldout"]=clustered(32771,8,8,2402)
        # A compute-heavy case makes actual kernel/CPU overlap observable even
        # when the first measured iteration must upload the initial GPU suffix.
        datasets["profile"]=clustered(65539,32,32,771)
        (output/"datasets").mkdir();(output/"golden").mkdir();metadata={}
        for name,(points,centers,provenance) in datasets.items():
            metadata[name]=write_dataset(output/"datasets"/(name+".kmdat"),points,centers,provenance)
        save_new(output/"manifest.json",dict(schema_version="4.0",started_utc=utc_now(),datasets=metadata,workers=workers,blocks=blocks,
            variant="Lloyd FP64 squared-Euclidean, lowest-index ties, keep-empty-centroid, seeded Forgy or declared fixtures",
            atol=ATOL,rtol=RTOL,shift_tolerance=SHIFT_TOL,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            source_hashes=json.loads((output/"environment/source_hashes.json").read_text()),energy_qualification="blocked_phase2",
            clock_policy="unchanged platform automatic policy",stopping="fixed budget or unchanged labels/shift<=1e-6 with cap"))
        index=[];results=[]

        def execute(dataset,counts,budget,mode,stage,block=None,golden=None,profile=False):
            case=dict(dataset=dataset,n=metadata[dataset]["n"],cpu_counts=counts,max_iterations=budget,mode=mode,
                      workers_requested=workers,stage=stage,block=block,record=f"trial-{len(index):03d}.json",golden=golden)
            argv=[str(binary),str((output/"datasets"/(dataset+".kmdat")).resolve()),",".join(map(str,counts)),str(budget),str(workers),mode]
            if profile:argv=["nsys","profile","--trace=cuda,nvtx","--sample=none","--cpuctxsw=none","--export=sqlite",f"--output={output/'profile'}",*argv]
            process=command(argv,timeout=120);save_new(output/case["record"],process);index.append(case)
            expected=json.loads((output/golden).read_text()) if golden else None
            result=assess(case,metadata[dataset],process,expected,profile);results.append(result)
            print(json.dumps(dict(trial=len(index)-1,dataset=dataset,stage=stage,cpu_counts=counts,status=result["status"])),flush=True)
            if result["status"]!="passed":raise RuntimeError(f"Kmeans trial failed: {result.get('reason')}")

        for name in fixtures():
            points,centers,_=datasets[name];n=len(points)
            for mode,budget in (("fixed",5),("converge",20)):
                golden=f"golden/{name}-{mode}.json";save_new(output/golden,trajectory(points,centers,budget,mode))
                counts=sorted({0,n,n//4,n//2,3*n//4,1,n-1}) if mode=="fixed" else sorted({0,n//2,n})
                for count in counts:execute(name,[count],budget,mode,"correctness",golden=golden)
                if mode=="fixed":execute(name,[n,n//2,0,n,0],budget,mode,"migration",golden=golden)

        n=metadata["tuning"]["n"]
        def stage(name,dataset,counts,seed):
            order=[];rng=random.Random(seed)
            for block in range(blocks):
                shuffled=sorted(set(counts));rng.shuffle(shuffled)
                order.extend(dict(block=block,cpu_items=cut) for cut in shuffled)
            save_new(output/(name+"-plan.json"),dict(dataset=dataset,iterations=6,mode="fixed",seed=seed,order=order))
            for entry in order:execute(dataset,[entry["cpu_items"]],6,"fixed",name,entry["block"])
        coarse=sorted({0,n//4,n//2,3*n//4,n});stage("coarse_tuning","tuning",coarse,483)
        winner=best_point(curves(results,"coarse_tuning"))
        refined=sorted({max(0,winner-n//8),winner,min(n,winner+n//8)})
        stage("refine_tuning","tuning",refined,484)
        selected=best_point(curves(results,"refine_tuning"))
        save_new(output/"selection.json",dict(cpu_items=selected,selected_on="refine_tuning",coarse_best=winner,refined=refined,rule="minimum mean active iteration time, ties favor lower CPU share"))
        stage("heldout_evaluation","heldout",set(coarse)|set(refined),485)
        save_new(output/"profiler-version.json",command(["nsys","--version"]))
        execute("profile",[3*metadata["profile"]["n"]//4],3,"fixed","profile",profile=True)
        save_new(output/"trial-index.json",index)
        summary=summarize(output);save_new(output/"summary.json",summary)
        save_new(output/"timing-curves.json",summary["timing_curves"]);save_new(output/"timeline.json",summary["concurrency"]["timeline"])
        save_new(output/"checksums.json",{p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.rglob("*")) if p.is_file()})
        return summary
    except BaseException as error:
        if "index" in locals() and not (output/"trial-index.json").exists():save_new(output/"trial-index.json",index)
        save_new(output/"failure.json",dict(reason=f"{type(error).__name__}: {error}",status="failed"))
        raise


def replay(root):
    root=Path(root);checksums=json.loads((root/"checksums.json").read_text())
    actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p!=root/"checksums.json"}
    if actual!=set(checksums):raise ValueError("Archive inventory changed")
    for name,expected in checksums.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError(f"Evidence checksum mismatch: {name}")
    return summarize(root)
