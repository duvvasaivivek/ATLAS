"""Phase 3 native adapter, exact partitioning, validation matrix and replay."""

import hashlib
import json
import os
import platform
import struct
from fractions import Fraction
from pathlib import Path

from .capabilities import collect, command, utc_now
from .concurrency import analyze_profile
from .experiment import save_new


def partition(elements, share):
    if type(elements) is not int or not 1<=elements<=1048579:
        raise ValueError("elements must be an integer in 1..1048579")
    fraction = Fraction(share)
    if not 0<=fraction<=1:
        raise ValueError("CPU share must be in [0,1]")
    cpu = elements*fraction.numerator//fraction.denominator
    return dict(elements=elements,cpu_items=cpu,gpu_items=elements-cpu,
                requested_share=str(fraction),realized_share=cpu/elements)


def reference_sum(elements,rounds,seed):
    """Small independent Python golden check, separate from native oracle."""
    total = 0
    for i in range(elements):
        value = i^seed
        for _ in range(rounds):
            value = (value ^ (value<<13)) & 0xffffffff
            value = (value ^ (value>>17)) & 0xffffffff
            value = (value ^ (value<<5)) & 0xffffffff
        total += value
    return total


def make_case(n,share,rounds=128,iterations=3,workers=4,seed=1729):
    case = partition(n,share)
    for name,value,low,high in (("rounds",rounds,1,4096),("iterations",iterations,1,100),
                                ("workers",workers,1,8),("seed",seed,1,0xffffffff)):
        if type(value) is not int or not low<=value<=high:
            raise ValueError(f"{name} outside {low}..{high}")
    case.update(rounds=rounds,iterations=iterations,workers_requested=workers,seed=seed,
                input_sha256=hashlib.sha256(b"".join(struct.pack("<I",i^seed) for i in range(n))).hexdigest())
    return case


def case_matrix(workers):
    cases = []
    for n in (1,2,255,256,257,4099,65539):
        seen = set()
        for share in ("0","1","0.25","0.5","0.75",f"1/{n}",f"{n-1}/{n}"):
            case = make_case(n,share,workers=workers,seed=1729)
            if case["cpu_items"] not in seen:
                cases.append(case)
                seen.add(case["cpu_items"])
    return cases


def native_args(binary,case):
    return [str(binary),*[str(case[k]) for k in ("elements","cpu_items","rounds","iterations","workers_requested","seed")]]


def summarize_case(case,process,profiled=False):
    result = dict(status="failed",case=case,iterations=[],energy_j=None,
                  energy_qualification="blocked_phase2",device_event_timing="not_used_phase2_block",
                  execution_mode="profiled_hardware" if profiled else "hardware_static_split")
    try:
        if process["status"]!="ok" or process["returncode"]!=0:
            raise ValueError("Native/profiler command failed or timed out")
        lines=process["stdout"].splitlines()
        records=[json.loads(line) for line in lines if line.startswith("{") or not profiled]
        if len(records)!=case["iterations"]+2 or records[0]["kind"]!="configuration":
            raise ValueError("Missing or extra native records")
        cfg=records[0]
        if cfg["schema_version"]!="3.0" or cfg["clock"]!="CLOCK_MONOTONIC":
            raise ValueError("Unsupported schema/clock")
        if any(cfg[k]!=case[k] for k in ("elements","cpu_items","gpu_items","rounds","iterations","workers_requested","seed")):
            raise ValueError("Executed configuration does not match requested work")
        if cfg["oracle"]!="masked_uint64_xorshift" or cfg["tolerance"]!=0 or cfg["buffer_capacity_items"]!=case["elements"] or cfg["allocation_policy"]!="allocate_once_reuse":
            raise ValueError("Missing correctness/allocation contract")
        if case["elements"]<=257 and cfg["reference_sum"]!=reference_sum(case["elements"],case["rounds"],case["seed"]):
            raise ValueError("Native oracle disagrees with independent Python golden sum")
        previous=cfg["preparation_start_ns"]
        for index,r in enumerate(records[1:]):
            if r["kind"]!="iteration" or r["iteration"]!=index or r["warmup"] is not (index==0):
                raise ValueError("Invalid warm-up/iteration sequence")
            if r["status"]!="passed" or r["mismatches"]!=0 or r["coverage_errors"]!=0 or r["elements_verified"]!=case["elements"]:
                raise ValueError("Output or exactly-once coverage validation failed")
            if r["combined_sum"]!=cfg["reference_sum"] or r["cpu_sum"]+r["gpu_sum"]!=r["combined_sum"]:
                raise ValueError("Reduction mismatch")
            times=[r[k] for k in ("reset_start_ns","reset_end_ns","release_ns","barrier_end_ns","merge_end_ns","validation_end_ns")]
            if any(type(t) is not int or t<0 for t in times) or times!=sorted(times) or times[0]<previous:
                raise ValueError("Invalid shared timing sequence")
            previous=r["validation_end_ns"]
            start=r["release_ns"]
            branches={}
            for branch,keys in (("cpu",("cpu_start_ns","cpu_end_ns")),
                                ("gpu",("gpu_dispatch_ns","h2d_end_ns","compute_end_ns","gpu_end_ns"))):
                stamps=[r[k] for k in keys]
                if case[branch+"_items"]:
                    if any(type(t) is not int for t in stamps) or stamps!=sorted(stamps) or stamps[0]<start or stamps[-1]>r["barrier_end_ns"] or stamps[-1]<=start:
                        raise ValueError(f"Invalid {branch} timing")
                    branches[branch]=(stamps[-1]-start)/1e9
                else:
                    if any(t is not None for t in stamps) or r[branch+"_sum"]!=0:
                        raise ValueError("Empty branch reported work/timing")
                    branches[branch]=None
            spans=r["worker_spans"]
            if case["cpu_items"]:
                if not 1<=r["workers_used"]<=case["workers_requested"] or len(spans)!=case["workers_requested"]:
                    raise ValueError("CPU worker contract failed")
                if [s["worker"] for s in spans]!=list(range(len(spans))) or sum(s["items"] for s in spans)!=case["cpu_items"]:
                    raise ValueError("CPU worker coverage failed")
                for span in spans:
                    if span["items"]>0 and not r["cpu_start_ns"]<=span["start_ns"]<span["end_ns"]<=r["cpu_end_ns"]:
                        raise ValueError("Invalid CPU compute span")
            elif spans or r["workers_used"]!=0:
                raise ValueError("Empty CPU branch started a team")
            if index:
                result["iterations"].append(dict(iteration=index,duration_s=(r["merge_end_ns"]-start)/1e9,
                    cpu_completion_s=branches["cpu"],gpu_completion_s=branches["gpu"],
                    merge_s=(r["merge_end_ns"]-r["barrier_end_ns"])/1e9,
                    final_host_wait_s=(r["barrier_end_ns"]-(r["gpu_end_ns"] or start))/1e9,
                    validation_s=(r["validation_end_ns"]-r["merge_end_ns"])/1e9,
                    imbalance=abs(branches["cpu"]-branches["gpu"])/max(branches.values()) if all(v is not None for v in branches.values()) else None))
        result.update(status="passed",warmup_iterations=1,exactly_once=True,
                      preparation_s=(records[1]["reset_start_ns"]-cfg["preparation_start_ns"])/1e9,
                      reference_sum=cfg["reference_sum"],concurrency_proof="requires_separate_profiler_evidence")
    except (KeyError,ValueError,TypeError,IndexError,ZeroDivisionError) as error:
        result["reason"]=str(error)
    return result


def summarize_archive(root):
    root=Path(root)
    manifest=json.loads((root/"manifest.json").read_text())
    cases=[]
    for index,case in enumerate(manifest["cases"]):
        process=json.loads((root/f"case-{index:03d}.json").read_text())
        cases.append(summarize_case(case,process))
    profiled=summarize_case(manifest["profile_case"],json.loads((root/"profile-command.json").read_text()),profiled=True)
    overlap=analyze_profile(root/"profile.sqlite",manifest["profile_case"]["iterations"])
    equivalence=[]
    for size in sorted({c["case"]["elements"] for c in cases}):
        group=[c for c in cases if c["case"]["elements"]==size]
        identical_input=len({(c["case"]["input_sha256"],c["case"]["rounds"],c["case"]["seed"]) for c in group})==1
        equivalent=identical_input and all(c["status"]=="passed" for c in group) and len({c.get("reference_sum") for c in group})==1
        equivalence.append(dict(elements=size,splits=len(group),identical_input=identical_input,equivalent_outputs=equivalent))
    passed=bool(cases) and all(e["equivalent_outputs"] for e in equivalence) and profiled["status"]=="passed" and overlap["status"]=="demonstrated"
    return dict(schema_version="3.0",status="passed" if passed else "incomplete",cases=cases,
                split_equivalence=equivalence,
                profiled_case=profiled,concurrency=overlap,
                energy_qualification="blocked_phase2",device_event_timing="not_used_phase2_block",
                hardware_settings_changed=False)


def run_static_checks(output,binary,workers=4):
    if platform.system()!="Linux":
        raise ValueError("Run the native adapter in Linux/WSL")
    if not 1<=workers<=min(8,max(1,(os.cpu_count() or 1)-2)):
        raise ValueError("Worker count must be 1..8 and reserve two visible CPUs when possible")
    binary=Path(binary).resolve(strict=True)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    try:
        collect(output/"environment","/usr/local/cuda-13.1/bin/nvcc",None)
        cases=case_matrix(workers)
        profile_case=make_case(1048579,"0.25",rounds=1024,iterations=3,workers=workers)
        save_new(output/"manifest.json",dict(schema_version="3.0",started_utc=utc_now(),cases=cases,
                 profile_case=profile_case,binary=str(binary),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                 execution="native subprocess with concurrent OpenMP CPU team and CUDA main thread",
                 input_generator="uint32 index xor seed; SHA256 of little-endian uint32 input",
                 cpu_policy="unchanged",gpu_policy="unchanged",energy_qualification="blocked_phase2",
                 source_hashes=json.loads((output/"environment/source_hashes.json").read_text())))
        cache=binary.parent/"CMakeCache.txt"
        if cache.is_file(): (output/"environment/raw/CMakeCache.txt").write_bytes(cache.read_bytes())
        for index,case in enumerate(cases):
            process=command(native_args(binary,case),timeout=60)
            save_new(output/f"case-{index:03d}.json",process)
            result=summarize_case(case,process)
            print(json.dumps(dict(case=index,elements=case["elements"],cpu_items=case["cpu_items"],status=result["status"])),flush=True)
            if result["status"]!="passed": raise RuntimeError(f"Case {index} failed: {result.get('reason')}")
        save_new(output/"profiler-version.json",command(["nsys","--version"]))
        process=command(["nsys","profile","--trace=cuda,nvtx","--sample=none","--cpuctxsw=none",
                         "--export=sqlite",f"--output={output/'profile'}",*native_args(binary,profile_case)],timeout=120)
        save_new(output/"profile-command.json",process)
        result=summarize_archive(output)
        save_new(output/"summary.json",result)
        save_new(output/"timeline.json",result["concurrency"]["timeline"])
        save_new(output/"checksums.json",{p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(output.rglob("*")) if p.is_file()})
        return result
    except BaseException as error:
        save_new(output/"failure.json",dict(status="failed",reason=f"{type(error).__name__}: {error}"))
        raise


def replay_static(root):
    root=Path(root)
    checksums=json.loads((root/"checksums.json").read_text())
    files={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p!=root/"checksums.json"}
    if files!=set(checksums): raise ValueError("Archive inventory changed")
    for name,expected in checksums.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f"Checksum mismatch: {name}")
    return summarize_archive(root)
