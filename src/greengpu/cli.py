"""Discovery and measurement command line; run from the repository root."""

import argparse
import json
from pathlib import Path

from .capabilities import collect


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="atlas")
    sub = parser.add_subparsers(dest="command", required=True)
    discover = sub.add_parser("discover", help="Read-only hardware and environment snapshot")
    discover.add_argument("--output", type=Path, required=True, help="New directory for retained evidence")
    discover.add_argument("--cuda-compiler", help="Explicit nvcc path, independent of interactive PATH")
    discover.add_argument("--smoke-binary", type=Path, help="Run the built CPU/CUDA correctness fixture")
    measure = sub.add_parser("calibrate", help="Record a pilot and paired fixed-work GPU measurement trials")
    measure.add_argument("--output", type=Path, required=True)
    measure.add_argument("--binary", type=Path, required=True)
    measure.add_argument("--blocks", type=int, default=3)
    measure.add_argument("--target-seconds", type=float, default=6.0)
    measure.add_argument("--period", type=float, default=0.1)
    measure.add_argument("--seed", type=int, default=1729)
    replay_parser = sub.add_parser("replay", help="Verify evidence checksums and regenerate a Phase 2 summary on stdout")
    replay_parser.add_argument("run", type=Path)
    static_parser = sub.add_parser("static-check", help="Validate static CPU/GPU partitions and capture concurrency evidence")
    static_parser.add_argument("--output", type=Path, required=True)
    static_parser.add_argument("--binary", type=Path, required=True)
    static_parser.add_argument("--workers", type=int, default=4)
    static_replay = sub.add_parser("static-replay", help="Verify and replay a Phase 3 validation archive")
    static_replay.add_argument("run", type=Path)
    kmeans_parser = sub.add_parser("kmeans-check", help="Validate kmeans, migration, and a held-out static split sweep")
    kmeans_parser.add_argument("--output",type=Path,required=True)
    kmeans_parser.add_argument("--binary",type=Path,required=True)
    kmeans_parser.add_argument("--workers",type=int,default=4)
    kmeans_parser.add_argument("--blocks",type=int,default=3)
    kmeans_replay = sub.add_parser("kmeans-replay",help="Verify and regenerate a Phase 4 archive")
    kmeans_replay.add_argument("run",type=Path)
    kmeans_plot = sub.add_parser("kmeans-plot",help="Plot verified timing curves outside the raw archive")
    kmeans_plot.add_argument("run",type=Path)
    kmeans_plot.add_argument("--output",type=Path,required=True)
    allocation = sub.add_parser("allocation-check",help="Validate Tier 1 policies and persistent kmeans feedback")
    allocation.add_argument("--output",type=Path,required=True)
    allocation.add_argument("--binary",type=Path,required=True)
    allocation.add_argument("--workers",type=int,default=4)
    allocation.add_argument("--blocks",type=int,default=3)
    allocation_replay = sub.add_parser("allocation-replay",help="Verify Phase 5 evidence and replay every decision")
    allocation_replay.add_argument("run",type=Path)
    allocation_plot = sub.add_parser("allocation-plot",help="Plot verified Phase 5 timing and allocation traces")
    allocation_plot.add_argument("run",type=Path)
    allocation_plot.add_argument("--output",type=Path,required=True)
    hotspot = sub.add_parser("hotspot-check",help="Validate thermal state migration and allocation policies")
    hotspot.add_argument("--output",type=Path,required=True)
    hotspot.add_argument("--binary",type=Path,required=True)
    hotspot.add_argument("--workers",type=int,default=4)
    hotspot.add_argument("--blocks",type=int,default=3)
    hotspot_replay = sub.add_parser("hotspot-replay",help="Verify and replay Phase 6 evidence")
    hotspot_replay.add_argument("run",type=Path)
    hotspot_plot = sub.add_parser("hotspot-plot",help="Plot verified Phase 6 thermal/timing evidence")
    hotspot_plot.add_argument("run",type=Path)
    hotspot_plot.add_argument("--output",type=Path,required=True)
    frequency = sub.add_parser("frequency-check",help="Validate offline Tier 2 policies; never change clocks")
    frequency.add_argument("--output",type=Path,required=True)
    frequency.add_argument("--hardware-trace",action="store_true",help="Also capture a read-only NVML trace")
    frequency_replay = sub.add_parser("frequency-replay",help="Verify and replay every Phase 7 decision")
    frequency_replay.add_argument("run",type=Path)
    frequency_plot = sub.add_parser("frequency-plot",help="Plot verified offline frequency diagnostics")
    frequency_plot.add_argument("run",type=Path)
    frequency_plot.add_argument("--output",type=Path,required=True)
    phase8 = sub.add_parser("actuation-check",help="Check restoration capability and characterize real GPU diagnostics")
    phase8.add_argument("--output",type=Path,required=True)
    phase8.add_argument("--binary",type=Path,required=True)
    phase8.add_argument("--blocks",type=int,default=3)
    phase8.add_argument("--target-seconds",type=float,default=2.)
    phase8_replay = sub.add_parser("actuation-replay",help="Verify and regenerate Phase 8 hardware evidence")
    phase8_replay.add_argument("run",type=Path)
    args = parser.parse_args(argv)
    if not Path("pyproject.toml").is_file() or not Path("src/greengpu").is_dir():
        parser.error("Run this command from the ATLAS repository root")
    if args.command in ("actuation-check","actuation-replay"):
        from .phase8 import run, replay
        try:
            result=run(args.output,args.binary,args.blocks,args.target_seconds) if args.command=="actuation-check" else replay(args.run)
            print(json.dumps({k:result[k] for k in ("diagnostic_status","hardware_exit_gate","reason","clock_setter_calls","measured_trials","matched_loaded_samples")},indent=2))
        except (ValueError,RuntimeError,OSError,ImportError) as error:
            parser.exit(1,f"Actuation validation failed: {error}\n")
        return 0
    if args.command in ("frequency-check", "frequency-replay", "frequency-plot"):
        from .frequency_experiment import run, replay
        try:
            if args.command == "frequency-plot":
                from .frequency_plot import render
                print(render(args.run,args.output))
            else:
                result = run(args.output,args.hardware_trace) if args.command == "frequency-check" else replay(args.run)
                print(json.dumps({k:result[k] for k in ("implementation_status","synthetic_observations","hardware","offline_gate","live_actuation")},indent=2))
        except (ValueError, OSError, ImportError, ArithmeticError) as error:
            parser.exit(1,f"Frequency validation failed: {error}\n")
        return 0
    if args.command == "replay":
        from .runner import replay
        print(json.dumps(replay(args.run), indent=2, allow_nan=False))
        return 0
    if args.command == "static-replay":
        from .static_execution import replay_static
        print(json.dumps(replay_static(args.run), indent=2, allow_nan=False))
        return 0
    if args.command == "kmeans-replay":
        from .kmeans import replay
        print(json.dumps(replay(args.run),indent=2,allow_nan=False))
        return 0
    if args.command == "kmeans-plot":
        from .kmeans_plot import render
        try:
            print(render(args.run,args.output))
        except (ValueError,OSError,ImportError) as error:
            parser.exit(1,f"Plot failed: {error}\n")
        return 0
    if args.command == "allocation-plot":
        from .allocation_plot import render
        print(render(args.run,args.output))
        return 0
    if args.command == "allocation-replay":
        from .allocation_experiment import replay
        result=replay(args.run)
        print(json.dumps(dict(status=result['implementation_status'],trials=len(result['results']),full_phase_gate=result['full_phase_gate']),indent=2))
        return 0
    if args.command == "hotspot-plot":
        from .hotspot_plot import render
        print(render(args.run,args.output))
        return 0
    if args.command == "hotspot-replay":
        from .hotspot import replay
        result=replay(args.run)
        print(json.dumps(dict(status=result['implementation_status'],trials=len(result['results']),steps=result['measured_steps']),indent=2))
        return 0 if result['implementation_status']=='passed' else 1
    if args.output.exists():
        parser.error("Output already exists; use a new directory to preserve earlier evidence")
    if args.command == "hotspot-check":
        from .hotspot import run
        try:
            result=run(args.output,args.binary,args.workers,args.blocks)
        except (ValueError,RuntimeError,OSError) as error:
            parser.exit(1,f"HotSpot validation failed: {error}\n")
        print(json.dumps(dict(output=str(args.output),status=result['implementation_status'],trials=len(result['results']),steps=result['measured_steps'],energy=result['energy_qualification']),indent=2))
        return 0 if result['implementation_status']=='passed' else 1
    if args.command == "allocation-check":
        from .allocation_experiment import run
        try:
            result=run(args.output,args.binary,args.workers,args.blocks)
        except (ValueError,RuntimeError,OSError) as error:
            parser.exit(1,f"Allocation validation failed: {error}\n")
        print(json.dumps(dict(output=str(args.output),status=result['implementation_status'],trials=len(result['results']),full_phase_gate=result['full_phase_gate']),indent=2))
        return 0 if result['implementation_status']=='passed' else 1
    if args.command == "kmeans-check":
        from .kmeans import run
        try:
            result=run(args.output,args.binary,args.workers,args.blocks)
        except (ValueError,RuntimeError,OSError) as error:
            parser.exit(1,f"Kmeans validation failed: {error}\n")
        print(json.dumps(dict(output=str(args.output),status=result["implementation_status"],full_phase_gate=result["full_phase_gate"],
                              trials=len(result["results"]),concurrency=result["concurrency"]["status"]),indent=2))
        return 0 if result["implementation_status"]=="passed" else 1
    if args.command == "static-check":
        from .static_execution import run_static_checks
        try:
            result = run_static_checks(args.output,args.binary,args.workers)
        except (ValueError,RuntimeError,OSError) as error:
            parser.exit(1,f"Static validation failed: {error}\n")
        print(json.dumps(dict(output=str(args.output),status=result["status"],cases=len(result["cases"]),
                              concurrency=result["concurrency"]["status"]),indent=2))
        return 0 if result["status"]=="passed" else 1
    if args.command == "calibrate":
        from .runner import run_calibration
        try:
            result = run_calibration(args.output, args.binary, args.blocks, args.target_seconds, args.period, args.seed)
        except (ValueError, RuntimeError, OSError, ImportError) as error:
            parser.exit(1, f"Calibration failed: {error}\n")
        print(json.dumps({"output":str(args.output), "overhead_percent":result["overhead_percent"],
                          "monitored_gpu_j":result["monitored_gpu_j"],
                          "energy_qualification":result["energy_qualification"],
                          "device_timing_qualification":result["device_timing_qualification"]}, indent=2))
        return 0
    result = collect(args.output, args.cuda_compiler, args.smoke_binary)
    print(json.dumps({"output": str(args.output), "gpu_query": result["gpu"]["status"],
                      "smoke": result["smoke"]["status"], "path": result["chosen_path"]}, indent=2))
    if args.smoke_binary and result["smoke"]["status"] != "passed":
        return 1
    return 0 if result["gpu"]["status"] == "available" else 1
