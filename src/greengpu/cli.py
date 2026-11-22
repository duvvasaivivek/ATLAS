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
    integration=sub.add_parser("integration-check",help="Real allocation/shadow integration and synthetic joint-control validation")
    integration.add_argument("--output",type=Path,required=True)
    integration.add_argument("--kmeans-binary",type=Path,required=True)
    integration.add_argument("--hotspot-binary",type=Path,required=True)
    integration.add_argument("--workers",type=int,default=4)
    integration.add_argument("--mode",choices=("shadow-validation","live-joint"),default="shadow-validation")
    integration_replay=sub.add_parser("integration-replay",help="Verify and regenerate integrated runtime evidence")
    integration_replay.add_argument("run",type=Path)
    qualification=sub.add_parser("qualification-check",help="Fresh read-only loaded telemetry and energy qualification")
    qualification.add_argument("--output",type=Path,required=True)
    qualification.add_argument("--build",type=Path,required=True)
    qualification.add_argument("--blocks",type=int,default=3)
    qualification_replay=sub.add_parser("qualification-replay",help="Verify and regenerate read-only hardware qualification")
    qualification_replay.add_argument("run",type=Path)
    performance=sub.add_parser("performance-check",help="Randomized matched remediation evaluation")
    performance.add_argument("--output",type=Path,required=True)
    performance.add_argument("--build",type=Path,required=True)
    performance.add_argument("--baseline-build",type=Path)
    performance.add_argument("--blocks",type=int,default=10)
    performance_replay=sub.add_parser("performance-replay",help="Verify and regenerate a matched evaluation archive")
    performance_replay.add_argument("run",type=Path)
    report=sub.add_parser("research-report",help="Verify archives and generate the final capability-limited research report")
    report.add_argument("--performance",type=Path,required=True)
    report.add_argument("--integration",type=Path,required=True)
    report.add_argument("--qualification",type=Path,required=True)
    report.add_argument("--output",type=Path,required=True)
    report.add_argument("--require-full",action="store_true",help="Return exit code 2 if full-project acceptance is incomplete")
    live=sub.add_parser('live-run',help='Run guarded clock control using an independently qualified platform provider')
    live.add_argument('--workload',choices=('kmeans','hotspot'),required=True)
    live.add_argument('--binary',type=Path,required=True)
    live.add_argument('--dataset',type=Path,required=True)
    live.add_argument('--output',type=Path,required=True)
    live.add_argument('--provider',help='Trusted installed Python module:factory; default NVML provider remains blocked')
    live.add_argument('--steps',type=int,default=1000)
    live.add_argument('--workers',type=int,default=4)
    live.add_argument('--cpu-fraction',type=float,default=.5)
    live.add_argument('--allocation-policy',default='paper_simple')
    live.add_argument('--frequency-policy',choices=('paper_additive','discounted_loss_v1'),default='paper_additive')
    live.add_argument('--initial-clock',nargs=2,type=int,metavar=('CORE_MHZ','MEMORY_MHZ'),
                      help='Select an initial pair from the verified settings')
    live.add_argument('--fixed-clock',action='store_true',help='Verify and hold the selected clock pair throughout the run')
    live.add_argument('--measure-energy',action='store_true',help='Use the provider’s independently calibrated power trace')
    live_replay=sub.add_parser('live-replay',help='Verify joint-control decisions, native correctness and restoration journals')
    live_replay.add_argument('run',type=Path)
    controlled=sub.add_parser('controlled-check',help='Run matched fixed-clock, frequency-only and joint energy trials')
    controlled.add_argument('--provider',required=True,help='Trusted qualified provider module:factory')
    controlled.add_argument('--output',type=Path,required=True)
    controlled.add_argument('--binary',type=Path,required=True)
    controlled.add_argument('--workload',choices=('kmeans','hotspot'),required=True)
    controlled.add_argument('--tuning',type=Path,required=True)
    controlled.add_argument('--held-out',type=Path,required=True)
    controlled.add_argument('--high-clock',nargs=2,type=int,metavar=('CORE_MHZ','MEMORY_MHZ'),required=True)
    controlled.add_argument('--clock',nargs=2,type=int,action='append',metavar=('CORE_MHZ','MEMORY_MHZ'),required=True)
    controlled.add_argument('--steps',type=int,default=10000)
    controlled.add_argument('--workers',type=int,default=4)
    controlled.add_argument('--blocks',type=int,default=10)
    controlled.add_argument('--initial-fraction',type=float,choices=(.15,.5,.9),default=.5)
    controlled_replay=sub.add_parser('controlled-replay',help='Regenerate matched energy and timing results from a retained archive')
    controlled_replay.add_argument('run',type=Path)
    args = parser.parse_args(argv)
    if not Path("pyproject.toml").is_file() or not Path("src/greengpu").is_dir():
        parser.error("Run this command from the ATLAS repository root")
    if args.command in ('live-run','live-replay'):
        from .live import load_provider,run_live,replay
        from .actuation import ActuationBlocked
        provider=None
        try:
            if args.command=='live-replay':result=replay(args.run)
            else:
                import math,struct
                from .integration import Config
                from .live_native import NativeWorkload
                if not math.isfinite(args.cpu_fraction) or not 0<=args.cpu_fraction<1:raise ValueError('CPU fraction must be in [0,1)')
                provider=load_provider(args.provider)
                qualification=provider.qualification()
                raw=args.dataset.read_bytes()
                if len(raw)<16 or raw[:8]!=(b'ATLSKM01' if args.workload=='kmeans' else b'ATLSHS01'):raise ValueError('Wrong or truncated input format')
                n=struct.unpack_from('<I',raw,8)[0]
                cfg=Config(args.output.name,n,int(n*args.cpu_fraction),tuple(map(tuple,qualification.get('verified_pairs',()))),
                           mode='allocation_shadow',allocation_policy=args.allocation_policy,frequency_policy=args.frequency_policy)
                workload=NativeWorkload(args.binary,args.dataset,args.workload,args.steps,args.workers,args.output/'native')
                result=run_live(provider,workload,cfg,args.output,initial_pair=args.initial_clock,
                                frequency_control=not args.fixed_clock,measure=args.measure_energy)
                replay(args.output)
            keys=('status','evidence_kind','physical_control_evidence','restoration','energy_savings','energy')
            print(json.dumps({k:result[k] for k in keys if k in result},indent=2))
            return 0 if result['status']=='passed' and result.get('energy',{}).get('status','ok')=='ok' else 2
        except ActuationBlocked as error:parser.exit(2,f'Live control blocked: {error}\n')
        except (ValueError,RuntimeError,OSError,ImportError,AttributeError,TypeError,KeyError) as error:parser.exit(1,f'Live execution failed: {error}\n')
        finally:
            if provider is not None:provider.close()
    if args.command in ('controlled-check','controlled-replay'):
        from .actuation import ActuationBlocked
        from .controlled_evaluation import replay, run
        from .live import load_provider
        provider=None
        try:
            if args.command=='controlled-replay':
                result=replay(args.run)
            else:
                provider=load_provider(args.provider)
                result=run(provider,args.output,args.binary,args.workload,args.tuning,args.held_out,
                           args.high_clock,args.clock,steps=args.steps,workers=args.workers,blocks=args.blocks,
                           initial_fraction=args.initial_fraction)
            keys=('status','domain','scope','failed_trials','selection','comparisons')
            print(json.dumps({key:result[key] for key in keys},indent=2))
            return 0 if result['status']=='passed' else 2
        except ActuationBlocked as error:parser.exit(2,f'Controlled evaluation blocked: {error}\n')
        except (ValueError,RuntimeError,OSError,ImportError,AttributeError,TypeError,KeyError) as error:
            parser.exit(1,f'Controlled evaluation failed: {error}\n')
        finally:
            if provider is not None:provider.close()
    if args.command=="research-report":
        from .research_report import run
        try:
            result=run(args.output,args.performance,args.integration,args.qualification)
            print(json.dumps(dict(status=result['status'],full_project_complete=result['full_project_complete'],
                                  report=str(args.output/'report.html'),blockers=result['blockers']),indent=2))
            return 2 if args.require_full and not result['full_project_complete'] else 0
        except (ValueError,RuntimeError,OSError,KeyError,ImportError) as error:
            parser.exit(1,f"Research report failed: {error}\n")
    if args.command in ("qualification-check","qualification-replay","performance-check","performance-replay"):
        try:
            if args.command=="performance-replay":
                from .performance import replay
                result=replay(args.run);result.pop('results',None)
            elif args.command=="qualification-check":
                from .qualification import run
                result=run(args.output,args.build,args.blocks)
            elif args.command=="qualification-replay":
                from .qualification import replay
                result=replay(args.run)
            else:
                from .performance import run
                result=run(args.output,args.build,args.baseline_build,args.blocks)
            print(json.dumps(result,indent=2));return 1 if result.get('status')=='incomplete_failed_trials' else 0
        except (ValueError,RuntimeError,OSError,ImportError) as error:parser.exit(1,f"Evaluation failed: {error}\n")
    if args.command in ("integration-check","integration-replay"):
        from .phase9 import run,replay
        if args.command=="integration-check" and args.mode=="live-joint":
            parser.exit(2,"Live joint control is blocked: authoritative prior-policy readback, verified actuation/restoration, and live telemetry qualification are unavailable. No fallback or workload launch.\n")
        try:
            result=run(args.output,args.kmeans_binary,args.hotspot_binary,args.workers) if args.command=="integration-check" else replay(args.run)
            print(json.dumps({k:result[k] for k in ("software_gate","live_joint_hardware_gate","hardware_trials","iterations","shadow_frequency_updates","synthetic_cases")},indent=2))
        except (ValueError,RuntimeError,OSError,ImportError) as error:parser.exit(1,f"Integration failed: {error}\n")
        return 0 if result["software_gate"]=="passed" else 1
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
