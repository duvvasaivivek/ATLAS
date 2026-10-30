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
    args = parser.parse_args(argv)
    if not Path("pyproject.toml").is_file() or not Path("src/greengpu").is_dir():
        parser.error("Run this command from the ATLAS repository root")
    if args.command == "replay":
        from .runner import replay
        print(json.dumps(replay(args.run), indent=2, allow_nan=False))
        return 0
    if args.output.exists():
        parser.error("Output already exists; use a new directory to preserve earlier evidence")
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
