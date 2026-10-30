"""Minimal Phase 1 command line; run from the repository root."""

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
    args = parser.parse_args(argv)
    if not Path("pyproject.toml").is_file() or not Path("src/greengpu").is_dir():
        parser.error("Run this command from the ATLAS repository root")
    if args.output.exists():
        parser.error("Output already exists; use a new directory to preserve earlier evidence")
    result = collect(args.output, args.cuda_compiler, args.smoke_binary)
    print(json.dumps({"output": str(args.output), "gpu_query": result["gpu"]["status"],
                      "smoke": result["smoke"]["status"], "path": result["chosen_path"]}, indent=2))
    if args.smoke_binary and result["smoke"]["status"] != "passed":
        return 1
    return 0 if result["gpu"]["status"] == "available" else 1
