#!/usr/bin/env python3
"""Run the reproducible acceptance workflow on Linux/WSL with a CUDA GPU.

Exit 0: full acceptance. Exit 2: evidence generated, capability gates incomplete.
Exit 1: execution/validation failed. Every invocation requires a new output path.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from greengpu.experiment import save_new
from greengpu.frequency_experiment import sha
from greengpu.research_report import run as report
from greengpu.supervision import capture_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True, help='Fresh CMake build directory')
    parser.add_argument('--cuda-compiler', type=Path, default=Path('/usr/local/cuda-13.1/bin/nvcc'))
    parser.add_argument('--blocks', type=int, default=10)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Run in native Linux or Ubuntu/WSL with the CUDA toolchain')
    if not 2 <= args.blocks <= 30:
        parser.error('--blocks must be 2..30; fewer than ten is exploratory only')
    output, build = args.output.resolve(), args.build.resolve()
    if output.exists() or build.exists():
        parser.error('Output and build must both be fresh paths; existing evidence is never overwritten')
    if output.is_relative_to(build) or build.is_relative_to(output):
        parser.error('Build and output directories must be separate')
    os.chdir(ROOT)
    os.environ['PYTHONPATH'] = str(ROOT/'src')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['OMP_NUM_THREADS'] = '4'
    output.mkdir(parents=True, exist_ok=False)
    stages = []
    def stage(name, argv, timeout, allowed=(0,)):
        print(json.dumps(dict(stage=name, status='running')), flush=True)
        started = time.monotonic()
        try:
            result = capture_command([str(a) for a in argv], timeout)
        except subprocess.TimeoutExpired as error:
            def decoded(value):
                return value.decode('utf-8',errors='replace') if isinstance(value,bytes) else value or ''
            (output/(name+'.txt')).write_text(decoded(error.output)+'\n'+decoded(error.stderr),encoding='utf-8')
            row=dict(stage=name,argv=[str(a) for a in argv],returncode=None,timed_out=True,
                     elapsed_s=time.monotonic()-started,log=name+'.txt')
            stages.append(row);save_new(output/(name+'.json'),row)
            raise RuntimeError(f'{name} exceeded {timeout}s; partial output retained') from error
        (output/(name+'.txt')).write_text(result.stdout+'\n'+result.stderr, encoding='utf-8')
        row = dict(stage=name, argv=[str(a) for a in argv], returncode=result.returncode,
                   elapsed_s=time.monotonic()-started, log=name+'.txt')
        stages.append(row)
        save_new(output/(name+'.json'), row)
        if result.returncode not in allowed:
            raise RuntimeError(f'{name} failed with exit code {result.returncode}; see {output/(name+".txt")}')
        print(json.dumps(dict(stage=name, status='recorded', returncode=result.returncode)), flush=True)
    cli = [sys.executable, '-m', 'greengpu']
    try:
        stage('unit-tests', [sys.executable,'-m','unittest','discover','-s','tests','-v'], 180)
        stage('configure', ['cmake','-S','.', '-B',build,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
                            f'-DCMAKE_CUDA_COMPILER={args.cuda_compiler}',f'-DPython3_EXECUTABLE={sys.executable}'], 300)
        stage('build', ['cmake','--build',build,'--parallel','2'], 600)
        stage('native-tests', ['ctest','--test-dir',build,'--output-on-failure'], 600)
        stage('sanitizers', [sys.executable,'tests/native_sanitizers.py',build,output/'sanitizers'], 1200)
        stage('offline-frequency', [*cli,'frequency-check','--output',output/'frequency'], 180)
        stage('qualification', [*cli,'qualification-check','--build',build,'--output',output/'qualification'], 600)
        # Incomplete concurrency is retained, then the allocation study proceeds.
        # Missing/malformed archives still fail during the report's strict replay.
        stage('integration', [*cli,'integration-check','--kmeans-binary',build/'atlas_kmeans',
                              '--hotspot-binary',build/'atlas_hotspot','--output',output/'integration'], 1200, (0,1))
        stage('performance', [*cli,'performance-check','--build',build,'--blocks',args.blocks,
                              '--output',output/'performance'], 1800)
        result = report(output/'report',output/'performance',output/'integration',output/'qualification')
        save_new(output/'acceptance.json',dict(status=result['status'],full_project_complete=result['full_project_complete'],
                                              blockers=result['blockers'],stages=stages))
        sources = [p for folder in ('src','native','tests','scripts') for p in (ROOT/folder).rglob('*')
                   if p.is_file() and p.suffix in ('.py','.cu','.cpp','.hpp','.sh')]
        sources += [ROOT/p for p in ('pyproject.toml','CMakeLists.txt','MANIFEST.in')]
        save_new(output/'release-source-hashes.json',{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)})
        print(json.dumps(dict(report=str(output/'report/report.html'),status=result['status']),indent=2))
        return 0 if result['full_project_complete'] else 2
    except Exception as error:
        save_new(output/'failure.json',dict(error=f'{type(error).__name__}: {error}',stages=stages))
        print(str(error),file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
