#!/usr/bin/env python3
"""Build distributions and verify the installed wheel in a fresh Python environment.

Downloads only the pinned build backend and declared runtime dependency. Does not
change the active environment. Native CMake tests belong to validate_release.py.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import venv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    environment = output/'environment'
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    env = dict(os.environ)
    env.pop('PYTHONPATH',None)
    env['PYTHONDONTWRITEBYTECODE']='1'
    records=[]
    def command(name,argv,cwd=ROOT):
        print(name,flush=True)
        with (output/(name+'.txt')).open('x',encoding='utf-8') as log:
            process=subprocess.run([str(a) for a in argv],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        records.append(dict(name=name,argv=[str(a) for a in argv],returncode=process.returncode))
        if process.returncode:raise RuntimeError(f'{name} failed; inspect {output/(name+".txt")}')
    try:
        command('dependencies',[python,'-m','pip','install','setuptools==80.9.0','nvidia-ml-py==13.615.71'])
        distributions=output/'distributions';distributions.mkdir()
        command('build-distributions',[python,'-c',
            'from setuptools.build_meta import build_sdist, build_wheel; '
            f'build_sdist({str(distributions)!r}); build_wheel({str(distributions)!r})'])
        sdist,=distributions.glob('*.tar.gz')
        wheel,=distributions.glob('*.whl')
        source=output/'source';source.mkdir()
        with tarfile.open(sdist) as archive:
            archive.extractall(source,filter='data')
        extracted,=source.iterdir()
        for name in ('CMakeLists.txt','native/kmeans/main.cu','native/hotspot/main.cu',
                     'tests/data/kmeans_ties.kmdat','tests/data/hotspot_edge.hsdat',
                     'scripts/validate_release.py','docs/reproducibility.md','references/rodinia/LICENSE'):
            if not (extracted/name).is_file():raise RuntimeError('Incomplete source distribution: '+name)
        command('install-wheel',[python,'-m','pip','install','--no-deps',wheel],extracted)
        command('installed-import',[python,'-c',
            'import greengpu; from pathlib import Path; '
            f'assert Path(greengpu.__file__).resolve().is_relative_to(Path({str(environment)!r})); '
            'print(greengpu.__file__)'],extracted)
        command('installed-cli',[python,'-m','greengpu','--help'],extracted)
        command('installed-tests',[python,'-m','unittest','discover','-s','tests','-v'],extracted)
        artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (sdist,wheel)}
        result=dict(status='passed',artifacts=artifacts,commands=records,
                    scope='fresh installed Python package and source inventory; native validation is separate')
        (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(result,indent=2))
        return 0
    except Exception as error:
        (output/'failure.json').write_text(json.dumps(dict(error=str(error),commands=records),indent=2)+'\n',encoding='utf-8')
        print(str(error),file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())
