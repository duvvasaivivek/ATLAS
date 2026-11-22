"""Explicit compatibility check for immutable pre-remediation archives (WSL Python 3.12)."""
from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from greengpu import runner,static_execution,kmeans,allocation_experiment,hotspot,frequency_experiment,phase8,phase9


def main():
    root=Path(__file__).resolve().parents[1]
    checks=[(2,'final-v2',runner.replay),(3,'final',static_execution.replay_static),
            (4,'final-v2',kmeans.replay),(5,'final',allocation_experiment.replay),
            (6,'final',hotspot.replay),(7,'final',frequency_experiment.replay),
            (8,'final-v2',phase8.replay),(9,'final-v3',phase9.replay)]
    for phase,suffix,replay in checks:
        path=root/f'results/phase{phase}/2026-10-31-{suffix}'
        replay(path)
        print(json.dumps(dict(phase=phase,archive=path.relative_to(root).as_posix(),replay='passed')),flush=True)


if __name__=='__main__':main()
