"""Reproduce additional paired command/loop comparisons from the final archive."""
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from greengpu.performance import interval

root=Path(sys.argv[1])
summary=json.loads((root/'summary.json').read_text())
results=[]
for workload in ('kmeans','hotspot'):
    for size in (0,1):
        rows=[r for r in summary['results'] if (r['workload'],r['size'],r['stage'],r['status'])==(workload,size,'evaluation','passed')]
        for candidate,baseline in (('static50','legacy_static50'),('batched_atomic','legacy_static50'),('tuned_static','gpu_only'),('cost_aware','gpu_only'),('overlap50','static50')):
            a={r['block']:r for r in rows if r['name']==candidate}
            b={r['block']:r for r in rows if r['name']==baseline}
            shared=sorted(a.keys()&b.keys())
            if not shared:continue
            for metric in ('primary_s','command_s'):
                changes=[100*(a[i][metric]/b[i][metric]-1) for i in shared]
                results.append(dict(workload=workload,size=size,candidate=candidate,baseline=baseline,metric=metric,
                                    pairs=len(changes),mean_percent=statistics.mean(changes),bootstrap_95_percent=interval(changes)))
print(json.dumps(dict(archive=str(root),scope='paired mean percentage; negative is faster; pointwise exploratory bootstrap, 2000 resamples',comparisons=results),indent=2))
