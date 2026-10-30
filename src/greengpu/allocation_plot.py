"""Standalone scientific Phase 5 figures, generated from verified raw decisions."""
import hashlib
import os
from pathlib import Path

from .allocation_experiment import replay
from .experiment import save_new


def render(run, output):
    run=Path(run).resolve();output=Path(output).resolve()
    if output.is_relative_to(run):raise ValueError('Figures must be outside immutable evidence')
    summary=replay(run)
    output.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault('MPLCONFIGDIR',str(Path('build/matplotlib-cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    trials=[r for r in summary['results'] if r['case']['stage']=='heldout']
    # Predeclared display subset: all policies initialized nearest 50%, block zero.
    selected=[r for r in trials if r['case']['block']==0 and r['case']['policy']!='static'
              and abs(r['shares'][0]-.5)<1/r['case']['n']]
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for r in selected:
        name=r['case']['policy'];x=list(range(1,r['iterations']+1))
        axes[0].plot(x,r['shares'],marker='.',label=name)
        axes[1].plot(x,r['diagnostic']['imbalance'],marker='.')
    axes[0].set_ylabel('Realized CPU share');axes[0].set_ylim(-.03,1.03)
    axes[0].legend(ncol=2,fontsize=8)
    axes[1].set_ylabel('Relative branch imbalance');axes[1].set_ylim(-.03,1.03)
    axes[1].axhline(.05,color='gray',ls='--',lw=1);axes[1].set_xlabel('Measured iteration')
    for ax in axes:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Kmeans allocation feedback: 50% starts, first randomized block')
    fig.text(.5,.015,'Dashed line: 5% diagnostic tolerance. Automatic clock policy; no energy claim.\n'
             'This is a preselected trace subset, not the best trial.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.95))
    for ext in ('png','svg'):fig.savefig(output/f'allocation-traces.{ext}',dpi=180)
    plt.close(fig)
    names={r['case']['name'] for r in selected}|{r['case']['name'] for r in trials if r['case']['policy']=='static'}
    rows=[r for r in summary['comparisons'] if r['name'] in names]
    rows.sort(key=lambda r:r['active_and_control_s']['mean'])
    fig,ax=plt.subplots(figsize=(11,6))
    ax.barh([r['name'].split('@')[0] for r in rows],[r['active_and_control_s']['mean']*1000 for r in rows],
            xerr=[(r['active_and_control_s']['stdev'] or 0)*1000 for r in rows],capsize=3,color='#4477aa')
    ax.invert_yaxis();ax.set_xlabel('24 complete iterations + feedback/export/IPC (ms)')
    ax.set_title('Held-out runtime: 50% policy starts and static baselines')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.2)
    fig.text(.5,.018,f"Mean ± sample SD across {rows[0]['trials']} trials. Validation gaps excluded; retained loop elapsed includes them.\n"
             'GPU-only and static-tuned share the selected endpoint, but use separate retained trials.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,1))
    for ext in ('png','svg'):fig.savefig(output/f'allocation-runtime.{ext}',dpi=180)
    plt.close(fig)
    save_new(output/'provenance.json',dict(summary_sha256=hashlib.sha256((run/'summary.json').read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),matplotlib_version=matplotlib.__version__,
        subset='nearest 50% policy starts; trace block 0; runtime all blocks; all static baselines',energy='unavailable'))
    return output
