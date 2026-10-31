"""Standalone thermal-field, timing, and allocation figures from verified evidence."""
import hashlib
import json
import os
from pathlib import Path
import struct
from .experiment import save_new
from .hotspot import replay
from .hotspot_reference import read_grid


def render(run,output):
    run=Path(run).resolve();output=Path(output).resolve()
    if output.is_relative_to(run):raise ValueError('Figures must be outside immutable evidence')
    s=replay(run)
    if s['implementation_status']!='passed':raise ValueError('Incomplete validation')
    output.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault('MPLCONFIGDIR',str(Path('build/matplotlib-cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True)
    for ax,stages,title in ((axes[0],('coarse','refine'),'Tuning input'),(axes[1],('heldout',),'Held-out static splits')):
        for stage in stages:
            rows=[r for r in s['comparisons'] if r['stage']==stage and r['name'].startswith('static@')]
            rows.sort(key=lambda r:int(r['name'].split('@')[1]))
            ax.errorbar([int(r['name'].split('@')[1])/257*100 for r in rows],[r['runtime_s']['mean']*1000 for r in rows],
                yerr=[(r['runtime_s']['stdev'] or 0)*1000 for r in rows],fmt='o-',capsize=3,label=stage)
        ax.set_title(title);ax.set_xlabel('CPU row share (%)');ax.set_xticks([0,25,50,75,100]);ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('24 timesteps + feedback + final gather (ms)');axes[0].set_ylim(bottom=0)
    fig.suptitle('HotSpot static partition timing: 257 × 259 grid')
    fig.text(.5,.02,'Mean ± sample SD across trials. Automatic clocks; energy unqualified.\nValidation copies are absent from timing loops.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.94))
    for ext in ('png','svg'):fig.savefig(output/f'hotspot-static.{ext}',dpi=180)
    plt.close(fig)
    selected=[r for r in s['results'] if r['case']['stage']=='heldout' and r['case']['block']==0
              and r['case']['policy']!='static' and r['case']['counts'][0]==128]
    fig,ax=plt.subplots(figsize=(11,4.8))
    for r in selected:ax.plot(range(1,len(r['shares'])+1),r['shares'],marker='.',label=r['case']['policy'])
    ax.set_ylim(-.03,1.03);ax.set_xlabel('Timestep');ax.set_ylabel('Realized CPU row share');ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.2)
    ax.set_title('HotSpot feedback: nearest-50% starts, randomized block zero')
    fig.text(.5,.02,'Preselected trace subset. Full starts, blocks, and imbalance diagnostics remain in the archive.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1))
    for ext in ('png','svg'):fig.savefig(output/f'hotspot-allocation.{ext}',dpi=180)
    plt.close(fig)
    example=next(r for r in s['results'] if r['case']['dataset']=='heldout' and r['case']['policy']=='static')
    raw=(run/'datasets/heldout.hsdat').read_bytes();_,h,w=struct.unpack_from('<8sII',raw)
    initial=np.frombuffer(raw,dtype='<f8',count=h*w,offset=16).reshape(h,w)
    final=np.array(read_grid(run/example['case']['output'])[2]).reshape(h,w)
    fig,axes=plt.subplots(1,3,figsize=(12,4.4))
    for ax,values,title in zip(axes,(initial,final,final-initial),('Initial temperature','After 24 timesteps','Temperature change')):
        kwargs=dict(cmap='coolwarm')
        if 'change' not in title:kwargs.update(vmin=min(initial.min(),final.min()),vmax=max(initial.max(),final.max()))
        im=ax.imshow(values,**kwargs);ax.set_title(title);ax.set_xlabel('Column');ax.set_ylabel('Row');fig.colorbar(im,ax=ax,shrink=.75,label='°C')
    fig.suptitle('Generated HotSpot input and validated final field')
    fig.tight_layout(rect=(0,0,1,.94))
    for ext in ('png','svg'):fig.savefig(output/f'hotspot-field.{ext}',dpi=180)
    plt.close(fig)
    save_new(output/'provenance.json',dict(summary_sha256=hashlib.sha256((run/'summary.json').read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),matplotlib_version=matplotlib.__version__,
        source_field=example['case']['output'],energy='unavailable'))
    return output
