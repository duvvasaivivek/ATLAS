"""Optional standalone scientific plots from checksum-verified Phase 4 evidence."""

import hashlib
import os
from pathlib import Path

from .experiment import save_new
from .kmeans import replay


def render(run,output):
    run=Path(run).resolve();output=Path(output).resolve()
    if output.is_relative_to(run):raise ValueError("Write figures outside the immutable run archive")
    summary=replay(run)
    if summary["implementation_status"]!="passed":raise ValueError("Cannot plot an incomplete validation as a successful sweep")
    output.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR",str(Path("build/matplotlib-cache").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.9),sharey=True)
    for ax,stages,title in ((axes[0],["coarse_tuning","refine_tuning"],"Tuning input · seed 2027"),
                            (axes[1],["heldout_evaluation"],"Held-out input · seed 2402")):
        for stage in stages:
            rows=summary["timing_curves"][stage]
            ax.errorbar([100*r["cpu_share"] for r in rows],[1000*r["active_s"]["mean"] for r in rows],
                        yerr=[1000*(r["active_s"]["stdev"] or 0) for r in rows],fmt="o-",capsize=4,
                        label={"coarse_tuning":"Coarse sweep","refine_tuning":"Refinement","heldout_evaluation":"Independent evaluation"}[stage])
        ax.set_title(title,fontsize=11);ax.set_xlabel("Realized CPU share (%)");ax.set_xticks([0,25,50,75,100])
        ax.set_xlim(-4,104);ax.grid(axis="y",alpha=.2);ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=8)
    n=next(r["case"]["n"] for r in summary["results"] if r["case"]["stage"]=="heldout_evaluation")
    selected=100*summary["tuning_selected_cpu_items"]/n
    axes[1].axvline(selected,color="#333333",linestyle=":",alpha=.65)
    axes[0].set_ylabel("Sum of 6 complete iteration intervals (ms)")
    axes[0].set_ylim(bottom=0)
    fig.suptitle("Kmeans static partition timing · 32,771 points · 8 dimensions · 8 clusters",fontsize=13)
    blocks=summary["timing_curves"]["heldout_evaluation"][0]["active_s"]["n"]
    fig.text(.5,.045,f"Mean ± sample SD; {blocks} trials per point. Dotted line: share selected on tuning data.\n"
             "Automatic GPU clock policy. Validation/logging gaps excluded. Energy remains unqualified.",ha="center",fontsize=8.5)
    fig.tight_layout(rect=(0,.12,1,.94))
    for suffix in ("png","svg"):fig.savefig(output/f"kmeans-runtime.{suffix}",dpi=180)
    plt.close(fig)
    save_new(output/"provenance.json",dict(archive=str(run),summary_sha256=hashlib.sha256((run/"summary.json").read_bytes()).hexdigest(),
              generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),matplotlib_version=matplotlib.__version__,
              energy_curve="unavailable; no zero-valued or synthetic replacement"))
    return output
