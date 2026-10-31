"""Export research diagnostics, kept outside the immutable evidence archive."""

from pathlib import Path

from .frequency_experiment import replay, read_lines


def render(root, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root, output = Path(root), Path(output)
    replay(root)
    if output.resolve() == root.resolve() or root.resolve() in output.resolve().parents:
        raise ValueError("Figures must be outside the immutable archive")
    output.mkdir(parents=True,exist_ok=False)
    fig, axes = plt.subplots(2,2,figsize=(12,7),layout="constrained")
    for policy, label in (("paper_additive","Paper additive"),("discounted_loss_v1","Discounted loss (lambda=0.95)")):
        records = read_lines(root/(policy+"-phase_change.decisions.jsonl"))
        for ax,domain,title in ((axes[0,0],0,"Core recommendation"),(axes[0,1],1,"Memory recommendation")):
            ax.step(range(len(records)),[r["recommended_pair"][domain] for r in records],where="post",label=label)
            ax.axvline(80,color="gray",linestyle=":"); ax.set(title=title,ylabel="Synthetic candidate MHz",xlabel="Observation")
        records = read_lines(root/(policy+"-closed_loop.decisions.jsonl"))
        axes[1,0].plot([r["observation"]["synthetic_model"]["runtime_units"] for r in records],label=label)
    axes[1,0].set(title="Assumed closed-loop capacity model",ylabel="Synthetic runtime units",xlabel="Observation")
    for beta in (.01,.2,.99):
        records = read_lines(root/f"beta-{beta}.decisions.jsonl")
        axes[1,1].plot([max(r["post_state"]["weights"]) for r in records],label=f"beta={beta}")
    axes[1,1].set(title="Different scores, identical selections",ylabel="Maximum additive score",xlabel="Observation")
    for ax in axes.flat: ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Phase 7: offline synthetic diagnostics — no measured hardware savings")
    for suffix in ("png","pdf"): fig.savefig(output/f"frequency-diagnostics.{suffix}",dpi=160)
    plt.close(fig)
    return str(output/"frequency-diagnostics.png")
