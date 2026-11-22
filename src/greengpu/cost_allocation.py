"""Explicit sampled-cost extension; no linear extrapolation to unmeasured endpoints."""
from copy import deepcopy
from dataclasses import asdict
import math
from .experiment import digest


def decide(config,state,observation):
    o=observation;post=deepcopy(state);count=o['cpu_items']
    samples=max(1,len(o.get('source_iterations',[])))
    cost=o.get('work_s',max(o.get('t_cpu') or 0,o.get('t_gpu') or 0))/samples
    migration=o.get('migration_s',0)
    remaining=o.get('remaining_iterations',config.dwell)
    if not all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in (cost,migration,remaining)) or cost<=0:
        raise ValueError('Invalid measured-cost observation')
    if post['epoch']!=o['epoch']:
        post.update(eligible=0,cost_samples={},migration_bound=0.)
    costs=post.setdefault('cost_samples',{})
    # String keys keep serialized and in-memory controller state identical.
    key=str(count);old=costs.get(key)
    costs[key]=cost if old is None else config.alpha*cost+(1-config.alpha)*old
    post['migration_bound']=max(post.get('migration_bound',0.),migration)
    candidates=sorted({config.nearest(x) for x in (0.,.25,.5,.75,1.)})
    missing=[c for c in candidates if str(c) not in costs]
    target=count;reason='hold_cost_deadband'
    if post['eligible']>=config.budget:reason='budget_exhausted'
    elif missing and remaining>=config.dwell*len(missing):
        target=missing[0];reason='probe_measured_candidate'
    elif missing:reason='hold_short_horizon'
    else:
        best=min(candidates,key=lambda c:(costs[str(c)],c))
        improvement=costs[key]-costs[str(best)]
        if improvement>config.deadband*costs[key] and improvement*remaining>post['migration_bound']:
            target=best;reason='amortized_measured_adjust'
    post.update(epoch=o['epoch'],last_iteration=o['iteration'],eligible=min(config.budget,post['eligible']+1))
    return dict(specification_version='cost-allocation-1.0',policy=config.policy,config_hash=digest(asdict(config)),
                observation=deepcopy(o),pre_state=deepcopy(state),post_state=post,reason=reason,
                requested_ratio=target/config.n,cpu_items=target,realized_ratio=target/config.n,
                cost_model='measured_candidates_ewma; communication proxy for transition cost; no speedup guarantee')
