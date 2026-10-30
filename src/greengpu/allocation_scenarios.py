"""Deterministic closed-loop diagnostics; these are not hardware measurements."""

import math
from .allocation import POLICIES, Config, initial_state, transition, balance_trace

SCENARIOS = ("stationary", "asymmetric", "noisy", "transfer_dominated", "phase_change")


def simulate(policy, scenario, start=.5, iterations=205, n=1000000):
    config = Config(policy, n)
    state = initial_state(f"synthetic:{policy}:{scenario}:{start}")
    count = config.nearest(start)
    trace, observations = [], []
    for i in range(iterations):
        ratio = count/n
        epoch = int(scenario == "phase_change" and i >= iterations//2)
        cpu_capacity, gpu_capacity = ((.2,.8) if epoch else (.8,.2))
        if scenario == "asymmetric":
            cpu_capacity, gpu_capacity = .05,.95
        a, b = ratio/cpu_capacity, (1-ratio)/gpu_capacity
        if scenario == "noisy":
            a *= 1+.20*math.sin(i*2.1)
            b *= 1+.20*math.cos(i*1.7)
        if scenario == "transfer_dominated":
            a += .4
            b += 8.0  # No interior balance for these capacities/transfer costs.
        o = dict(run_id=state["run_id"], epoch=epoch, iteration=i, cpu_items=count,
                 t_cpu=a if count else None, t_gpu=b if count<n else None,
                 valid=True, common_origin=True)
        decision = transition(config,state,o)
        trace.append(decision);observations.append(o)
        state = decision["post_state"];count = decision["cpu_items"]
    return dict(policy=policy, scenario=scenario, start=start, trace=trace,
                diagnostic=balance_trace(observations), energy_j=None,
                evidence_kind="analytic_assumed_time_functions_not_hardware")


def suite():
    return [simulate(policy,scenario,start,205 if policy=='paper_binary_literal' and scenario=='stationary' else 64) for policy in POLICIES
            for scenario in SCENARIOS for start in (.15,.5,.9,.79)]
