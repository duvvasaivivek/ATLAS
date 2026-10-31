"""Deterministic frequency diagnostics with explicit synthetic assumptions."""

from dataclasses import asdict
import random

from .frequency import Config, initial_state, transition

PAIRS = ((600, 1000), (900, 1000), (900, 3000), (1350, 3000), (1350, 6000), (1800, 6000))


def observation(run_id, sample_id, core, memory, epoch=0):
    window = dict(kind="synthetic_interval", start=sample_id, end=sample_id+1)
    return dict(run_id=run_id, sample_id=sample_id, epoch=epoch, core=core, memory=memory,
                valid=True, source_ref=f"synthetic:{run_id}:{sample_id}",
                semantics="synthetic_busy_fraction", core_window=dict(window), memory_window=dict(window))


def definitions():
    cases = []
    for policy in ("paper_additive", "discounted_loss_v1"):
        for name in ("idle", "saturated", "core_heavy", "memory_heavy", "phase_change", "noisy", "gaps", "closed_loop"):
            cases.append(dict(run_id=f"{policy}-{name}", name=name, count=160,
                              config=asdict(Config(PAIRS, policy=policy))))
    for beta in (.01, .2, .99):
        cases.append(dict(run_id=f"beta-{beta}", name="phase_change", count=512,
                          config=asdict(Config(PAIRS, beta=beta))))
    for name, pairs in (("one_core", ((900,1000),(900,3000),(900,6000))),
                        ("one_memory", ((600,3000),(900,3000),(1800,3000))),
                        ("one_pair", ((900,3000),)), ("empty", ())):
        cases.append(dict(run_id=name, name=name, count=12, config=asdict(Config(pairs))))
    return cases


def simulate(case):
    config = Config(**case["config"])
    state = initial_state(config, case["run_id"])
    rng = random.Random(1729)
    observations, decisions = [], []
    for i in range(case["count"]):
        name = case["name"]
        core, memory = .4, .7
        epoch = int(i >= case["count"]//2)
        if name == "idle": core, memory = 0., 0.
        if name == "saturated": core, memory = 1., 1.
        if name == "core_heavy": core, memory = .95, .1
        if name == "memory_heavy": core, memory = .1, .95
        if name == "phase_change": core = memory = .95 if not epoch else .05
        if name == "noisy": core, memory = rng.random(), rng.random()
        model = None
        if name == "closed_loop":
            # Dimensionless illustrative capacities. These are not device measurements.
            c, m = config.pairs[state["selected"]]
            cc, mc = c/1800, m/6000
            dc, dm = (.65,.15) if not epoch else (.15,.65)
            core, memory = min(1.,dc/cc), min(1.,dm/mc)
            runtime = max(dc/cc, dm/mc)
            power = 10+20*cc**3+5*mc**3
            model = dict(kind="assumed_dimensionless_capacity_model", applied_synthetic_pair=[c,m],
                         core_demand=dc, memory_demand=dm, runtime_units=runtime,
                         power_units=power, energy_units=runtime*power)
        obs = observation(case["run_id"], i, core, memory, epoch)
        if model is not None: obs["synthetic_model"] = model
        if name == "gaps":
            if i % 8 == 1: obs["memory"] = None
            if i % 8 == 2: obs["memory_window"]["end"] += 1
            if i % 8 == 3: obs["valid"] = False
            if i % 8 == 4: obs["sample_id"] = 0
        decision = transition(config, state, obs)
        state = decision["post_state"]
        observations.append(obs); decisions.append(decision)
    return observations, decisions


def summarize(config, decisions):
    reasons = {}
    switches = 0
    for d in decisions:
        reasons[d["reason"]] = reasons.get(d["reason"], 0)+1
        switches += d["pre_state"]["selected"] != d["post_state"]["selected"]
    return dict(observations=len(decisions), updates=decisions[-1]["post_state"]["updates"] if decisions else 0,
                reasons=reasons, switches=switches,
                final_pair=decisions[-1]["recommended_pair"] if decisions else None,
                final_weights=decisions[-1]["post_state"]["weights"] if decisions else [],
                candidate_provenance=config.provenance, synthetic_hardware_savings_claim=False)
