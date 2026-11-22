"""Phase 8 verification and frequency-only orchestration, separate from Tier 1."""

from .actuation import ActuationBlocked, ClockTransaction
from .frequency import Config, finite, fraction, initial_state, transition

SPEC_VERSION="frequency-hardware-1.0"


def verify_setting(frames, pair, request_end_ns, minimum_span_ns=400_000_000, require_latest=False):
    """Require three fresh paired samples under load after a conservative 1 s guard.

    Each frame supplies acquisition bounds and an exact NVML buffer timestamp
    join. A 15 MHz core / 2 MHz memory tolerance accommodates clock reporting
    granularity; these tolerances are fixed before examining experimental results.
    Live control sets require_latest so a later bad sample invalidates an earlier
    streak. The default preserves historical one-time qualification replay.
    """
    streak=[]; last_stamp=-1
    for index,f in enumerate(frames):
        stamp=f.get("sample_timestamp_us")
        good=(type(stamp) is int and f.get("sample_valid") is True and
              type(f.get("acquisition_start_ns")) is int and type(f.get("acquisition_end_ns")) is int and
              f["acquisition_end_ns"]>=f["acquisition_start_ns"]>=request_end_ns+1_000_000_000 and
              finite(f.get("sample_age_us")) and 0<=f["sample_age_us"]<=500_000 and
              type(f.get("sample_monotonic_estimate_ns")) is int and f["sample_monotonic_estimate_ns"]>=request_end_ns+1_000_000_000 and
              fraction(f.get("core_utilization")) and f["core_utilization"]>=.1 and
              fraction(f.get("memory_utilization")) and finite(f.get("temperature_c")) and 0<=f["temperature_c"]<85 and
              f.get("clock_event_reasons") == 0 and
              finite(f.get("core_mhz")) and abs(f["core_mhz"]-pair[0])<=15 and
              finite(f.get("memory_mhz")) and abs(f["memory_mhz"]-pair[1])<=2)
        if type(stamp) is int and stamp<=last_stamp:
            if not good:streak=[]
            continue
        if not good:
            streak=[]
        else:
            streak.append(f)
        if type(stamp) is int: last_stamp=max(last_stamp,stamp)
        if (len(streak)>=3 and streak[-1]["acquisition_start_ns"]-streak[0]["acquisition_start_ns"]>=minimum_span_ns
                and (not require_latest or index==len(frames)-1)):
            return dict(status="verified_under_load",pair=list(pair),samples=len(streak),
                        first_sample_ns=streak[0]["acquisition_start_ns"],verified_ns=streak[-1]["acquisition_end_ns"],
                        settling_ns=streak[-1]["acquisition_end_ns"]-request_end_ns,
                        qualification="conservative_timestamp_and_1s_window_guard")
    return dict(status="unverified",pair=list(pair),reason="no_fresh_stable_loaded_window")


def controlled_sweep(backend,directory,pairs,measure,blocks=3):
    """Callback returns correctness plus telemetry for a bounded real load.

    Restoration is tested in a separate short transaction before measurement.
    This path cannot run with the default NVML backend's unknown prior policy.
    """
    from pathlib import Path
    if type(blocks) is not int or not 1<=blocks<=10 or not pairs:
        raise ValueError("Require candidates and 1..10 blocks")
    root=Path(directory); results=[]; verified=[]
    with ClockTransaction(backend,root/"restoration-preflight",pairs) as control:
        control.apply(pairs[0])
    # Only a successful verified restoration allows a longer transaction.
    with ClockTransaction(backend,root/"static-sweep",pairs) as control:
        for pair in pairs:
            end=control.apply(pair)
            probe=measure("settle",pair,0)
            check=verify_setting(probe["frames"],pair,end)
            results.append(dict(stage="settle",pair=list(pair),trial=probe,verification=check))
            if not probe["correct"]: raise RuntimeError("Incorrect probe output")
            if check["status"]=="verified_under_load": verified.append(tuple(pair))
        # Counterbalance setting order by block; no unsupported pair enters comparisons.
        for block in range(blocks):
            for pair in (verified if block%2==0 else list(reversed(verified))):
                end=control.apply(pair)
                settling=measure("settle",pair,block)
                check=verify_setting(settling["frames"],pair,end)
                results.append(dict(stage="settle_repeat",pair=list(pair),block=block,trial=settling,verification=check))
                if not settling["correct"]: raise RuntimeError("Incorrect settling output")
                if check["status"]!="verified_under_load": continue
                trial=measure("measure",pair,block)
                check=verify_setting(trial["frames"],pair,end)
                results.append(dict(stage="measure",pair=list(pair),block=block,trial=trial,verification=check))
                if not trial["correct"]: raise RuntimeError("Incorrect measured output")
    return dict(results=results,verified_pairs=[list(p) for p in verified],restoration="verified",
                energy_surface=None,energy_reason="requires_separate_measurement_qualification")


def frequency_only(backend,directory,pairs,policy,observe,settle,budget=20):
    """Bounded Tier 2 runner; caller supplies pairs verified by the static sweep.

    No Tier 1 allocation changes. observe returns a normalized observation and
    fresh loaded frames; settle verifies each request.
    """
    if type(budget) is not int or not 1<=budget<=1000:
        raise ValueError("Invalid controller budget")
    cfg=Config(pairs,policy=policy,provenance="hardware_reported_unverified")
    if not cfg.pairs: raise ActuationBlocked("No verified pairs")
    state=initial_state(cfg,"phase8-frequency-only"); decisions=[]
    with ClockTransaction(backend,directory,cfg.pairs) as control:
        current=cfg.pairs[0]
        end=control.apply(current)
        if verify_setting(settle(current),current,end)["status"]!="verified_under_load":
            raise ActuationBlocked("Initial setting did not settle")
        for i in range(budget):
            obs,frames=observe(i)
            qualification=verify_setting(frames,current,end)
            eligible=qualification["status"]=="verified_under_load"
            obs=dict(obs,valid=obs.get("valid") is True and eligible is True)
            decision=transition(cfg,state,obs)
            state=decision["post_state"]
            requested=tuple(decision["recommended_pair"])
            entry=dict(decision=decision,requested_pair=None,acknowledged_pair=None,verification=None,
                       input_verification=qualification)
            if decision["reason"]=="recommend_pair" and requested!=current:
                end=control.apply(requested)
                entry.update(requested_pair=list(requested),acknowledged_pair=list(requested),
                             verification=verify_setting(settle(requested),requested,end))
                if entry["verification"]["status"]!="verified_under_load":
                    control.record("decision",entry=entry)
                    raise ActuationBlocked("Recommended setting did not settle")
                current=requested
            control.record("decision",entry=entry); decisions.append(entry)
    return decisions
