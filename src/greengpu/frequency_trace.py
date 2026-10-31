"""Read-only NVML snapshot and conservative timestamp-paired trace adapter."""

import time


SEMANTICS = "nvml_kernel_and_memory_busy_time_fraction"


def pair_samples(raw, run_id="hardware-offline"):
    """Exact driver timestamp matching; never infer missing memory utilization.

    NVML gives sample timestamps, not start/end integration bounds. Matching timestamps
    is the explicitly named offline eligibility assumption, not a latency qualification.
    Duplicate timestamp values must agree. Ignore pre-collection buffered history.
    """
    indexes = []
    for domain in ("core_samples", "memory_samples"):
        index = {}
        conflicts = set()
        for row in raw[domain]:
            stamp, value = row["timestamp_us"], row["percent"]
            if type(stamp) is not int or not raw["start_us"] <= stamp <= raw["end_us"]:
                continue
            if stamp in index and index[stamp] != value:
                conflicts.add(stamp)
            index[stamp] = value
        for stamp in conflicts: index.pop(stamp, None)
        indexes.append(index)
    core, memory = indexes
    observations = []
    for i, stamp in enumerate(sorted(core.keys() & memory.keys())):
        window = dict(kind="nvml_buffer_timestamp_us", start=None, end=stamp)
        def normalized(value):
            return value/100 if type(value) in (int,float) else None
        observations.append(dict(run_id=run_id, sample_id=i, epoch=0,
                                 core=normalized(core[stamp]), memory=normalized(memory[stamp]), valid=True,
                                 source_ref=f"hardware.json:driver_timestamp_us:{stamp}", semantics=SEMANTICS,
                                 core_window=dict(window), memory_window=dict(window)))
    return observations


def capture(polls=6, period=.25):
    """No workload dispatch or setters; the trace may be entirely idle."""
    import pynvml as n
    raw = dict(start_us=time.time_ns()//1000, end_us=None, core_samples=[], memory_samples=[],
               reported_pairs=[], selected_pairs=[], errors=[], rates=[], clock_observations=[],
               hardware_settings_changed=False, workload="uncontrolled_desktop_or_idle",
               eligibility="exact_matched_driver_timestamps_offline_assumption",
               sample_interval_bounds="not_exposed", actuation_verified=False)
    initialized = False
    try:
        n.nvmlInit(); initialized = True
        if n.nvmlDeviceGetCount() != 1:
            raise ValueError("This capture requires exactly one visible GPU")
        h = n.nvmlDeviceGetHandleByIndex(0)
        raw["device"] = dict(name=n.nvmlDeviceGetName(h), uuid=n.nvmlDeviceGetUUID(h), driver=n.nvmlSystemGetDriverVersion())
        try:
            for memory in n.nvmlDeviceGetSupportedMemoryClocks(h):
                cores = sorted(set(n.nvmlDeviceGetSupportedGraphicsClocks(h, memory)))
                raw["reported_pairs"].extend([[c,memory] for c in cores])
                # Bounded explicit subset, each queried for this memory setting.
                raw["selected_pairs"].extend([[cores[i],memory] for i in sorted({0,len(cores)//2,len(cores)-1})])
        except n.NVMLError as error:
            raw["errors"].append(dict(query="supported_pairs", error=str(error)))
        for poll in range(polls):
            for key, sample_type in (("core_samples",n.NVML_GPU_UTILIZATION_SAMPLES),
                                     ("memory_samples",n.NVML_MEMORY_UTILIZATION_SAMPLES)):
                try:
                    value_type, samples = n.nvmlDeviceGetSamples(h,sample_type,raw["start_us"])
                    if value_type != n.NVML_VALUE_TYPE_UNSIGNED_INT:
                        raise ValueError("Unexpected utilization sample value type")
                    raw[key].extend(dict(timestamp_us=s.timeStamp,percent=s.sampleValue.uiVal) for s in samples)
                except (n.NVMLError,ValueError) as error:
                    raw["errors"].append(dict(query=key,poll=poll,error=str(error)))
            try:
                before=time.monotonic_ns(); u=n.nvmlDeviceGetUtilizationRates(h); after=time.monotonic_ns()
                raw["rates"].append(dict(core_percent=u.gpu,memory_percent=u.memory,
                                         acquisition_start_ns=before,acquisition_end_ns=after,
                                         replay_eligible=False,reason="no_sample_timestamp"))
                raw["clock_observations"].append(dict(acquired_monotonic_ns=time.monotonic_ns(),
                                                      graphics_mhz=n.nvmlDeviceGetClockInfo(h,n.NVML_CLOCK_GRAPHICS),
                                                      memory_mhz=n.nvmlDeviceGetClockInfo(h,n.NVML_CLOCK_MEM),
                                                      joined_to_utilization_sample=False))
            except n.NVMLError as error:
                raw["errors"].append(dict(query="rates_or_clocks",poll=poll,error=str(error)))
            if poll+1 < polls: time.sleep(period)
    except (n.NVMLError,ValueError) as error:
        raw["errors"].append(dict(query="initialization",error=str(error)))
    finally:
        if initialized: n.nvmlShutdown()
        raw["end_us"] = time.time_ns()//1000
    raw["selected_pairs"].sort()
    return raw
