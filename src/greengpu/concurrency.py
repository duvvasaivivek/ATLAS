"""Read-only Nsight SQLite analysis. Host launch intervals are not GPU activity."""

import sqlite3
from pathlib import Path


def union_length(intervals):
    total = 0
    end = None
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        total += hi-max(lo,end) if end is not None and hi > max(lo,end) else (hi-lo if end is None else 0)
        end = max(end,hi) if end is not None else hi
    return total


def analyze_profile(path, expected_iterations):
    """Correlate actual CUPTI kernel activity with measured CPU NVTX regions.

    All timestamps here belong to the profiler's correlated timeline, not the
    application's CLOCK_MONOTONIC trace. Do not combine their numeric timestamps.
    """
    try:
        with sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro",uri=True) as db:
            db.row_factory = sqlite3.Row
            ranges = [dict(r) for r in db.execute("""
                SELECT n.start,n.end,n.globalTid,COALESCE(n.text,s.value) AS name
                FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
                WHERE n.end>n.start AND n.eventType=59 ORDER BY n.start
            """)]
            kernels = [dict(r) for r in db.execute("""
                SELECT k.start,k.end,k.deviceId,k.streamId,k.correlationId,k.globalPid,
                       s.value AS name,r.start AS launch_start,r.end AS launch_end,r.globalTid
                FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
                JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId=r.correlationId
                    AND k.globalPid=(r.globalTid & -16777216)
                WHERE s.value='atlas_static_transform_reduce' AND k.end>k.start
                ORDER BY k.start
            """)]
        iterations = [r for r in ranges if r["name"]=="atlas_iteration"]
        if len(iterations) != expected_iterations:
            raise ValueError("Measured NVTX iteration count does not match native configuration")
        evidence = []
        events = []
        origin = min(r["start"] for r in iterations)
        for index, iteration in enumerate(iterations,1):
            lo, hi = iteration["start"], iteration["end"]
            cpu = [r for r in ranges if r["name"]=="atlas_cpu_compute" and lo<=r["start"]<r["end"]<=hi
                   and r["globalTid"]>>24 == iteration["globalTid"]>>24]
            gpu = [k for k in kernels if k["globalTid"]==iteration["globalTid"]
                   and lo<=k["launch_start"]<=k["launch_end"]<=hi and lo<=k["start"]<k["end"]<=hi
                   and k["start"]>=k["launch_start"]]
            intersections = [(max(c["start"],g["start"]),min(c["end"],g["end"])) for c in cpu for g in gpu]
            overlap = union_length(intersections)
            evidence.append(dict(iteration=index, cpu_regions=cpu, kernels=gpu,
                                 overlap_union_ns=overlap, demonstrated=bool(cpu and len(gpu)==1 and overlap>0)))
            for region in cpu+gpu:
                is_gpu = "deviceId" in region
                events.append(dict(name=region["name"],cat="GPU" if is_gpu else "CPU",ph="X",
                    pid=2 if is_gpu else 1,tid=region["streamId"] if is_gpu else region["globalTid"] & 0xffffff,
                    ts=(region["start"]-origin)/1000,dur=(region["end"]-region["start"])/1000,
                    args=dict(iteration=index,source="Nsight correlated timeline")))
        passed = bool(evidence) and all(e["demonstrated"] for e in evidence)
        return dict(status="demonstrated" if passed else "not_demonstrated", iterations=evidence,
                    clock="Nsight correlated timeline", energy_claim_eligible=False,
                    limitation="Qualitative profiler-observed overlap of kernel activity and CPU compute regions; OS scheduling and absolute clock rates are not independently calibrated",
                    timeline=dict(traceEvents=events,displayTimeUnit="ms"))
    except (sqlite3.Error,ValueError,KeyError,TypeError) as error:
        return dict(status="unavailable",reason=str(error),iterations=[],timeline=dict(traceEvents=[]))
