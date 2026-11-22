"""Non-overlapping host accounting; branch completion is kept distinct."""


def breakdown(row, workload):
    release,end=row["release_ns"],row["end_ns"]
    if workload=="hotspot":
        shared=row.get("compute_release_ns",row["communication_end_ns"])-release
        merge=0
        origin=row.get("compute_release_ns",row["communication_end_ns"])
    else:
        shared=0
        merge=end-row["barrier_end_ns"]
        origin=release
    cpu=row.get("cpu_end_ns");gpu=row.get("gpu_end_ns")
    branch_end=max(x for x in (cpu,gpu) if x is not None)
    tail=end-merge-branch_end
    if min(shared,merge,tail,branch_end-origin)<0:
        raise ValueError("Inconsistent timing decomposition")
    return dict(active_ns=end-release,shared_communication_ns=shared,
                parallel_envelope_ns=branch_end-origin,join_tail_ns=tail,merge_ns=merge,
                cpu_completion_ns=cpu-release if cpu is not None else None,
                gpu_completion_ns=gpu-release if gpu is not None else None,
                cpu_after_shared_ns=cpu-origin if cpu is not None else None,
                gpu_after_shared_ns=gpu-origin if gpu is not None else None,
                branch_wait_opportunity_ns=abs(cpu-gpu) if cpu is not None and gpu is not None else None,
                waiting_scope="inferred_branch_completion_difference_not_measured_thread_wait")
