"""Independent scalar Lloyd oracle and deterministic, locally generated fixtures."""

import hashlib
import math
import random
import struct
from pathlib import Path

ATOL=1e-8
RTOL=1e-10
SHIFT_TOL=1e-6


def close(a,b):
    return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=ATOL+RTOL*abs(b)


def label_hash(labels):
    value=14695981039346656037
    for label in labels:
        for byte in struct.pack("<I",label):
            value=((value^byte)*1099511628211)&0xffffffffffffffff
    return f"{value:016x}"


def validate_data(points,centers):
    if not points or not centers or not 1<=len(centers)<=min(32,len(points)) or len(points)>262147:
        raise ValueError("Require 1<=K<=min(N,32), 1<=N<=262147")
    d=len(points[0])
    if not 1<=d<=32 or any(len(row)!=d for row in points+centers):
        raise ValueError("Inconsistent or unsupported dimensions")
    if any(not math.isfinite(v) or abs(v)>1e6 for row in points+centers for v in row):
        raise ValueError("Coordinates must be finite and bounded by 1e6")


def write_dataset(path,points,centers,provenance):
    validate_data(points,centers)
    n,d,k=len(points),len(points[0]),len(centers)
    data=bytearray(struct.pack("<8sIII",b"ATLSKM01",n,d,k))
    for row in points+centers:
        data.extend(struct.pack(f"<{d}d",*row))
    with Path(path).open("xb") as f:f.write(data)
    return dict(n=n,d=d,k=k,sha256=hashlib.sha256(data).hexdigest(),provenance=provenance,
                initial_centroids=[v for row in centers for v in row])


def read_dataset(path):
    raw=Path(path).read_bytes()
    if len(raw)<20: raise ValueError("Truncated dataset")
    magic,n,d,k=struct.unpack_from("<8sIII",raw)
    if magic!=b"ATLSKM01" or not 1<=n<=262147 or not 1<=d<=32 or not 1<=k<=min(n,32) or len(raw)!=20+(n+k)*d*8:
        raise ValueError("Invalid dataset header/length")
    rows=[list(struct.unpack_from(f"<{d}d",raw,20+i*d*8)) for i in range(n+k)]
    validate_data(rows[:n],rows[n:])
    return rows[:n],rows[n:]


def step(points,centers,previous=None):
    """Ordered FP64 distances, math.fsum aggregates, immutable caller inputs."""
    validate_data(points,centers)
    k,d=len(centers),len(centers[0])
    labels=[];costs=[];groups=[[] for _ in centers]
    for index,point in enumerate(points):
        distances=[]
        for center in centers:
            total=0.0
            for x,y in zip(point,center):total+=(x-y)*(x-y)
            distances.append(total)
        winner=min(range(k),key=distances.__getitem__)
        labels.append(winner);costs.append(distances[winner]);groups[winner].append(index)
    counts=[len(group) for group in groups]
    sums=[[math.fsum(points[i][j] for i in group) for j in range(d)] for group in groups]
    updated=[[sums[c][j]/counts[c] if counts[c] else centers[c][j] for j in range(d)] for c in range(k)]
    shift=max(math.sqrt(sum((a-b)**2 for a,b in zip(old,new))) for old,new in zip(centers,updated))
    changed=len(points) if previous is None else sum(a!=b for a,b in zip(labels,previous))
    quality=math.fsum((points[i][j]-updated[labels[i]][j])**2 for i in range(len(points)) for j in range(d))
    return dict(labels=labels,label_hash=label_hash(labels),counts=counts,sums=[v for row in sums for v in row],
                centroids=[v for row in updated for v in row],objective_before_update=math.fsum(costs),
                objective_after_update=quality,max_centroid_shift=shift,changed_labels=changed,
                empty_clusters=sum(c==0 for c in counts),converged=changed==0 or shift<=SHIFT_TOL)


def trajectory(points,centers,iterations,mode="fixed"):
    if mode not in ("fixed","converge") or not 1<=iterations<=100:raise ValueError("Invalid iteration mode/budget")
    centers=[row[:] for row in centers];previous=None;records=[];d=len(centers[0])
    for _ in range(iterations):
        result=step(points,centers,previous)
        records.append(result)
        centers=[result["centroids"][i:i+d] for i in range(0,len(result["centroids"]),d)]
        previous=result["labels"]
        if mode=="converge" and result["converged"]:break
    return records


def clustered(n,d,k,seed):
    rng=random.Random(seed)
    # Local synthetic data; no third-party dataset/license or paper-replication claim.
    points=[[12.0*(i%k)+rng.uniform(-2,2)+0.3*j for j in range(d)] for i in range(n)]
    rng.shuffle(points)
    indices=rng.sample(range(n),k)  # Seeded Forgy: distinct row indices; duplicate coordinates permitted.
    centers=[points[i][:] for i in indices]
    return points,centers,dict(generator="clustered_uniform_v1",seed=seed,initialization="seeded_Forgy",
                               initial_indices=indices,source="generated locally by ATLAS",license="project-generated fixture")


def fixtures():
    return {
        "single":([[3.25,-2.5]],[[3.25,-2.5]],dict(source="handwritten",purpose="single point and cluster")),
        "ties":([[-2.0],[0.0],[2.0]],[[-2.0],[2.0]],dict(source="handwritten",purpose="exact equidistance chooses lowest cluster")),
        "empty":([[1.0,1.0] for _ in range(17)],[[1.0,1.0] for _ in range(4)],dict(source="handwritten",purpose="duplicate centers and empty clusters")),
        "k_equals_n":([[0.0,0.0],[1.0,2.0],[-2.0,3.0]],[[0.0,0.0],[1.0,2.0],[-2.0,3.0]],dict(source="handwritten",purpose="K=N")),
        "one_cluster":([[-5.5,2.25],[3.125,-4.0],[0.0,1.0]],[[3.125,-4.0]],dict(source="handwritten",purpose="mean with signed fractional data")),
        "awkward":clustered(257,3,4,1729),
        "multidimensional":clustered(513,7,5,829),
    }
