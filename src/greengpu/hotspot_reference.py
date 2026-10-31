"""Independent FP64 thermal reference and local HotSpot inputs (Rodinia CUDA model)."""
import hashlib
import math
from pathlib import Path
import random
import struct

ATOL=1e-8
RTOL=1e-10


def model(rows,cols):
    if type(rows) is not int or type(cols) is not int or not 1<=rows<=2048 or not 1<=cols<=2048:
        raise ValueError('Grid dimensions must be integers in [1,2048]')
    dy,dx=.016/rows,.016/cols
    cap=.5*1.75e6*.0005*dx*dy
    rx=dx/(2*100*.0005*dy);ry=dy/(2*100*.0005*dx);rz=.0005/(100*dy*dx)
    nominal=.001/(3e6/(.5*.0005*1.75e6))
    stable=.9*cap/(2/rx+2/ry+1/rz)
    dt=min(nominal,stable)
    return dict(cap=cap,rx=rx,ry=ry,rz=rz,dt=dt,nominal_dt=nominal,ambient=80.,
                stability_sum=dt/cap*(2/rx+2/ry+1/rz),limited=dt<nominal)


def step(temp,power,rows,cols):
    m=model(rows,cols);out=[]
    if len(temp)!=rows*cols or len(power)!=len(temp):raise ValueError('Grid length mismatch')
    for r in range(rows):
        for c in range(cols):
            i=r*cols+c;t=temp[i]
            neighbors=((max(0,r-1)*cols+c,m['ry']),(min(rows-1,r+1)*cols+c,m['ry']),
                       (r*cols+max(0,c-1),m['rx']),(r*cols+min(cols-1,c+1),m['rx']))
            flux=math.fsum([power[i],(80.-t)/m['rz']]+[(temp[j]-t)/resistance for j,resistance in neighbors])
            out.append(t+m['dt']/m['cap']*flux)
    return out


def trajectory(temp,power,rows,cols,steps):
    result=[]
    for _ in range(steps):
        temp=step(temp,power,rows,cols);result.append(temp)
    return result


def generated(rows,cols,seed,kind='random'):
    model(rows,cols);rng=random.Random(seed);n=rows*cols
    limit=3e6*.016*.016/n
    temp=[65.+30*rng.random() for _ in range(n)]
    power=[limit*rng.random() for _ in range(n)]
    if kind=='equilibrium':temp=[80.]*n;power=[0.]*n
    elif kind=='impulse':
        temp=[80.]*n;temp[(rows//2)*cols+cols//2]=140.;power=[0.]*n
    elif kind=='edge':
        temp=[80.+40*(r in (0,rows-1) or c in (0,cols-1)) for r in range(rows) for c in range(cols)]
    elif kind!='random':raise ValueError('Unknown input kind')
    return temp,power,dict(generator='atlas_thermal_v1',seed=seed,kind=kind,source='locally generated; not historical paper data')


def write_input(path,rows,cols,temp,power,provenance):
    m=model(rows,cols);n=rows*cols
    if len(temp)!=n or len(power)!=n or any(not math.isfinite(v) or not 0<=v<=500 for v in temp):
        raise ValueError('Invalid temperatures')
    if any(not math.isfinite(v) or not 0<=v<=3e6*.016*.016/n for v in power):
        raise ValueError('Invalid cell power')
    data=struct.pack('<8sII',b'ATLSHS01',rows,cols)+struct.pack(f'<{2*n}d',*(temp+power))
    with Path(path).open('xb') as f:f.write(data)
    return dict(rows=rows,cols=cols,model=m,sha256=hashlib.sha256(data).hexdigest(),provenance=provenance)


def read_grid(path):
    data=Path(path).read_bytes()
    if len(data)<16:raise ValueError('Truncated temperature field')
    magic,r,c=struct.unpack_from('<8sII',data)
    model(r,c)
    if magic!=b'ATLSTP01' or len(data)!=16+8*r*c:raise ValueError('Invalid temperature field')
    values=list(struct.unpack_from(f'<{r*c}d',data,16))
    if any(not math.isfinite(v) for v in values):raise ValueError('Nonfinite temperature field')
    return r,c,values


def close(a,b):return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=ATOL+RTOL*abs(b)


def communication(rows,cols,old,cut,iteration,power_begin):
    """Ownership state always migrates; immutable power alone may be cached."""
    size=cols*8
    return dict(initial_h2d_bytes=(rows-cut)*size if iteration==0 else 0,
        migration_h2d_bytes=max(0,old-cut)*size if iteration else 0,
        migration_d2h_bytes=max(0,cut-old)*size if iteration else 0,
        halo_h2d_bytes=size if 0<cut<rows else 0,halo_d2h_bytes=size if 0<cut<rows else 0,
        power_h2d_bytes=max(0,power_begin-cut)*size,power_begin_after=min(power_begin,cut))
