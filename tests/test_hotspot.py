import copy
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

from greengpu.hotspot_reference import model,step,trajectory,generated,write_input,read_grid,communication,close
from greengpu.hotspot import assess


class ThermalTests(unittest.TestCase):
    def test_model_matches_physical_hand_calculation(self):
        m=model(1,1)
        self.assertAlmostEqual(m['cap'],.112)
        self.assertAlmostEqual(m['rx'],10.)
        self.assertAlmostEqual(m['rz'],.01953125)
        self.assertAlmostEqual(m['nominal_dt'],1.4583333333333335e-7)
        self.assertAlmostEqual(step([100.],[0.],1,1)[0],100.+m['dt']/m['cap']*(80.-100.)/m['rz'])

    def test_stability_limiter_and_degenerate_shapes(self):
        self.assertTrue(model(2048,2048)['limited'])
        for r,c in ((1,1),(1,2048),(2048,1),(2048,2048),(19,23)):
            m=model(r,c)
            self.assertLessEqual(m['stability_sum'],.900000000000001)
            self.assertGreater(m['dt'],0)
        for r,c in ((0,1),(1,0),(2049,1),(1.5,2)):
            with self.assertRaises(ValueError):model(r,c)

    def test_equilibrium_and_diffusion_maximum_principle(self):
        for r,c in ((1,1),(1,9),(9,1),(7,11)):
            self.assertEqual(step([80.]*(r*c),[0.]*(r*c),r,c),[80.]*(r*c))
            t,p,_=generated(r,c,4,'impulse')
            for state in trajectory(t,p,r,c,100):
                self.assertTrue(all(80.-1e-12<=v<=140.+1e-12 for v in state))

    def test_lateral_flux_cancels_and_symmetry(self):
        r,c=5,7;t,p,_=generated(r,c,5)
        m=model(r,c);new=step(t,p,r,c)
        expected=m['dt']/m['cap']*(math.fsum(p)+math.fsum(80.-x for x in t)/m['rz'])
        self.assertAlmostEqual(math.fsum(new)-math.fsum(t),expected,places=10)
        t,p,_=generated(5,5,1,'impulse');new=step(t,p,5,5)
        self.assertEqual(new[7],new[11]);self.assertEqual(new[13],new[17])

    def test_migration_reuploads_current_state_not_cached_temperatures(self):
        a=communication(10,7,10,5,0,10)
        self.assertEqual(a['initial_h2d_bytes'],280)
        self.assertEqual(a['halo_h2d_bytes'],56)
        b=communication(10,7,5,0,1,5)
        self.assertEqual(b['migration_h2d_bytes'],280)
        self.assertEqual(b['halo_d2h_bytes'],0)
        c=communication(10,7,0,10,2,0)
        self.assertEqual(c['migration_d2h_bytes'],560)
        d=communication(10,7,10,0,3,0)
        self.assertEqual(d['migration_h2d_bytes'],560)
        self.assertEqual(d['power_h2d_bytes'],0)

    def test_input_constraints_and_output_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.hsdat';t,p,meta=generated(2,3,7)
            a=write_input(path,2,3,t,p,meta)
            self.assertEqual(len(path.read_bytes()),16+6*16)
            self.assertEqual(a['model'],model(2,3))
            with self.assertRaises(FileExistsError):write_input(path,2,3,t,p,meta)
            with self.assertRaises(ValueError):write_input(Path(directory)/'bad',2,3,[float('nan')]*6,p,meta)
            out=Path(directory)/'out.hstemp';out.write_bytes(struct.pack('<8sII6d',b'ATLSTP01',2,3,*t))
            self.assertEqual(read_grid(out),(2,3,t))
            out.write_bytes(out.read_bytes()+b'x')
            with self.assertRaises(ValueError):read_grid(out)


def records():
    m=model(2,1);gold=trajectory([100.,80.],[0.,0.],2,1,2)
    case=dict(run_id='fixture',policy='scripted',rows=2,cols=1,steps=2,workers=1,mode='verify',counts=[2,0],stage='test')
    cfg=dict(kind='configuration',schema_version='6.0',clock='CLOCK_MONOTONIC',feedback=True,rows=2,cols=1,
             steps=2,workers=1,mode='verify',atol=1e-8,rtol=1e-10,**{k:v for k,v in m.items() if k not in ('ambient','limited')})
    items=[cfg]
    old,power_begin=2,2
    for j,cut in enumerate((2,2,0)):
        i=j-1;base=100+100*j;warmup=j==0
        expected=communication(2,1,old,cut,max(i,0),power_begin)
        items.append(dict(kind='iteration',index=i,warmup=warmup,cpu_rows=cut,old_cpu_rows=old,input_epoch=max(0,i),output_epoch=max(0,i)+1,
            status='completed',validation='checked',errors=0,max_error=0.,**expected,validation_d2h_bytes=(2-cut)*8,
            release_ns=base,communication_end_ns=base+2,cpu_start_ns=base+3 if cut else None,cpu_end_ns=base+8 if cut else None,
            gpu_end_ns=base+8 if cut<2 else None,end_ns=base+10,validation_end_ns=base+20,control_ns=10 if i==1 else 0,
            temperatures=gold[max(0,i)]))
        if not warmup:old=cut;power_begin=min(power_begin,cut)
    items.append(dict(kind='summary',status='passed',errors=0,max_error=0.,steps=2,active_ns=20,control_ns=10,gather_ns=1,
        gather_d2h_bytes=0,validation_d2h_bytes=16,work_and_control_and_gather_ns=31,loop_elapsed_ns=110,simulated_seconds=2*m['dt']))
    proc=dict(status='ok',returncode=0,stdout='\n'.join(json.dumps(r) for r in items),configuration=None,elapsed_ns=1000,
        decisions=[dict(decision=dict(reason='scripted_schedule',iteration=0,cpu_items=0),controller_ns=2)])
    return case,dict(rows=2,cols=1,model=m),proc,gold,items


class HotspotRecordTests(unittest.TestCase):
    def test_valid_record_and_corruption_guards(self):
        case,meta,proc,gold,items=records()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'out';path.write_bytes(struct.pack('<8sII2d',b'ATLSTP01',2,1,*gold[-1]))
            result=assess(case,meta,proc,path,gold)
            self.assertEqual(result['status'],'passed')
            self.assertEqual(result['work_and_control_and_gather_s'],31e-9)
            self.assertIsNone(result['energy_j'])
            for index,key,value in ((3,'input_epoch',0),(3,'migration_h2d_bytes',0),(3,'halo_d2h_bytes',8),
                                   (2,'validation_d2h_bytes',16),(3,'temperatures',[80.,80.]),
                                   (3,'gpu_end_ns',1),(4,'active_ns',100),(4,'gather_d2h_bytes',16),(3,'control_ns',500)):
                bad=copy.deepcopy(items);bad[index][key]=value;changed=dict(proc,stdout='\n'.join(json.dumps(r) for r in bad))
                with self.subTest(key=key):
                    with self.assertRaises(ValueError):assess(case,meta,changed,path,gold)
            bad=copy.deepcopy(proc);bad['decisions'][0]['decision']['cpu_items']=1
            with self.assertRaises(ValueError):assess(case,meta,bad,path,gold)


if __name__=='__main__':unittest.main()
