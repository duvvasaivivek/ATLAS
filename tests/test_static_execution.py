import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from greengpu.concurrency import analyze_profile, union_length
from greengpu.static_execution import case_matrix, make_case, partition, reference_sum, summarize_case


def fixture():
    case=make_case(2,"0.5",rounds=1,iterations=1,workers=1,seed=1)
    cpu=reference_sum(1,1,1)
    total=reference_sum(2,1,1)
    cfg=dict(kind="configuration",schema_version="3.0",clock="CLOCK_MONOTONIC",**{k:case[k] for k in
             ("elements","cpu_items","gpu_items","rounds","iterations","workers_requested","seed")},
             preparation_start_ns=1,reference_sum=total,oracle="masked_uint64_xorshift",tolerance=0,
             buffer_capacity_items=2,allocation_policy="allocate_once_reuse")
    records=[cfg]
    for i in range(2):
        t=100+i*100
        records.append(dict(kind="iteration",iteration=i,warmup=i==0,status="passed",reset_start_ns=t,
            reset_end_ns=t+1,release_ns=t+2,cpu_start_ns=t+3,cpu_end_ns=t+10,
            gpu_dispatch_ns=t+4,h2d_end_ns=t+5,compute_end_ns=t+7,gpu_end_ns=t+8,
            barrier_end_ns=t+11,merge_end_ns=t+12,validation_end_ns=t+15,
            cpu_sum=cpu,gpu_sum=total-cpu,combined_sum=total,mismatches=0,coverage_errors=0,
            elements_verified=2,workers_used=1,worker_spans=[dict(worker=0,start_ns=t+4,end_ns=t+9,items=1)]))
    return case,records


def process(records):
    return dict(status="ok",returncode=0,stdout="\n".join(json.dumps(r) for r in records))


class PartitionTests(unittest.TestCase):
    def test_floor_disjoint_and_exhaustive(self):
        for n in (1,2,255,256,257,4099):
            for ratio in ("0","1","0.25","0.5","0.75",f"1/{n}",f"{n-1}/{n}"):
                p=partition(n,ratio)
                self.assertEqual(p["cpu_items"]+p["gpu_items"],n)
                self.assertEqual(list(range(p["cpu_items"]))+list(range(p["cpu_items"],n)),list(range(n)))
        self.assertEqual(partition(257,"0.5")["cpu_items"],128)
        self.assertEqual(partition(65539,"1/65539")["cpu_items"],1)

    def test_invalid_inputs(self):
        for n in (0,-1,True,1048580):
            with self.assertRaises(ValueError): partition(n,"0.5")
        for share in ("-0.01","1.01","NaN"):
            with self.assertRaises(ValueError): partition(10,share)
        for kwargs in (dict(rounds=0),dict(workers=9),dict(iterations=0),dict(seed=-1)):
            with self.assertRaises(ValueError): make_case(10,"0.5",**kwargs)

    def test_matrix_endpoints_tiny_partitions_and_reuse(self):
        cases=case_matrix(4)
        self.assertEqual(len(cases),40)
        for n in {c["elements"] for c in cases}:
            counts={c["cpu_items"] for c in cases if c["elements"]==n}
            self.assertTrue({0,n,1,n-1}<=counts)
            self.assertEqual(len({c["input_sha256"] for c in cases if c["elements"]==n}),1)
            self.assertEqual(len({(c["seed"],c["rounds"]) for c in cases if c["elements"]==n}),1)
        self.assertTrue(all(c["iterations"]==3 for c in cases))


class NativeRecordTests(unittest.TestCase):
    def test_valid_timing_and_reduction(self):
        case,records=fixture()
        r=summarize_case(case,process(records))
        self.assertEqual(r["status"],"passed")
        self.assertEqual(len(r["iterations"]),1)
        self.assertTrue(r["exactly_once"])
        self.assertEqual(r["iterations"][0]["cpu_completion_s"],8e-9)
        self.assertEqual(r["iterations"][0]["gpu_completion_s"],6e-9)
        self.assertEqual(r["energy_qualification"],"blocked_phase2")

    def test_coverage_wrong_work_and_missing_iterations_fail(self):
        case,records=fixture()
        for index,key,value in ((2,"coverage_errors",1),(2,"mismatches",1),(2,"combined_sum",1),
                                 (2,"gpu_end_ns",0),(0,"cpu_items",0),(0,"reference_sum",0),
                                 (2,"warmup",True),(2,"workers_used",2)):
            altered=copy.deepcopy(records)
            altered[index][key]=value
            self.assertEqual(summarize_case(case,process(altered))["status"],"failed",key)
        self.assertEqual(summarize_case(case,process(records[:-1]))["status"],"failed")
        p=process(records);p["returncode"]=1
        self.assertEqual(summarize_case(case,p)["status"],"failed")

    def test_empty_cpu_does_not_create_fake_timing(self):
        case,records=fixture()
        case.update(cpu_items=0,gpu_items=2,requested_share="0",realized_share=0)
        records[0].update(cpu_items=0,gpu_items=2)
        for r in records[1:]:
            r.update(cpu_start_ns=None,cpu_end_ns=None,cpu_sum=0,gpu_sum=records[0]["reference_sum"],workers_used=0,worker_spans=[])
        r=summarize_case(case,process(records))
        self.assertEqual(r["status"],"passed")
        self.assertIsNone(r["iterations"][0]["cpu_completion_s"])
        self.assertIsNone(r["iterations"][0]["imbalance"])
        records[-1]["cpu_end_ns"]=0
        self.assertEqual(summarize_case(case,process(records))["status"],"failed")


class ProfilerTests(unittest.TestCase):
    def database(self,path,kernel_start=30,kernel_end=50,with_kernel=True):
        pid=1<<24
        with sqlite3.connect(path) as db:
            db.executescript('''
              CREATE TABLE StringIds(id INTEGER,value TEXT);
              CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,eventType INTEGER,text TEXT,textId INTEGER);
              CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,streamId INTEGER,correlationId INTEGER,globalPid INTEGER,shortName INTEGER);
              CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER);
            ''')
            db.execute("INSERT INTO StringIds VALUES(1,'atlas_static_transform_reduce')")
            db.executemany("INSERT INTO NVTX_EVENTS VALUES(?,?,?,59,?,NULL)",
                [(10,100,pid+1,'atlas_iteration'),(20,80,pid+2,'atlas_cpu_compute'),(25,70,pid+3,'atlas_cpu_compute')])
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(21,22,?,42)",(pid+1,))
            if with_kernel: db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,0,7,42,?,1)",(kernel_start,kernel_end,pid))

    def test_kernel_activity_overlap_not_double_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"profile.sqlite"
            self.database(path)
            r=analyze_profile(path,1)
            self.assertEqual(r["status"],"demonstrated")
            self.assertEqual(r["iterations"][0]["overlap_union_ns"],20)
            self.assertEqual(len(r["timeline"]["traceEvents"]),3)
        self.assertEqual(union_length([(1,10),(2,5),(8,12),(15,17)]),13)

    def test_queued_launch_without_activity_does_not_prove_overlap(self):
        for kwargs in (dict(with_kernel=False),dict(kernel_start=85,kernel_end=95)):
            with self.subTest(kwargs=kwargs),tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/"profile.sqlite"
                self.database(path,**kwargs)
                self.assertEqual(analyze_profile(path,1)["status"],"not_demonstrated")

    def test_missing_schema_or_iterations_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"profile.sqlite"
            self.assertEqual(analyze_profile(path,1)["status"],"unavailable")
            self.database(path)
            self.assertEqual(analyze_profile(path,2)["status"],"unavailable")

    def test_waiting_range_cannot_prove_compute_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"profile.sqlite"
            self.database(path)
            with sqlite3.connect(path) as db:
                db.execute("UPDATE NVTX_EVENTS SET text='hotspot_cpu_wait' WHERE text='atlas_cpu_compute'")
            self.assertEqual(analyze_profile(path,1)["status"],"not_demonstrated")


if __name__=="__main__": unittest.main()
