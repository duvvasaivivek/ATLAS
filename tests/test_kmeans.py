import copy
import json
from pathlib import Path
import tempfile
import unittest

from greengpu.kmeans import assess,best_point,curves
from greengpu.kmeans_reference import close,clustered,fixtures,label_hash,read_dataset,step,trajectory,validate_data,write_dataset


class ReferenceTests(unittest.TestCase):
    def test_exact_tie_and_updated_mean(self):
        r=step([[-2.0],[0.0],[2.0]],[[-2.0],[2.0]])
        self.assertEqual(r["labels"],[0,0,1])
        self.assertEqual(r["counts"],[2,1])
        self.assertEqual(r["centroids"],[-1.0,2.0])
        self.assertEqual(r["objective_before_update"],4)
        self.assertEqual(r["objective_after_update"],2)

    def test_empty_clusters_keep_their_previous_centers(self):
        r=step([[1.0],[1.0],[1.0]],[[1.0],[10.0],[1.0]])
        self.assertEqual(r["counts"],[3,0,0])
        self.assertEqual(r["centroids"],[1.0,10.0,1.0])
        self.assertEqual(r["empty_clusters"],2)
        self.assertTrue(r["converged"])

    def test_fixed_budget_does_not_stop_at_convergence(self):
        self.assertEqual(len(trajectory([[1.0]],[[1.0]],5)),5)
        self.assertEqual(len(trajectory([[1.0]],[[1.0]],20,"converge")),1)

    def test_objective_monotonicity_and_repeatable_initialization(self):
        points,centers,meta=clustered(257,3,4,1729)
        self.assertEqual((points,centers,meta),clustered(257,3,4,1729))
        self.assertEqual(len(set(meta["initial_indices"])),4)
        previous=float("inf")
        for r in trajectory(points,centers,12):
            self.assertLessEqual(r["objective_after_update"],r["objective_before_update"]+1e-8)
            self.assertLessEqual(r["objective_before_update"],previous+1e-8)
            self.assertEqual(sum(r["counts"]),257)
            previous=r["objective_after_update"]

    def test_bad_inputs_and_binary_identity(self):
        for points,centers in (([],[[0.0]]),([[float("nan")]],[[0.0]]),([[1.0]],[[1.0],[2.0]]),([[1.0,2.0]],[[1.0]])):
            with self.assertRaises(ValueError):validate_data(points,centers)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"input.kmdat"
            points,centers,_=fixtures()["ties"]
            meta=write_dataset(p,points,centers,{"source":"test"})
            self.assertEqual(read_dataset(p),(points,centers))
            self.assertEqual(meta["n"],3)
            with self.assertRaises(FileExistsError):write_dataset(p,points,centers,{})
            p.write_bytes(p.read_bytes()+b"x")
            with self.assertRaises(ValueError):read_dataset(p)


def records_fixture():
    points,centers,_=fixtures()["ties"]
    expected=trajectory(points,centers,2)
    meta=dict(n=3,d=1,k=2,initial_centroids=[-2.0,2.0])
    case=dict(dataset="ties",cpu_counts=[3,0],mode="fixed",max_iterations=2,workers_requested=1,stage="migration",n=3)
    cfg=dict(kind="configuration",schema_version="4.0",clock="CLOCK_MONOTONIC",n=3,d=1,k=2,
             initial_centroids=[-2.0,2.0],mode="fixed",max_iterations=2,workers_requested=1,
             atol=1e-8,rtol=1e-10,shift_tolerance=1e-6,preparation_start_ns=0)
    rows=[]
    for index,(cut,ref) in enumerate([(3,expected[0]),(3,expected[0]),(0,expected[1])]):
        t=100+100*index;gpu=3-cut
        r=dict(kind="iteration",index=index-1,warmup=index==0,status="passed",cpu_items=cut,gpu_items=gpu,
               resident_before=3,resident_after=cut,ownership_changed_points=3-cut,h2d_point_bytes=(3-cut)*8,
               h2d_centroid_bytes=16 if gpu else 0,d2h_label_bytes=gpu*4,d2h_aggregate_bytes=32 if gpu else 0,
               release_ns=t,cpu_start_ns=t+1 if cut else None,cpu_end_ns=t+5 if cut else None,
               gpu_dispatch_ns=t+1 if gpu else None,h2d_end_ns=t+3 if gpu else None,
               compute_end_ns=t+4 if gpu else None,gpu_end_ns=t+5 if gpu else None,
               barrier_end_ns=t+6,end_ns=t+10,validation_end_ns=t+20,workers_used=1 if cut else 0,
               label_errors=0,count_errors=0,numeric_errors=0,**ref)
        rows.append(r)
    summary=dict(kind="summary",status="passed",iterations=2,active_ns=20,loop_elapsed_ns=110,
                 final_objective=expected[-1]["objective_after_update"],stop_reason="fixed_budget",
                 final_labels=expected[-1]["labels"],final_centroids=expected[-1]["centroids"])
    return case,meta,[cfg]+rows+[summary],expected


def process(records):return dict(status="ok",returncode=0,stdout="\n".join(json.dumps(r) for r in records))


class RecordTests(unittest.TestCase):
    def test_valid_native_record_and_python_oracle(self):
        case,meta,records,golden=records_fixture()
        r=assess(case,meta,process(records),golden)
        self.assertEqual(r["status"],"passed",r)
        self.assertEqual(r["active_s"],20e-9)
        self.assertEqual(r["loop_elapsed_s"],110e-9)
        self.assertIsNone(r["energy_j"])

    def test_migration_stale_labels_and_wrong_aggregates_rejected(self):
        case,meta,records,golden=records_fixture()
        for index,key,value in ((3,"h2d_point_bytes",0),(3,"resident_before",0),(2,"label_hash","bad"),
                                 (2,"counts",[1,1]),(2,"sums",[0,0]),(2,"centroids",[0,0]),
                                 (2,"objective_after_update",100),(2,"cpu_end_ns",1),(0,"mode","converge"),
                                 (4,"active_ns",1),(3,"gpu_items",0),(4,"final_labels",[1,1,1])):
            with self.subTest(key=key):
                changed=copy.deepcopy(records);changed[index][key]=value
                self.assertEqual(assess(case,meta,process(changed),golden)["status"],"failed")

    def test_incomplete_or_failed_worker_rejected(self):
        case,meta,records,golden=records_fixture()
        self.assertEqual(assess(case,meta,process(records[:-1]),golden)["status"],"failed")
        p=process(records);p["returncode"]=1
        self.assertEqual(assess(case,meta,p,golden)["status"],"failed")

    def test_curve_statistics_use_trials_and_keep_regressions(self):
        results=[]
        for cut,values in ((0,[1,3]),(5,[4,6])):
            for value in values:
                results.append(dict(status="passed",case=dict(stage="heldout",cpu_counts=[cut],n=10),active_s=value,loop_elapsed_s=value+1))
        series=curves(results,"heldout")
        self.assertEqual(series[0]["active_s"]["n"],2)
        self.assertEqual(series[0]["active_s"]["mean"],2)
        self.assertEqual(series[1]["active_s"]["mean"],5)
        self.assertEqual(best_point(series),0)
        self.assertIsNone(series[1]["energy_j"])


if __name__=="__main__":unittest.main()
