from copy import deepcopy
from dataclasses import replace
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from greengpu.integration import Config,Coordinator,Lifecycle,sample_for_window
from greengpu.integration_runner import execute,replay_decisions
from greengpu.integration_scenarios import definitions,simulate,PAIRS
from greengpu.cli import main


def obs(engine,i,start=0):
    cut=engine.cut
    return dict(run_id=engine.config.run_id,epoch=0,iteration=i,cpu_items=cut,t_cpu=2. if cut else None,
                t_gpu=1. if cut<engine.config.n else None,release_ns=start,valid=True,common_origin=True)


def frame(i=0,stamp=2_000_000,value=60):
    return dict(frame_id=i,acquisition_start_ns=stamp*1000-1,acquisition_end_ns=stamp*1000,wall_end_us=stamp,
                buffer_samples=dict(core=[dict(timestamp_us=stamp,percent=70)],memory=[dict(timestamp_us=stamp,percent=value)]))


class CoordinationTests(unittest.TestCase):
    def test_native_batch_interior_never_dispatches(self):
        engine=Coordinator(Config("t",100,50,mode="allocation_only",allocation_iterations=1,allocation_interval_ns=0))
        result=engine.barrier(obs(engine,0),100,action_boundary=False)
        self.assertIsNone(result['allocation']);self.assertEqual(result['next_cpu_items'],50)
        result=engine.barrier(obs(engine,1),200,action_boundary=True)
        self.assertIsNotNone(result['allocation'])

    def test_new_clock_guard_rejects_wall_clock_motion(self):
        a=frame();b=frame(1,stamp=2_010_000)
        self.assertIsNotNone(sample_for_window([a,b],0,2_100_000_000,-1,require_stable_clock=True))
        b['wall_end_us']+=2000
        self.assertIsNone(sample_for_window([a,b],0,2_100_000_000,-1,require_stable_clock=True))

    def test_modes_and_live_gate(self):
        with self.assertRaises(ValueError):Config("t",10,5,mode="live_joint")
        with self.assertRaises(ValueError):Config("t",10,5,allocation_policy="bracketed_bisection")
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"not-created"
            with self.assertRaises(SystemExit) as result:
                main(["integration-check","--mode","live-joint","--output",str(output),"--kmeans-binary","missing","--hotspot-binary","missing"])
            self.assertEqual(result.exception.code,2);self.assertFalse(output.exists())

    def test_allocation_window_and_post_migration_settling(self):
        engine=Coordinator(Config("t",100,50,mode="allocation_only",allocation_iterations=2,allocation_interval_ns=0))
        first=engine.barrier(obs(engine,0),1)
        self.assertIsNone(first["allocation"])
        second=engine.barrier(obs(engine,1),2)
        self.assertEqual(second["next_cpu_items"],33)
        self.assertEqual(second["allocation"]["observation"]["source_iterations"],[0,1])
        third=engine.barrier(obs(engine,2),3)
        self.assertEqual(third["allocation_reason"],"settling_iteration_excluded")
        self.assertIsNone(third["frequency"])
        engine.barrier(obs(engine,3),4)
        fifth=engine.barrier(obs(engine,4),5)
        self.assertEqual(fifth["allocation"]["observation"]["source_iterations"],[3,4])

    def test_paper_ratio_uses_time_not_sleep_or_iteration_count(self):
        cfg=Config("t",100,50,mode="allocation_only",allocation_iterations=1,allocation_interval_ns=0,cadence_profile="paper_ratio_40_v1")
        engine=Coordinator(cfg)
        self.assertIsNone(engine.barrier(obs(engine,0),7_999_999_999)["allocation"])
        self.assertIsNotNone(engine.barrier(obs(engine,1),8_000_000_000)["allocation"])

    def test_join_missing_stale_conflicting_and_future_samples(self):
        good=frame()
        result=sample_for_window([good],1_000_000_000,2_100_000_000,-1)
        self.assertEqual(result["core"],.7);self.assertEqual(result["frame_ids"],[0])
        for frames,since,now,last in (([good],0,3_000_000_000,-1),([good],0,1_000_000_000,-1),
                                      ([good],2_000_000_001,2_100_000_000,-1),([good],0,2_100_000_000,2_000_000)):
            self.assertIsNone(sample_for_window(frames,since,now,last))
        missing=deepcopy(good);missing["buffer_samples"]["memory"]=[]
        self.assertIsNone(sample_for_window([missing],0,2_100_000_000,-1))
        conflict=frame(1,value=40)
        self.assertIsNone(sample_for_window([good,conflict],0,2_100_000_000,-1))

    def test_join_can_span_two_poll_frames(self):
        a=frame();b=frame(1)
        a["buffer_samples"]["memory"]=[];b["buffer_samples"]["core"]=[]
        self.assertEqual(sample_for_window([a,b],0,2_000_000_000,-1)["frame_ids"],[0,1])

    def test_driver_clock_ahead_of_host_is_not_corrected(self):
        raw=frame();raw["wall_end_us"]-=300_000
        self.assertIsNone(sample_for_window([raw],0,2_100_000_000,-1))

    def test_shadow_decision_is_bound_and_not_applied(self):
        cfg=Config("t",100,50,PAIRS,allocation_iterations=100)
        engine=Coordinator(cfg)
        result=engine.barrier(obs(engine,0),2_100_000_000,[frame()])
        self.assertEqual(result["frequency"]["post_state"]["updates"],1)
        self.assertEqual(result["frequency"]["observation"]["source_ref"],"telemetry_frames:0")
        self.assertIsNone(result["hardware_requested_pair"])
        self.assertIsNone(result["synthetic_applied_pair"])
        self.assertEqual(engine.logical_pair,PAIRS[0])
        second=engine.barrier(obs(engine,1),2_400_000_000)
        self.assertIsNone(second["frequency"]);self.assertEqual(engine.fstate["updates"],1)

    def test_phase_and_allocation_do_not_reset_frequency_history(self):
        cfg=Config("t",100,50,PAIRS,"synthetic_joint",allocation_iterations=1,allocation_interval_ns=0,frequency_guard_ns=0)
        engine=Coordinator(cfg)
        engine.barrier(obs(engine,0),1_000_000_000,synthetic_utilization=[.9,.9])
        first=engine.fstate["updates"]
        engine.barrier(obs(engine,1,1_000_000_000),2_000_000_000,phase=1,synthetic_utilization=[.9,.9])
        self.assertEqual(engine.fstate["updates"],first+1)

    def test_synthetic_frequency_change_invalidates_bracket_window(self):
        cfg=Config("t",100,50,PAIRS,"synthetic_joint",allocation_policy="bracketed_bisection",
                   allocation_iterations=1,allocation_interval_ns=0,frequency_guard_ns=0)
        engine=Coordinator(cfg)
        result=engine.barrier(obs(engine,0),1_000_000_000,synthetic_utilization=[1.,1.])
        self.assertIsNotNone(result["synthetic_applied_pair"])
        self.assertEqual(engine.window,[]);self.assertEqual(engine.capacity_epoch,1)
        self.assertIsNone(result["allocation"])

    def test_identity_ownership_and_correctness_guards(self):
        for key,value in (("run_id","other"),("iteration",1),("cpu_items",0),("valid",False),("t_cpu",float("nan"))):
            engine=Coordinator(Config("t",100,50));bad=obs(engine,0);bad[key]=value
            with self.assertRaises(ValueError):engine.barrier(bad,1)
        engine=Coordinator(Config("t",100,50))
        with self.assertRaises(ValueError):engine.barrier(obs(engine,0),2_100_000_000,[frame(1)])

    def test_final_barrier_never_dispatches_new_action(self):
        engine=Coordinator(Config("t",100,50,PAIRS,allocation_iterations=1,allocation_interval_ns=0))
        result=engine.barrier(obs(engine,0),2_100_000_000,[frame()],final=True)
        self.assertIsNone(result["allocation"]);self.assertIsNone(result["frequency"])
        self.assertEqual(result["events"],["observe","finalize"])

    def test_barrier_clock_cannot_go_backwards_between_decisions(self):
        engine=Coordinator(Config("t",100,50))
        engine.barrier(obs(engine,0),100)
        with self.assertRaises(ValueError):engine.barrier(obs(engine,1),50)

    def test_lifecycle_rejects_skipping_restoration(self):
        life=Lifecycle();life.move("failed")
        with self.assertRaises(ValueError):life.move("complete")
        life.move("restore");life.move("stopped")

    def test_synthetic_model_uses_preceding_actions_and_restores(self):
        for case in definitions():
            result=simulate(case)
            self.assertEqual(result["restored_synthetic_pair"],list(PAIRS[0]))
            self.assertEqual(result["hardware_setter_calls"],0)
            records=result["records"]
            for old,new in zip(records,records[1:]):
                expected=old["decision"]["synthetic_applied_pair"] or old["model"]["applied_pair"]
                self.assertEqual(new["model"]["applied_pair"],expected)
                self.assertEqual(new["input"]["observation"]["cpu_items"],old["decision"]["next_cpu_items"])
            self.assertEqual(result["status"],"expected_failure_restored" if case["name"] in ("actuation_failure","interruption") else "passed")


class FakePump:
    def __init__(self,*args):self.count=0;self.error=None;self.wall_ns=0;self.cpu_ns=0
    def start(self):pass
    def drain(self):return []
    def stop(self):pass


class WorkerFailureTests(unittest.TestCase):
    def trial(self,source,timeout=2):
        with tempfile.TemporaryDirectory() as tmp:
            script=Path(tmp)/"worker.py";script.write_text(source)
            case=dict(workload="kmeans",steps=10,workers=1,native_mode="fixed",integration=dict(run_id="failure",n=10,initial_cpu_items=5))
            with patch("greengpu.integration_runner.TelemetryPump",FakePump):
                return execute(sys.executable,script,None,case,None,timeout=timeout)
    def test_malformed_worker_cleanup_retains_failure(self):
        result=self.trial("print('{broken',flush=True)")
        self.assertEqual(result["status"],"failed");self.assertIn("JSONDecodeError",result["failure"])
        self.assertEqual(result["lifecycle"][-2:],["restore","stopped"])
    def test_worker_timeout_kills_process_group(self):
        result=self.trial("import time; time.sleep(10)",timeout=.1)
        self.assertTrue(result["timed_out"]);self.assertNotEqual(result["returncode"],0)
        self.assertEqual(result["hardware_setter_calls"],0)


class ReplayTests(unittest.TestCase):
    def fixture(self):
        cfg=Config("t",100,50,mode="allocation_only")
        case=dict(workload="kmeans",integration=asdict(cfg))
        row=dict(kind="iteration",warmup=False,index=0,cpu_items=50,gpu_items=50,release_ns=0,
                 cpu_end_ns=2,gpu_end_ns=1,validation_end_ns=3,status="passed",label_errors=0,count_errors=0,numeric_errors=0)
        from greengpu.allocation_experiment import observation
        o=observation(row,"t");o["release_ns"]=0
        inp=dict(observation=o,now_ns=4,new_frames=[],final=True,phase=0)
        decision=Coordinator(cfg).barrier(**inp)
        record=dict(status="ok",returncode=0,failure=None,polling=dict(error=None,frames=0),configuration=json.loads(json.dumps(asdict(cfg))),
                    stdout=json.dumps(row),barriers=[dict(input=inp,decision=decision,controller_ns=0)],trailing_frames=[],
                    lifecycle=["prepare","warmup","execute","observe","finalize","restore","complete"],
                    hardware_setter_calls=0,restoration="not_needed_no_settings_changed")
        return case,record
    def test_replay_rejects_changed_decision(self):
        case,record=self.fixture();self.assertEqual(len(replay_decisions(case,record)),1)
        record["barriers"][0]["decision"]["next_cpu_items"]=99
        with self.assertRaisesRegex(ValueError,"replay mismatch"):replay_decisions(case,record)
    def test_replay_rejects_unbound_observation(self):
        case,record=self.fixture();record["barriers"][0]["input"]["observation"]["t_cpu"]=123.
        with self.assertRaisesRegex(ValueError,"not bound"):replay_decisions(case,record)
    def test_replay_rejects_shadow_actuation_claim(self):
        case,record=self.fixture();record["hardware_setter_calls"]=1
        with self.assertRaisesRegex(ValueError,"shadow-actuation"):replay_decisions(case,record)


if __name__=="__main__":unittest.main()
