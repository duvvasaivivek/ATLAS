from pathlib import Path
import tempfile
import unittest
import sys

from greengpu.energy import combine_components, counter_delta, integrate_power
from greengpu.experiment import Recorder, digest, read_records, save_new, summarize_run, summarize_trial
from greengpu.telemetry import RaplTelemetry, acquire
from greengpu.runner import trial, replay


def sample(seconds, value, status="ok"):
    return dict(time_ns=round(seconds*1e9), value=value, status=status, timestamp_uncertainty_ns=1)


def trace():
    settings = dict(elements=1, repetitions=2, rounds=3, seed=4, max_gap_ns=2_000_000_000)
    return [
        dict(kind="trial_config", trial_id="test", block=0, monitored=True, settings=settings, lifecycle_start_ns=0),
        dict(kind="ready", clock="CLOCK_MONOTONIC", preparation_start_ns=0, warmup_start_ns=0, warmup_end_ns=0),
        dict(kind="counter_before", reading=sample(1, 10)),
        dict(kind="complete", start_ns=1_000_000_000, h2d_end_ns=1_100_000_000, compute_end_ns=1_900_000_000,
             end_ns=2_000_000_000, gpu_event_ms=800, elements=1, repetitions=2, rounds=3, seed=4),
        dict(kind="counter_after", reading=sample(2, 30)),
        dict(kind="validation", status="passed", mismatches=0, elements_verified=1, oracle="masked_uint64_xorshift", tolerance=0),
        dict(kind="finish", returncode=0, sampler_error=None, time_ns=3_000_000_000),
        *[dict(kind="telemetry", acquisition_start_ns=0, acquisition_end_ns=100, query_cpu_ns=10,
               readings=dict(power_w=sample(t, 20), energy_j=sample(t, 10+20*(t-1)))) for t in (1, 1.5, 2)],
    ]


class EnergyTests(unittest.TestCase):
    def test_nonuniform_fixture_and_clipped_boundaries(self):
        power = [sample(0,10),sample(1,20),sample(3,10)]
        self.assertEqual(integrate_power(power,0,3_000_000_000,2_000_000_000)["joules"],45)
        self.assertAlmostEqual(integrate_power(power,500_000_000,2_000_000_000,2_000_000_000)["joules"],26.25)

    def test_missing_stale_negative_nan_not_bridged(self):
        for bad in (sample(1,None,"unavailable"),sample(1,10,"stale"),sample(1,-1),sample(1,float("nan"))):
            r=integrate_power([sample(0,10),bad,sample(2,10)],0,2_000_000_000,2_000_000_000)
            self.assertEqual(r["status"],"invalid")
            self.assertIsNone(r["joules"])

    def test_gap_bracketing_order_and_zero_interval(self):
        cases=[([sample(0,10),sample(3,10)],0,3_000_000_000,1_000_000_000),
               ([sample(1,10),sample(3,10)],0,3_000_000_000,4_000_000_000),
               ([sample(0,10),sample(0,10)],0,1,10),
               ([sample(1,10),sample(0,10)],0,1,10),
               ([sample(0,10),sample(1,10)],0,0,10)]
        for args in cases:
            self.assertEqual(integrate_power(*args)["status"],"invalid")
        self.assertEqual(integrate_power([dict(time_ns=float("nan"),value=10,status="ok"),sample(1,10)],0,1,10)["status"],"invalid")

    def test_repeated_zero_and_irrelevant_bad_samples(self):
        r=integrate_power([sample(0,None,"unavailable"),sample(1,0),sample(2,0)],1_000_000_000,2_000_000_000,1_000_000_000)
        self.assertEqual(r["joules"],0)

    def test_wrap_fixture(self):
        r=counter_delta(sample(0,95e-6),sample(1,4e-6),same_epoch=True,range_j=100e-6,max_power_w=10e-6)
        self.assertAlmostEqual(r["joules"],9e-6)

    def test_wrap_ambiguity_and_reset(self):
        for kwargs in (dict(same_epoch=False), dict(same_epoch=True),
                       dict(same_epoch=True,range_j=100),
                       dict(same_epoch=True,range_j=100,max_power_w=100),
                       dict(same_epoch=True,range_j=100,max_power_w=1)):
            self.assertEqual(counter_delta(sample(0,95),sample(1,4),**kwargs)["status"],"invalid")
        # Even an increasing counter could have wrapped multiple times.
        self.assertEqual(counter_delta(sample(0,4),sample(1,5),same_epoch=True,range_j=100,max_power_w=101)["status"],"invalid")

    def test_component_sum_requires_domain_and_alignment(self):
        cpu=dict(status="ok",domain="cpu_package",joules=5,start_ns=0,end_ns=1)
        gpu=dict(status="ok",domain="gpu_board",joules=10,start_ns=0,end_ns=1)
        self.assertEqual(combine_components(cpu,gpu)["status"],"invalid")
        r=combine_components(cpu,gpu,disjoint_verified=True)
        self.assertEqual(r["joules"],15)
        self.assertNotEqual(r["domain"],"system_wall")
        self.assertEqual(combine_components(cpu,dict(gpu,end_ns=2),disjoint_verified=True)["status"],"invalid")
        self.assertEqual(combine_components(cpu,cpu,disjoint_verified=True)["status"],"invalid")


class TelemetryTests(unittest.TestCase):
    def test_acquisition_units_bounds_and_errors(self):
        r=acquire(lambda: 1000,"mJ",.001)
        self.assertEqual(r["value"],1)
        self.assertLessEqual(r["acquisition_start_ns"],r["time_ns"])
        self.assertLessEqual(r["time_ns"],r["acquisition_end_ns"])
        def fail(): raise PermissionError("denied")
        r=acquire(fail,"W")
        self.assertEqual(r["status"],"unavailable")
        self.assertIsNone(r["value"])
        self.assertIn("PermissionError",r["reason"])
        self.assertEqual(acquire(lambda:101,"%",.01)["status"],"invalid")
        self.assertEqual(acquire(lambda:float("nan"),"W")["raw"],"nan")

    def test_rapl_packages_only_and_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.assertEqual(RaplTelemetry(root).metadata["status"],"unavailable")
            for folder,name in (("intel-rapl:0","package-0"),("intel-rapl:0/intel-rapl:0:0","core"),("intel-rapl:1","package-1")):
                p=root/folder
                p.mkdir(parents=True)
                (p/"name").write_text(name)
                (p/"energy_uj").write_text("5000000")
                (p/"max_energy_range_uj").write_text("10000000")
            rapl=RaplTelemetry(root)
            self.assertEqual(len(rapl.zones),2)
            self.assertTrue(all(x["energy_j"]["value"]==5 for x in rapl.sample()))


class RecordingTests(unittest.TestCase):
    def test_counter_preferred_and_power_crosscheck(self):
        r=summarize_trial(trace())
        self.assertEqual(r["status"],"ok")
        self.assertEqual(r["energy"]["method"],"counter_delta")
        self.assertEqual(r["energy"]["joules"],20)
        self.assertEqual(r["sampled_power_energy"]["joules"],20)
        self.assertFalse(r["energy_claim_eligible"])

    def test_power_fallback_only_for_unsupported_counter(self):
        records=trace()
        for r in records:
            if r["kind"].startswith("counter_"):
                r["reading"].update(status="unavailable",value=None)
        self.assertEqual(summarize_trial(records)["energy"]["method"],"trapezoid")
        records=trace()
        records[4]["reading"]["value"]=1
        self.assertEqual(summarize_trial(records)["energy"]["status"],"invalid")

    def test_midrun_counter_reset_not_hidden_by_endpoints(self):
        records=trace()
        records[-2]["readings"]["energy_j"]["value"]=1
        self.assertEqual(summarize_trial(records)["energy"]["status"],"invalid")

    def test_wrong_work_output_clock_and_partial_runs_invalid(self):
        for index,key,value in ((3,"elements",2),(5,"status","failed"),(1,"clock","other"),(6,"returncode",1)):
            records=trace()
            records[index][key]=value
            self.assertEqual(summarize_trial(records)["status"],"invalid")
        self.assertEqual(summarize_trial(trace()[:4])["status"],"invalid")

    def test_exclusive_records_deterministic_replay_and_sequence(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"trial-000.jsonl"
            writer=Recorder(path)
            for record in trace(): writer.write(record)
            writer.close()
            with self.assertRaises(FileExistsError): Recorder(path)
            result=summarize_run(d)
            self.assertEqual(result,summarize_run(d))
            self.assertEqual(result["monitored_gpu_j"]["n"],1)
            self.assertEqual(result["overhead_percent"]["n"],0)
            lines=path.read_text().splitlines()
            path.write_text("\n".join(lines[1:]))
            with self.assertRaises(ValueError): read_records(path)

    def test_frozen_configuration_hash_and_exclusive_manifest(self):
        self.assertEqual(digest(dict(a=1,b=2)),digest(dict(b=2,a=1)))
        self.assertNotEqual(digest(dict(a=1)),digest(dict(a=2)))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"config.json"
            save_new(p,dict(a=1))
            with self.assertRaises(FileExistsError): save_new(p,dict(a=2))

    def test_long_interval_counter_disagreement_blocks_energy(self):
        records=trace()
        records[0]["settings"].update(counter_power_crosscheck=dict(min_duration_s=5,relative_allowance=.1),averaging_window_ns=1_000_000_000)
        records[4]["reading"]=sample(7,410)
        records[-1]["readings"]["power_w"]=sample(7,20)
        records[-1]["readings"]["energy_j"]=sample(7,410)
        records[0]["settings"]["max_gap_ns"]=10_000_000_000
        r=summarize_trial(records)
        self.assertEqual(r["status"],"ok")
        self.assertEqual(r["counter_power_crosscheck"]["status"],"inconsistent")
        self.assertEqual(r["energy"]["status"],"invalid")
        self.assertEqual(r["counter_energy"]["joules"],400)

    def test_short_window_cannot_pass_consistency_screen(self):
        records=trace()
        records[0]["settings"].update(counter_power_crosscheck=dict(min_duration_s=5,relative_allowance=.1),averaging_window_ns=1_000_000_000)
        r=summarize_trial(records)
        self.assertEqual(r["energy"]["status"],"invalid")
        self.assertEqual(r["counter_power_crosscheck"]["status"],"unresolved")

    def test_device_event_cannot_exceed_enclosing_host_interval(self):
        records=trace()
        records[3]["gpu_event_ms"]=1500
        result=summarize_trial(records)
        self.assertEqual(result["status"],"ok")
        self.assertEqual(result["gpu_event_quality"]["status"],"inconsistent")
        self.assertEqual(result["gpu_event_reported_s"],1.5)
        self.assertIsNone(result["gpu_event_s"])
        self.assertEqual(summarize_trial(trace())["gpu_event_quality"]["status"],"consistent")

    def test_replay_rejects_changed_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"raw.json"
            p.write_text("original")
            import hashlib
            save_new(Path(d)/"checksums.json",{"raw.json":hashlib.sha256(p.read_bytes()).hexdigest()})
            p.write_text("changed")
            with self.assertRaises(ValueError): replay(d)

    @unittest.skipUnless(sys.platform == "linux", "POSIX executable protocol test")
    def test_worker_protocol_failure_and_timeout_retained(self):
        for script in ("print('not-json', flush=True)", "import time; time.sleep(5)"):
            with self.subTest(script=script), tempfile.TemporaryDirectory() as d:
                worker=Path(d)/"worker"
                worker.write_text(f"#!{sys.executable}\n"+script+"\n")
                worker.chmod(0o700)
                path=Path(d)/"trial.jsonl"
                result=trial(path,worker,None,None,dict(elements=1,repetitions=1,rounds=1,seed=1,timeout_s=.1),"failure",0,False)
                self.assertEqual(result["status"],"invalid")
                records=read_records(path)
                self.assertTrue(any(r["kind"]=="failure" for r in records))
                self.assertEqual(records[-1]["kind"],"finish")
                self.assertIsNotNone(records[-1]["returncode"])


if __name__ == "__main__":
    unittest.main()
