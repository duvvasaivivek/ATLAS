from copy import deepcopy
import json
from pathlib import Path
import signal
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from greengpu.actuation import ActuationBlocked, ClockTransaction, NvmlClockBackend, RestorationFailed, valid_policy
from greengpu.frequency_hardware import controlled_sweep, frequency_only, verify_setting
from greengpu.frequency_scenarios import observation
from greengpu.phase8 import candidate_subset, characterize, evaluate, replay
from greengpu.experiment import save_new
from greengpu.frequency_experiment import sha
from greengpu.frequency_hardware import SPEC_VERSION


PAIRS=((600,1000),(1200,1000),(1200,3000))


class FakeBackend:
    """Failure-injection device; never hardware evidence."""
    def __init__(self):
        self.device_uuid="test-only-device";self.policy={"core":[900,900],"memory":None}
        self.original=deepcopy(self.policy);self.calls=[];self.fail_at=set();self.readback_wrong=False
    def snapshot(self):
        policy=deepcopy(self.policy)
        if self.readback_wrong:policy["core"]=[999,999]
        return dict(restorable=True,device_uuid=self.device_uuid,policy_source="authoritative_policy_readback",policy=policy)
    def set_domain(self,domain,bounds):
        self.calls.append((domain,deepcopy(bounds)));self.policy[domain]=deepcopy(bounds)
        if len(self.calls) in self.fail_at:raise RuntimeError("Injected partial setter failure")


def frames(pair,end=0):
    return [dict(sample_timestamp_us=1000000+i*200000,sample_valid=True,
                 acquisition_start_ns=end+1_200_000_000+i*200_000_000,
                 acquisition_end_ns=end+1_210_000_000+i*200_000_000,
                 sample_age_us=10000,sample_monotonic_estimate_ns=end+1_200_000_000+i*200_000_000,
                 core_utilization=.9,memory_utilization=.5,temperature_c=60,clock_event_reasons=0,
                 core_mhz=pair[0],memory_mhz=pair[1]) for i in range(3)]


class TransactionTests(unittest.TestCase):
    def test_second_owner_blocked_until_transaction_exits(self):
        backend=FakeBackend()
        with tempfile.TemporaryDirectory() as tmp:
            with ClockTransaction(backend,Path(tmp)/"one",PAIRS):
                with self.assertRaises(ActuationBlocked):
                    with ClockTransaction(backend,Path(tmp)/"two",PAIRS):self.fail("second owner entered")
            with ClockTransaction(backend,Path(tmp)/"three",PAIRS):pass

    def test_success_and_durable_restoration(self):
        backend=FakeBackend()
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/"tx"
            with ClockTransaction(backend,folder,PAIRS) as tx:
                tx.apply(PAIRS[0]);self.assertEqual(backend.policy["core"],[600,600])
                self.assertTrue((folder/"recovery.json").exists())
                journal=[json.loads(s) for s in (folder/"journal.jsonl").read_text().splitlines()]
                self.assertEqual(journal[1]["kind"],"request")
            self.assertEqual(backend.policy,backend.original)
            self.assertEqual(tx.events[-1]["status"],"verified")
            self.assertEqual(backend.calls[-2:],[("memory",None),("core",[900,900])])

    def test_partial_apply_failure_restores_both(self):
        backend=FakeBackend();backend.fail_at={2}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,"Injected"):
                with ClockTransaction(backend,Path(tmp)/"tx",PAIRS) as tx:tx.apply(PAIRS[-1])
            self.assertEqual(backend.policy,backend.original)
            self.assertEqual(tx.events[-1]["status"],"verified")

    def test_first_setter_failure_also_restores(self):
        backend=FakeBackend();backend.fail_at={1}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                with ClockTransaction(backend,Path(tmp)/"tx",PAIRS) as tx:tx.apply(PAIRS[-1])
            self.assertEqual(backend.policy,backend.original)
            self.assertEqual(len(backend.calls),3)

    def test_handled_interrupt_restores(self):
        backend=FakeBackend()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(KeyboardInterrupt):
                with ClockTransaction(backend,Path(tmp)/"tx",PAIRS) as tx:
                    tx.apply(PAIRS[-1]);raise KeyboardInterrupt()
            self.assertEqual(backend.policy,backend.original)

    def test_sigterm_restores_and_returns_previous_handler(self):
        backend=FakeBackend();previous=signal.getsignal(signal.SIGTERM)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(KeyboardInterrupt):
                with ClockTransaction(backend,Path(tmp)/"tx",PAIRS) as tx:
                    tx.apply(PAIRS[-1]);signal.raise_signal(signal.SIGTERM)
            self.assertEqual(backend.policy,backend.original)
            self.assertEqual(signal.getsignal(signal.SIGTERM),previous)

    def test_restoration_failure_attempts_other_domain(self):
        backend=FakeBackend();backend.fail_at={3}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RestorationFailed):
                with ClockTransaction(backend,Path(tmp)/"tx",PAIRS) as tx:tx.apply(PAIRS[-1])
            self.assertEqual(len(backend.calls),4)
            self.assertEqual(tx.events[-1]["status"],"failed")

    def test_wrong_restoration_readback_is_failure(self):
        backend=FakeBackend()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RestorationFailed):
                with ClockTransaction(backend,Path(tmp)/"tx",PAIRS) as tx:
                    tx.apply(PAIRS[-1]);backend.readback_wrong=True

    def test_unknown_baseline_never_calls_setter(self):
        backend=FakeBackend();backend.snapshot=lambda:dict(restorable=False,policy=None,device_uuid=backend.device_uuid)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ActuationBlocked):
                with ClockTransaction(backend,Path(tmp)/"tx",PAIRS):self.fail("entered blocked transaction")
            self.assertEqual(backend.calls,[])

    def test_unknown_candidate_does_not_mutate(self):
        backend=FakeBackend()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ActuationBlocked):
                with ClockTransaction(backend,Path(tmp)/"tx",PAIRS) as tx:tx.apply((600,3000))
            self.assertEqual(backend.calls,[])

    def test_journal_failure_cannot_skip_restore(self):
        backend=FakeBackend()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(OSError):
                with ClockTransaction(backend,Path(tmp)/"tx",PAIRS) as tx:
                    tx.apply(PAIRS[-1]);tx.record=Mock(side_effect=OSError("disk full"))
            self.assertEqual(backend.policy,backend.original)

    def test_prior_policy_requires_explicit_known_mode(self):
        backend=FakeBackend();snap=backend.snapshot()
        self.assertTrue(valid_policy(snap))
        for key,value in (("restorable",False),("policy_source","observed_frequency"),("policy",None),
                          ("policy",dict(core=[0,100],memory=None))):
            bad=dict(snap);bad[key]=value;self.assertFalse(valid_policy(bad))

    def test_nvml_adapter_guard_and_dispatch(self):
        adapter=NvmlClockBackend.__new__(NvmlClockBackend)
        adapter.nv=Mock();adapter.handle="test";adapter.device_uuid="test-only-device"
        adapter.snapshot=lambda:dict(restorable=False)
        with self.assertRaises(ActuationBlocked):adapter.set_domain("core",[600,600])
        adapter.nv.nvmlDeviceSetGpuLockedClocks.assert_not_called()
        adapter.snapshot=FakeBackend().snapshot
        adapter.set_domain("core",[600,600]);adapter.set_domain("memory",None)
        adapter.nv.nvmlDeviceSetGpuLockedClocks.assert_called_once_with("test",600,600)
        adapter.nv.nvmlDeviceResetMemoryLockedClocks.assert_called_once_with("test")


class HardwareOrchestrationTests(unittest.TestCase):
    def test_freshness_tolerance_load_and_invalid_frames(self):
        good=frames(PAIRS[0]);self.assertEqual(verify_setting(good,PAIRS[0],0)["status"],"verified_under_load")
        for key,value in (("sample_timestamp_us",None),("core_mhz",float("nan")),("memory_mhz",None),
                          ("core_mhz",700),("memory_mhz",1010),("sample_age_us",500001),
                          ("sample_valid",False),("core_utilization",0),("memory_utilization",None),
                          ("temperature_c",85),("clock_event_reasons",1),("acquisition_start_ns",None)):
            bad=deepcopy(good)
            for f in bad:f[key]=value
            self.assertEqual(verify_setting(bad,PAIRS[0],0)["status"],"unverified",key)
        duplicate=[good[0],good[0],good[1],good[1],good[2]]
        self.assertEqual(verify_setting(duplicate,PAIRS[0],0)["samples"],3)
        self.assertEqual(verify_setting([good[0]]*10,PAIRS[0],0)["status"],"unverified")
        self.assertEqual(verify_setting(good,PAIRS[0],1_000_000_000)["status"],"unverified")

    def test_sparse_candidate_subset(self):
        pairs=[(c,m) for c in (600,900,1200,1500,1800) for m in (1000,3000,6000) if (c,m)!=(900,3000)]
        chosen=candidate_subset(pairs)
        self.assertTrue(chosen);self.assertTrue(all(tuple(p) in pairs for p in chosen))
        self.assertEqual(candidate_subset([]),[])

    def test_sweep_preflight_and_counterbalanced_trials(self):
        backend=FakeBackend()
        def measure(stage,pair,block):return dict(correct=True,frames=frames(pair,time.monotonic_ns()),duration_s=1.)
        with tempfile.TemporaryDirectory() as tmp:
            result=controlled_sweep(backend,Path(tmp),PAIRS,measure,blocks=2)
            measured=[r for r in result["results"] if r["stage"]=="measure"]
            self.assertEqual([r["pair"] for r in measured[:3]],[list(p) for p in PAIRS])
            self.assertEqual([r["pair"] for r in measured[3:]],[list(p) for p in reversed(PAIRS)])
            self.assertEqual(result["restoration"],"verified")
            self.assertEqual(backend.policy,backend.original)
            self.assertIsNone(result["energy_surface"])

    def test_unverified_pair_excluded_and_failed_preflight_stops(self):
        backend=FakeBackend()
        with tempfile.TemporaryDirectory() as tmp:
            result=controlled_sweep(backend,Path(tmp),PAIRS,lambda *args:dict(correct=True,frames=[]))
            self.assertEqual(result["verified_pairs"],[])
        backend=FakeBackend();backend.fail_at={3};callback=Mock()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RestorationFailed):controlled_sweep(backend,Path(tmp),PAIRS,callback)
            callback.assert_not_called()

    def test_incorrect_work_restores_and_aborts(self):
        backend=FakeBackend()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,"Incorrect"):
                controlled_sweep(backend,Path(tmp),PAIRS,lambda *args:dict(correct=False,frames=[]))
            self.assertEqual(backend.policy,backend.original)

    def test_frequency_only_and_missing_input_hold(self):
        backend=FakeBackend()
        def settle(pair):return frames(pair,time.monotonic_ns())
        def observe(i):
            pair=(backend.policy["core"][0],backend.policy["memory"][0])
            return observation("phase8-frequency-only",i,1.,None if i==1 else 1.),frames(pair,time.monotonic_ns())
        with tempfile.TemporaryDirectory() as tmp:
            result=frequency_only(backend,Path(tmp)/"live",PAIRS,"paper_additive",observe,settle,budget=3)
            self.assertEqual(result[0]["requested_pair"],[1200,3000])
            self.assertEqual(result[1]["decision"]["reason"],"hold_invalid_observation")
            self.assertEqual(result[-1]["decision"]["post_state"]["updates"],2)
            self.assertEqual(backend.policy,backend.original)

    def test_failed_live_settling_restores(self):
        backend=FakeBackend()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ActuationBlocked):
                frequency_only(backend,Path(tmp)/"live",PAIRS,"paper_additive",Mock(),lambda p:[],budget=1)
            self.assertEqual(backend.policy,backend.original)

    def test_characterization_never_fills_missing_memory(self):
        trial=dict(host_start_ns=0,host_end_ns=3_000_000_000,frames=[],native=dict(mode="stream",mismatches=0,
                   elements_verified=1024,elements=1024,rounds=1,repetitions=10,oracle="masked_uint64_reference",tolerance=0,
                   start_ns=1,compute_end_ns=2_000_000_001,download_end_ns=2_100_000_000,validation_end_ns=2_200_000_000))
        result=characterize(trial)
        self.assertEqual(result["host_compute_s"],2.)
        self.assertEqual(result["guarded_matched_samples"],0)
        self.assertIsNone(result["memory_busy_percent_median"])
        self.assertIsNone(result["energy_j"])
        trial["native"]["mismatches"]=1
        with self.assertRaises(ValueError):characterize(trial)


class ArchiveTests(unittest.TestCase):
    def fixture(self,root):
        root.mkdir();(root/"actuation-gate").mkdir()
        names=[]
        for stage in ("pilot","measured"):
            for mode in ("compute","stream"):
                name=f"{mode}.pilot.json" if stage=="pilot" else f"{mode}.block-0.json"
                batch=dict(mode=mode,elements=257,rounds=1,seed=1729,batch=0 if stage=="pilot" else 1,
                           repetitions=1 if stage=="pilot" else 100,mismatches=0,elements_verified=257,
                           oracle="masked_uint64_reference",tolerance=0,start_ns=1,compute_end_ns=2_000_000_001,
                           download_end_ns=2_100_000_000,validation_end_ns=2_200_000_000,checksum=12345)
                save_new(root/name,dict(native=batch,frames=[],host_start_ns=0,host_end_ns=3_000_000_000,
                                       stage=stage,block=-1 if stage=="pilot" else 0,requested_pair=None,acknowledged_pair=None))
                names.append(name)
                if stage=="pilot":save_new(root/(mode+".ready.json"),{k:batch[k] for k in ("mode","elements","rounds","seed")})
        save_new(root/"capabilities.json",dict(prior=dict(restorable=False),pairs=[],selected_pairs=[]))
        (root/"actuation-gate/journal.jsonl").write_text(json.dumps(dict(kind="snapshot",snapshot=dict(restorable=False)))+'\n'+json.dumps(dict(kind="blocked"))+'\n')
        save_new(root/"manifest.json",dict(spec_version=SPEC_VERSION,native_exit_codes=dict(compute=0,stream=0),blocks=1,trials=names))
        save_new(root/"summary.json",evaluate(root))
        self.hashes(root)

    def hashes(self,root):
        path=root/"checksums.json"
        value={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p!=path}
        path.write_text(json.dumps(value))

    def test_archive_replay_checksum_and_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"archive";self.fixture(root)
            self.assertEqual(replay(root)["hardware_exit_gate"],"blocked")
            (root/"extra").write_text("extra")
            with self.assertRaisesRegex(ValueError,"inventory"):replay(root)
            (root/"extra").unlink()
            path=root/"compute.block-0.json";path.write_text(path.read_text()+" ")
            with self.assertRaisesRegex(ValueError,"checksum"):replay(root)

    def test_rehashed_incorrect_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"archive";self.fixture(root)
            path=root/"compute.block-0.json";trial=json.loads(path.read_text());trial["native"]["mismatches"]=1
            path.write_text(json.dumps(trial));self.hashes(root)
            with self.assertRaisesRegex(ValueError,"Incorrect"):replay(root)

    def test_external_trial_path_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"archive";self.fixture(root)
            path=root/"manifest.json";manifest=json.loads(path.read_text());manifest["trials"][0]="../outside.json"
            path.write_text(json.dumps(manifest));self.hashes(root)
            with self.assertRaisesRegex(ValueError,"inventory/order"):replay(root)


if __name__=="__main__":unittest.main()
