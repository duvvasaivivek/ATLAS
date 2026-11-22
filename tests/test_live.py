from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from greengpu.actuation import ActuationBlocked, RestorationFailed
from greengpu.integration import Config
from greengpu.live import LiveController, preflight, run_live, replay
from greengpu.frequency_experiment import sha

PAIRS=((600,1000),(1200,3000))


class Clock:
    def __init__(self):self.now=time.monotonic_ns()
    def __call__(self):return self.now


class Backend:
    def __init__(self):
        self.device_uuid='software-test-device';self.original={'core':None,'memory':[1500,1500]}
        self.policy=deepcopy(self.original);self.calls=[];self.fail_at=None
    def snapshot(self):return dict(device_uuid=self.device_uuid,policy=deepcopy(self.policy),restorable=True,policy_source='authoritative_policy_readback')
    def set_domain(self,domain,bounds):
        self.calls.append((domain,bounds));self.policy[domain]=bounds
        if len(self.calls)==self.fail_at:raise RuntimeError('injected setter failure')


class Provider:
    def __init__(self,clock):
        self.clock=clock;self.backend=Backend()
        self.q=dict(device_uuid=self.backend.device_uuid,pci_bus_id='0000:01:00.0',evidence_kind='software_test',
                    telemetry_clock='qualified',restoration_probe='qualified',device_timing='qualified',
                    evidence_id='test-only-fixture',qualified_at_ns=clock(),valid_until_ns=clock()+60_000_000_000,
                    verified_pairs=PAIRS)
    def qualification(self):return deepcopy(self.q)
    def samples(self):
        pair=(self.backend.policy['core'][0],self.backend.policy['memory'][0])
        return frames(self.clock(),pair,self.backend.device_uuid)


def frames(now,pair,uuid='software-test-device'):
    return [dict(device_uuid=uuid,sample_timestamp_us=(now-delta)//1000,sample_valid=True,
                 sample_monotonic_estimate_ns=now-delta-10000,sample_age_us=10,
                 acquisition_start_ns=now-delta-1000,acquisition_end_ns=now-delta,
                 core_utilization=.9,memory_utilization=.9,temperature_c=60,clock_event_reasons=0,
                 core_mhz=pair[0],memory_mhz=pair[1]) for delta in (400_000_000,200_000_000,0)]


class Workload:
    def __init__(self,clock,fail=False):self.clock=clock;self.started=False;self.closed=False;self.i=0;self.fail=fail
    def start(self,cfg,q):self.started=True;self.cfg=cfg
    def release(self,cut):self.cut=cut
    def observe(self,run_id):
        if self.fail:raise RuntimeError('injected workload failure')
        self.clock.now+=2_000_000_000
        obs=dict(run_id=run_id,iteration=self.i,cpu_items=self.cut,t_cpu=.02 if self.cut else None,
                 t_gpu=.01 if self.cut<100 else None,valid=True,common_origin=True,release_ns=self.clock()-100_000_000)
        self.i+=1
        return obs,self.i==4,0
    def finish(self):return {'status':'passed','fixture':'software_only'}
    def close(self):self.closed=True


def config():return Config('live-test',100,50,PAIRS,allocation_iterations=1,allocation_interval_ns=0,frequency_guard_ns=0,settle_iterations=0)


class LiveTests(unittest.TestCase):
    def test_unknown_stale_or_wrong_device_cannot_start_or_write(self):
        for change in ({'telemetry_clock':'blocked'},{'device_uuid':'other'},{'valid_until_ns':0},
                       {'verified_pairs':[]},{'restoration_probe':'blocked'},{'evidence_kind':'hardware'}):
            clock=Clock();provider=Provider(clock);provider.q.update(change);work=Workload(clock)
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder)/'run'
                with self.assertRaises(ActuationBlocked):run_live(provider,work,config(),root,clock)
                self.assertFalse(root.exists());self.assertFalse(work.started);self.assertEqual(provider.backend.calls,[])

    def test_joint_changes_replay_and_restore_without_hardware_claim(self):
        clock=Clock();provider=Provider(clock);work=Workload(clock)
        with tempfile.TemporaryDirectory() as folder,patch('greengpu.actuation.time.monotonic_ns',clock):
            root=Path(folder)/'run';result=run_live(provider,work,config(),root,clock)
            self.assertEqual(result['status'],'passed');self.assertTrue(work.closed)
            self.assertEqual(provider.backend.policy,provider.backend.original)
            self.assertFalse(result['physical_control_evidence'])
            self.assertTrue(any(r['decision']['requested_pair'] for r in result['records']))
            self.assertTrue(any(r['decision']['next_cpu_items']!=50 for r in result['records']))
            self.assertEqual(replay(root),result)
            result['records'][1]['decision']['next_cpu_items']=99
            (root/'result.json').write_text(json.dumps(result))
            hashes=json.loads((root/'checksums.json').read_text());hashes['result.json']=sha(root/'result.json')
            (root/'checksums.json').write_text(json.dumps(hashes))
            with self.assertRaisesRegex(ValueError,'replay mismatch'):replay(root)

    def test_workload_and_partial_setter_failures_restore(self):
        for setter in (False,True):
            clock=Clock();provider=Provider(clock);work=Workload(clock,fail=not setter)
            if setter:provider.backend.fail_at=2
            with tempfile.TemporaryDirectory() as folder,patch('greengpu.actuation.time.monotonic_ns',clock):
                root=Path(folder)/'run'
                with self.assertRaisesRegex(RuntimeError,'injected'):run_live(provider,work,config(),root,clock)
                self.assertTrue(work.closed);self.assertEqual(provider.backend.policy,provider.backend.original)
                self.assertTrue((root/'failure.json').exists());self.assertTrue((root/'checksums.json').exists())

    def test_qualification_expiring_while_native_arms_blocks_first_setter(self):
        clock = Clock()
        provider = Provider(clock)
        work = Workload(clock)
        work.start = lambda cfg, qualification: setattr(clock, 'now', qualification['valid_until_ns'])
        with tempfile.TemporaryDirectory() as folder, patch('greengpu.actuation.time.monotonic_ns', clock):
            root = Path(folder)/'expired'
            with self.assertRaisesRegex(ActuationBlocked, 'expired before clock actuation'):
                run_live(provider, work, config(), root, clock)
            self.assertEqual(provider.backend.calls, [])
            self.assertTrue(work.closed)

    def test_fixed_pair_and_calibrated_power_trace_replay(self):
        clock = Clock()
        provider = Provider(clock)
        provider.q.update(energy_domain='gpu_board', energy_evidence_id='separate-meter', energy_max_gap_ns=2_000_000_000,
                          energy_accuracy_w=.125, energy_accuracy_relative=.01)
        def power(start, end):
            return [dict(time_ns=time_ns, value=12.5, status='ok', domain='gpu_board', evidence_id='separate-meter')
                    for time_ns in range(start, end+1, 1_000_000_000)]
        provider.energy_samples = power
        with tempfile.TemporaryDirectory() as folder, patch('greengpu.actuation.time.monotonic_ns', clock):
            root = Path(folder)/'fixed'
            result = run_live(provider, Workload(clock), config(), root, clock,
                              initial_pair=PAIRS[1], frequency_control=False, measure=True)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['energy']['joules'], 100)
            self.assertEqual(result['energy']['meter_accuracy_j'], 2)
            self.assertFalse(result['energy']['physical_measurement'])
            self.assertTrue(all(row['decision']['frequency'] is None for row in result['records']))
            self.assertEqual(len(provider.backend.calls), 4)
            self.assertEqual(replay(root), result)
            trace = json.loads((root/'power.json').read_text())
            trace['end_ns'] -= 1
            (root/'power.json').write_text(json.dumps(trace))
            checksums = json.loads((root/'checksums.json').read_text())
            checksums['power.json'] = sha(root/'power.json')
            (root/'checksums.json').write_text(json.dumps(checksums))
            with self.assertRaisesRegex(ValueError, 'Energy window'):
                replay(root)

    def test_unbracketed_power_yields_no_energy_and_missing_meter_blocks_clock_writes(self):
        clock = Clock()
        provider = Provider(clock)
        with tempfile.TemporaryDirectory() as folder, patch('greengpu.actuation.time.monotonic_ns', clock):
            root = Path(folder)/'energy'
            with self.assertRaises(ActuationBlocked):
                run_live(provider, Workload(clock), config(), root, clock, measure=True)
            self.assertFalse(root.exists())
            self.assertEqual(provider.backend.calls, [])
            provider.q.update(energy_domain='gpu_board', energy_evidence_id='separate-meter', energy_max_gap_ns=2_000_000_000,
                              energy_accuracy_w=.125, energy_accuracy_relative=.01)
            provider.energy_samples = lambda start, end: []
            result = run_live(provider, Workload(clock), config(), root, clock, measure=True)
            self.assertEqual(result['energy']['status'], 'invalid')
            self.assertIsNone(result['energy']['joules'])
            self.assertEqual(replay(root), result)

    def test_settling_timeout_and_bad_frames_stop_control(self):
        clock=Clock();provider=Provider(clock);cfg=config()
        def ready():
            controller=LiveController(cfg,provider.q);controller.applied(PAIRS[0],clock());return controller
        obs=dict(run_id=cfg.run_id,iteration=0,cpu_items=50,t_cpu=.02,t_gpu=.01,valid=True,common_origin=True,release_ns=clock())
        with self.assertRaisesRegex(ActuationBlocked,'deadline'):ready().barrier(obs,clock()+11_000_000_000)
        for key,value in (('device_uuid','other'),('temperature_c',85),('acquisition_end_ns',clock()+3_000_000_000)):
            data=frames(clock()+2_000_000_000,PAIRS[0]);data[-1][key]=value
            with self.assertRaises(ActuationBlocked):ready().barrier(obs,clock()+2_000_000_000,data)

    def test_short_job_does_not_certify_unsettled_setting(self):
        clock=Clock();provider=Provider(clock);cfg=config();controller=LiveController(cfg,provider.q)
        controller.applied(PAIRS[0],clock())
        obs=dict(run_id=cfg.run_id,iteration=0,cpu_items=50,t_cpu=.02,t_gpu=.01,valid=True,common_origin=True,release_ns=clock())
        result=controller.barrier(obs,clock()+100000,final=True)
        self.assertEqual(result['reason'],'incomplete_settling');self.assertIsNone(result['requested_pair'])

    def test_lost_verification_cannot_finish_or_continue_indefinitely(self):
        clock=Clock();provider=Provider(clock);cfg=config();controller=LiveController(cfg,provider.q)
        controller.applied(PAIRS[0],clock())
        obs=dict(run_id=cfg.run_id,iteration=0,cpu_items=50,t_cpu=.02,t_gpu=.01,valid=True,common_origin=True,release_ns=clock())
        controller.barrier(obs,clock()+2_000_000_000,frames(clock()+2_000_000_000,PAIRS[0]))
        self.assertFalse(controller.pending)
        result=controller.barrier(dict(obs,iteration=1),clock()+3_000_000_000,final=True)
        self.assertEqual(result['reason'],'incomplete_verification')
        with self.assertRaisesRegex(ActuationBlocked,'verification was lost'):
            controller.barrier(dict(obs,iteration=2),clock()+13_000_000_000)

    def test_failed_final_correctness_and_restoration_are_not_success(self):
        for restoration_failure in (False,True):
            clock=Clock();provider=Provider(clock);work=Workload(clock,fail=restoration_failure)
            if restoration_failure:provider.backend.fail_at=3
            work.finish=lambda:dict(status='failed')
            error=RestorationFailed if restoration_failure else RuntimeError
            with tempfile.TemporaryDirectory() as folder,patch('greengpu.actuation.time.monotonic_ns',clock):
                root=Path(folder)/'run'
                with self.assertRaises(error):run_live(provider,work,config(),root,clock)
                self.assertTrue(work.closed);self.assertFalse((root/'result.json').exists())
                failure=json.loads((root/'failure.json').read_text())
                self.assertEqual(failure['restoration_events'][-1]['status'],'failed' if restoration_failure else 'verified')

    def test_later_clock_drift_invalidates_earlier_good_streak(self):
        clock=Clock();provider=Provider(clock);cfg=config();controller=LiveController(cfg,provider.q)
        controller.applied(PAIRS[0],clock())
        now=clock()+2_000_000_000
        data=frames(now,PAIRS[0])+frames(now+10_000_000,PAIRS[1])[-1:]
        obs=dict(run_id=cfg.run_id,iteration=0,cpu_items=50,t_cpu=.02,t_gpu=.01,valid=True,common_origin=True,release_ns=clock())
        result=controller.barrier(obs,now+10_000_000,data,final=True)
        self.assertEqual(result['verification']['status'],'unverified')
        self.assertEqual(result['reason'],'incomplete_settling')

    def test_native_device_mismatch_is_reaped_before_any_setter(self):
        from greengpu import kmeans_reference as kr
        from greengpu.live_native import NativeWorkload
        clock=Clock();provider=Provider(clock)
        with tempfile.TemporaryDirectory() as folder,patch('greengpu.live_native.Supervisor') as supervisor:
            root=Path(folder)/'run';dataset=Path(folder)/'input.kmdat'
            kr.write_dataset(dataset,*kr.clustered(100,2,2,33))
            supervisor.return_value.receive.return_value=json.dumps(dict(kind='configuration',armed=True,feedback_batch=1,pci_bus_id='0000:ff:00.0'))+'\n'
            work=NativeWorkload(__file__,dataset,'kmeans',4,4,root/'native')
            with self.assertRaisesRegex(ActuationBlocked,'device mismatch'):run_live(provider,work,config(),root,clock)
            self.assertEqual(provider.backend.calls,[])
            supervisor.return_value.close.assert_called_once()
            failure=json.loads((root/'failure.json').read_text())
            self.assertEqual(failure['restoration_events'][-1]['status'],'not_needed')


if __name__=='__main__':unittest.main()
