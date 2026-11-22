import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from greengpu.supervision import Supervisor
from greengpu.sampling import IsolatedSampler
from greengpu.integration_runner import execute
from greengpu.capabilities import command


class StuckSampler:
    def sample(self):
        time.sleep(60)


class SlowStartingSampler:
    def __init__(self):time.sleep(.6)
    def sample(self):return {'value':42}


class FailedStartingSampler:
    def __init__(self):raise RuntimeError('factory failed')


def exited_within(pid, seconds=.5):
    status=Path(f'/proc/{pid}/stat')
    deadline=time.monotonic()+seconds
    while True:
        try:
            if status.read_text().split()[2] in ('Z','X'):return True
        except FileNotFoundError:
            return True
        if time.monotonic()>=deadline:return False
        time.sleep(.01)


class FailedPump:
    def __init__(self,*args):
        self.error=None;self.count=0;self.wall_ns=0;self.cpu_ns=0
    def start(self):self.error="temperature_limit_85C"
    def drain(self):return []
    def stop(self):pass


@unittest.skipUnless(os.name=="posix", "Linux/WSL process groups")
class SupervisionTests(unittest.TestCase):
    def test_startup_has_separate_deadline_from_queries(self):
        sampler=IsolatedSampler(SlowStartingSampler,timeout=.2,startup_timeout=5)
        try:self.assertEqual(sampler.sample()['value'],42)
        finally:sampler.close()

    def test_startup_timeout_and_failure_are_bounded(self):
        start=time.monotonic()
        with self.assertRaisesRegex(TimeoutError,'initialization'):
            IsolatedSampler(SlowStartingSampler,startup_timeout=.1)
        self.assertLess(time.monotonic()-start,2)
        with self.assertRaisesRegex(RuntimeError,'factory failed'):
            IsolatedSampler(FailedStartingSampler)

    def test_invalid_sampler_deadlines_rejected_before_spawn(self):
        for value in (0,-1,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):IsolatedSampler(timeout=value)
            with self.assertRaises(ValueError):IsolatedSampler(startup_timeout=value)

    def test_noninteractive_leader_exit_reaps_descendants(self):
        source='import subprocess,sys; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]); print(p.pid,flush=True)'
        start=time.monotonic()
        result=command([sys.executable,'-c',source],timeout=5)
        self.assertLess(time.monotonic()-start,3)
        self.assertEqual(result['status'],'ok')
        child=int(result['stdout'])
        self.assertTrue(exited_within(child))

    def test_noninteractive_timeout_retains_output(self):
        result=command([sys.executable,'-c','import time;print("partial",flush=True);time.sleep(60)'],timeout=2)
        self.assertEqual(result['status'],'timeout')
        self.assertEqual(result['stdout'],'partial\n')

    def test_exited_leader_does_not_disable_group_timeout(self):
        with tempfile.TemporaryFile(mode="w+t") as errors:
            source='import subprocess,sys; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]); print(p.pid,flush=True)'
            worker=Supervisor([sys.executable,"-c",source],errors,2)
            start=time.monotonic()
            try:
                child=int(worker.receive())
                with self.assertRaises(TimeoutError):list(worker.lines())
            finally:worker.close()
            self.assertLess(time.monotonic()-start,4)
            # killpg queues SIGKILL; the orphan may not have been scheduled to
            # exit when its already-exited leader has been reaped. Require death
            # within a bounded grace period, as in the noninteractive test.
            self.assertTrue(exited_within(child))

    def test_temperature_failure_does_not_wait_for_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"silent.py";path.write_text("import time;time.sleep(60)")
            case=dict(workload="kmeans",steps=10,workers=1,native_mode="fixed",
                      integration=dict(run_id="failure",n=10,initial_cpu_items=5))
            start=time.monotonic()
            with patch("greengpu.integration_runner.TelemetryPump",FailedPump):
                result=execute(sys.executable,path,None,case,None,timeout=10)
            self.assertLess(time.monotonic()-start,2)
            self.assertIn("temperature_limit",result["failure"])
            self.assertFalse(result["timed_out"])

    def test_stalled_sampler_is_terminated_and_reaped(self):
        sampler=IsolatedSampler(StuckSampler,timeout=.5)
        start=time.monotonic()
        with self.assertRaises(TimeoutError):sampler.sample()
        self.assertLess(time.monotonic()-start,2)
        self.assertFalse(sampler.process.is_alive())
        sampler.close()
