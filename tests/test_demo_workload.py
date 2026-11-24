"""Presenter must not publish a successful comparison after failed or unequal work."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('demo_workload', Path(__file__).resolve().parents[1] / 'scripts/demo_workload.py')
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def record(labels=(0, 1, 2, 3), valid=True):
    rows = [{'kind': 'iteration', 'warmup': False, 'index': i, 'cpu_items': 2, 'centroids': list(range(8))} for i in range(32)]
    rows.append({'kind': 'summary', 'status': 'passed', 'independent_validation': 'passed' if valid else 'failed',
                 'iterations': 32, 'final_labels': list(labels), 'final_centroids': list(range(8)), 'loop_elapsed_ns': 1000000})
    return {'status': 'ok', 'stderr': '', 'stdout': '\n'.join(json.dumps(row) for row in rows), 'elapsed_ns': 2000000}


class DemoWorkloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.build = self.root / 'build'
        self.build.mkdir()
        (self.build / 'atlas_kmeans').write_bytes(b'test binary')
        self.output = self.root / 'run'

    def execute(self, records):
        with patch.object(demo, 'execute', side_effect=records), patch.object(demo, 'replay_decisions'):
            return demo.run(self.build, self.output, 4)

    def test_failed_native_trial_is_retained_without_success_summary(self):
        with self.assertRaisesRegex(RuntimeError, 'driver failure'):
            self.execute([{'status': 'failed', 'stderr': 'driver failure', 'failure': 'failed'}])
        self.assertTrue((self.output / 'trial-00.json').is_file())
        self.assertTrue((self.output / 'failure.json').is_file())
        self.assertFalse((self.output / 'summary.json').exists())
        self.assertFalse((self.output / 'demo.html').exists())

    def test_different_answers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outputs differ'):
            self.execute([record(), record(labels=(1, 0, 2, 3))])
        self.assertFalse((self.output / 'demo.html').exists())

    def test_independent_validation_is_required(self):
        with self.assertRaisesRegex(ValueError, 'independent correctness'):
            self.execute([record(valid=False)])
        self.assertFalse((self.output / 'summary.json').exists())

    def test_success_has_real_trace_and_verified_portable_presentation(self):
        result = self.execute([record() for _ in range(9)])
        self.assertEqual(len(result['trials']), 9)
        self.assertEqual(len(result['trace']), 32)
        self.assertEqual(result['energy_savings'], None)
        inventory = json.loads((self.output / 'checksums.json').read_text())
        for name, expected in inventory.items():
            self.assertEqual(demo.hashlib.sha256((self.output / name).read_bytes()).hexdigest(), expected)
        page = (self.output / 'demo.html').read_text(encoding='utf-8')
        self.assertIn('RECORDED MEASUREMENT', page)
        self.assertNotIn('__DATA__', page)
        self.assertNotIn('src="http', page)

    def test_live_snapshots_arrive_before_trial_validation(self):
        observed = []
        def streaming_execute(*args, on_iteration, **kwargs):
            result = record()
            for row in map(json.loads, result['stdout'].splitlines()):
                if row['kind'] == 'iteration':
                    on_iteration(row)
            snapshot = json.loads((self.output / 'live.json').read_text())
            observed.append(snapshot)
            return result
        with patch.object(demo, 'execute', side_effect=streaming_execute), patch.object(demo, 'replay_decisions'):
            demo.run(self.build, self.output, 4)
        self.assertEqual(len(observed), 9)
        self.assertTrue(all(s['phase'] == 'computing' for s in observed))
        self.assertTrue(all(s['labels'] is None for s in observed))
        self.assertTrue(all(len(s['trace']) == 32 for s in observed))
        self.assertEqual(observed[0]['trace'][-1]['cpu_percent'], 50)
        self.assertEqual(json.loads((self.output / 'live.json').read_text())['phase'], 'complete')


if __name__ == '__main__':
    unittest.main()
