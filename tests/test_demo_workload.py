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
        def streaming_execute(*args, on_feedback, **kwargs):
            result = record()
            for row in map(json.loads, result['stdout'].splitlines()):
                if row['kind'] == 'iteration':
                    on_feedback(row, {})
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

    def test_decision_window_and_history_survive_skipped_snapshots(self):
        observed = []

        def streaming_execute(*args, on_feedback, **kwargs):
            result = record()
            sharing = args[3]['integration']['allocation_policy'] == 'paper_proportional'
            barriers = []
            for row in map(json.loads, result['stdout'].splitlines()):
                if row['kind'] != 'iteration':
                    continue
                barrier = {}
                if sharing and row['index'] in (3, 7):
                    reason = 'proportional_adjust' if row['index'] == 3 else 'hold_quantized'
                    barrier = {'input': {'observation': {'t_cpu': .0001, 't_gpu': .0009}},
                               'decision': {'allocation_reason': reason,
                                            'next_cpu_items': 1 if row['index'] == 3 else 2,
                                            'allocation': {'policy': 'paper_proportional',
                                                           'observation': {'t_cpu': .004, 't_gpu': .002,
                                                                           'source_iterations': [0, 1, 2, 3]}}}}
                barriers.append(barrier)
                on_feedback(row, barrier)
            result['barriers'] = barriers
            observed.append(json.loads((self.output / 'live.json').read_text()))
            return result

        with patch.object(demo, 'execute', side_effect=streaming_execute), patch.object(demo, 'replay_decisions'), \
             patch.object(demo.time, 'monotonic', return_value=1):
            result = demo.run(self.build, self.output, 4, repeats=1)
        snapshot = next(value for value in observed if value['mode'] == 'ATLAS sharing')
        self.assertEqual(snapshot['round'], 32)
        self.assertEqual([item['step'] for item in snapshot['decision_history']], [4, 8])
        explanation = snapshot['decision_history'][0]['explanation']
        self.assertIn('shifts 1 row to the GPU', explanation)
        self.assertIn('CPU 4.000 ms, GPU 2.000 ms', explanation)
        self.assertIn('rounded to the same', snapshot['decision_history'][1]['explanation'])
        step = next(t for t in result['step_trials'] if t['mode'] == 'ATLAS sharing')['steps'][3]
        self.assertEqual(step['cpu_completion_ms'], .1)
        self.assertEqual(step['explanation'], explanation)
        final_live = json.loads((self.output / 'live.json').read_text())
        self.assertEqual(final_live['decision_history'], snapshot['decision_history'])
        self.assertIn('Lowest median whole-command time:', final_live['comparison'][1])

    def test_import_preserves_row_identity_in_export(self):
        dataset = self.root / 'my locations.csv'
        dataset.write_text('id,x,y\na,100,200\nb,3,4\nc,-5,6\nd,7,8\n')
        with patch.object(demo, 'execute', return_value=record()), patch.object(demo, 'replay_decisions'):
            result = demo.run(self.build, self.output, 2000, repeats=1, input_csv=dataset)
        self.assertEqual(result['size'], 4)
        self.assertEqual(result['input']['sha256'], demo.hashlib.sha256(dataset.read_bytes()).hexdigest())
        self.assertIn('1,a,100.0,200.0,1', (self.output / 'groups.csv').read_text())
        self.assertEqual(result['step_trials'][0]['steps'][0]['gpu_items'], 2)

    def test_invalid_coordinates_are_rejected_before_running(self):
        dataset = self.root / 'invalid.csv'
        dataset.write_text('id,x,y\na,nan,200\n')
        with patch.object(demo, 'execute') as execute:
            with self.assertRaisesRegex(ValueError, 'CSV line 2'):
                demo.run(self.build, self.output, 2000, input_csv=dataset)
            execute.assert_not_called()

    def test_live_performance_traces_use_real_timestamps_without_inventing_missing_modes(self):
        def timed_record(step_ns):
            rows = [{
                'kind': 'iteration', 'warmup': False, 'index': i, 'cpu_items': 2,
                'centroids': list(range(8)), 'release_ns': 1_000_000 * (i + 1),
                'end_ns': 1_000_000 * (i + 1) + step_ns * (i + 1)
            } for i in range(32)]
            rows.append({'kind': 'summary', 'status': 'passed', 'independent_validation': 'passed',
                         'iterations': 32, 'final_labels': [0, 1, 2, 3], 'final_centroids': list(range(8)),
                         'loop_elapsed_ns': 1000000})
            return {'status': 'ok', 'stderr': '', 'stdout': '\n'.join(json.dumps(row) for row in rows), 'elapsed_ns': 2000000}

        records = [timed_record(500_000), timed_record(200_000), timed_record(300_000)]
        snapshots = []
        def streaming_execute(*args, on_feedback, **kwargs):
            res = records.pop(0)
            for row in map(json.loads, res['stdout'].splitlines()):
                if row['kind'] == 'iteration':
                    on_feedback(row, {})
            snapshots.append(json.loads((self.output / 'live.json').read_text()))
            return res

        with patch.object(demo, 'execute', side_effect=streaming_execute), patch.object(demo, 'replay_decisions'):
            demo.run(self.build, self.output, 4, repeats=1)

        # First snapshot only has the first executed mode; unexecuted modes remain empty.
        first_perf = snapshots[0]['performance_traces']
        non_empty_first = [mode for mode, pts in first_perf.items() if pts]
        self.assertEqual(len(non_empty_first), 1)
        self.assertEqual(len(first_perf[non_empty_first[0]]), 32)
        self.assertAlmostEqual(first_perf[non_empty_first[0]][0]['step_ms'], 0.5)

        # Final snapshot has real measured durations for all three modes without fake data.
        final_live = json.loads((self.output / 'live.json').read_text())
        self.assertEqual(final_live['phase'], 'complete')
        for mode in ('CPU only', 'GPU only', 'ATLAS sharing'):
            self.assertEqual(len(final_live['performance_traces'][mode]), 32)
        self.assertEqual(final_live['trace'][0]['cpu_percent'], 50)


if __name__ == '__main__':
    unittest.main()
