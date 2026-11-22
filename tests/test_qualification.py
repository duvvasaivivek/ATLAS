import unittest
import json
import tempfile
from pathlib import Path
from greengpu.qualification import clock_diagnostic,replay
from greengpu.performance import interval,comparisons_for


class QualificationTests(unittest.TestCase):
    def test_modified_qualification_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'summary.json').write_text('{}')
            (root/'checksums.json').write_text(json.dumps({'summary.json':'wrong-hash'}))
            with self.assertRaisesRegex(ValueError,'Qualification archive integrity failure'):
                replay(root)

    def test_future_clock_is_rejected_without_fitting(self):
        frame=dict(wall_end_us=100,acquisition_end_ns=100000,buffer_samples=dict(core=[dict(timestamp_us=200)],memory=[]))
        result=clock_diagnostic([frame])
        self.assertEqual(result['future_samples'],1)
        self.assertEqual(result['status'],'rejected_future_driver_time')
        self.assertFalse(result['offset_fitted']);self.assertFalse(result['live_qualified'])

    def test_missing_and_valid_samples_are_not_physical_qualification(self):
        self.assertEqual(clock_diagnostic([])['status'],'no_samples')
        frame=dict(wall_end_us=100,acquisition_end_ns=100000,buffer_samples=dict(core=[dict(timestamp_us=90)],memory=[]))
        self.assertFalse(clock_diagnostic([frame])['live_qualified'])

    def test_bootstrap_retains_regressions_and_is_reproducible(self):
        values=[10.,20.,30.]
        self.assertEqual(interval(values),interval(values))
        self.assertGreater(interval(values)[0],0)

    def test_failed_baseline_is_not_imputed_or_paired(self):
        def row(name,block,status='passed'):
            return dict(name=name,block=block,status=status,workload='kmeans',size=0,stage='evaluation',primary_s=1.,command_s=2.,feedback_s=.1)
        result=comparisons_for([row('gpu_only',0,'failed'),row('gpu_only',1),row('adaptive',0),row('adaptive',1)])
        candidate=next(r for r in result if r['name']=='adaptive')
        self.assertEqual(candidate['requested_trials'],2)
        self.assertEqual(candidate['trials'],1)
        self.assertEqual(candidate['unpaired_trials'],1)
        self.assertEqual(candidate['paired_runtime_change_percent'],0)

    def test_all_failed_pairs_remain_unavailable(self):
        rows=[dict(name='gpu_only',block=0,status='failed',workload='hotspot',size=0,stage='evaluation')]
        result=comparisons_for(rows)[0]
        self.assertEqual(result['failed_trials'],1)
        self.assertIsNone(result['paired_runtime_change_percent'])
        self.assertIsNone(result['paired_bootstrap_95_percent'])
