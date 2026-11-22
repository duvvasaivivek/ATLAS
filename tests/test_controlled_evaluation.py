from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from greengpu.actuation import ActuationBlocked
from greengpu.controlled_evaluation import evaluation_plan, paired_comparisons, run, select_static, tuning_plan


class ControlledEvaluationTests(unittest.TestCase):
    def test_independent_static_selection_and_matched_arm_configuration(self):
        plan = dict(items=101, initial_cpu_items=15, pairs=[[600, 1000], [1200, 3000]],
                    high_pair=[1200, 3000], blocks=3, seed=1)
        pilot = tuning_plan(plan)
        self.assertEqual(len(pilot), 16)
        self.assertEqual({(row['arm'], row['block']) for row in pilot},
                         {(row['arm'], block) for row in pilot for block in range(2)})
        tuning = [dict(row, status='passed', joules=(1 if row['cpu_items'] == 25 else 2)) for row in pilot]
        best = select_static(tuning)
        self.assertEqual(best['cpu_items'], 25)
        trials = evaluation_plan(plan, best)
        self.assertEqual(len(trials), 18)
        by_arm = {trial['arm']: trial for trial in trials if trial['block'] == 0}
        self.assertEqual(by_arm['B0']['cpu_items'], 0)
        self.assertEqual(by_arm['B3']['cpu_items'], by_arm['B5']['cpu_items'])
        self.assertFalse(by_arm['B3']['change_clocks'])
        self.assertTrue(by_arm['B4']['change_clocks'])
        self.assertEqual(by_arm['B4']['allocation'], 'static')
        self.assertEqual(by_arm['B6']['pair'], best['pair'])
        self.assertIsNone(select_static([*tuning, dict(tuning[0], status='failed')]))

    def test_comparison_uses_matched_complete_blocks_and_keeps_negative_results(self):
        def result(arm, block, joules, runtime, status='passed'):
            return dict(arm=arm, block=block, joules=joules, measured_s=runtime, status=status)

        rows = [result('B0', 0, 10, 1), result('B0', 1, None, None, 'failed'),
                result('B5', 0, 12, .5), result('B5', 1, 8, 4)]
        comparison = next(row for row in paired_comparisons(rows) if (row['arm'], row['reference']) == ('B5', 'B0'))
        self.assertEqual(comparison['requested'], 2)
        self.assertEqual(comparison['pairs'], 1)
        self.assertAlmostEqual(comparison['energy_saving_percent']['mean'], -20)
        self.assertEqual(comparison['runtime_change_percent']['mean'], -50)
        self.assertIsNone(comparison['energy_bootstrap_95_percent'])

    def test_software_qualification_cannot_launch_physical_benchmark(self):
        with tempfile.TemporaryDirectory() as folder, patch('greengpu.controlled_evaluation.preflight') as preflight:
            root = Path(folder)
            tuning, held_out = root/'tuning', root/'held-out'
            header = struct.pack('<8sIII', b'ATLSKM01', 4, 2, 2)
            tuning.write_bytes(header+b'tuning')
            held_out.write_bytes(header+b'held-out')
            preflight.return_value = {'evidence_kind': 'software_test'}
            output = root/'evaluation'
            with self.assertRaisesRegex(ActuationBlocked, 'physical control'):
                run(object(), output, __file__, 'kmeans', tuning, held_out, (1200,3000), [(600,1000)],
                    steps=4, workers=4, blocks=2, initial_fraction=.5)
            self.assertFalse(output.exists())
