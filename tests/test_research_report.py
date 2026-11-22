"""Report claims must preserve missing evidence and independent trial pairing."""
import copy
from pathlib import Path
import tempfile
import unittest

from greengpu.research_report import assemble, command_comparisons, render, run


def fixture():
    results=[]
    datasets={}
    selections={}
    for workload in ('kmeans','hotspot'):
        for size in (0,1):
            datasets[f'datasets/{workload}-{size}-evaluation.data']={'n':100,'rows':100}
            selections[f'{workload}-{size}']=25
            for name in ('gpu_only','cpu_only','static50','tuned_static','cost_aware'):
                for block in range(10):
                    results.append(dict(workload=workload,size=size,name=name,block=block,
                                        stage='evaluation',status='passed',command_s=2. if name=='gpu_only' else 3.))
    performance=dict(status='passed',trials=len(results),failed_trials=0,results=results,
                     primary_scope='workload',baseline='automatic GPU-only',comparisons=[])
    integration=dict(software_gate='incomplete_concurrency_evidence',
                     live_joint_hardware_gate='blocked',hardware_trials=13,
                     allocation_decisions=10,ownership_changes=5,
                     concurrency={'kmeans':dict(status='not_demonstrated',iterations=[{'demonstrated':True},{'demonstrated':False}])})
    qualification=dict(actuation_gate='blocked',clock_mapping={'live_qualified':False},
                       device_timing_qualification='requires_external_validation',energy_qualification='blocked')
    plan=dict(blocks=10,datasets=datasets,selections=selections)
    return performance,integration,qualification,plan


class ReportTests(unittest.TestCase):
    def test_correctness_pass_does_not_imply_project_or_energy_success(self):
        report=assemble(*fixture())
        self.assertFalse(report['full_project_complete'])
        self.assertIsNone(report['energy_savings_percent'])
        self.assertTrue(report['evaluation']['primary_matrix_complete'])
        self.assertEqual(report['integration']['concurrency']['kmeans']['overlapping_iterations'],1)
        self.assertTrue(any('frequency-only' in b for b in report['blockers']))
        self.assertTrue(any('Physical energy' in b for b in report['blockers']))

    def test_command_regression_cannot_be_hidden_by_workload_speedup(self):
        performance,*_=fixture()
        comparisons=command_comparisons(performance['results'])
        row=next(r for r in comparisons if r['name']=='tuned_static')
        self.assertEqual(row['change_percent'],50.)
        self.assertEqual(row['bootstrap_95_percent'],[50.,50.])

    def test_failed_baseline_excludes_corresponding_pair(self):
        performance,*_=fixture()
        performance['results'][0]['status']='failed'
        rows=command_comparisons(performance['results'])
        row=next(r for r in rows if (r['workload'],r['size'],r['name'])==('kmeans',0,'tuned_static'))
        self.assertEqual(row['pairs'],9)
        self.assertEqual(row['unpaired'],1)
        performance['results']=[dict(r,status='failed') for r in performance['results']]
        self.assertTrue(all(r['change_percent'] is None for r in command_comparisons(performance['results'])))

    def test_missing_or_duplicate_blocks_do_not_pass_matrix(self):
        for mode in ('missing','duplicate'):
            args=list(fixture())
            if mode=='missing':args[0]['results'].pop()
            else:args[0]['results'][-1]['block']=8
            self.assertFalse(assemble(*args)['evaluation']['primary_matrix_complete'])

    def test_report_does_not_mutate_input_and_escapes_source_text(self):
        args=fixture();before=copy.deepcopy(args)
        report=assemble(*args)
        self.assertEqual(args,before)
        report['sources']={'performance':dict(path='<script>alert(1)</script>',checksums_sha256='abc')}
        document=render(report)
        self.assertIn('&lt;script&gt;',document)
        self.assertNotIn('<script>',document)
        self.assertIn('Full project acceptance remains incomplete',document)

    def test_output_cannot_modify_source_archive_or_overwrite_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaisesRegex(ValueError,'outside immutable'):
                run(root/'source'/'report',root/'source',root/'integration',root/'qualification')
            with self.assertRaises(FileExistsError):
                run(root,root/'source',root/'integration',root/'qualification')


if __name__=='__main__':unittest.main()
