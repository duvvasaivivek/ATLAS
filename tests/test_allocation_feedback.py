import copy
from dataclasses import asdict
import unittest

from greengpu.allocation import Config, initial_state, transition
from greengpu.allocation_experiment import observation, assess_feedback
from test_kmeans import records_fixture, process


def fixture():
    case,meta,rows,golden=records_fixture()
    case.update(run_id='feedback-test',policy='static',initial_cpu_items=3,name='static',record='test.json')
    rows[0]['feedback']=True
    for r in rows[1:-1]:
        r.update(cpu_items=3,gpu_items=0,resident_before=3,resident_after=3,ownership_changed_points=0,
                 h2d_point_bytes=0,h2d_centroid_bytes=0,d2h_label_bytes=0,d2h_aggregate_bytes=0,
                 cpu_start_ns=r['release_ns']+1,cpu_end_ns=r['release_ns']+5,
                 gpu_dispatch_ns=None,h2d_end_ns=None,compute_end_ns=None,gpu_end_ns=None,workers_used=1,
                 control_ns=7 if r['index']==1 else 0)
    rows[-1].update(control_ns=7,active_and_control_ns=27)
    cfg=Config('static',3)
    decision=transition(cfg,initial_state(case['run_id']),observation(rows[2],case['run_id']))
    p=process(rows)
    p.update(configuration=asdict(cfg),decisions=[dict(decision=decision,controller_ns=2)],elapsed_ns=1000)
    return case,meta,p,golden


class FeedbackTests(unittest.TestCase):
    def test_feedback_replay_accounts_without_double_counting(self):
        c,m,p,g=fixture()
        result=assess_feedback(c,m,p,g)
        self.assertEqual(result['active_and_control_s'],27e-9)
        self.assertEqual(result['controller_compute_s'],2e-9)
        self.assertIsNone(result['energy_j'])

    def test_corrupt_decision_and_missing_observation_rejected(self):
        c,m,p,g=fixture()
        bad=copy.deepcopy(p);bad['decisions'][0]['decision']['cpu_items']=0
        with self.assertRaisesRegex(ValueError,'Decision replay'):assess_feedback(c,m,bad,g)
        bad=copy.deepcopy(p);bad['decisions']=[]
        with self.assertRaisesRegex(ValueError,'Missing/extra'):assess_feedback(c,m,bad,g)

    def test_bad_control_accounting_and_config_rejected(self):
        import json
        c,m,p,g=fixture()
        bad=copy.deepcopy(p);rows=[json.loads(line) for line in bad['stdout'].splitlines()]
        rows[-1]['active_and_control_ns']=29
        bad['stdout']='\n'.join(json.dumps(r) for r in rows)
        with self.assertRaisesRegex(ValueError,'timing total'):assess_feedback(c,m,bad,g)
        bad=copy.deepcopy(p);bad['configuration']['step']=.2
        with self.assertRaisesRegex(ValueError,'configuration'):assess_feedback(c,m,bad,g)


if __name__=='__main__':unittest.main()
