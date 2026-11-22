import unittest
from greengpu.allocation import Config,initial_state,transition


class CostTests(unittest.TestCase):
    def test_probes_endpoints_and_selects_measured_gpu_only(self):
        cfg=Config("cost_aware_v1",100);state=initial_state('t');cut=50
        for i in range(8):
            cost=1+cut/100
            obs=dict(run_id='t',epoch=0,iteration=i,cpu_items=cut,t_cpu=cost if cut else None,
                     t_gpu=cost if cut<100 else None,valid=True,common_origin=True,work_s=cost,migration_s=.1,remaining_iterations=100)
            decision=transition(cfg,state,obs);state=decision['post_state'];cut=decision['cpu_items']
        self.assertEqual(cut,0)
        self.assertTrue({'0','100'}.issubset(state['cost_samples']))

    def test_short_horizon_avoids_exploratory_migrations(self):
        obs=dict(run_id='t',epoch=0,iteration=0,cpu_items=50,t_cpu=1.,t_gpu=2.,valid=True,common_origin=True,
                 work_s=2.,migration_s=10.,remaining_iterations=1)
        result=transition(Config('cost_aware_v1',100),initial_state('t'),obs)
        self.assertEqual(result['cpu_items'],50)
        self.assertEqual(result['reason'],'hold_short_horizon')
