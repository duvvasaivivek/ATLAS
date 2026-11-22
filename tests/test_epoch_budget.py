import unittest
from greengpu.allocation import Config,initial_state,transition


class EpochBudgetTests(unittest.TestCase):
    def test_extension_resumes_on_new_epoch_but_literal_policy_stays_exhausted(self):
        for policy,resumes in (("robust_proportional_epoch_v1",True),("robust_proportional_v1",False)):
            cfg=Config(policy,100,budget=1);state=initial_state("t")
            first=transition(cfg,state,dict(run_id="t",epoch=0,iteration=0,cpu_items=50,t_cpu=2.,t_gpu=1.,valid=True,common_origin=True))
            obs=dict(run_id="t",epoch=0,iteration=1,cpu_items=first['cpu_items'],t_cpu=1.,t_gpu=3.,valid=True,common_origin=True)
            held=transition(cfg,first['post_state'],obs)
            self.assertEqual(held['reason'],'budget_exhausted')
            obs.update(epoch=1,iteration=2)
            resumed=transition(cfg,held['post_state'],obs)
            self.assertEqual(resumed['reason']!='budget_exhausted',resumes)
