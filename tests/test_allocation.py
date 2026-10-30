import copy
import math
import unittest

from greengpu.allocation import Config, initial_state, transition, balance_trace, POLICIES
from greengpu.allocation_scenarios import simulate


def observation(r=.3,a=6.,b=3.,i=0,n=1000000,epoch=0):
    return dict(run_id="test",epoch=epoch,iteration=i,cpu_items=round(r*n),
                t_cpu=a,t_gpu=b,valid=True,common_origin=True)


class AllocationTests(unittest.TestCase):
    def test_hand_calculations(self):
        s=initial_state("test")
        d=transition(Config("paper_simple",1000000),s,observation(.1,1.,1.4))
        self.assertEqual(d["reason"],"hold_predicted_oscillation")
        self.assertEqual(d["cpu_items"],100000)
        d=transition(Config("paper_proportional",1000000),s,observation())
        self.assertAlmostEqual(d["requested_ratio"],3/17)
        s.update(sign=1,step=.15)
        d=transition(Config("paper_adaptive",1000000),s,observation(.15,.6,.8))
        self.assertEqual(d["post_state"]["step"],.075)
        self.assertAlmostEqual(d["requested_ratio"],.225)

    def test_purity_and_invalid_identity_or_timing(self):
        cfg=Config("paper_proportional",1000000)
        s=initial_state("test");o=observation();old=copy.deepcopy((s,o))
        d=transition(cfg,s,o)
        self.assertEqual((s,o),old)
        for key,value in (("run_id","other"),("t_cpu",0),("t_gpu",float('nan')),
                          ("t_cpu",float('inf')),("t_gpu",-1),("valid",False),("common_origin",False)):
            bad=dict(o);bad[key]=value
            rejected=transition(cfg,s,bad)
            self.assertEqual(rejected["reason"],"hold_invalid_observation")
            self.assertEqual(rejected["post_state"],s)
        self.assertEqual(transition(cfg,d['post_state'],o)['post_state'],d['post_state'])
        d['post_state']['epoch']=2
        self.assertEqual(transition(cfg,d['post_state'],observation(i=1,epoch=1))['reason'],'hold_invalid_observation')

    def test_endpoints_and_no_interior(self):
        for policy in POLICIES:
            cfg=Config(policy,10)
            for ratio,a,b in ((0,None,1.),(1,1.,None)):
                d=transition(cfg,initial_state('test'),observation(ratio,a,b,n=10))
                self.assertEqual(d['reason'],'needs_interior_probe')
                self.assertEqual(d['cpu_items'],5)
            cfg=Config(policy,1)
            d=transition(cfg,initial_state('test'),observation(0,None,1.,n=1))
            self.assertEqual(d['reason'],'unsupported_partition')

    def test_legal_rounding_and_sparse_grid(self):
        cfg=Config('paper_proportional',10,legal_counts=(0,2,6,10))
        self.assertEqual(cfg.nearest(.4),2)
        self.assertEqual(cfg.nearest(.5),6)
        self.assertEqual(Config('paper_simple',3).nearest(.5),1)
        with self.assertRaises(ValueError):Config('paper_simple',10,legal_counts=(1,1))

    def test_literal_binary_counterexample_and_budget(self):
        run=simulate('paper_binary_literal','stationary',.5,n=1048576)
        values=[d['observation']['cpu_items']/1048576 for d in run['trace'][:7]]
        self.assertEqual(values,[.5,.75,.875,.4375,.71875,.859375,.4296875])
        self.assertEqual(run['trace'][-1]['reason'],'budget_exhausted')
        self.assertIsNone(run['diagnostic']['first_sustained'])

    def test_bisection_bound_stationary_and_no_bracket(self):
        run=simulate('bracketed_bisection','stationary',.5,n=1048576)
        stopped=next(d for d in run['trace'] if d['reason']=='stop_bracket_width')
        b=stopped['post_state']['bracket']
        self.assertLessEqual((b['hi']-b['lo'])/1048576,.01)
        self.assertLessEqual(b['updates'],math.ceil(math.log2(1/.01)))
        self.assertTrue(b['lo']/1048576 <= .8 <= b['hi']/1048576)
        run=simulate('bracketed_bisection','transfer_dominated')
        self.assertIn('no_bracket',[d['reason'] for d in run['trace']])

    def test_bisection_resolution_and_invalid_sign(self):
        cfg=Config('bracketed_bisection',100,legal_counts=(0,20,80,100),precision=.001)
        s=initial_state('test')
        d=transition(cfg,s,observation(.2,1.,4.,n=100))
        d=transition(cfg,d['post_state'],observation(.8,4.,1.,i=1,n=100))
        self.assertEqual(d['reason'],'stop_resolution')
        cfg=Config('bracketed_bisection',100)
        s=initial_state('test')
        d=transition(cfg,s,observation(.01,1.,4.,n=100))
        d=transition(cfg,d['post_state'],observation(.99,4.,1.,i=1,n=100))
        # Feed a contradictory repeated endpoint instead of the requested midpoint.
        d=transition(cfg,d['post_state'],observation(.01,5.,1.,i=2,n=100))
        self.assertEqual(d['reason'],'invalid_bracket')

    def test_precision_and_quantization_are_not_balance(self):
        cfg=Config('paper_adaptive',100)
        s=initial_state('test');s.update(step=.015,sign=-1)
        d=transition(cfg,s,observation(.5,4.,1.,n=100))
        self.assertEqual(d['reason'],'stop_precision')
        self.assertFalse(balance_trace([d['observation']])['balanced_at_end'])
        cfg=Config('paper_proportional',2)
        d=transition(cfg,initial_state('test'),observation(.5,1.,1.01,n=2))
        self.assertEqual(d['reason'],'hold_quantized')

    def test_diagnostics_reset_and_do_not_freeze_proportional(self):
        obs=[observation(.5,1.,1.,i=i) for i in range(5)]
        obs[3]['valid']=False
        diag=balance_trace(obs)
        self.assertEqual(diag['sustained_at'],[2])
        self.assertFalse(diag['balanced_at_end'])
        obs[3]['valid']=True;obs[2]['epoch']=1
        self.assertNotIn(2,balance_trace(obs)['sustained_at'])
        d=transition(Config('paper_proportional',1000000),initial_state('test'),obs[0])
        self.assertIsNone(d['post_state']['frozen'])

    def test_robust_limits_dwell_and_phase_reset(self):
        cfg=Config('robust_proportional_v1',1000000)
        s=initial_state('test')
        d=transition(cfg,s,observation(.5,100.,1.))
        self.assertAlmostEqual(d['requested_ratio'],.4)
        d=transition(cfg,d['post_state'],observation(.4,80.,1.2,i=1))
        self.assertEqual(d['reason'],'hold_dwell')
        d=transition(cfg,d['post_state'],observation(.4,1.,80.,i=2,epoch=1))
        self.assertEqual(d['reason'],'reset_phase_and_adjust')
        self.assertIsNone(d['post_state']['bracket'])

    def test_bracket_restarts_after_declared_phase_change(self):
        run=simulate('bracketed_bisection','phase_change')
        changed=run['trace'][102]
        self.assertEqual(changed['reason'],'probe_bracket_low')
        self.assertTrue(run['diagnostic']['balanced_at_end'])

    def test_completed_bracket_rejects_later_sign_reversal(self):
        cfg=Config('bracketed_bisection',100,legal_counts=(0,20,80,100),precision=.001)
        d=transition(cfg,initial_state('test'),observation(.2,1.,4.,n=100))
        d=transition(cfg,d['post_state'],observation(.8,4.,1.,i=1,n=100))
        self.assertEqual(d['reason'],'stop_resolution')
        self.assertEqual(d['cpu_items'],20)
        d=transition(cfg,d['post_state'],observation(.2,4.,1.,i=2,n=100))
        self.assertEqual(d['reason'],'invalid_bracket')

    def test_robust_detects_undeclared_capacity_change(self):
        cfg=Config('robust_proportional_v1',1000000)
        d=transition(cfg,initial_state('test'),observation(.5,10.,1.))
        d=transition(cfg,d['post_state'],observation(.4,1.,20.,i=1))
        d=transition(cfg,d['post_state'],observation(.4,1.,20.,i=2))
        self.assertEqual(d['reason'],'reset_phase_and_adjust')

    def test_all_synthetic_policies_respect_bounds_and_replay(self):
        for policy in POLICIES:
            run=simulate(policy,'noisy',iterations=30,n=100)
            for d in run['trace']:
                self.assertTrue(0<=d['cpu_items']<=100)
                self.assertEqual(transition(Config(policy,100),d['pre_state'],d['observation']),d)

    def test_large_finite_timings_do_not_overflow_proportional_targets(self):
        for policy in ('paper_proportional','robust_proportional_v1'):
            d=transition(Config(policy,1000000),initial_state('test'),observation(.5,5e307,7e307))
            self.assertTrue(math.isfinite(d['requested_ratio']))
            self.assertGreater(d['requested_ratio'],.5)
        s=initial_state('test')
        d=transition(Config('robust_proportional_v1',1000000),s,observation(.000001,1e308,1e308))
        self.assertEqual(d['reason'],'hold_invalid_observation')
        self.assertEqual(d['post_state'],s)


if __name__=='__main__':unittest.main()
