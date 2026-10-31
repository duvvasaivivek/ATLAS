import copy
from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from greengpu.frequency import Config, choose, initial_state, loss, losses, target, transition
from greengpu.frequency_experiment import run, replay, sha
from greengpu.frequency_scenarios import PAIRS, definitions, observation, simulate
from greengpu.frequency_trace import pair_samples


class FrequencyTests(unittest.TestCase):
    def test_f01_hand_fixture(self):
        cfg=Config(((600,1000),(1200,3000),(1600,5000)))
        self.assertAlmostEqual(loss(.4,.6,.15),.03)
        self.assertAlmostEqual(loss(.9,.5,.02),.392)
        self.assertAlmostEqual(losses(cfg,.4,.9)[1],.2834)
        d=transition(cfg,initial_state(cfg,"test"),observation("test",0,.4,.9))
        self.assertAlmostEqual(d["post_state"]["weights"][1],.77328)
        self.assertEqual(d["requested_pair"],None)
        self.assertEqual(d["acknowledged_pair"],None)

    def test_nonuniform_targets_and_sparse_candidates(self):
        self.assertEqual(target(900,[600,900,1800]),.25)
        cfg=Config(PAIRS)
        for case in definitions()[:8]:
            _,decisions=simulate(case)
            self.assertTrue(all(tuple(d["recommended_pair"]) in cfg.pairs for d in decisions))
        self.assertNotIn((600,6000),cfg.pairs)

    def test_configuration_validation(self):
        for pairs in (((0,1),),((True,1),),((1.,2),),((1,2,3),),((1,2),(1,2))):
            with self.assertRaises(ValueError): Config(pairs)
        for key,value in (("beta",0),("beta",1),("beta",float("nan")),("gamma",2),
                          ("alpha_memory",-1),("alpha_core",True),("discount",1),
                          ("policy","unknown"),("provenance","hardware_verified")):
            with self.assertRaises(ValueError): Config(PAIRS,**{key:value})
        self.assertEqual(Config(tuple(reversed(PAIRS))).identity,Config(PAIRS).identity)
        self.assertNotEqual(Config(PAIRS).identity,Config(PAIRS,beta=.3).identity)

    def test_loss_endpoints(self):
        self.assertEqual(loss(0,1,.15),.15)
        self.assertEqual(loss(1,0,.15),.85)
        self.assertEqual(loss(.2,.2,.15),0)
        for value in (-1,1.01,None,True,float("nan"),float("inf")):
            with self.assertRaises(ValueError): loss(value,.5,.2)

    def test_single_domains_and_empty(self):
        cfg=Config(((900,1000),(900,3000)))
        self.assertEqual(losses(cfg,0,.5),losses(cfg,1,.5))
        cfg=Config(((600,3000),(1800,3000)))
        self.assertEqual(losses(cfg,.5,0),losses(cfg,.5,1))
        cfg=Config(((900,3000),))
        d=transition(cfg,initial_state(cfg,"t"),observation("t",0,1,1))
        self.assertEqual(d["post_state"]["weights"],[1.])
        self.assertTrue(d["reduced_actuation"])
        bad=transition(cfg,d["post_state"],observation("t",1,.5,None))
        self.assertEqual(bad["post_state"],d["post_state"])
        cfg=Config(())
        d=transition(cfg,initial_state(cfg,"t"),observation("t",0,1,1))
        self.assertEqual(d["reason"],"unsupported_no_candidates")
        self.assertIsNone(d["recommended_pair"])

    def test_ties_and_initialization(self):
        cfg=Config(PAIRS)
        s=initial_state(cfg,"t")
        self.assertEqual(s["weights"],[0.]*len(PAIRS))
        self.assertEqual(s["selected"],0)
        self.assertEqual(choose([1,1,1],0,PAIRS[:3]),0)
        self.assertEqual(choose([0,1,1],0,PAIRS[:3]),2)
        self.assertEqual(choose([1,1+1e-12,1],0,PAIRS[:3]),1)

    def test_invalid_observations_and_purity(self):
        cfg=Config(PAIRS)
        state=initial_state(cfg,"t"); obs=observation("t",0,.4,.9)
        original=copy.deepcopy((state,obs))
        good=transition(cfg,state,obs)
        self.assertEqual((state,obs),original)
        for key,value in (("run_id","other"),("memory",None),("core",True),("core",float("nan")),
                          ("memory",float("inf")),("memory",1.1),("valid",False),
                          ("sample_id",True),("epoch",-1),("source_ref",""),
                          ("semantics","allocated_memory_fraction"),("core_window",None)):
            bad=dict(obs);bad[key]=value
            d=transition(cfg,state,bad)
            self.assertEqual(d["reason"],"hold_invalid_observation",key)
            self.assertEqual(d["post_state"],state)
            json.dumps(d,allow_nan=False)
        good["post_state"]["weights"][0]=1000
        self.assertEqual(state,original[0])

    def test_freshness_phase_history_and_config_isolation(self):
        cfg=Config(PAIRS); s=initial_state(cfg,"t")
        d=transition(cfg,s,observation("t",0,.9,.9)); s=d["post_state"]
        for obs in (observation("t",0,.9,.9),observation("t",0,.1,.1,1)):
            self.assertEqual(transition(cfg,s,obs)["post_state"],s)
        obs=observation("t",1,.1,.1,1)
        d=transition(cfg,s,obs)
        expected=[w+1-.8*v for w,v in zip(s["weights"],losses(cfg,.1,.1))]
        for a,b in zip(d["post_state"]["weights"],expected): self.assertAlmostEqual(a,b)
        self.assertEqual(d["post_state"]["updates"],2)
        self.assertEqual(transition(cfg,d["post_state"],observation("t",2,.1,.1,0))["post_state"],d["post_state"])
        with self.assertRaises(ValueError): transition(replace(cfg,beta=.3),s,obs)
        self.assertEqual(initial_state(cfg,"new")["updates"],0)

    def test_incompatible_and_overlapping_windows(self):
        cfg=Config(PAIRS);s=initial_state(cfg,"t")
        obs=observation("t",0,.5,.5)
        obs["memory_window"]["end"]=2
        self.assertFalse(transition(cfg,s,obs)["input_valid"])
        s=transition(cfg,s,observation("t",0,.5,.5))["post_state"]
        obs=observation("t",1,.5,.5)
        obs["core_window"]["start"]=obs["memory_window"]["start"]=0
        self.assertEqual(transition(cfg,s,obs)["post_state"],s)

    def test_independent_decimal_oracle_and_beta_invariance(self):
        # Independent absolute-loss expression, no production loss/target calls.
        rng=random.Random(73)
        inputs=[(Decimal(rng.randrange(1001))/1000,Decimal(rng.randrange(1001))/1000) for _ in range(600)]
        cumulative=[Decimal(0) for _ in PAIRS]
        states=[initial_state(Config(PAIRS,beta=b),"t") for b in (.01,.2,.99)]
        selections=[[],[],[]]
        for i,(u,v) in enumerate(inputs):
            for j,(c,m) in enumerate(PAIRS):
                tc=Decimal(c-600)/1200; tm=Decimal(m-1000)/5000
                lc=(Decimal('.15') if tc>=u else Decimal('.85'))*abs(tc-u)
                lm=(Decimal('.02') if tm>=v else Decimal('.98'))*abs(tm-v)
                cumulative[j]+=Decimal('.3')*lc+Decimal('.7')*lm
            for k,beta in enumerate((.01,.2,.99)):
                cfg=Config(PAIRS,beta=beta)
                states[k]=transition(cfg,states[k],observation("t",i,float(u),float(v)))["post_state"]
                oracle=[Decimal(i+1)-(1-Decimal(str(beta)))*c for c in cumulative]
                for actual,expected in zip(states[k]["weights"],oracle):
                    self.assertAlmostEqual(actual,float(expected),delta=2e-10)
                selections[k].append(states[k]["selected"])
        self.assertEqual(selections[0],selections[1]);self.assertEqual(selections[1],selections[2])

    def test_long_run_unclipped_scores(self):
        cfg=Config(((600,1000),(1200,3000),(1600,5000)))
        s=initial_state(cfg,"t")
        for i in range(10000): s=transition(cfg,s,observation("t",i,.4,.9))["post_state"]
        self.assertAlmostEqual(s["weights"][1],7732.8,delta=2e-8)
        self.assertTrue(all(w>1 for w in s["weights"]))

    def test_discount_extension_and_literal_history(self):
        cfg=Config(PAIRS,policy="discounted_loss_v1",discount=.9)
        s=initial_state(cfg,"t")
        for i in range(2): s=transition(cfg,s,observation("t",i,.4,.9))["post_state"]
        for w,v in zip(s["weights"],losses(cfg,.4,.9)): self.assertAlmostEqual(w,-1.9*v)
        literal=next(c for c in definitions() if c["run_id"]=="paper_additive-phase_change")
        extension=next(c for c in definitions() if c["run_id"]=="discounted_loss_v1-phase_change")
        _,a=simulate(literal);_,b=simulate(extension)
        self.assertNotEqual(a[-1]["recommended_pair"],b[-1]["recommended_pair"])
        self.assertEqual(a[-1]["post_state"]["updates"],160)

    def test_closed_loop_uses_previous_recommendation(self):
        case=next(c for c in definitions() if c["run_id"]=="paper_additive-closed_loop")
        obs,decisions=simulate(case)
        for i in range(1,len(obs)):
            self.assertEqual(obs[i]["synthetic_model"]["applied_synthetic_pair"],decisions[i-1]["recommended_pair"])

    def test_trace_pairing_no_fabricated_memory_or_freshness(self):
        raw=dict(start_us=10,end_us=30,
                 core_samples=[dict(timestamp_us=t,percent=50) for t in (5,10,20,20,30,40)],
                 memory_samples=[dict(timestamp_us=t,percent=20) for t in (10,20,30)])
        obs=pair_samples(raw)
        self.assertEqual([o["core_window"]["end"] for o in obs],[10,20,30])
        self.assertEqual(obs[0]["memory"],.2)
        raw["memory_samples"].append(dict(timestamp_us=20,percent=40))
        self.assertEqual(len(pair_samples(raw)),2)
        raw["memory_samples"]=[]
        self.assertEqual(pair_samples(raw),[])


class FrequencyArchiveTests(unittest.TestCase):
    def test_hardware_replay_and_unavailable_memory(self):
        for memory_available in (True,False):
            raw=dict(start_us=10,end_us=30,core_samples=[dict(timestamp_us=20,percent=40)],
                     memory_samples=[dict(timestamp_us=20,percent=70)] if memory_available else [],
                     reported_pairs=[[600,1000],[1800,6000]],selected_pairs=[[600,1000],[1800,6000]],
                     eligibility="exact_matched_driver_timestamps_offline_assumption",workload="test_fixture")
            with tempfile.TemporaryDirectory() as temporary, patch("greengpu.frequency_experiment.capture",return_value=raw):
                result=run(Path(temporary)/"archive",hardware=True)
                self.assertEqual(result["hardware"]["updates"],int(memory_available))
                self.assertFalse(result["hardware"]["actuation_verified"])
                if not memory_available:
                    self.assertEqual(result["hardware"]["status"],"blocked_no_eligible_observations_or_candidates")

    def test_replay_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/"evidence"
            result=run(root)
            self.assertEqual(result,replay(root))
            self.assertEqual(len(result["synthetic_cases"]),23)
            self.assertEqual(result["hardware"]["status"],"not_requested")
            with self.assertRaises(FileExistsError): run(root)
            extra=root/"unexpected.txt";extra.write_text("extra")
            with self.assertRaisesRegex(ValueError,"inventory"): replay(root)
            extra.unlink()
            path=root/"paper_additive-idle.decisions.jsonl"
            content=path.read_text();path.write_text(content.replace('"recommended_pair": [600, 1000]','"recommended_pair": [1800, 6000]',1))
            with self.assertRaisesRegex(ValueError,"checksum"): replay(root)
            hashes=json.loads((root/"checksums.json").read_text());hashes[path.name]=sha(path)
            (root/"checksums.json").write_text(json.dumps(hashes))
            with self.assertRaisesRegex(ValueError,"decision mismatch"): replay(root)


if __name__ == "__main__": unittest.main()
