"""Causal synthetic joint-control scenarios. No hardware timing or energy claims."""
from dataclasses import asdict
from .integration import Config,Coordinator,Lifecycle

PAIRS=((600,1000),(1200,1000),(1200,3000),(1800,6000))


def definitions():
    cases=[dict(name=name,allocation_policy=a,frequency_policy=f,steps=60)
            for a in ("paper_proportional","paper_binary_literal","bracketed_bisection")
            for f in ("paper_additive","discounted_loss_v1")
            for name in ("stationary","phase_change","missing_memory","actuation_failure","interruption")]
    cases.extend(dict(name="stationary",allocation_policy="paper_proportional",frequency_policy=f,steps=60,
                      cadence_profile="paper_ratio_40_v1") for f in ("paper_additive","discounted_loss_v1"))
    return cases


def simulate(case):
    name="-".join(case[k] for k in ("name","allocation_policy","frequency_policy"))+"-"+case.get("cadence_profile","observed_cadence_v1")
    cfg=Config(name,1000,500,PAIRS,"synthetic_joint",case["allocation_policy"],case["frequency_policy"],
               allocation_iterations=2,allocation_interval_ns=2_000_000_000,frequency_guard_ns=0,
               cadence_profile=case.get("cadence_profile","observed_cadence_v1"))
    engine=Coordinator(cfg);lifecycle=Lifecycle();lifecycle.move("warmup");lifecycle.move("execute")
    now=0;records=[];current=PAIRS[0];initial=current;failed=False;signs=[]
    try:
        for i in range(case["steps"]):
            if case["name"] in ("actuation_failure","interruption") and i==12:
                raise RuntimeError("injected_"+case["name"])
            phase=int(case["name"]=="phase_change" and i>=30)
            cpu_capacity=.6 if not phase else .2
            gpu_capacity=.8*(current[0]/1800)**.7*(current[1]/6000)**.3
            share=engine.cut/1000
            tc=share/cpu_capacity if share else None;tg=(1-share)/gpu_capacity if share<1 else None
            duration=max(tc or 0,tg or 0);start=now;now+=max(1,round(duration*1e9))
            core=min(1.,(1-share)*1800/current[0]);memory=min(1.,(1-share)*6000/current[1])
            if case["name"]=="missing_memory" and i%4==0:memory=None
            obs=dict(run_id=cfg.run_id,epoch=phase,iteration=i,cpu_items=engine.cut,t_cpu=tc,t_gpu=tg,
                     release_ns=start,valid=True,common_origin=True)
            inputs=dict(observation=obs,now_ns=now,final=i+1==case["steps"],phase=phase,synthetic_utilization=[core,memory])
            lifecycle.move("observe");decision=engine.barrier(**inputs)
            records.append(dict(input=inputs,decision=decision,model=dict(applied_pair=list(current),cpu_capacity=cpu_capacity,
                                                                        gpu_capacity=gpu_capacity,duration_units=duration)))
            if tc and tg:signs.append((tc>tg)-(tc<tg))
            if decision["synthetic_applied_pair"]:current=tuple(decision["synthetic_applied_pair"])
            lifecycle.move("finalize" if inputs["final"] else "execute")
    except RuntimeError:
        failed=True;lifecycle.move("failed")
    finally:
        lifecycle.move("restore");current=initial;lifecycle.move("stopped" if failed else "complete")
    return dict(case=case,configuration=asdict(cfg),records=records,lifecycle=lifecycle.events,
                status="expected_failure_restored" if failed else "passed",restored_synthetic_pair=list(current),
                hardware_setter_calls=0,energy_j=None,model="dimensionless_capacity_v1",
                sign_reversals=sum(x*y<0 for x,y in zip(signs,signs[1:])),
                frequency_updates=engine.fstate["updates"],allocation_updates=engine.astate["eligible"])
