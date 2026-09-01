import asyncio
from fastapi import APIRouter,Depends,HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db import get_db,SessionLocal
from app.domain import db_models as m
from app.api.schemas import *
from app.orchestration.orchestrator import Orchestrator
from app.arena.engine import ArenaEngine
from app.models.adapters import build_adapter
router=APIRouter(prefix="/api")
@router.get("/health")
def health(): return {"status":"ok","version":"0.6.0","database":"sqlite"}
@router.get("/models")
def models(db:Session=Depends(get_db)): return db.query(m.ModelEndpoint).order_by(m.ModelEndpoint.id).all()
@router.post("/models")
def add_model(data:ModelCreate,db:Session=Depends(get_db)):
 if data.adapter not in {"mock","ollama"}: raise HTTPException(400,"Only mock and local Ollama adapters are enabled")
 row=m.ModelEndpoint(**data.model_dump()); db.add(row); db.commit(); db.refresh(row); return row
@router.get("/experiments")
def experiments(db:Session=Depends(get_db)): return db.query(m.Experiment).order_by(m.Experiment.id.desc()).all()
@router.post("/experiments")
def create_experiment(data:ExperimentCreate,db:Session=Depends(get_db)):
 if set(data.categories)-set(CATEGORIES): raise HTTPException(400,"Unsupported category")
 if data.policy not in {"epsilon_greedy","ucb1","thompson","contextual_ucb","linucb"}: raise HTTPException(400,"Unsupported policy")
 row=m.Experiment(**data.model_dump()); db.add(row); db.commit(); db.refresh(row); return row
@router.get("/experiments/{eid}")
def experiment(eid:int,db:Session=Depends(get_db)):
 row=db.get(m.Experiment,eid)
 if not row: raise HTTPException(404,"Not found")
 return row
@router.post("/experiments/{eid}/start")
async def start(eid:int,db:Session=Depends(get_db)):
 if not db.get(m.Experiment,eid): raise HTTPException(404,"Not found")
 asyncio.create_task(Orchestrator(SessionLocal).run(eid)); return {"status":"scheduled"}
@router.post("/experiments/{eid}/{action}")
def control(eid:int,action:str,db:Session=Depends(get_db)):
 if action not in {"pause","cancel"}: raise HTTPException(400,"Bad action")
 row=db.get(m.Experiment,eid); row.status="paused" if action=="pause" else "cancelled"; db.commit(); return row
@router.get("/experiments/{eid}/results")
def results(eid:int,db:Session=Depends(get_db)):
 cases=db.query(m.TestCaseRecord).filter_by(experiment_id=eid).all(); failures=db.query(m.FailureRecord).filter_by(experiment_id=eid).all(); rewards=db.query(m.RewardRecord).filter_by(experiment_id=eid).all(); outputs=db.query(m.ModelOutputRecord).join(m.TestCaseRecord).filter(m.TestCaseRecord.experiment_id==eid).all(); correct=max(0,len(outputs)-len(failures)); bycat={}
 for c in cases:
  bycat.setdefault(c.category,{"tests":0,"failures":0}); bycat[c.category]["tests"]+=1
 for f in failures:
  c=db.get(m.TestCaseRecord,f.test_case_id); bycat[c.category]["failures"]+=1
 for v in bycat.values(): v["accuracy"]=1-v["failures"]/max(1,v["tests"]*max(1,len(set(x.model_id for x in outputs))))
 return {"tests":len(cases),"outputs":len(outputs),"accuracy":correct/max(1,len(outputs)),"failures":len(failures),"mean_reward":sum(x.total for x in rewards)/max(1,len(rewards)),"by_category":bycat,"reward_curve":[x.total for x in rewards[-200:]]}
@router.get("/experiments/{eid}/events")
def events(eid:int,db:Session=Depends(get_db)): return db.query(m.AgentEvent).filter_by(experiment_id=eid).order_by(m.AgentEvent.id.desc()).limit(300).all()
@router.get("/agents")
def agents(): return [{"name":x,"role":y} for x,y in [("policy","Adaptive strategy selection"),("generator","Typed test generation"),("model_agent","Local model execution"),("judge","Panel evaluation"),("critic","Failure diagnosis"),("verifier","Controlled experiments"),("research","Pattern mining")]]
@router.get("/failures")
def failures(experiment_id:int|None=None,db:Session=Depends(get_db)):
 q=db.query(m.FailureRecord)
 if experiment_id:q=q.filter_by(experiment_id=experiment_id)
 return q.order_by(m.FailureRecord.id.desc()).limit(500).all()
@router.get("/failures/{fid}")
def failure(fid:int,db:Session=Depends(get_db)):
 f=db.get(m.FailureRecord,fid)
 if not f: raise HTTPException(404,"Not found")
 return {"failure":f,"evidence":db.query(m.FailureEvidence).filter_by(failure_id=fid).all(),"verification":db.query(m.VerificationRun).filter_by(failure_id=fid).all()}
@router.get("/research/hypotheses")
def hypotheses(db:Session=Depends(get_db)): return db.query(m.HypothesisRecord).order_by(m.HypothesisRecord.id.desc()).all()
@router.get("/reviews")
def reviews(db:Session=Depends(get_db)): return db.query(m.HumanReview).order_by(m.HumanReview.id.desc()).all()
@router.post("/reviews/{rid}")
def review(rid:int,data:ReviewUpdate,db:Session=Depends(get_db)):
 row=db.get(m.HumanReview,rid)
 if not row: raise HTTPException(404,"Not found")
 row.verdict=data.verdict; row.notes=data.notes; row.status="completed"; db.commit(); return row
@router.get("/arena/leaderboard")
def leaderboard(db:Session=Depends(get_db)): return db.query(m.ModelEndpoint).order_by(m.ModelEndpoint.elo.desc()).all()
@router.get("/arena/matches")
def matches(db:Session=Depends(get_db)): return db.query(m.ArenaMatch).order_by(m.ArenaMatch.id.desc()).all()
@router.post("/arena/matches")
async def match(data:MatchCreate,db:Session=Depends(get_db)):
 a,b=db.get(m.ModelEndpoint,data.model_a_id),db.get(m.ModelEndpoint,data.model_b_id)
 if not a or not b or a.id==b.id: raise HTTPException(400,"Select two distinct models")
 judge_ep=db.get(m.ModelEndpoint,data.judge_model_id) if data.judge_model_id else None
 return await ArenaEngine(db,build_adapter(judge_ep,data.seed+9000) if judge_ep else None).match(a,b,data.tests,data.seed,data.mode)



@router.get("/rl/environment")
def rl_environment_spec():
    return {
        "version": "0.6.0",
        "observation_dimensions": 13,
        "action_space": {
            "category": "Discrete(6)",
            "parameters": "Box(0, 1, shape=(5,))",
        },
        "ppo_baseline": "optional discrete Level-2 wrapper",
        "training_command": "python -m scripts.train_ppo",
    }

@router.post("/benchmarks")
async def create_benchmark(data:BenchmarkCreate,db:Session=Depends(get_db)):
 model=db.get(m.ModelEndpoint,data.model_id)
 if not model: raise HTTPException(404,"Model not found")
 allowed={"epsilon_greedy","ucb1","thompson","contextual_ucb","linucb"}
 if set(data.policies)-allowed: raise HTTPException(400,"Unsupported benchmark policy")
 bench=m.PolicyBenchmark(name=data.name,model_id=data.model_id,budget=data.budget,seeds=data.seeds,policies=data.policies,status="running");db.add(bench);db.commit();db.refresh(bench)
 from app.research.benchmark import PolicyBenchmarkRunner
 runner=PolicyBenchmarkRunner(model,CATEGORIES)
 for policy in data.policies:
  for seed in data.seeds:
   result=await runner.run_once(policy,data.budget,seed);db.add(m.BenchmarkResult(benchmark_id=bench.id,policy=policy,seed=seed,**result))
 bench.status="completed";db.commit();return {"benchmark_id":bench.id,"status":bench.status}
@router.get("/benchmarks")
def benchmarks(db:Session=Depends(get_db)): return db.query(m.PolicyBenchmark).order_by(m.PolicyBenchmark.id.desc()).all()
@router.get("/benchmarks/{bid}/results")
def benchmark_results(bid:int,db:Session=Depends(get_db)):
 import statistics
 rows=db.query(m.BenchmarkResult).filter_by(benchmark_id=bid).all();groups={}
 for row in rows:
  groups.setdefault(row.policy,[]).append(row)
 
 def stats(values):
  import statistics,math
  mean=statistics.mean(values);std=statistics.stdev(values) if len(values)>1 else 0.;half=1.96*std/math.sqrt(max(1,len(values)));return {"mean":mean,"std":std,"ci_low":mean-half,"ci_high":mean+half}
 baseline=groups.get("epsilon_greedy",[])
 base=statistics.mean([x.mean_reward for x in baseline]) if baseline else 0
 summary=[]
 for policy,values in groups.items():
  reward_stats=stats([x.mean_reward for x in values]);failure_stats=stats([x.failures for x in values]);coverage_stats=stats([x.coverage for x in values]);ig_stats=stats([x.mean_information_gain for x in values]);pooled=statistics.stdev([x.mean_reward for x in values]+[x.mean_reward for x in baseline]) if len(values)+len(baseline)>2 else 0
  summary.append({"policy":policy,"runs":len(values),"reward":reward_stats,"failures":failure_stats,"coverage":coverage_stats,"information_gain":ig_stats,"effect_size_vs_epsilon":(reward_stats["mean"]-base)/pooled if pooled else 0})
 return {"runs":rows,"summary":summary}
@router.get("/test-space")
def test_space(experiment_id:int|None=None,db:Session=Depends(get_db)):
 q=db.query(m.TestCaseRecord)
 if experiment_id:q=q.filter_by(experiment_id=experiment_id)
 cases=q.order_by(m.TestCaseRecord.id.desc()).limit(2000).all(); failure_ids={x.test_case_id for x in db.query(m.FailureRecord).all()}
 return [{"id":x.id,"category":x.category,"difficulty":x.difficulty,"context_load":x.metadata_json.get("test_space",{}).get("context_load",0),"distractor_density":x.metadata_json.get("test_space",{}).get("distractor_density",0),"region":x.metadata_json.get("region"),"failed":x.id in failure_ids} for x in cases]
@router.get("/agents/status/{eid}")
def agent_status(eid:int,db:Session=Depends(get_db)):
 rows=db.query(m.AgentEvent).filter_by(experiment_id=eid,event_type="status").order_by(m.AgentEvent.id).all(); status={}
 for row in rows: status[row.agent]={"status":row.message.lower(),"time":row.created_at,"message":row.message}
 return status
@router.get("/dashboard/analytics")
def dashboard_analytics(db:Session=Depends(get_db)):
 failures=db.query(m.FailureRecord).all(); rewards=db.query(m.RewardRecord).order_by(m.RewardRecord.id).all(); policies=db.query(m.PolicyUpdate).order_by(m.PolicyUpdate.id.desc()).limit(100).all(); dist={}
 for f in failures: dist[f.failure_type]=dist.get(f.failure_type,0)+1
 return {"failure_distribution":dist,"reward_curve":[{"id":x.id,"reward":x.total} for x in rewards[-300:]],"policy_evolution":[{"step":x.step,"strategy":x.selected_strategy,"reward":x.reward} for x in reversed(policies)]}
@router.get("/dashboard")
def dashboard(db:Session=Depends(get_db)):
 return {"models":db.query(m.ModelEndpoint).count(),"experiments":db.query(m.Experiment).count(),"tests":db.query(m.TestCaseRecord).count(),"failures":db.query(m.FailureRecord).count(),"hypotheses":db.query(m.HypothesisRecord).count(),"reviews":db.query(m.HumanReview).filter_by(status="pending").count()}
