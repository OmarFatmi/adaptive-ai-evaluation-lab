from datetime import datetime
from app.domain import db_models as m
from app.domain.contracts import ExperimentState
from app.agents.core import GeneratorAgent,PolicyAgent
from app.agents.runtime_agents import ModelExecutionAgent,AsyncJudgeAgent,DiagnosticAgent
from app.agents.verifier import VerifierAgent
from app.agents.researcher import ResearchAgent
from app.agents.runtime import AgentRuntime
from app.orchestration.runner import ExperimentRunner
from app.models.adapters import build_adapter
from app.policies.core import create_policy
from app.rewards.engine import RewardEngine
from app.memory.service import AgentMemory
from app.orchestration.events import bus
class Orchestrator:
 def __init__(self,sf): self.sf=sf
 async def emit(self,db,eid,agent,kind,message,payload=None):
  db.add(m.AgentEvent(experiment_id=eid,agent=agent,event_type=kind,message=message,payload=payload or {})); db.commit(); await bus.publish(eid,{"agent":agent,"type":kind,"message":message,"payload":payload or {},"time":datetime.utcnow().isoformat()})
 async def run(self,eid):
  db=self.sf()
  try:
   exp=db.get(m.Experiment,eid); exp.status="running"; db.commit(); endpoints=[db.get(m.ModelEndpoint,i) for i in exp.model_ids]; endpoints=[x for x in endpoints if x]; memory=AgentMemory(db); reward=RewardEngine(); from app.evaluation.coverage import CoverageTracker
   coverage_tracker=CoverageTracker(); from app.research.beliefs import RegionBeliefStore
   beliefs=RegionBeliefStore(); policy=create_policy(exp.policy,exp.categories,exp.seed); judge_ep=db.get(m.ModelEndpoint,int(exp.judge_model)) if exp.judge_model and str(exp.judge_model).isdigit() else None; judge_adapter=build_adapter(judge_ep,exp.seed+9999) if judge_ep else None
   agents=[PolicyAgent(policy),GeneratorAgent(exp.seed),ModelExecutionAgent(endpoints,exp.seed),AsyncJudgeAgent(judge_adapter),DiagnosticAgent(),VerifierAgent()]; runtime=AgentRuntime(self.emit); runner=ExperimentRunner(runtime, agents); state=ExperimentState(eid); await self.emit(db,eid,"orchestrator","started","Experiment started")
   for step in range(exp.budget):
    db.refresh(exp)
    if exp.status in {"paused","cancelled"}: break
    state.step=step; state.difficulty=min(.95,.15+.8*step/max(1,exp.budget-1)); state.model_outputs={}; state.judgments={}; state.aggregated={}; state.failures=[]; state.memory=memory.snapshot(eid,exp.model_ids,exp.categories); remaining=1-step/max(1,exp.budget); rates=state.memory.get('domain_rates',{}); state.context=[state.difficulty,state.memory['failure_rate'],state.memory['uncertainty'],state.memory['coverage'],remaining,rates.get('reasoning',0),rates.get('context',0),rates.get('coding',0),rates.get('instruction',0),state.memory.get('judge_disagreement',0),max(-1,min(1,state.memory.get('recent_reward',0)/10)),state.memory.get('discrimination_rate',0),coverage_tracker.novelty('candidate')]
    state=await runner.execute_step(db,state)
    case=state.test_case; case.metadata['region']=case.test_space.region(); tc=m.TestCaseRecord(experiment_id=eid,external_id=case.id,category=case.category,difficulty=case.difficulty,prompt=case.prompt,reference=case.reference,generation_strategy=case.generation_strategy,adversarial=case.adversarial,metadata_json={**case.metadata,"test_space":case.test_space.__dict__}); db.add(tc); db.commit(); db.refresh(tc)
    failed_models=set(x.model_id for x in state.failures); discrimination=1. if len(endpoints)>1 and 0<len(failed_models)<len(endpoints) else 0.; step_rewards=[]
    for mid,out in state.model_outputs.items():
     mr=m.ModelOutputRecord(test_case_id=tc.id,model_id=mid,content=out.content,latency_ms=out.latency_ms,estimated_cost=out.estimated_cost); db.add(mr); db.commit(); db.refresh(mr)
     for j in state.judgments[mid]: db.add(m.JudgmentRecord(output_id=mr.id,judge_name=j.judge_name,score=j.score,confidence=j.confidence,reliability=j.reliability,reason=j.reason))
     agg=state.aggregated[mid]; diag=next((x for x in state.failures if x.model_id==mid),None); verified=False
     if diag:
      prior=db.query(m.FailureRecord).filter_by(model_id=mid,failure_type=diag.failure_type).count(); fr=m.FailureRecord(experiment_id=eid,test_case_id=tc.id,output_id=mr.id,model_id=mid,failure_type=diag.failure_type,severity=diag.severity,confidence=diag.confidence,diagnosis=' | '.join(diag.evidence),hypothesis=diag.hypothesis); db.add(fr); db.commit(); db.refresh(fr)
      for text in diag.evidence: db.add(m.FailureEvidence(failure_id=fr.id,evidence_type="observation",statement=text,supports=True,score=diag.confidence))
      vr=state.verifications.get(mid); verified=bool(vr and vr['supported']); fr.verification_status="supported" if verified else "not_supported"; db.add(m.VerificationRun(failure_id=fr.id,**vr))
     if agg['human_review']: db.add(m.HumanReview(output_id=mr.id))
     coverage=coverage_tracker.novelty(case.test_space.region()); coverage_tracker.visit(case.test_space.region()); total=db.query(m.ModelOutputRecord).filter_by(model_id=mid).count(); fails=db.query(m.FailureRecord).filter_by(model_id=mid,failure_type=diag.failure_type if diag else '').count(); ig=beliefs.information_gain(mid,diag.failure_type if diag else 'pass',case.test_space.region(),bool(diag)); beliefs.observe(mid,diag.failure_type if diag else 'pass',case.test_space.region(),bool(diag)); total_reward,parts=reward.compute(failed=bool(diag),novelty=1/(1+fails),verified=verified,information_gain=ig,coverage=coverage,difficulty=case.difficulty,cost=out.estimated_cost,disagreement=agg['disagreement'],discrimination=discrimination); db.add(m.RewardRecord(experiment_id=eid,test_case_id=tc.id,strategy=case.category,total=total_reward,components=parts)); step_rewards.append(total_reward)
    avg=sum(step_rewards)/max(1,len(step_rewards)); policy.update(state.selected_strategy,avg,{"vector":state.context}); db.add(m.PolicyUpdate(experiment_id=eid,step=step,selected_strategy=state.selected_strategy,reward=avg,state_json=policy.snapshot())); exp.completed_steps=step+1; db.commit(); await self.emit(db,eid,"reward","computed","Reward computed",{"reward":avg,"discrimination":discrimination})
   allf=db.query(m.FailureRecord).filter_by(experiment_id=eid).all(); allv=db.query(m.VerificationRun).join(m.FailureRecord).filter(m.FailureRecord.experiment_id==eid).all(); state.metadata['all_failures']=allf; state.metadata['all_verifications']=allv; state=await runtime.execute(db,state,ResearchAgent())
   for h in state.hypotheses: db.add(m.HypothesisRecord(experiment_id=eid,**h))
   if exp.status=="running": exp.status="completed"; exp.completed_at=datetime.utcnow()
   db.commit(); await self.emit(db,eid,"orchestrator","completed",exp.status)
  except Exception as exc:
   db.rollback(); exp=db.get(m.Experiment,eid)
   if exp: exp.status="failed"; db.commit(); await self.emit(db,eid,"orchestrator","failed",str(exc))
  finally: db.close()
