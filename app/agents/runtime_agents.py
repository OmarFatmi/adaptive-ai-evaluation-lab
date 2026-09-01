from app.agents.base import Agent
from app.models.adapters import build_adapter
from app.evaluation.judges import RuleJudge,SemanticJudge,LocalLLMJudge,JudgeAggregator
from app.evaluation.failures import DiagnosticEngine
from app.domain.contracts import VerificationTask
class ModelExecutionAgent(Agent):
 name="model"; role="Execute target models"
 def __init__(self,endpoints,seed): super().__init__(); self.endpoints=endpoints; self.seed=seed
 async def run(self,state):
  state.adapters={x.id:build_adapter(x,self.seed+state.step*31+x.id) for x in self.endpoints}
  for mid,a in state.adapters.items(): state.model_outputs[mid]=await a.predict(state.test_case.prompt,state.test_case.reference,{**state.test_case.metadata,"category":state.test_case.category,"difficulty":state.test_case.difficulty})
  return state
class AsyncJudgeAgent(Agent):
 name="judge"; role="Run asynchronous judge panel"
 def __init__(self,judge_adapter=None): super().__init__(); self.panel=[RuleJudge(),SemanticJudge()]+([LocalLLMJudge(judge_adapter)] if judge_adapter else []); self.agg=JudgeAggregator()
 async def run(self,state):
  for mid,out in state.model_outputs.items():
   items=[await j.judge(state.test_case,out) for j in self.panel]; state.judgments[mid]=items; state.aggregated[mid]=self.agg.aggregate(items)
  return state
class DiagnosticAgent(Agent):
 name="critic"; role="Diagnose and hypothesize"
 def __init__(self): super().__init__(); self.engine=DiagnosticEngine()
 async def run(self,state):
  state.failures=[]; state.verification_tasks=[]
  for mid,agg in state.aggregated.items():
   if not agg['correct']:
    d=self.engine.diagnose(state.test_case,state.model_outputs[mid],agg); d.model_id=mid; state.failures.append(d); state.verification_tasks.append(VerificationTask(mid,d))
  return state
