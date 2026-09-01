from app.agents.base import Agent
from app.research.statistics import bootstrap_rate
class ResearchAgent(Agent):
 name="research";role="Mine patterns and evaluate evidence"
 async def run(self,state):
  grouped={}
  for failure in state.metadata.get('all_failures',[]):grouped.setdefault(failure.failure_type,[]).append(failure)
  verifications=state.metadata.get('all_verifications',[]);state.hypotheses=[]
  for kind,items in grouped.items():
   ids={x.id for x in items};relevant=[v for v in verifications if v.failure_id in ids and v.outcome in {'EFFECT_SUPPORTED','NO_EFFECT'}];observations=[int(v.outcome=='EFFECT_SUPPORTED') for v in relevant];stats=bootstrap_rate(observations,seed=state.experiment_id,samples=2000)
   status='proposed'
   if stats['n']>0:status='testing'
   if stats['n']>=3 and stats['ci_low']>.5:status='supported'
   if stats['n']>=3 and stats['ci_high']<.5:status='rejected'
   state.hypotheses.append({"statement":items[0].hypothesis,"variable":kind,"confidence":stats['mean'],"support_rate":stats['mean'],"ci_low":stats['ci_low'],"ci_high":stats['ci_high'],"status":status,"supporting_count":sum(observations),"contradicting_count":len(observations)-sum(observations)})
  return state
