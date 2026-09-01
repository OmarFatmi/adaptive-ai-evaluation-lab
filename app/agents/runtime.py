from datetime import datetime
class AgentRuntime:
 def __init__(self,emit): self.emit=emit; self.status={}
 async def execute(self,db,state,agent):
  self.status[agent.name]="running"; await self.emit(db,state.experiment_id,agent.name,"status","RUNNING")
  try:
   state=await agent.run(state); self.status[agent.name]="success"; await self.emit(db,state.experiment_id,agent.name,"status","SUCCESS"); return state
  except Exception:
   self.status[agent.name]="failed"; await self.emit(db,state.experiment_id,agent.name,"status","FAILED"); raise
