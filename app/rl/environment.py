from dataclasses import dataclass,asdict
@dataclass
class EvaluationAction:
 category:str;difficulty:float;context_load:float=0.;distractor_density:float=0.;constraint_count:int=0;adversarial_strength:float=0.
@dataclass
class Transition:
 observation:list[float];action:EvaluationAction;reward:float;next_observation:list[float];done:bool;info:dict
class EvaluationEnvironment:
 """RL-compatible transition store. Model execution remains delegated to the agent runtime."""
 def __init__(self,categories,budget): self.categories=categories;self.budget=budget;self.step_count=0;self._observation=[];self.transitions=[]
 def reset(self,observation): self.step_count=0;self.transitions=[];self._observation=list(observation);return self.observation()
 def observation(self): return list(self._observation)
 def reward(self,components): return float(sum(components.values()))
 def step(self,action,reward,next_observation,info=None):
  self.step_count+=1;done=self.step_count>=self.budget;t=Transition(self.observation(),action,float(reward),list(next_observation),done,info or {});self.transitions.append(t);self._observation=list(next_observation);return self.observation(),float(reward),done,{**(info or {}),"transition":asdict(t)}
class TestSpaceSampler:
 def sample(self,category,difficulty,memory):
  weaknesses=dict(memory.get('weaknesses',[]));strength=min(1.,.2+difficulty+.03*sum(weaknesses.values()))
  return EvaluationAction(category,difficulty,difficulty if category=='long_context' else 0.,difficulty if category=='distractor' else 0.,2+int(6*difficulty) if category=='multi_constraint' else 0.,strength if category in {'contradiction','distractor','long_context'} else 0.)
