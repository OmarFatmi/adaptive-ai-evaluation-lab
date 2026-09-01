from abc import ABC,abstractmethod
import math,random
class Policy(ABC):
 def __init__(self,arms,seed=42): self.arms=arms; self.rng=random.Random(seed); self.counts={a:0 for a in arms}; self.values={a:0. for a in arms}
 @abstractmethod
 def select(self,context=None): ...
 def update(self,arm,reward,context=None): self.counts[arm]+=1; n=self.counts[arm]; self.values[arm]+=(reward-self.values[arm])/n
 def snapshot(self): return {"counts":self.counts.copy(),"values":self.values.copy()}
class EpsilonGreedy(Policy):
 def select(self,context=None): return self.rng.choice(self.arms) if self.rng.random()<.15 else max(self.arms,key=self.values.get)
class UCB1(Policy):
 def select(self,context=None):
  for a in self.arms:
   if not self.counts[a]: return a
  t=sum(self.counts.values()); return max(self.arms,key=lambda a:self.values[a]+math.sqrt(2*math.log(t)/self.counts[a]))
class Thompson(Policy):
 def __init__(self,arms,seed=42): super().__init__(arms,seed); self.a={x:1 for x in arms}; self.b={x:1 for x in arms}
 def select(self,context=None): return max(self.arms,key=lambda x:self.rng.betavariate(self.a[x],self.b[x]))
 def update(self,arm,reward,context=None): super().update(arm,reward,context); self.a[arm]+=reward>2; self.b[arm]+=reward<=2
class ContextualUCB(UCB1):
 def select(self,context=None):
  base=super().select(context); difficulty=(context or {}).get("difficulty",0); under=[a for a in self.arms if self.counts[a]<2]
  return self.rng.choice(under) if difficulty>.65 and under else base
def create_policy(name,arms,seed):
 if name=="linucb":
  from app.policies.linucb import LinUCB
  return LinUCB(arms,seed)
 return {"epsilon_greedy":EpsilonGreedy,"ucb1":UCB1,"thompson":Thompson,"contextual_ucb":ContextualUCB}.get(name,UCB1)(arms,seed)
