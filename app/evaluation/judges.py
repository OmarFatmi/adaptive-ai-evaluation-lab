from abc import ABC,abstractmethod
import json,re,statistics
from app.domain.contracts import Judgment
class BaseJudge(ABC):
 name="base"; reliability=.5
 @abstractmethod
 async def judge(self,case,output)->Judgment: ...
class RuleJudge(BaseJudge):
 name="rule"; reliability=.98
 async def judge(self,case,output):
  a,r=output.content.strip().casefold(),case.reference.strip().casefold(); ok=a==r
  if re.fullmatch(r"-?\d+(?:\.\d+)?",r):
   nums=re.findall(r"-?\d+(?:\.\d+)?",a); ok=bool(nums) and abs(float(nums[-1])-float(r))<1e-9
  return Judgment(self.name,float(ok),.99,self.reliability,"Deterministic normalized comparison")
class SemanticJudge(BaseJudge):
 name="semantic"; reliability=.64
 async def judge(self,case,output):
  norm=lambda s:set(re.findall(r"[a-z0-9]+",s.casefold())); a,b=norm(output.content),norm(case.reference); p=len(a&b)/max(1,len(a)); q=len(a&b)/max(1,len(b)); f=2*p*q/max(1e-9,p+q); contradiction=bool((b-a) and (a-b)); return Judgment(self.name,max(0,f-.35*contradiction),.68,self.reliability,"Token F1 with contradiction penalty")
class LocalLLMJudge(BaseJudge):
 name="local_llm"; reliability=.72
 def __init__(self,adapter): self.adapter=adapter
 async def judge(self,case,output):
  prompt=("Act as an evaluation judge. Return JSON only with score (0..1), confidence (0..1), and reason.\n"+f"QUESTION: {case.prompt}\nREFERENCE: {case.reference}\nANSWER: {output.content}")
  raw=await self.adapter.predict(prompt,"",{"category":"judge","difficulty":0})
  try:
   text=raw.content; obj=json.loads(text[text.index('{'):text.rindex('}')+1]); return Judgment(self.name,float(obj['score']),float(obj.get('confidence',.7)),self.reliability,str(obj.get('reason','Local LLM judgment')))
  except Exception: return Judgment(self.name,0.,.15,.2,"Invalid JSON from local LLM judge")
class JudgeAggregator:
 def aggregate(self,items):
  weights=[max(.01,x.confidence*x.reliability) for x in items]; score=sum(x.score*w for x,w in zip(items,weights))/sum(weights); disagreement=statistics.pstdev(x.score for x in items) if len(items)>1 else 0.; return {"score":score,"correct":score>=.7,"confidence":max(0.,min(1.,1-disagreement)),"disagreement":disagreement,"human_review":disagreement>.3 or max(x.confidence for x in items)<.55}
