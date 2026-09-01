from abc import ABC,abstractmethod
import asyncio,random,time,httpx
from app.domain.contracts import ModelOutput
from app.config import settings
class ModelAdapter(ABC):
 @abstractmethod
 async def predict(self,prompt,reference,metadata)->ModelOutput: ...
class MockAdapter(ModelAdapter):
 def __init__(self,profile,seed): self.profile=profile; self.rng=random.Random(seed)
 async def predict(self,prompt,reference,metadata):
  start=time.perf_counter(); rates={"mock-strong":.88,"mock-balanced":.70,"mock-fragile":.48}; penalties={"long_context":.18,"distractor":.12,"contradiction":.10,"multi_constraint":.08,"coding":.06}; p=rates.get(self.profile,.7)-penalties.get(metadata.get("category"),0)*metadata.get("difficulty",.5); await asyncio.sleep(.004)
  answer=reference if self.rng.random()<p else (str(int(reference)+1) if reference.lstrip('-').isdigit() else "UNKNOWN")
  return ModelOutput(answer,(time.perf_counter()-start)*1000,0)
class OllamaAdapter(ModelAdapter):
 def __init__(self,name,base_url=None): self.name=name; self.base=(base_url or settings.ollama_base_url).rstrip('/')
 async def predict(self,prompt,reference,metadata):
  start=time.perf_counter()
  async with httpx.AsyncClient(timeout=180) as c:
   r=await c.post(f"{self.base}/api/chat",json={"model":self.name,"stream":False,"messages":[{"role":"system","content":"Follow the task and answer concisely."},{"role":"user","content":prompt}],"options":{"temperature":0}}); r.raise_for_status(); text=r.json()["message"]["content"].strip()
  return ModelOutput(text,(time.perf_counter()-start)*1000,0)
def build_adapter(endpoint,seed): return OllamaAdapter(endpoint.model_name,endpoint.base_url) if endpoint.adapter=="ollama" else MockAdapter(endpoint.model_name,seed)
