import asyncio
class EventBus:
 def __init__(self): self.listeners={}
 def subscribe(self,eid): q=asyncio.Queue(); self.listeners.setdefault(eid,set()).add(q); return q
 def unsubscribe(self,eid,q): self.listeners.get(eid,set()).discard(q)
 async def publish(self,eid,item):
  for q in list(self.listeners.get(eid,set())): await q.put(item)
bus=EventBus()
