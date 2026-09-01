from abc import ABC,abstractmethod
from app.domain.contracts import ExperimentState
class Agent(ABC):
 name="agent"; role=""
 def __init__(self,tools=None,memory=None): self.tools=tools or []; self.memory=memory
 @abstractmethod
 async def run(self,state:ExperimentState)->ExperimentState: ...
