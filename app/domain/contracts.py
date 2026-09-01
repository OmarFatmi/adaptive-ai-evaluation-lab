from dataclasses import dataclass,field
from typing import Any
from uuid import uuid4
@dataclass
class TestSpace:
 category:str; difficulty:float=.2; context_load:float=0.; distractor_density:float=0.; constraint_count:int=0; adversarial_strength:float=0.; adversarial:bool=False
 def vector(self): return [self.difficulty,self.context_load,self.distractor_density,min(1,self.constraint_count/10),self.adversarial_strength]
 def region(self): return f"{self.category}:{int(self.difficulty*4)}:{int(self.context_load*3)}:{int(self.distractor_density*3)}:{self.constraint_count//2}:{int(self.adversarial_strength*3)}"
@dataclass
class TestCase:
 category:str; difficulty:float; prompt:str; reference:str; metadata:dict[str,Any]; generation_strategy:str; adversarial:bool=False; expected_failure_modes:list[str]=field(default_factory=list); id:str=field(default_factory=lambda:f"test_{uuid4().hex[:12]}"); test_space:TestSpace|None=None
@dataclass
class ModelOutput: content:str; latency_ms:float; estimated_cost:float=0.
@dataclass
class Judgment: judge_name:str; score:float; confidence:float; reliability:float; reason:str
@dataclass
class FailureDiagnosis: failure_type:str; severity:str; confidence:float; evidence:list[str]; hypothesis:str; recommended_test:str; model_id:int|None=None
@dataclass
class VerificationTask: model_id:int; failure:FailureDiagnosis
@dataclass
class ExperimentState:
 experiment_id:int; step:int=0; selected_strategy:str|None=None; difficulty:float=.2; context:list[float]=field(default_factory=list); test_case:TestCase|None=None; model_outputs:dict[int,ModelOutput]=field(default_factory=dict); adapters:dict[int,Any]=field(default_factory=dict); judgments:dict[int,list[Judgment]]=field(default_factory=dict); aggregated:dict[int,dict]=field(default_factory=dict); failures:list[FailureDiagnosis]=field(default_factory=list); verification_tasks:list[VerificationTask]=field(default_factory=list); verifications:dict[int,dict]=field(default_factory=dict); rewards:dict[int,float]=field(default_factory=dict); policy_state:dict=field(default_factory=dict); memory:dict=field(default_factory=dict); hypotheses:list[dict]=field(default_factory=list); metadata:dict=field(default_factory=dict)
