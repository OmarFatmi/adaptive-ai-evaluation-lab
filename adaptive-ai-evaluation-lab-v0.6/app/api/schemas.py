from pydantic import BaseModel,Field
CATEGORIES=["arithmetic","multi_constraint","contradiction","distractor","long_context","coding"]
class ModelCreate(BaseModel): name:str; adapter:str="mock"; model_name:str="mock-balanced"; base_url:str|None=None; config:dict={}
class ExperimentCreate(BaseModel): name:str; model_ids:list[int]=Field(min_length=1); categories:list[str]=Field(default_factory=lambda:CATEGORIES.copy()); budget:int=Field(30,ge=1,le=5000); policy:str="ucb1"; seed:int=42; judge_model:str|None=None
class ReviewUpdate(BaseModel): verdict:str; notes:str|None=None
class MatchCreate(BaseModel): model_a_id:int; model_b_id:int; tests:int=Field(20,ge=2,le=1000); seed:int=42; mode:str="random"; judge_model_id:int|None=None

class BenchmarkCreate(BaseModel):
 name:str="Policy comparison"; model_id:int; budget:int=Field(100,ge=10,le=5000); seeds:list[int]=[11,22,33]; policies:list[str]=["epsilon_greedy","ucb1","thompson","contextual_ucb","linucb"]
