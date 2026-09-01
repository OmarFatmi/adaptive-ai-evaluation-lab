import asyncio,random
from app.policies.linucb import LinUCB,inverse,mat_vec
from app.research.beliefs import RegionBeliefStore
from app.evaluation.coverage import CoverageTracker
from app.rewards.engine import RewardEngine
from app.agents.verifier import VerifierAgent
from app.evaluation.generators import GeneratorRegistry

def test_full_matrix_linucb_has_off_diagonal_updates():
 p=LinUCB(["a"],dimensions=3);x={"vector":[1,.5,.25]};p.select(x);p.update("a",2,x);assert p.A["a"][0][1]==.5 and p.snapshot()["implementation"]=="full_matrix"
def test_matrix_inverse_identity():
 a=[[2.,0.],[0.,4.]];inv=inverse(a);assert mat_vec(inv,[2,4])==[1,1]
def test_region_beliefs_are_isolated():
 b=RegionBeliefStore();b.observe(1,"context.retention","hard",True);assert b.probability(1,"context.retention","hard")!=b.probability(1,"context.retention","easy")
def test_reward_difficulty_not_unconditional():
 e=RewardEngine();low,_=e.compute(failed=False,novelty=0,verified=False,information_gain=0,coverage=0,difficulty=0,cost=0,disagreement=0);high,_=e.compute(failed=False,novelty=0,verified=False,information_gain=0,coverage=0,difficulty=1,cost=0,disagreement=0);assert low==high==0
def test_multi_constraint_pair_changes_only_constraints():
 case=GeneratorRegistry().generate("multi_constraint",.9,random.Random(3));control,treatment,var=VerifierAgent().make_pair(case);assert var=="constraint_count" and control!=treatment and case.reference in case.metadata["ordered_names"]
