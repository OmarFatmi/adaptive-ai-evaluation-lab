import asyncio
from app.agents.verifier import VerifierAgent
from app.research.beliefs import RegionBeliefStore
from app.research.benchmark import PolicyBenchmarkRunner
from app.domain.contracts import ModelOutput
class Model: id=1;adapter="mock";model_name="mock-balanced";base_url=None
def test_verification_outcomes():
 assert VerifierAgent.classify(True,False)=="EFFECT_SUPPORTED"
 assert VerifierAgent.classify(True,True)=="NO_EFFECT"
 assert VerifierAgent.classify(False,False)=="BASELINE_FAILURE"
 assert VerifierAgent.classify(False,True)=="REVERSE_EFFECT"
 assert VerifierAgent.classify(True,False,False)=="INCONCLUSIVE"
def test_region_information_gain_changes_region_only():
 b=RegionBeliefStore();before=b.probability(1,"x","a");ig=b.information_gain(1,"x","a",True);b.observe(1,"x","a",True);assert ig>=0 and b.probability(1,"x","a")!=before and b.probability(1,"x","b")==before
def test_benchmark_uses_full_testspace_and_curves():
 async def run():
  result=await PolicyBenchmarkRunner(Model(),["arithmetic","distractor"]).run_once("linucb",10,42);assert len(result["curves"])==10 and "mean_information_gain" in result and result["unique_regions"]>0
 asyncio.run(run())
