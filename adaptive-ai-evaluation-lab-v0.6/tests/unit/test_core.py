import asyncio
import random
from app.evaluation.generators import GeneratorRegistry
from app.evaluation.judges import RuleJudge, SemanticJudge, JudgeAggregator
from app.domain.contracts import ModelOutput, ExperimentState, FailureDiagnosis, VerificationTask
from app.policies.linucb import LinUCB
from app.rewards.engine import RewardEngine, BeliefInformationGain
from app.agents.verifier import VerifierAgent

def test_all_generators_and_judges():
    async def scenario():
        registry = GeneratorRegistry()
        for category in registry.items:
            case = registry.generate(category, .7, random.Random(42))
            assert case.test_space and case.test_space.region()
            out = ModelOutput(case.reference, 1)
            items = [await RuleJudge().judge(case, out), await SemanticJudge().judge(case, out)]
            assert JudgeAggregator().aggregate(items)["correct"]
    asyncio.run(scenario())

def test_reward_has_discrimination():
    total, parts = RewardEngine().compute(failed=True, novelty=1, verified=True, information_gain=.2, coverage=.5, difficulty=.8, cost=0, disagreement=0, discrimination=1)
    assert parts["model_discrimination"] == 2 and total > 8

def test_belief_ig():
    assert BeliefInformationGain().compute(5, 1, True) >= 0

def test_linucb_context():
    p = LinUCB(["a", "b"], 42)
    x = {"vector": [.8, .2, .4, .1, .2, 1]}
    a = p.select(x)
    p.update(a, 4, x)
    assert p.select(x) in p.arms

class Adapter:
    async def predict(self, prompt, reference, metadata):
        return ModelOutput(reference, 1)

def test_verifier_run_is_real_agent():
    async def scenario():
        case = GeneratorRegistry().generate("distractor", .8, random.Random(2))
        diagnosis = FailureDiagnosis("x", "high", .9, [], "h", "r", 1)
        state = ExperimentState(1, test_case=case, adapters={1: Adapter()}, model_outputs={1: ModelOutput("wrong", 1)}, verification_tasks=[VerificationTask(1, diagnosis)])
        await VerifierAgent().run(state)
        assert 1 in state.verifications
    asyncio.run(scenario())
