import random,statistics,math
from app.policies.core import create_policy
from app.evaluation.generators import GeneratorRegistry
from app.evaluation.judges import RuleJudge,SemanticJudge,JudgeAggregator
from app.models.adapters import build_adapter
from app.evaluation.coverage import CoverageTracker
from app.rewards.engine import RewardEngine
from app.research.beliefs import RegionBeliefStore
from app.rl.environment import TestSpaceSampler

class ParameterPolicy:
    """Selects category through a policy and parameters through uncertainty-aware exploration."""
    def __init__(self,policy,seed): self.policy=policy;self.rng=random.Random(seed)
    def select(self,context,memory):
        category=self.policy.select({"vector":context,"difficulty":context[0]})
        uncertainty=context[2]; novelty=context[-1]
        difficulty=max(.05,min(.98,.2+.55*uncertainty+.25*novelty+self.rng.uniform(-.1,.1)))
        return TestSpaceSampler().sample(category,difficulty,memory)
    def update(self,category,reward,context): self.policy.update(category,reward,{"vector":context})

class PolicyBenchmarkRunner:
    def __init__(self,model,categories): self.model=model;self.categories=categories
    async def run_once(self,policy_name,budget,seed):
        rng=random.Random(seed);base=create_policy(policy_name,self.categories,seed);selector=ParameterPolicy(base,seed);gen=GeneratorRegistry();panel=[RuleJudge(),SemanticJudge()];agg=JudgeAggregator();coverage=CoverageTracker();beliefs=RegionBeliefStore();reward_engine=RewardEngine();failures=[];rewards=[];igs=[];disagreements=[];cost=0.;verified=0;discrimination_history=0.;context=[.2,0,.5,0,1,0,0,0,0,0,0,0,1]
        curves=[]
        for step in range(budget):
            memory={"weaknesses":[],"failure_rate":sum(x[2] for x in failures)/max(1,len(failures))}
            action=selector.select(context,memory);case=gen.generate(action.category,action.difficulty,rng)
            case.test_space.context_load=action.context_load;case.test_space.distractor_density=action.distractor_density;case.test_space.constraint_count=action.constraint_count;case.test_space.adversarial_strength=action.adversarial_strength
            region=case.test_space.region();out=await build_adapter(self.model,seed+step).predict(case.prompt,case.reference,{"category":action.category,"difficulty":action.difficulty});result=agg.aggregate([await j.judge(case,out) for j in panel]);failed=not result['correct'];failure_type=(case.expected_failure_modes[0] if failed and case.expected_failure_modes else f"{action.category}.failure")
            novelty=coverage.novelty(region);ig=beliefs.information_gain(self.model.id,failure_type,region,failed);beliefs.observe(self.model.id,failure_type,region,failed);coverage.visit(region)
            reward,_=reward_engine.compute(failed=failed,novelty=novelty,verified=False,information_gain=ig,coverage=novelty,difficulty=action.difficulty,cost=out.estimated_cost,disagreement=result['disagreement'],discrimination=0);selector.update(action.category,reward,context)
            failures.append((failure_type,region,failed));rewards.append(reward);igs.append(ig);disagreements.append(result['disagreement']);cost+=out.estimated_cost
            rate=sum(x[2] for x in failures)/len(failures);domain=lambda prefix:sum(x[2] and x[0].startswith(prefix) for x in failures)/len(failures);remaining=1-(step+1)/budget
            context=[action.difficulty,rate,1/(step+2),coverage.ratio(len(self.categories)*24),remaining,domain('reasoning'),domain('context'),domain('coding'),domain('instruction'),result['disagreement'],max(-1,min(1,reward/10)),discrimination_history,novelty]
            curves.append({"step":step+1,"reward":reward,"cumulative_failures":sum(x[2] for x in failures),"coverage":coverage.ratio(len(self.categories)*24),"information_gain":ig})
        failed=[x for x in failures if x[2]];unique_types=len({x[0] for x in failed});return {"failures":len(failed),"verified_failures":verified,"unique_failure_types":unique_types,"unique_regions":len({x[1] for x in failures}),"coverage":coverage.ratio(len(self.categories)*24),"mean_information_gain":statistics.mean(igs),"mean_reward":statistics.mean(rewards),"mean_disagreement":statistics.mean(disagreements),"total_cost":cost,"efficiency":unique_types/max(1,budget),"curves":curves}
