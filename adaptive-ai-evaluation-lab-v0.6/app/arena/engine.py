from app.domain import db_models as m
from app.evaluation.generators import GeneratorRegistry
from app.evaluation.judges import RuleJudge,SemanticJudge,LocalLLMJudge,JudgeAggregator
from app.models.adapters import build_adapter
import random
class ArenaEngine:
    def __init__(self,db,judge_adapter=None): self.db=db; self.judge_adapter=judge_adapter
    async def evaluate(self,case,out):
        judges=[RuleJudge(),SemanticJudge()]+([LocalLLMJudge(self.judge_adapter)] if self.judge_adapter else [])
        items=[await j.judge(case,out) for j in judges]
        return JudgeAggregator().aggregate(items)['score']
    async def match(self,a,b,tests=20,seed=42,mode="random"):
        rng=random.Random(seed);gen=GeneratorRegistry();sa=sb=0.;values={c:1. for c in gen.items}
        for i in range(tests):
            cat=rng.choice(list(gen.items)) if mode=="random" else max(values,key=lambda c:values[c]+(rng.random()*.1 if mode=="discriminative" else 0))
            case=gen.generate(cat,min(.95,.2+i/tests*.7),rng)
            oa=await build_adapter(a,seed+i).predict(case.prompt,case.reference,{"category":cat,"difficulty":case.difficulty})
            ob=await build_adapter(b,seed+1000+i).predict(case.prompt,case.reference,{"category":cat,"difficulty":case.difficulty})
            ca,cb=await self.evaluate(case,oa),await self.evaluate(case,ob);sa+=ca;sb+=cb;values[cat]=.7*values[cat]+.3*abs(ca-cb)
        result=1 if sa>sb else 0 if sa<sb else .5;expected=1/(1+10**((b.elo-a.elo)/400));delta=32*(result-expected);a.elo+=delta;b.elo-=delta;winner=a.id if sa>sb else b.id if sb>sa else None
        row=m.ArenaMatch(model_a_id=a.id,model_b_id=b.id,tests=tests,score_a=sa/tests,score_b=sb/tests,winner_id=winner,elo_delta=delta,discrimination=abs(sa-sb)/tests);self.db.add(row);self.db.commit();self.db.refresh(row);return row
