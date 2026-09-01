import math
from collections import defaultdict
class RegionBeliefStore:
    """Beta-Bernoulli beliefs keyed by model, failure type and TestSpace region."""
    def __init__(self,rows=None):
        self.counts=defaultdict(lambda:[1,1])
        for model_id,failure_type,region,failed in rows or []: self.observe(model_id,failure_type,region,failed)
    def observe(self,model_id,failure_type,region,failed):
        key=(model_id,failure_type,region); self.counts[key][1 if failed else 0]+=1
    def probability(self,model_id,failure_type,region):
        success,failure=self.counts[(model_id,failure_type,region)]; return failure/(success+failure)
    @staticmethod
    def entropy(p):
        p=max(1e-12,min(1-1e-12,p)); return -(p*math.log2(p)+(1-p)*math.log2(1-p))
    def information_gain(self,model_id,failure_type,region,failed):
        key=(model_id,failure_type,region); before=self.probability(*key); clone=list(self.counts[key]); clone[1 if failed else 0]+=1; after=clone[1]/sum(clone)
        return max(0.0,self.entropy(before)-self.entropy(after))
    def snapshot(self): return {"|".join(map(str,k)):v for k,v in self.counts.items()}
