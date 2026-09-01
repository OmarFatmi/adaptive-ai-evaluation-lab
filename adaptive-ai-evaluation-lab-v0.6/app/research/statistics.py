import random,math
def bootstrap_rate(values,seed=42,samples=500):
    if not values:return {"mean":0.0,"ci_low":0.0,"ci_high":0.0,"n":0}
    rng=random.Random(seed); means=[]
    for _ in range(samples):
        draw=[rng.choice(values) for _ in values];means.append(sum(draw)/len(draw))
    means.sort();return {"mean":sum(values)/len(values),"ci_low":means[int(.025*samples)],"ci_high":means[min(samples-1,int(.975*samples))],"n":len(values)}
def phi_effect(x,y):
    if len(x)<2 or len(y)<2:return 0.0
    return sum(x)/len(x)-sum(y)/len(y)
