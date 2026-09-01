from abc import ABC,abstractmethod
import random
from app.domain.contracts import TestCase,TestSpace
class BaseGenerator(ABC):
 category:str
 @abstractmethod
 def generate(self,difficulty:float,rng:random.Random)->TestCase: ...
class ArithmeticGenerator(BaseGenerator):
 category="arithmetic"
 def generate(self,difficulty,rng):
  high=20+int(980*difficulty); a,b=rng.randint(2,high),rng.randint(2,high)
  return TestCase(self.category,difficulty,f"Compute {a} * {b}. Return only the integer.",str(a*b),{"a":a,"b":b},"scaled_multiplication",False,["reasoning.arithmetic_error","instruction.format_violation"])
class ConstraintGenerator(BaseGenerator):
 category="multi_constraint"
 def generate(self,difficulty,rng):
  names=["Amina","Bilal","Chafik","Dalia","Elias","Farah"][:4+int(2*difficulty)]; rng.shuffle(names); rules=" ".join(f"{names[i]} is before {names[i+1]}." for i in range(len(names)-1)); pos=min(len(names)-1,1+int(difficulty*2))
  return TestCase(self.category,difficulty,f"{rules} Who is in position {pos+1}? Return only the name.",names[pos],{"constraints":len(names)-1,"position":pos+1,"ordered_names":names},"ordered_constraints",False,["reasoning.constraint_omission","reasoning.multi_step"])
class ContradictionGenerator(BaseGenerator):
 category="contradiction"
 def generate(self,difficulty,rng):
  value=rng.randint(10,99); wrong=value+rng.choice([-7,-3,4,8]); extra=" Several informal comments repeat the unverified value." if difficulty>.5 else ""
  return TestCase(self.category,difficulty,f"The audited record reports {value}. An unverified note reports {wrong}.{extra} Use only the audited record. What value must be reported?",str(value),{"verified":value,"unverified":wrong},"source_priority_conflict",True,["reasoning.contradiction","instruction.source_priority"])
class DistractorGenerator(BaseGenerator):
 category="distractor"
 def generate(self,difficulty,rng):
  a,b=rng.randint(5,90),rng.randint(5,90); n=1+int(8*difficulty); noise=" The archive has seven windows and a blue folder arrived on Tuesday."*n
  return TestCase(self.category,difficulty,f"{noise} Ignore unrelated facts. Compute {a}+{b}. Return only the integer.",str(a+b),{"distractor_density":round(n/(n+2),3),"a":a,"b":b},"distractor_density",True,["context.distractor_sensitivity","instruction.following"])
class LongContextGenerator(BaseGenerator):
 category="long_context"
 def generate(self,difficulty,rng):
  code=rng.choice(["KAPPA","SIGMA","OMEGA","DELTA"]); n=8+int(90*difficulty); filler=" Background material is unrelated to the requested code."*n
  return TestCase(self.category,difficulty,f"Memorize the code {code}.{filler} What was the code? Return only the code.",code,{"context_tokens_approx":n*8},"retention_under_load",True,["context.retention","context.long_context"])
class CodingGenerator(BaseGenerator):
 category="coding"
 def generate(self,difficulty,rng):
  nums=[rng.randint(-9,20) for _ in range(4+int(5*difficulty))]; ref=str(max(nums))
  return TestCase(self.category,difficulty,f"Given the Python list {nums}, return only its maximum integer. Do not execute external tools.",ref,{"language":"python","edge_cases":"negative_values"},"coding_edge_case",False,["coding.algorithm","coding.edge_case"])
class GeneratorRegistry:
 def __init__(self):
  self.items={g.category:g for g in [ArithmeticGenerator(),ConstraintGenerator(),ContradictionGenerator(),DistractorGenerator(),LongContextGenerator(),CodingGenerator()]}
 def generate(self,category,difficulty,rng):
  if category not in self.items: raise ValueError(f"Unknown category: {category}")
  case=self.items[category].generate(max(0,min(1,difficulty)),rng)
  md=case.metadata; case.test_space=TestSpace(category=case.category,difficulty=case.difficulty,context_load=min(1,md.get("context_tokens_approx",0)/800),distractor_density=md.get("distractor_density",0),constraint_count=md.get("constraints",0),adversarial_strength=case.difficulty if case.adversarial else 0,adversarial=case.adversarial); return case
