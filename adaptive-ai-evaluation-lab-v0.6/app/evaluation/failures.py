from app.domain.contracts import FailureDiagnosis,TestCase,ModelOutput
TAXONOMY={
 "arithmetic":"reasoning.arithmetic_error","multi_constraint":"reasoning.constraint_omission","contradiction":"instruction.source_priority","distractor":"context.distractor_sensitivity","long_context":"context.retention","coding":"coding.edge_case"}
class DiagnosticEngine:
 def diagnose(self,case:TestCase,output:ModelOutput,aggregate:dict)->FailureDiagnosis:
  kind=TAXONOMY.get(case.category,"unknown.unclassified"); severity="high" if case.difficulty>=.65 else "medium"; conf=max(.55,aggregate["confidence"]*.9); evidence=[f"Expected '{case.reference}' but observed '{output.content[:160]}'",f"Judge score={aggregate['score']:.3f}",f"Difficulty={case.difficulty:.2f}"]
  hypotheses={"context.distractor_sensitivity":"Irrelevant context decreases answer accuracy.","context.retention":"Increasing context load reduces key retention.","reasoning.constraint_omission":"Increasing simultaneous constraints reduces correctness.","instruction.source_priority":"Conflicting unverified claims override trusted evidence.","reasoning.arithmetic_error":"Arithmetic difficulty increases numeric error rate.","coding.edge_case":"Negative-value edge cases reduce coding-task correctness."}
  return FailureDiagnosis(kind,severity,conf,evidence,hypotheses.get(kind,"The tested condition causes failure."),"Run a matched control/treatment pair varying only the suspected factor.")
