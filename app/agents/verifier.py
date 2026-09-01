from app.agents.base import Agent
class VerifierAgent(Agent):
 name="verifier";role="Execute matched control/treatment experiments"
 @staticmethod
 def classify(control_correct,treatment_correct,comparable=True):
  if not comparable:return "INCONCLUSIVE"
  if control_correct and not treatment_correct:return "EFFECT_SUPPORTED"
  if control_correct and treatment_correct:return "NO_EFFECT"
  if not control_correct and not treatment_correct:return "BASELINE_FAILURE"
  if not control_correct and treatment_correct:return "REVERSE_EFFECT"
  return "INCONCLUSIVE"
 def make_pair(self,case):
  m=case.metadata
  if case.category=="distractor":return f"Compute {m['a']} + {m['b']}. Return only the integer.",case.prompt,"distractor_density"
  if case.category=="long_context":return f"Memorize the code {case.reference}. What was the code? Return only the code.",case.prompt,"context_load"
  if case.category=="contradiction":return f"The audited record reports {m['verified']}. Use only the audited record. What value must be reported?",case.prompt,"conflicting_source"
  if case.category=="arithmetic":return case.prompt,case.prompt,"difficulty_observation_only"
  if case.category=="multi_constraint":
   names=m.get('ordered_names',[]);position=m.get('position',2)
   if len(names)>=4:
    shorter=names[:max(position,3)];rules=' '.join(f"{shorter[i]} is before {shorter[i+1]}." for i in range(len(shorter)-1));return f"{rules} Who is in position {position}? Return only the name.",case.prompt,"constraint_count"
  return case.prompt,case.prompt,"task_condition"
 async def run(self,state):
  state.verifications={}
  for task in state.verification_tasks:
   adapter=state.adapters[task.model_id];control,treatment,variable=self.make_pair(state.test_case);c=await adapter.predict(control,state.test_case.reference,{"category":state.test_case.category,"difficulty":max(0,state.test_case.difficulty-.3)});t=state.model_outputs[task.model_id];expected=state.test_case.reference.strip().casefold();norm=lambda x:x.content.strip().casefold()==expected;cc,tc=norm(c),norm(t);comparable=control!=treatment;outcome=self.classify(cc,tc,comparable)
   state.verifications[task.model_id]={"control_prompt":control,"treatment_prompt":treatment,"control_correct":cc,"treatment_correct":tc,"effect":float(cc)-float(tc),"supported":outcome=="EFFECT_SUPPORTED","outcome":outcome,"variable":variable,"design":"single_matched_pair" if comparable else "observational_repeat"}
  return state
