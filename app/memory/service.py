from collections import Counter
from app.domain import db_models as m
class AgentMemory:
    def __init__(self,db): self.db=db
    def snapshot(self,experiment_id,model_ids,categories):
        failures=self.db.query(m.FailureRecord).filter(m.FailureRecord.model_id.in_(model_ids)).all()
        tests=self.db.query(m.TestCaseRecord).filter_by(experiment_id=experiment_id).all()
        outputs=self.db.query(m.ModelOutputRecord).join(m.TestCaseRecord).filter(m.TestCaseRecord.experiment_id==experiment_id).all()
        events=self.db.query(m.AgentEvent).filter_by(experiment_id=experiment_id,agent="judge").all()
        groups=Counter((x.failure_type.split('.')[0] if '.' in x.failure_type else x.failure_type) for x in failures)
        regions=[x.metadata_json.get('region') for x in tests if x.metadata_json.get('region')]
        rewards=self.db.query(m.RewardRecord).filter_by(experiment_id=experiment_id).order_by(m.RewardRecord.id.desc()).limit(10).all()
        return {"total_outputs":len(outputs),"total_failures":len(failures),"failure_rate":len(failures)/max(1,len(outputs)),"uncertainty":1/(1+len(outputs)),"coverage":len(set(regions))/max(1,len(categories)*24),"regions":regions,"weaknesses":Counter(x.failure_type for x in failures).most_common(8),"domain_rates":{k:v/max(1,len(failures)) for k,v in groups.items()},"recent_reward":sum(x.total for x in rewards)/max(1,len(rewards)),"judge_disagreement":0.0,"discrimination_rate":0.0}
