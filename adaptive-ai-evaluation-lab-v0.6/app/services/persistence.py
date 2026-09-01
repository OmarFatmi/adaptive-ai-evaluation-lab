from sqlalchemy.orm import Session

from app.domain import db_models as models


class PersistenceService:
    """Own all database writes for an experiment step."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def save_test_case(self, experiment_id: int, case):
        case.metadata["region"] = case.test_space.region()
        record = models.TestCaseRecord(
            experiment_id=experiment_id,
            external_id=case.id,
            category=case.category,
            difficulty=case.difficulty,
            prompt=case.prompt,
            reference=case.reference,
            generation_strategy=case.generation_strategy,
            adversarial=case.adversarial,
            metadata_json={**case.metadata, "test_space": case.test_space.__dict__},
        )
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def save_model_output(self, test_case_id: int, model_id: int, output):
        record = models.ModelOutputRecord(
            test_case_id=test_case_id,
            model_id=model_id,
            content=output.content,
            latency_ms=output.latency_ms,
            estimated_cost=output.estimated_cost,
        )
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def save_judgments(self, output_id: int, judgments) -> None:
        for judgment in judgments:
            self.session.add(
                models.JudgmentRecord(
                    output_id=output_id,
                    judge_name=judgment.judge_name,
                    score=judgment.score,
                    confidence=judgment.confidence,
                    reliability=judgment.reliability,
                    reason=judgment.reason,
                )
            )

    def commit(self) -> None:
        self.session.commit()
