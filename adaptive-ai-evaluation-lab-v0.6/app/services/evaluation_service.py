from app.domain import db_models as models


class EvaluationService:
    """Persist diagnoses and verification evidence without orchestration concerns."""

    def __init__(self, session) -> None:
        self.session = session

    def save_failure(
        self,
        experiment_id: int,
        test_case_id: int,
        output_id: int,
        model_id: int,
        diagnosis,
        verification: dict | None,
    ):
        record = models.FailureRecord(
            experiment_id=experiment_id,
            test_case_id=test_case_id,
            output_id=output_id,
            model_id=model_id,
            failure_type=diagnosis.failure_type,
            severity=diagnosis.severity,
            confidence=diagnosis.confidence,
            diagnosis=" | ".join(diagnosis.evidence),
            hypothesis=diagnosis.hypothesis,
            verification_status=(verification or {}).get("outcome", "INCONCLUSIVE"),
        )
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)

        for statement in diagnosis.evidence:
            self.session.add(
                models.FailureEvidence(
                    failure_id=record.id,
                    evidence_type="observation",
                    statement=statement,
                    supports=True,
                    score=diagnosis.confidence,
                )
            )
        if verification:
            self.session.add(models.VerificationRun(failure_id=record.id, **verification))
        return record
