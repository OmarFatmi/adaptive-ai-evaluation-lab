class RewardEngine:
    """Compute the multi-objective reward from explicit, testable components."""

    def compute(
        self,
        *,
        failed: bool,
        novelty: float,
        verified: bool,
        information_gain: float,
        coverage: float,
        difficulty: float,
        cost: float,
        disagreement: float,
        discrimination: float = 0.0,
        valid: bool = True,
    ) -> tuple[float, dict[str, float]]:
        difficulty_weight = 0.75 + 0.5 * difficulty
        components = {
            "failure_discovery": 3.0 * float(failed) * difficulty_weight,
            "novelty": 3.0 * novelty,
            "verification": 2.0 * float(verified),
            "information_gain": 2.0 * information_gain * difficulty_weight,
            "coverage": coverage,
            "model_discrimination": 2.0 * discrimination,
            "cost": -cost,
            "disagreement": -1.5 * disagreement,
            "invalid": -5.0 * float(not valid),
        }
        return round(sum(components.values()), 4), components
