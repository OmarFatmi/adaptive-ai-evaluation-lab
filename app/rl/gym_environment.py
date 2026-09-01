from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.evaluation.coverage import CoverageTracker
from app.research.beliefs import RegionBeliefStore


class AdaptiveEvaluationGymEnv(gym.Env):
    """Fast synthetic Gymnasium environment for RL policy research.

    It models known weakness surfaces and does not call an LLM. Policies trained
    here must be validated later through the real AgentRuntime environment.
    """

    metadata = {"render_modes": []}

    def __init__(self, categories: list[str] | None = None, budget: int = 100):
        super().__init__()
        self.categories = categories or [
            "arithmetic",
            "multi_constraint",
            "contradiction",
            "distractor",
            "long_context",
            "coding",
        ]
        self.budget = budget
        self.action_space = spaces.Dict(
            {
                "category": spaces.Discrete(len(self.categories)),
                "parameters": spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32),
            }
        )
        self.observation_space = spaces.Box(0.0, 1.0, shape=(13,), dtype=np.float32)
        self.rng = np.random.default_rng(42)
        self.coverage = CoverageTracker()
        self.beliefs = RegionBeliefStore()
        self.steps = 0
        self.failures = 0
        self.last_reward = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self.rng = np.random.default_rng(seed)
        self.coverage = CoverageTracker()
        self.beliefs = RegionBeliefStore()
        self.steps = 0
        self.failures = 0
        self.last_reward = 0.0
        return self._observation(), {}

    def step(self, action: dict[str, Any]):
        category_index = int(action["category"])
        parameters = np.asarray(action["parameters"], dtype=np.float32)
        category = self.categories[category_index]
        difficulty, context, distractors, constraints, adversarial = parameters.tolist()
        region = self._region(category, parameters)

        base = {
            "arithmetic": 0.12,
            "multi_constraint": 0.18,
            "contradiction": 0.22,
            "distractor": 0.20,
            "long_context": 0.24,
            "coding": 0.16,
        }[category]
        risk = base + 0.25 * difficulty
        risk += 0.22 * context if category == "long_context" else 0.0
        risk += 0.22 * distractors if category == "distractor" else 0.0
        risk += 0.15 * constraints if category == "multi_constraint" else 0.0
        risk += 0.12 * adversarial
        probability = float(np.clip(risk, 0.02, 0.98))
        failed = bool(self.rng.random() < probability)

        failure_type = f"{category}.failure"
        information_gain = self.beliefs.information_gain(0, failure_type, region, failed)
        self.beliefs.observe(0, failure_type, region, failed)
        novelty = self.coverage.novelty(region)
        self.coverage.visit(region)

        reward = 3.0 * float(failed) + 3.0 * novelty + 2.0 * information_gain
        self.failures += int(failed)
        self.steps += 1
        self.last_reward = reward
        terminated = False
        truncated = self.steps >= self.budget
        info = {
            "category": category,
            "region": region,
            "failed": failed,
            "information_gain": information_gain,
            "coverage": self.coverage.ratio(len(self.categories) * 24),
        }
        return self._observation(), reward, terminated, truncated, info

    def _observation(self) -> np.ndarray:
        failure_rate = self.failures / max(1, self.steps)
        remaining = 1.0 - self.steps / max(1, self.budget)
        values = [
            self.steps / max(1, self.budget),
            failure_rate,
            1.0 / (self.steps + 1),
            self.coverage.ratio(len(self.categories) * 24),
            remaining,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            min(1.0, self.last_reward / 8.0),
            0.0,
            1.0,
        ]
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _region(category: str, parameters: np.ndarray) -> str:
        buckets = [min(3, int(float(value) * 4)) for value in parameters]
        return f"{category}:" + ":".join(map(str, buckets))
