from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.rl.gym_environment import AdaptiveEvaluationGymEnv


class DiscreteEvaluationWrapper(gym.ActionWrapper):
    """Level-2 PPO action space: category plus one of five difficulty templates."""

    def __init__(self, environment: AdaptiveEvaluationGymEnv):
        super().__init__(environment)
        self.levels = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
        self.action_space = spaces.Discrete(len(environment.categories) * len(self.levels))

    def action(self, action: int) -> dict:
        category = int(action) // len(self.levels)
        level = float(self.levels[int(action) % len(self.levels)])
        return {
            "category": category,
            "parameters": np.asarray([level, level, level, level, level], dtype=np.float32),
        }


def train_ppo(total_timesteps: int = 10_000, budget: int = 100, seed: int = 42):
    """Train an optional PPO baseline. Install with `pip install -e .[rl]`."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.evaluation import evaluate_policy
    except ImportError as exc:
        raise RuntimeError("Install the optional RL dependencies with: pip install -e .[rl]") from exc

    environment = DiscreteEvaluationWrapper(AdaptiveEvaluationGymEnv(budget=budget))
    model = PPO("MlpPolicy", environment, seed=seed, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    mean_reward, std_reward = evaluate_policy(model, environment, n_eval_episodes=10)
    return model, {"mean_reward": float(mean_reward), "std_reward": float(std_reward)}
