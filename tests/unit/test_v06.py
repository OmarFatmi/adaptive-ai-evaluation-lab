import numpy as np

from app.policies.linucb import LinUCB, identity, mat_vec
from app.rl.gym_environment import AdaptiveEvaluationGymEnv
from app.rl.ppo import DiscreteEvaluationWrapper
from app.rewards.engine import RewardEngine


def test_sherman_morrison_inverse_identity():
    policy = LinUCB(["a"], dimensions=3)
    context = {"vector": [1.0, 0.5, 0.25]}
    policy.select(context)
    policy.update("a", 2.0, context)
    product = [
        [sum(policy.A["a"][i][k] * policy.a_inv["a"][k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]
    assert np.allclose(product, identity(3), atol=1e-8)


def test_gym_environment_contract_and_seed():
    env = AdaptiveEvaluationGymEnv(budget=3)
    first, _ = env.reset(seed=7)
    action = {"category": 0, "parameters": np.full(5, 0.5, dtype=np.float32)}
    next_observation, reward, terminated, truncated, info = env.step(action)
    assert first.shape == (13,)
    assert next_observation.shape == (13,)
    assert reward >= 0
    assert not terminated and not truncated
    assert "information_gain" in info


def test_gym_environment_truncates_at_budget():
    env = AdaptiveEvaluationGymEnv(budget=2)
    env.reset(seed=1)
    action = {"category": 1, "parameters": np.zeros(5, dtype=np.float32)}
    env.step(action)
    *_, truncated, _ = env.step(action)
    assert truncated


def test_ppo_level_two_mapping():
    wrapped = DiscreteEvaluationWrapper(AdaptiveEvaluationGymEnv(budget=5))
    action = wrapped.action(7)
    assert action["category"] == 1
    assert np.allclose(action["parameters"], np.full(5, 0.5, dtype=np.float32))


def test_no_dead_coverage_engine():
    engine = RewardEngine()
    assert not hasattr(engine, "coverage")
