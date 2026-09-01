# Adaptive AI Evaluation Lab v0.6

V0.6 focuses on maintainability, end-to-end testability, RL compatibility, and computational efficiency. The project remains local-first with SQLite, mock models, and optional Ollama. No paid API is required.

## Main corrections

- Critical modules are reformatted into readable functions and services.
- `PersistenceService`, `EvaluationService`, and `ExperimentRunner` begin the decomposition of the old orchestrator.
- The unused `CoverageEngine` was removed. `CoverageTracker` is the single coverage implementation.
- Full-matrix LinUCB now maintains `A_inv` incrementally with Sherman-Morrison. Selection no longer runs a Gauss-Jordan inversion for every arm.
- A Gymnasium environment exposes a real `reset(seed=...)` and `step(action)` contract.
- The Gym environment computes its own next observation, reward, termination flags, and diagnostics.
- An optional PPO baseline is included through Stable-Baselines3.
- PPO Level 2 uses a discrete hierarchical action: category plus difficulty template.
- The original hybrid TestSpace action remains available for future hybrid RL algorithms.
- Scientific tests now cover the Gymnasium contract, deterministic seeding, Sherman-Morrison correctness, PPO action mapping, services, API, benchmark, and orchestration runner.

## Install

Backend and tests:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item .env.example .env
python -m pytest -q
python -m uvicorn app.main:app --reload
```

Optional PPO dependencies:

```powershell
pip install -e ".[rl]"
python -m scripts.train_ppo --timesteps 10000 --budget 100 --seed 42
```

Frontend:

```powershell
cd web
npm install
npm run build
npm run dev
```

## RL levels

- Level 1: category only, evaluated with bandits and LinUCB.
- Level 2: category plus one of five parameter templates, implemented by the PPO wrapper.
- Level 3: full hybrid TestSpace action, reserved for a future hybrid PPO or SAC implementation.

## Scientific limitations

The Gymnasium environment is a fast synthetic weakness surface for algorithm development. A policy trained there is not automatically valid for a real LLM. It must be evaluated against the real AgentRuntime and local models. PPO is optional and is presented as a baseline, not as proof that RL outperforms contextual bandits.
