import argparse
import json

from app.rl.ppo import train_ppo


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the V0.6 PPO baseline")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="ppo_evaluation_lab.zip")
    args = parser.parse_args()

    model, metrics = train_ppo(args.timesteps, args.budget, args.seed)
    model.save(args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
