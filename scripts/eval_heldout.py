#!/usr/bin/env python3
"""
Zero-shot evaluation of a meta-optimizer policy on held-out tasks.

Run from repo root with:
  uv run --project my_env python scripts/eval_heldout.py
  uv run --project my_env python scripts/eval_heldout.py --seed 123 --episodes 5

Uses MetaOptimizerEnvironment directly (no server required). By default uses a
random policy; pass --policy path to load a trained policy (not implemented here;
extend to load your RL model).
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Add my_env directory so "models" and "server" resolve (same as when server runs)
_my_env = Path(__file__).resolve().parent.parent / "my_env"
if str(_my_env) not in sys.path:
    sys.path.insert(0, str(_my_env))

from models import MetaOptimizerAction
from server.meta_optimizer_environment import MetaOptimizerEnvironment
from server.tasks import EVAL_TASK_IDS


def random_policy(obs) -> MetaOptimizerAction:
    """Sample random actions within valid bounds."""
    return MetaOptimizerAction(
        lr_scale=random.uniform(1e-3, 0.1),
        momentum_coef=random.uniform(0.0, 0.95),
        grad_clip_threshold=random.uniform(0.0, 2.0),
        weight_decay_this_step=random.uniform(0.0, 1e-4),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate meta-optimizer on held-out tasks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for eval")
    parser.add_argument("--episodes", type=int, default=2, help="Episodes per held-out task")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--save-logs", type=Path, default=None, help="Directory to save episode action logs (JSON) for viz")
    args = parser.parse_args()

    random.seed(args.seed)
    env = MetaOptimizerEnvironment(max_steps=args.max_steps)

    returns: list[float] = []
    successes: list[bool] = []

    for task_id in EVAL_TASK_IDS:
        for ep in range(args.episodes):
            obs = env.reset(seed=args.seed + task_id * 1000 + ep, task_id=task_id)
            done = obs.done
            while not done:
                action = random_policy(obs)
                obs = env.step(action)  # step returns observation directly
                done = obs.done
            # Final reward is in last observation
            r = obs.reward if obs.reward is not None else -args.max_steps
            returns.append(r)
            # Success = reached threshold (reward > -max_steps means we reached it in fewer steps)
            successes.append(obs.steps_to_threshold is not None)
            # Save action log for emergent-behavior visualization
            if args.save_logs and obs.metadata and "action_log" in obs.metadata:
                args.save_logs.mkdir(parents=True, exist_ok=True)
                out = args.save_logs / f"episode_task{task_id}_ep{ep}.json"
                with open(out, "w") as f:
                    json.dump(obs.metadata["action_log"], f, indent=0)

    n = len(returns)
    mean_return = sum(returns) / n
    variance = sum((r - mean_return) ** 2 for r in returns) / n
    std_return = variance ** 0.5
    success_rate = sum(successes) / n

    print("Held-out evaluation (zero-shot)")
    print(f"  Tasks: {EVAL_TASK_IDS}, Episodes per task: {args.episodes}")
    print(f"  Mean return (reward = -steps_to_threshold): {mean_return:.2f}")
    print(f"  Std return: {std_return:.2f}")
    print(f"  Success rate (reached loss threshold): {success_rate:.1%}")
    print(f"  Per-episode returns: {returns}")


if __name__ == "__main__":
    main()
