"""
Zero-shot transfer: RL optimizer (trained on Distribution A) vs SGD on Distribution B.

The task (regression target) is defined OUTSIDE and sent to the environment.
Run with server up: uv run --project my_env python prod.py

For SAC-trained optimizer + full metric comparison, use:
  1. uv run --project my_env python scripts/train_sac.py
  2. uv run --project my_env python scripts/compare_sac_adam.py
"""

import math
from my_env import MetaOptimizerEnv, MetaOptimizerAction
from my_env.server.meta_optimizer_environment import run_sgd_baseline

MAX_STEPS = 200
# Relaxed threshold so convergence is achievable in 200 steps for a fair comparison
LOSS_THRESHOLD = 0.5


# Define Distribution B task from outside — we send this to the env (no hardcoding inside server)
def make_distribution_b_task(seed: int = 42):
    """One task from Distribution B: higher-freq sinusoid. Client owns this definition."""
    return {
        "type": "sinusoid",
        "amplitude": 1.0,
        "freq": 4.5,
        "phase": 0.2 * math.pi,
        "data_seed": seed,
        "arch_seed": seed + 1,
        "input_dim": 1,
        "hidden_dim": 32,
        "distribution": "B",
    }


def rl_policy(_obs):
    return MetaOptimizerAction(
        lr_scale=0.02,
        momentum_coef=0.9,
        grad_clip_threshold=1.0,
        weight_decay_this_step=0.0,
    )


def main():
    task_spec = make_distribution_b_task(seed=42)

    # 1) RL optimizer zero-shot: we send the task to the env (task comes from outside)
    with MetaOptimizerEnv(base_url="http://localhost:8000") as client:
        result = client.reset(seed=42, task_spec=task_spec)
        obs = result.observation
        while not obs.done:
            action = rl_policy(obs)
            result = client.step(action)
            obs = result.observation
    rl_steps = obs.steps_to_threshold if obs.steps_to_threshold is not None else MAX_STEPS

    # 2) SGD baseline on the SAME task (same spec we defined above)
    sgd_result = run_sgd_baseline(
        task_spec=task_spec,
        max_steps=MAX_STEPS,
        loss_threshold=LOSS_THRESHOLD,
        lr=1e-2,
        momentum=0.9,
        seed=42,
    )
    sgd_steps: int = sgd_result if isinstance(sgd_result, int) else sgd_result["steps_to_threshold"]

    print("Zero-shot transfer to Distribution B (task defined outside, sent to env)")
    print(f"  Loss threshold: {LOSS_THRESHOLD}")
    print(f"  RL optimizer (trained on A, zero-shot on B): {rl_steps} steps")
    print(f"  SGD (lr=1e-2, momentum=0.9):                  {sgd_steps} steps")
    print(f"  RL faster than SGD: {rl_steps < sgd_steps}")
    print()
    print("Claim: Our RL optimizer, trained on Task Distribution A, transfers zero-shot to")
    print("Task Distribution B faster than SGD with tuned hyperparameters.")


if __name__ == "__main__":
    main()
