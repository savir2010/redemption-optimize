#!/usr/bin/env python3
"""
Compare SAC-trained meta-optimizer vs SGD on Distribution B with many metrics.

Loads SAC from saved_models/sac_meta_optimizer.zip, runs both on the same B tasks,
and reports which method is better on each metric.
Run: uv run --project my_env python scripts/compare_sac_adam.py
"""

import math
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
from my_env.env_gym import obs_to_vector, vector_to_action
from my_env.models import MetaOptimizerAction
from my_env.server.meta_optimizer_environment import (
    run_sgd_baseline,
    run_meta_optimizer_trajectory,
)

MAX_STEPS = 200
LOSS_THRESHOLD = 0.5
N_TASKS_B = 5  # number of Distribution B tasks to evaluate
SEEDS = [42, 43, 44]


def make_b_task(seed: int):
    return {
        "type": "sinusoid",
        "amplitude": 0.5 + 0.5 * (seed % 10) / 10,
        "freq": 4.0 + 0.5 * (seed % 5),
        "phase": 0.2 * math.pi * (seed % 5),
        "data_seed": seed,
        "arch_seed": seed + 1,
        "input_dim": 1,
        "hidden_dim": 32,
        "distribution": "B",
    }


def sac_policy_from_model(model, max_steps: int):
    """Build a policy callable (obs -> MetaOptimizerAction) from an SB3 model."""

    def policy(obs) -> MetaOptimizerAction:
        vec = obs_to_vector(obs, max_steps)
        action_vec, _ = model.predict(vec, deterministic=True)
        return vector_to_action(np.array(action_vec, dtype=np.float32))

    return policy


def main():
    saved_path = _root / "saved_models" / "sac_meta_optimizer.zip"
    if not saved_path.exists():
        print("SAC model not found. Run: uv run --project my_env python scripts/train_sac.py")
        print("Proceeding with SGD-only comparison (SAC metrics will be missing).")
        sac_model = None
    else:
        from stable_baselines3 import SAC
        sac_model = SAC.load(str(saved_path))

    tasks_b = [make_b_task(seed=s) for s in range(N_TASKS_B)]
    metric_names = [
        "steps_to_threshold",
        "success",
        "final_loss",
        "mean_last_10_loss",
        "loss_auc",
    ]

    sac_results: dict = {m: [] for m in metric_names}
    sgd_results: dict = {m: [] for m in metric_names}

    for task_spec in tasks_b:
        for seed in SEEDS:
            # SGD
            sgd_m = run_sgd_baseline(
                task_spec=task_spec,
                max_steps=MAX_STEPS,
                loss_threshold=LOSS_THRESHOLD,
                lr=1e-2,
                momentum=0.9,
                seed=seed,
                return_metrics=True,
            )
            for k in metric_names:
                v = sgd_m[k]
                if k == "success":
                    v = float(v)
                sgd_results[k].append(v)

            # SAC (or default policy if no model)
            if sac_model is not None:
                policy = sac_policy_from_model(sac_model, MAX_STEPS)
            else:
                policy = None
            sac_m = run_meta_optimizer_trajectory(
                task_spec=task_spec,
                max_steps=MAX_STEPS,
                loss_threshold=LOSS_THRESHOLD,
                seed=seed,
                policy_callable=policy,
            )
            for k in metric_names:
                v = sac_m[k]
                if k == "success":
                    v = float(v)
                sac_results[k].append(v)

    # Report
    print("=" * 70)
    print("SAC (trained on A, zero-shot on B) vs SGD (lr=1e-2, momentum=0.9) — multiple metrics")
    print("=" * 70)
    print(f"  Tasks (Distribution B): {N_TASKS_B}, seeds per task: {len(SEEDS)}")
    print(f"  Threshold: {LOSS_THRESHOLD}, max_steps: {MAX_STEPS}")
    print()

    better = {}
    for m in metric_names:
        sac_vals = np.array(sac_results[m])
        sgd_vals = np.array(sgd_results[m])
        sac_mean, sac_std = sac_vals.mean(), sac_vals.std()
        sgd_mean, sgd_std = sgd_vals.mean(), sgd_vals.std()
        # Lower is better for steps_to_threshold, final_loss, mean_last_10_loss, loss_auc
        # Higher is better for success
        if m == "success":
            sac_better = sac_mean > sgd_mean
            diff = sac_mean - sgd_mean
        else:
            sac_better = sac_mean < sgd_mean
            diff = sgd_mean - sac_mean  # positive when SAC is better (SAC lower)
        better[m] = ("SAC" if sac_better else "SGD", diff)
        print(f"  {m}")
        print(f"    SAC:  {sac_mean:.4f} ± {sac_std:.4f}")
        print(f"    SGD:  {sgd_mean:.4f} ± {sgd_std:.4f}")
        print(f"    → Better: {better[m][0]} (Δ = {better[m][1]:.4f})")
        print()

    print("Summary: which is potentially better per metric?")
    for m in metric_names:
        print(f"  {m}: {better[m][0]}")
    sac_wins = sum(1 for m in metric_names if better[m][0] == "SAC")
    print(f"\n  SAC wins on {sac_wins}/{len(metric_names)} metrics.")


if __name__ == "__main__":
    main()
