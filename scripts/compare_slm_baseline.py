#!/usr/bin/env python3
"""
Compare meta-optimizer (default or SAC policy) vs AdamW baseline on SLM tasks.

Runs both on the same SLM tasks (Distribution A or B), reports metrics:
steps_to_threshold, success, final_loss, perplexity, mean_last_10_loss, loss_auc.
Run: uv run --project my_env python scripts/compare_slm_baseline.py
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
from my_env.env_gym import obs_to_vector, vector_to_action
from my_env.models import MetaOptimizerAction
from my_env.server.meta_optimizer_environment import (
    run_adamw_baseline,
    run_meta_optimizer_trajectory,
)
from my_env.server.tasks import SLM_EVAL_TASK_IDS

MAX_STEPS = 100
LOSS_THRESHOLD = 1.5  # SLM cross-entropy threshold
N_TASKS = 5  # number of SLM tasks to evaluate (use eval task IDs)
SEEDS = [42, 43, 44]

METRIC_NAMES = [
    "steps_to_threshold",
    "success",
    "final_loss",
    "perplexity",
    "mean_last_10_loss",
    "loss_auc",
]


def sac_policy_from_model(model, max_steps: int):
    """Build a policy callable (obs -> MetaOptimizerAction) from an SB3 model."""

    def policy(obs) -> MetaOptimizerAction:
        vec = obs_to_vector(obs, max_steps)
        action_vec, _ = model.predict(vec, deterministic=True)
        return vector_to_action(np.array(action_vec, dtype=np.float32))

    return policy


def main():
    saved_path = _root / "my_env" / "saved_models" / "sac_meta_optimizer.zip"
    if not saved_path.exists():
        saved_path = _root / "saved_models" / "sac_meta_optimizer.zip"
    if not saved_path.exists():
        print("SAC model not found. Using default policy for meta-optimizer.")
        sac_model = None
    else:
        from stable_baselines3 import SAC
        sac_model = SAC.load(str(saved_path))

    task_ids = list(SLM_EVAL_TASK_IDS)[:N_TASKS] if N_TASKS <= len(SLM_EVAL_TASK_IDS) else list(range(N_TASKS))
    meta_results: dict = {m: [] for m in METRIC_NAMES}
    baseline_results: dict = {m: [] for m in METRIC_NAMES}

    for task_id in task_ids:
        for seed in SEEDS:
            task_spec = {"type": "slm", "task_id": task_id}
            # AdamW baseline
            bl = run_adamw_baseline(
                task_id=task_id,
                max_steps=MAX_STEPS,
                loss_threshold=LOSS_THRESHOLD,
                seed=seed,
                return_metrics=True,
            )
            for k in METRIC_NAMES:
                v = bl.get(k)
                if v is None:
                    continue
                if k == "success":
                    v = float(v)
                baseline_results[k].append(v)

            # Meta-optimizer (SAC or default policy)
            if sac_model is not None:
                policy = sac_policy_from_model(sac_model, MAX_STEPS)
            else:
                policy = None
            meta_m = run_meta_optimizer_trajectory(
                task_id=task_id,
                max_steps=MAX_STEPS,
                loss_threshold=LOSS_THRESHOLD,
                seed=seed,
                policy_callable=policy,
            )
            for k in METRIC_NAMES:
                v = meta_m.get(k)
                if v is None:
                    continue
                if k == "success":
                    v = float(v)
                meta_results[k].append(v)

    # Report
    print("=" * 70)
    print("Meta-optimizer vs AdamW baseline on SLM tasks")
    print("=" * 70)
    print(f"  Task IDs: {task_ids}, seeds per task: {len(SEEDS)}")
    print(f"  Threshold: {LOSS_THRESHOLD}, max_steps: {MAX_STEPS}")
    print()

    better = {}
    for m in METRIC_NAMES:
        meta_vals = [x for x in meta_results[m] if x is not None]
        bl_vals = [x for x in baseline_results[m] if x is not None]
        if not meta_vals or not bl_vals:
            continue
        meta_arr = np.array(meta_vals)
        bl_arr = np.array(bl_vals)
        meta_mean, meta_std = meta_arr.mean(), meta_arr.std()
        bl_mean, bl_std = bl_arr.mean(), bl_arr.std()
        if m == "success":
            meta_better = meta_mean > bl_mean
            diff = meta_mean - bl_mean
        else:
            meta_better = meta_mean < bl_mean
            diff = bl_mean - meta_mean
        better[m] = ("Meta" if meta_better else "AdamW", diff)
        print(f"  {m}")
        print(f"    Meta:   {meta_mean:.4f} ± {meta_std:.4f}")
        print(f"    AdamW:  {bl_mean:.4f} ± {bl_std:.4f}")
        print(f"    → Better: {better[m][0]} (Δ = {better[m][1]:.4f})")
        print()

    print("Summary: which is potentially better per metric?")
    for m in METRIC_NAMES:
        if m in better:
            print(f"  {m}: {better[m][0]}")
    meta_wins = sum(1 for m in better if better[m][0] == "Meta")
    print(f"\n  Meta wins on {meta_wins}/{len(better)} metrics.")


if __name__ == "__main__":
    main()
