#!/usr/bin/env python3
"""
Train Soft Actor-Critic (SAC) on the meta-optimizer env over Distribution A (50 tasks).

Saves the trained policy to saved_models/sac_meta_optimizer.zip.
Run from repo root: uv run --project my_env python scripts/train_sac.py
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from my_env.env_gym import MetaOptimizerGymEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

def main():
    out_dir = _root / "saved_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "sac_meta_optimizer.zip"

    env = MetaOptimizerGymEnv(max_steps=100, loss_threshold=0.1, task_ids=list(range(50)))
    env = DummyVecEnv([lambda: env])

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        verbose=1,
        seed=42,
    )
    total_timesteps = 50_000
    model.learn(total_timesteps=total_timesteps)
    model.save(str(save_path))
    print(f"Saved SAC policy to {save_path}")


if __name__ == "__main__":
    main()
