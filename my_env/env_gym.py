# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Gymnasium wrapper for MetaOptimizerEnvironment for use with Stable-Baselines3 (e.g. SAC).
"""

import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from my_env.models import MetaOptimizerAction
from my_env.server.meta_optimizer_environment import MetaOptimizerEnvironment
from my_env.server.tasks import get_task

# Bounds for normalization / clipping
LOSS_LOG_MAX = 2.0   # log10(loss+1e-8) capped for obs
GRAD_NORM_SCALE = 10.0


def obs_to_vector(obs: Any, max_steps: int) -> np.ndarray:
    """Convert MetaOptimizerObservation to a fixed-size vector for SAC."""
    loss = getattr(obs, "loss", 0.0) or 0.0
    step_count = getattr(obs, "step_count", 0)
    grad_norm = getattr(obs, "grad_norm", None)
    # Normalize: log loss (bounded), step ratio, grad norm scale
    loss_feat = min(math.log10(loss + 1e-8), LOSS_LOG_MAX) / LOSS_LOG_MAX
    step_feat = step_count / max(1, max_steps)
    grad_feat = (grad_norm / GRAD_NORM_SCALE) if grad_norm is not None else 0.0
    grad_feat = min(max(grad_feat, 0.0), 1.0)
    return np.array([loss_feat, step_feat, grad_feat], dtype=np.float32)


def vector_to_action(vec: np.ndarray) -> MetaOptimizerAction:
    """Map [0,1]^4 to action bounds: lr [1e-4, 1], momentum [0,1], clip [0, 2], wd [0, 1e-3]."""
    lr = 1e-4 + (1.0 - 1e-4) * float(np.clip(vec[0], 0, 1))
    momentum = float(np.clip(vec[1], 0, 1))
    clip = 2.0 * float(np.clip(vec[2], 0, 1))
    wd = 1e-3 * float(np.clip(vec[3], 0, 1))
    return MetaOptimizerAction(
        lr_scale=lr,
        momentum_coef=momentum,
        grad_clip_threshold=clip,
        weight_decay_this_step=wd,
    )


class MetaOptimizerGymEnv(gym.Env):
    """
    Gymnasium env wrapping MetaOptimizerEnvironment for SAC.
    Samples tasks from Distribution A (task_id 0..49) on each reset.
    """

    def __init__(
        self,
        max_steps: int = 100,
        loss_threshold: float = 0.1,
        task_ids: Optional[list] = None,
    ):
        super().__init__()
        self._max_steps = max_steps
        self._loss_threshold = loss_threshold
        self._task_ids = task_ids or list(range(50))
        self._env = MetaOptimizerEnvironment(
            max_steps=max_steps,
            loss_threshold=loss_threshold,
        )
        # Obs: loss (norm), step (norm), grad_norm (norm) = 3
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )
        # Action: lr, momentum, grad_clip, weight_decay (all [0,1] mapped to bounds in vector_to_action)
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        import random
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            idx = self._np_random.integers(0, len(self._task_ids))
            task_id = self._task_ids[idx]
        else:
            task_id = random.choice(self._task_ids)
        obs = self._env.reset(seed=seed, task_id=task_id)
        vec = obs_to_vector(obs, self._max_steps)
        return vec, {"task_id": task_id}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        act = vector_to_action(action)
        obs = self._env.step(act)
        vec = obs_to_vector(obs, self._max_steps)
        reward = float(obs.reward if obs.reward is not None else 0.0)
        done = bool(obs.done)
        truncated = False
        info = {
            "loss": obs.loss,
            "step_count": obs.step_count,
            "steps_to_threshold": obs.steps_to_threshold,
        }
        return vec, reward, done, truncated, info
