# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Meta-Optimizer Environment Client (OpenEnv WebSocket client)."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import MetaOptimizerAction, MetaOptimizerObservation


class MetaOptimizerEnv(
    EnvClient[MetaOptimizerAction, MetaOptimizerObservation, State]
):
    """
    Client for the Meta-Optimizer Environment.

    Connects to the meta-optimizer server over WebSocket. Use reset(seed=..., task_id=...)
    for training (task_id=None samples from 50 train tasks) or eval (task_id in EVAL_TASK_IDS).
    """

    def _step_payload(self, action: MetaOptimizerAction) -> Dict:
        return {
            "lr_scale": action.lr_scale,
            "momentum_coef": action.momentum_coef,
            "grad_clip_threshold": action.grad_clip_threshold,
            "weight_decay_this_step": action.weight_decay_this_step,
        }

    def _parse_result(
        self, payload: Dict
    ) -> StepResult[MetaOptimizerObservation]:
        obs_data = payload.get("observation", {})
        observation = MetaOptimizerObservation(
            loss=obs_data.get("loss", 0.0),
            step_count=obs_data.get("step_count", 0),
            grad_norm=obs_data.get("grad_norm"),
            steps_to_threshold=obs_data.get("steps_to_threshold"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
