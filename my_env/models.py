# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the My Env Environment.

The my_env environment is a simple test environment that echoes back messages.
Meta-optimizer models support the meta-learning RL optimizer environment.
"""

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class MyAction(Action):
    """Action for the My Env environment - just a message to echo."""

    message: str = Field(..., description="Message to echo back")


class MyObservation(Observation):
    """Observation from the My Env environment - the echoed message."""

    echoed_message: str = Field(default="", description="The echoed message")
    message_length: int = Field(default=0, description="Length of the echoed message")


# --- Meta-optimizer environment (meta-learning RL optimizer) ---


class MetaOptimizerAction(Action):
    """Action for the meta-optimizer environment: control optimizer hyperparameters per step."""

    lr_scale: float = Field(
        ...,
        ge=1e-4,
        le=1.0,
        description="Learning rate scale for this step (e.g. 1e-4 to 1.0)",
    )
    momentum_coef: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Momentum coefficient (0 = no momentum, 1 = full carry)",
    )
    grad_clip_threshold: float = Field(
        ...,
        ge=0.0,
        description="Gradient clipping threshold (0 = no clipping)",
    )
    weight_decay_this_step: float = Field(
        ...,
        ge=0.0,
        description="Weight decay (L2) scale for this step (0 = no weight decay)",
    )


class MetaOptimizerObservation(Observation):
    """Observation from the meta-optimizer environment: loss, step, and optional grad norm."""

    loss: float = Field(..., description="Current loss after last update")
    step_count: int = Field(..., description="Current step in the episode")
    grad_norm: float | None = Field(
        default=None,
        description="Global gradient norm before last update (if available)",
    )
    steps_to_threshold: int | None = Field(
        default=None,
        description="Step at which loss first reached threshold (None if not yet reached)",
    )
    perplexity: float | None = Field(
        default=None,
        description="exp(loss) for language modeling (None for regression)",
    )
    