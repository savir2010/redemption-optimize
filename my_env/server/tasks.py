# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task registry for meta-learning.

Tasks can be from the internal registry (get_task(task_id)) or provided from outside
via task_spec_from_dict() — the client sends the task definition to the environment.
Supports sinusoid (regression) and SLM (next-token prediction) task types.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import math

from .slm_model import DEFAULT_VOCAB_SIZE as SLM_DEFAULT_VOCAB_SIZE

# Distribution A: 50 training tasks (low-freq sinusoids)
TRAIN_TASK_IDS: List[int] = list(range(50))

# Distribution B: held-out eval tasks (high-freq sinusoids — different distribution)
EVAL_TASK_IDS: List[int] = [50, 51]

# SLM: 50 train tasks, 2 eval (different corpus split or seed range)
SLM_TRAIN_TASK_IDS: List[int] = list(range(50))
SLM_EVAL_TASK_IDS: List[int] = [50, 51]

# Fixed small corpus for SLM (character-level). ~10KB so tasks are reproducible.
DEFAULT_CORPUS: str = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "Sphinx of black quartz, judge my vow. "
    "The five boxing wizards jump quickly. "
) * 200  # repeat to get enough length for sampling

# Bounds for each distribution (freq, amplitude, phase)
DIST_A_FREQ = (1.0, 3.0)
DIST_A_AMP = (0.5, 2.0)
DIST_B_FREQ = (4.0, 6.0)
DIST_B_AMP = (0.3, 1.5)


@dataclass
class TaskSpec:
    """Spec for one sinusoidal regression task."""

    task_id: int
    input_dim: int  # 1 for scalar sinusoid input
    hidden_dim: int
    output_dim: int
    data_seed: int
    arch_seed: int
    # Sinusoidal target: y = amplitude * sin(2*pi*freq*x + phase)
    amplitude: float
    freq: float
    phase: float
    distribution: str  # "A" or "B"


@dataclass
class SLMTaskSpec:
    """Spec for one SLM (next-token prediction) task."""

    task_id: int
    data_seed: int
    arch_seed: int
    vocab_size: int
    context_len: int  # block size
    n_layer: int
    n_head: int
    n_embd: int
    corpus_id: str  # e.g. "default"
    distribution: str  # "A" or "B" or "external"


def get_task(task_id: int) -> TaskSpec:
    """
    Return the task spec for the given task_id.
    Task IDs 0..49 = Distribution A (train), 50+ = Distribution B (eval).
    """
    if task_id < 0:
        raise ValueError(f"task_id must be >= 0, got {task_id}")
    r = task_id * 7919 + 1
    data_seed = task_id * 31337
    arch_seed = task_id * 131 + 7
    hidden_dim = 32 + (r % 33)

    if task_id < 50:
        # Distribution A
        f_lo, f_hi = DIST_A_FREQ
        a_lo, a_hi = DIST_A_AMP
        distribution = "A"
    else:
        # Distribution B
        f_lo, f_hi = DIST_B_FREQ
        a_lo, a_hi = DIST_B_AMP
        distribution = "B"

    # Deterministic but varied per task
    freq = f_lo + (r % 1000) / 1000.0 * (f_hi - f_lo)
    amplitude = a_lo + ((r * 3) % 1000) / 1000.0 * (a_hi - a_lo)
    phase = ((r * 7) % 1000) / 1000.0 * 2 * math.pi

    return TaskSpec(
        task_id=task_id,
        input_dim=1,
        hidden_dim=hidden_dim,
        output_dim=1,
        data_seed=data_seed,
        arch_seed=arch_seed,
        amplitude=amplitude,
        freq=freq,
        phase=phase,
        distribution=distribution,
    )


def get_slm_task(task_id: int) -> SLMTaskSpec:
    """
    Return the SLM task spec for the given task_id.
    Task IDs 0..49 = Distribution A (train), 50+ = Distribution B (eval).
    """
    if task_id < 0:
        raise ValueError(f"task_id must be >= 0, got {task_id}")
    r = task_id * 7919 + 1
    data_seed = task_id * 31337
    arch_seed = task_id * 131 + 7
    if task_id < 50:
        distribution = "A"
    else:
        distribution = "B"
    return SLMTaskSpec(
        task_id=task_id,
        data_seed=data_seed,
        arch_seed=arch_seed,
        vocab_size=SLM_DEFAULT_VOCAB_SIZE,
        context_len=64,
        n_layer=2,
        n_head=4,
        n_embd=128,
        corpus_id="default",
        distribution=distribution,
    )


def slm_task_spec_from_dict(d: Dict[str, Any]) -> SLMTaskSpec:
    """Build an SLMTaskSpec from an external dict (type='slm')."""
    task_id = int(d.get("task_id", 0))
    return SLMTaskSpec(
        task_id=task_id,
        data_seed=int(d.get("data_seed", task_id * 31337)),
        arch_seed=int(d.get("arch_seed", task_id * 131 + 7)),
        vocab_size=int(d.get("vocab_size", SLM_DEFAULT_VOCAB_SIZE)),
        context_len=int(d.get("context_len", 64)),
        n_layer=int(d.get("n_layer", 2)),
        n_head=int(d.get("n_head", 4)),
        n_embd=int(d.get("n_embd", 128)),
        corpus_id=str(d.get("corpus_id", "default")),
        distribution=d.get("distribution", "external"),
    )


def task_spec_from_dict(d: Dict[str, Any]) -> TaskSpec | SLMTaskSpec:
    """
    Build a TaskSpec or SLMTaskSpec from an external dict (sent by the client).

    For type "sinusoid": amplitude, freq, phase, data_seed (optional), arch_seed (optional), etc.
    For type "slm": vocab_size, context_len, n_layer, n_head, n_embd (all optional with defaults).
    """
    task_type = d.get("type", "slm")
    if task_type == "sinusoid":
        task_id = d.get("task_id", 0)
        return TaskSpec(
            task_id=task_id,
            input_dim=int(d.get("input_dim", 1)),
            hidden_dim=int(d.get("hidden_dim", 32)),
            output_dim=1,
            data_seed=int(d.get("data_seed", task_id * 31337)),
            arch_seed=int(d.get("arch_seed", task_id * 131 + 7)),
            amplitude=float(d["amplitude"]),
            freq=float(d["freq"]),
            phase=float(d["phase"]),
            distribution=d.get("distribution", "external"),
        )
    if task_type == "slm":
        return slm_task_spec_from_dict(d)
    raise ValueError(f"Unknown task type: {task_type!r}")
