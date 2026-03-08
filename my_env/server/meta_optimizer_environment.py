# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Meta-optimizer environment: train an RL agent to act as an optimizer on inner tasks.

Supports (1) SLM: next-token prediction with a tiny transformer; (2) sinusoid regression.
Rich action space (LR, momentum, grad clip, weight decay), convergence-speed reward.
Action log and loss/perplexity are exposed for dashboard visualization.
"""

import math
import random
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import torch
import torch.nn as nn

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from my_env.models import MetaOptimizerAction, MetaOptimizerObservation
from .tasks import (
    DEFAULT_CORPUS,
    SLM_TRAIN_TASK_IDS,
    TRAIN_TASK_IDS,
    get_slm_task,
    get_task,
    task_spec_from_dict,
    TaskSpec,
    SLMTaskSpec,
)
from .slm_model import (
    TinyLM,
    build_vocab,
    get_corpus_tensor,
    sample_batch_slm,
)

# Defaults
LOSS_THRESHOLD = 0.1
MAX_STEPS = 100
BATCH_SIZE = 32
# Dense reward scale: reward += DENSE_REWARD_SCALE * (prev_loss - current_loss) each step (potential-based, helps credit assignment)
DENSE_REWARD_SCALE = 0.2
# SLM loss threshold (cross-entropy); early termination when loss < this
SLM_LOSS_THRESHOLD = 1.5


def _default_device() -> torch.device:
    """Use CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(spec: TaskSpec) -> nn.Module:
    """Build a 2-layer MLP for the given task spec."""
    torch.manual_seed(spec.arch_seed)
    return nn.Sequential(
        nn.Linear(spec.input_dim, spec.hidden_dim),
        nn.ReLU(),
        nn.Linear(spec.hidden_dim, spec.output_dim),
    )


def _get_batch(spec: TaskSpec, step: int, device: torch.device):
    """Sinusoidal regression: X in [0,1], y = amplitude * sin(2*pi*freq*x + phase) + noise."""
    g = torch.Generator(device=device)
    g.manual_seed(spec.data_seed + step)
    X = torch.rand(BATCH_SIZE, spec.input_dim, device=device, generator=g)
    # y = amplitude * sin(2*pi*freq*x + phase); x is first column
    x = X[:, 0:1]
    y = spec.amplitude * torch.sin(2 * math.pi * spec.freq * x + spec.phase)
    y = y + 0.05 * torch.randn_like(y, device=device, generator=g)
    return X, y


def _build_slm(spec: SLMTaskSpec) -> TinyLM:
    """Build a tiny decoder-only transformer for the given SLM task spec."""
    torch.manual_seed(spec.arch_seed)
    return TinyLM(
        vocab_size=spec.vocab_size,
        context_len=spec.context_len,
        n_layer=spec.n_layer,
        n_head=spec.n_head,
        n_embd=spec.n_embd,
    )


def _get_batch_slm(
    spec: SLMTaskSpec,
    step: int,
    device: torch.device,
    corpus_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch for next-token prediction. Returns input_ids [B,T], target_ids [B,T]."""
    return sample_batch_slm(
        corpus_ids,
        BATCH_SIZE,
        spec.context_len,
        step,
        spec.data_seed,
        device,
    )


def run_adam_baseline(
    task_id: Optional[int] = None,
    task_spec: Optional[Dict[str, Any]] = None,
    max_steps: int = MAX_STEPS,
    loss_threshold: float = LOSS_THRESHOLD,
    lr: float = 1e-2,
    seed: Optional[int] = None,
    return_metrics: bool = False,
):
    """
    Run Adam on one task. Returns steps to threshold, or full metrics dict if return_metrics=True.
    """
    if (task_id is None) == (task_spec is None):
        raise ValueError("Provide exactly one of task_id or task_spec")
    if seed is not None:
        torch.manual_seed(seed)
    device = _default_device()
    spec = task_spec_from_dict(task_spec) if task_spec is not None else get_task(task_id)
    if isinstance(spec, SLMTaskSpec):
        raise ValueError("Use run_adamw_baseline for SLM tasks")
    model = _build_model(spec).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_trajectory: List[float] = []
    steps_to_threshold: Optional[int] = None
    for step in range(max_steps):
        X, y = _get_batch(spec, step, device)
        model.train()
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(X), y)
        loss.backward()
        opt.step()
        with torch.no_grad():
            L = nn.functional.mse_loss(model(X), y).item()
        loss_trajectory.append(L)
        if steps_to_threshold is None and L < loss_threshold:
            steps_to_threshold = step + 1
    final_loss = loss_trajectory[-1] if loss_trajectory else float("inf")
    if not return_metrics:
        return steps_to_threshold if steps_to_threshold is not None else max_steps
    last_k = min(10, len(loss_trajectory))
    mean_last_k = sum(loss_trajectory[-last_k:]) / last_k if loss_trajectory else final_loss
    return {
        "steps_to_threshold": steps_to_threshold if steps_to_threshold is not None else max_steps,
        "success": steps_to_threshold is not None,
        "final_loss": final_loss,
        "mean_last_10_loss": mean_last_k,
        "loss_auc": sum(loss_trajectory) / len(loss_trajectory) if loss_trajectory else final_loss,
        "loss_trajectory": loss_trajectory,
    }


def run_sgd_baseline(
    task_id: Optional[int] = None,
    task_spec: Optional[Dict[str, Any]] = None,
    max_steps: int = MAX_STEPS,
    loss_threshold: float = LOSS_THRESHOLD,
    lr: float = 1e-2,
    momentum: float = 0.9,
    seed: Optional[int] = None,
    return_metrics: bool = False,
):
    """
    Run SGD (with optional momentum) on one task. Returns steps to threshold, or full metrics dict if return_metrics=True.
    """
    if (task_id is None) == (task_spec is None):
        raise ValueError("Provide exactly one of task_id or task_spec")
    if seed is not None:
        torch.manual_seed(seed)
    device = _default_device()
    spec = task_spec_from_dict(task_spec) if task_spec is not None else get_task(task_id)
    if isinstance(spec, SLMTaskSpec):
        raise ValueError("Use run_adamw_baseline for SLM tasks")
    model = _build_model(spec).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_trajectory = []
    steps_to_threshold = None
    for step in range(max_steps):
        X, y = _get_batch(spec, step, device)
        model.train()
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(X), y)
        loss.backward()
        opt.step()
        with torch.no_grad():
            L = nn.functional.mse_loss(model(X), y).item()
        loss_trajectory.append(L)
        if steps_to_threshold is None and L < loss_threshold:
            steps_to_threshold = step + 1
    final_loss = loss_trajectory[-1] if loss_trajectory else float("inf")
    if not return_metrics:
        return steps_to_threshold if steps_to_threshold is not None else max_steps
    last_k = min(10, len(loss_trajectory))
    mean_last_k = sum(loss_trajectory[-last_k:]) / last_k if loss_trajectory else final_loss
    return {
        "steps_to_threshold": steps_to_threshold if steps_to_threshold is not None else max_steps,
        "success": steps_to_threshold is not None,
        "final_loss": final_loss,
        "mean_last_10_loss": mean_last_k,
        "loss_auc": sum(loss_trajectory) / len(loss_trajectory) if loss_trajectory else final_loss,
        "loss_trajectory": loss_trajectory,
    }


def run_adamw_baseline(
    task_id: Optional[int] = None,
    task_spec: Optional[Dict[str, Any]] = None,
    max_steps: int = MAX_STEPS,
    loss_threshold: float = SLM_LOSS_THRESHOLD,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    seed: Optional[int] = None,
    return_metrics: bool = False,
):
    """
    Run AdamW on one SLM task. Returns steps to threshold, or full metrics dict if return_metrics=True.
    """
    if (task_id is None) == (task_spec is None):
        raise ValueError("Provide exactly one of task_id or task_spec")
    if seed is not None:
        torch.manual_seed(seed)
    device = _default_device()
    spec = task_spec_from_dict(task_spec) if task_spec is not None else get_slm_task(task_id)
    if isinstance(spec, TaskSpec):
        raise ValueError("Use run_adam_baseline or run_sgd_baseline for sinusoid tasks")
    assert isinstance(spec, SLMTaskSpec)
    model = _build_slm(spec).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    char2idx, _ = build_vocab()
    corpus_ids = get_corpus_tensor(DEFAULT_CORPUS, char2idx, device)
    loss_trajectory: List[float] = []
    steps_to_threshold: Optional[int] = None
    for step in range(max_steps):
        inp, tgt = _get_batch_slm(spec, step, device, corpus_ids)
        model.train()
        opt.zero_grad()
        logits = model(inp)
        loss = nn.functional.cross_entropy(logits.view(-1, spec.vocab_size), tgt.view(-1))
        loss.backward()
        opt.step()
        with torch.no_grad():
            L = nn.functional.cross_entropy(
                model(inp).view(-1, spec.vocab_size), tgt.view(-1)
            ).item()
        loss_trajectory.append(L)
        if steps_to_threshold is None and L < loss_threshold:
            steps_to_threshold = step + 1
    final_loss = loss_trajectory[-1] if loss_trajectory else float("inf")
    perplexity = math.exp(min(final_loss, 20.0))
    if not return_metrics:
        return steps_to_threshold if steps_to_threshold is not None else max_steps
    last_k = min(10, len(loss_trajectory))
    mean_last_k = sum(loss_trajectory[-last_k:]) / last_k if loss_trajectory else final_loss
    return {
        "steps_to_threshold": steps_to_threshold if steps_to_threshold is not None else max_steps,
        "success": steps_to_threshold is not None,
        "final_loss": final_loss,
        "perplexity": perplexity,
        "mean_last_10_loss": mean_last_k,
        "loss_auc": sum(loss_trajectory) / len(loss_trajectory) if loss_trajectory else final_loss,
        "loss_trajectory": loss_trajectory,
    }


def run_sgd_baseline_slm(
    task_id: Optional[int] = None,
    task_spec: Optional[Dict[str, Any]] = None,
    max_steps: int = MAX_STEPS,
    loss_threshold: float = SLM_LOSS_THRESHOLD,
    lr: float = 1e-2,
    momentum: float = 0.9,
    seed: Optional[int] = None,
    return_metrics: bool = False,
):
    """
    Run SGD (with momentum) on one SLM task. Same return shape as run_adamw_baseline.
    For use in graphs / comparison only.
    """
    if (task_id is None) == (task_spec is None):
        raise ValueError("Provide exactly one of task_id or task_spec")
    if seed is not None:
        torch.manual_seed(seed)
    device = _default_device()
    spec = task_spec_from_dict(task_spec) if task_spec is not None else get_slm_task(task_id)
    if isinstance(spec, TaskSpec):
        raise ValueError("Use run_sgd_baseline for sinusoid tasks")
    assert isinstance(spec, SLMTaskSpec)
    model = _build_slm(spec).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    char2idx, _ = build_vocab()
    corpus_ids = get_corpus_tensor(DEFAULT_CORPUS, char2idx, device)
    loss_trajectory: List[float] = []
    steps_to_threshold: Optional[int] = None
    for step in range(max_steps):
        inp, tgt = _get_batch_slm(spec, step, device, corpus_ids)
        model.train()
        opt.zero_grad()
        logits = model(inp)
        loss = nn.functional.cross_entropy(logits.view(-1, spec.vocab_size), tgt.view(-1))
        loss.backward()
        opt.step()
        with torch.no_grad():
            L = nn.functional.cross_entropy(
                model(inp).view(-1, spec.vocab_size), tgt.view(-1)
            ).item()
        loss_trajectory.append(L)
        if steps_to_threshold is None and L < loss_threshold:
            steps_to_threshold = step + 1
    final_loss = loss_trajectory[-1] if loss_trajectory else float("inf")
    perplexity = math.exp(min(final_loss, 20.0))
    if not return_metrics:
        return steps_to_threshold if steps_to_threshold is not None else max_steps
    last_k = min(10, len(loss_trajectory))
    mean_last_k = sum(loss_trajectory[-last_k:]) / last_k if loss_trajectory else final_loss
    return {
        "steps_to_threshold": steps_to_threshold if steps_to_threshold is not None else max_steps,
        "success": steps_to_threshold is not None,
        "final_loss": final_loss,
        "perplexity": perplexity,
        "mean_last_10_loss": mean_last_k,
        "loss_auc": sum(loss_trajectory) / len(loss_trajectory) if loss_trajectory else final_loss,
        "loss_trajectory": loss_trajectory,
    }


def run_meta_optimizer_trajectory(
    task_id: Optional[int] = None,
    task_spec: Optional[Dict[str, Any]] = None,
    max_steps: int = MAX_STEPS,
    loss_threshold: float = LOSS_THRESHOLD,
    seed: Optional[int] = None,
    policy_callable: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run the meta-optimizer env with a policy (obs -> MetaOptimizerAction) and return metrics dict.
    If policy_callable is None, uses a fixed default policy.
    """
    if (task_id is None) == (task_spec is None):
        raise ValueError("Provide exactly one of task_id or task_spec")
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    env = MetaOptimizerEnvironment(max_steps=max_steps, loss_threshold=loss_threshold)
    obs = env.reset(seed=seed, task_id=task_id, task_spec=task_spec)
    loss_trajectory: List[float] = [obs.loss]
    if policy_callable is None:
        def _default_policy(o):  # type: ignore
            return MetaOptimizerAction(
                lr_scale=0.02, momentum_coef=0.9,
                grad_clip_threshold=1.0, weight_decay_this_step=0.0,
            )
        policy_callable = _default_policy
    while not obs.done:
        action = policy_callable(obs)
        obs = env.step(action)
        loss_trajectory.append(obs.loss)
    final_loss = obs.loss
    steps_to_threshold = obs.steps_to_threshold if obs.steps_to_threshold is not None else max_steps
    last_k = min(10, len(loss_trajectory))
    mean_last_k = sum(loss_trajectory[-last_k:]) / last_k
    return {
        "steps_to_threshold": steps_to_threshold,
        "success": obs.steps_to_threshold is not None,
        "final_loss": final_loss,
        "mean_last_10_loss": mean_last_k,
        "loss_auc": sum(loss_trajectory) / len(loss_trajectory),
        "loss_trajectory": loss_trajectory,
    }


class MetaOptimizerEnvironment(Environment[MetaOptimizerAction, MetaOptimizerObservation, State]):
    """
    Meta-learning optimizer environment: agent chooses LR scale, momentum, grad clip, weight decay per step.
    Reward: dense term = scale * (prev_loss - current_loss) each step (loss decrease); terminal = -steps_to_threshold
    when episode ends. Episode ends at max_steps or as soon as loss < threshold (early termination). Supports 50 train
    tasks and held-out eval.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        loss_threshold: float = LOSS_THRESHOLD,
        max_steps: int = MAX_STEPS,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.loss_threshold = loss_threshold
        self.max_steps = max_steps
        self._device = _default_device()

        # Episode state (set in reset)
        self._task_spec: Optional[Union[TaskSpec, SLMTaskSpec]] = None
        self._model: Optional[nn.Module] = None
        self._velocities: Optional[List[torch.Tensor]] = None
        self._step_count: int = 0
        self._current_loss: float = 0.0
        self._prev_loss: float = 0.0  # for dense reward (loss decrease)
        self._steps_to_threshold: Optional[int] = None
        self._action_log: List[Dict[str, Any]] = []
        self._episode_id: Optional[str] = None
        self._corpus_ids: Optional[torch.Tensor] = None  # for SLM only
        self._is_slm: bool = False
        self._slm_loss_threshold: float = SLM_LOSS_THRESHOLD

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[int] = None,
        task_spec: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> MetaOptimizerObservation:
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        if task_spec is not None:
            self._task_spec = task_spec_from_dict(task_spec)
        else:
            tid = task_id if task_id is not None else random.choice(SLM_TRAIN_TASK_IDS)
            self._task_spec = get_slm_task(tid)
        self._is_slm = isinstance(self._task_spec, SLMTaskSpec)

        if self._is_slm:
            spec = self._task_spec
            assert isinstance(spec, SLMTaskSpec)
            self._model = _build_slm(spec).to(self._device)
            self._velocities = [torch.zeros_like(p) for p in self._model.parameters()]
            char2idx, _ = build_vocab()
            self._corpus_ids = get_corpus_tensor(DEFAULT_CORPUS, char2idx, self._device)
            self._step_count = 0
            self._steps_to_threshold = None
            self._action_log = []
            self._episode_id = episode_id or str(uuid4())
            inp, tgt = _get_batch_slm(spec, 0, self._device, self._corpus_ids)
            with torch.no_grad():
                logits = self._model(inp)
                self._current_loss = nn.functional.cross_entropy(
                    logits.view(-1, spec.vocab_size), tgt.view(-1)
                ).item()
            self._prev_loss = self._current_loss
            return self._observation(reward=None, grad_norm=None, perplexity=math.exp(min(self._current_loss, 20.0)))
        else:
            spec = self._task_spec
            assert isinstance(spec, TaskSpec)
            self._model = _build_model(spec).to(self._device)
            self._velocities = [torch.zeros_like(p) for p in self._model.parameters()]
            self._step_count = 0
            self._steps_to_threshold = None
            self._action_log = []
            self._episode_id = episode_id or str(uuid4())
            X, y = _get_batch(spec, 0, self._device)
            with torch.no_grad():
                out = self._model(X)
                self._current_loss = nn.functional.mse_loss(out, y).item()
            self._prev_loss = self._current_loss
            return self._observation(reward=None, grad_norm=None)

    def step(
        self,
        action: MetaOptimizerAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MetaOptimizerObservation:
        assert self._model is not None and self._task_spec is not None
        prev_loss = self._prev_loss
        lr = action.lr_scale
        momentum = action.momentum_coef
        clip = action.grad_clip_threshold
        wd = action.weight_decay_this_step

        self._action_log.append({
            "step": self._step_count,
            "lr_scale": lr,
            "momentum_coef": momentum,
            "grad_clip_threshold": clip,
            "weight_decay_this_step": wd,
        })

        if self._is_slm:
            spec = self._task_spec
            assert isinstance(spec, SLMTaskSpec)
            inp, tgt = _get_batch_slm(spec, self._step_count + 1, self._device, self._corpus_ids)
            self._model.train()
            logits = self._model(inp)
            loss = nn.functional.cross_entropy(logits.view(-1, spec.vocab_size), tgt.view(-1))
        else:
            spec = self._task_spec
            assert isinstance(spec, TaskSpec)
            X, y = _get_batch(spec, self._step_count + 1, self._device)
            self._model.train()
            loss = nn.functional.mse_loss(self._model(X), y)

        self._model.zero_grad()
        loss.backward()

        grads = [p.grad.clone() for p in self._model.parameters()]
        grad_norm = sum(g.pow(2).sum() for g in grads).sqrt().item()

        if clip > 0:
            total_norm = sum(g.pow(2).sum() for g in grads).sqrt()
            if total_norm > clip:
                scale = clip / (total_norm + 1e-8)
                grads = [g * scale for g in grads]

        with torch.no_grad():
            for i, p in enumerate(self._model.parameters()):
                g = grads[i]
                v = self._velocities[i]
                v.mul_(momentum).add_(g)
                p.sub_(v, alpha=lr)
                if wd > 0:
                    p.sub_(p, alpha=wd)

        if self._is_slm:
            spec = self._task_spec
            assert isinstance(spec, SLMTaskSpec)
            with torch.no_grad():
                logits = self._model(inp)
                self._current_loss = nn.functional.cross_entropy(
                    logits.view(-1, spec.vocab_size), tgt.view(-1)
                ).item()
            loss_threshold = self._slm_loss_threshold
            perp = math.exp(min(self._current_loss, 20.0))
        else:
            spec = self._task_spec
            assert isinstance(spec, TaskSpec)
            with torch.no_grad():
                X, y = _get_batch(spec, self._step_count + 1, self._device)
                self._current_loss = nn.functional.mse_loss(self._model(X), y).item()
            loss_threshold = self.loss_threshold
            perp = None

        self._step_count += 1
        if self._steps_to_threshold is None and self._current_loss < loss_threshold:
            self._steps_to_threshold = self._step_count

        dense_reward = DENSE_REWARD_SCALE * (prev_loss - self._current_loss)
        self._prev_loss = self._current_loss

        done = self._step_count >= self.max_steps or self._steps_to_threshold is not None
        if done:
            terminal = -(self._steps_to_threshold if self._steps_to_threshold is not None else self.max_steps)
            reward = dense_reward + terminal
        else:
            reward = dense_reward

        return self._observation(reward=reward, grad_norm=grad_norm, done=done, perplexity=perp)

    def _observation(
        self,
        reward: Optional[float] = None,
        grad_norm: Optional[float] = None,
        done: bool = False,
        perplexity: Optional[float] = None,
    ) -> MetaOptimizerObservation:
        meta: Dict[str, Any] = {}
        if self._steps_to_threshold is not None:
            meta["steps_to_threshold"] = self._steps_to_threshold
        if done and self._action_log:
            meta["action_log"] = self._action_log
        return MetaOptimizerObservation(
            loss=self._current_loss,
            step_count=self._step_count,
            grad_norm=grad_norm,
            steps_to_threshold=self._steps_to_threshold,
            done=done,
            reward=reward,
            metadata=meta,
            perplexity=perplexity,
        )

    @property
    def state(self) -> State:
        return State(
            episode_id=self._episode_id,
            step_count=self._step_count,
        )

    def get_episode_action_log(self) -> List[Dict[str, Any]]:
        """Return the action log for the current episode (for in-process viz or eval)."""
        return list(self._action_log)

    def get_current_task_spec(self) -> Optional[Dict[str, Any]]:
        """Return current task spec as a dict for dashboard / run-baseline. None if no episode started."""
        if self._task_spec is None:
            return None
        if isinstance(self._task_spec, SLMTaskSpec):
            return {"type": "slm", **asdict(self._task_spec)}
        return {"type": "sinusoid", **asdict(self._task_spec)}

    def run_baseline(self) -> Dict[str, Any]:
        """Run the appropriate baseline (AdamW for SLM, Adam for sinusoid) for current task. Returns loss_trajectory and steps."""
        spec_dict = self.get_current_task_spec()
        if spec_dict is None:
            return {"loss_trajectory": [], "steps": [], "error": "No task"}
        if spec_dict.get("type") == "slm":
            result = run_adamw_baseline(task_spec=spec_dict, max_steps=self.max_steps, return_metrics=True)
        else:
            result = run_adam_baseline(task_spec=spec_dict, max_steps=self.max_steps, return_metrics=True)
        traj = result.get("loss_trajectory", [])
        return {"loss_trajectory": traj, "steps": list(range(len(traj)))}
