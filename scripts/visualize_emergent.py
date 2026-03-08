#!/usr/bin/env python3
"""
Visualize emergent optimizer behavior: heatmap of agent actions over an episode.

Loads episode action logs (JSON: list of {step, lr_scale, momentum_coef,
grad_clip_threshold, weight_decay_this_step}) and produces a heatmap plus
optional one-liner summary for the "emergent behavior" slide.

Usage:
  uv run --project my_env python scripts/visualize_emergent.py episode_logs/
  uv run --project my_env python scripts/visualize_emergent.py episode_logs/ep1.json -o emergent_heatmap.png
"""

import argparse
import json
import os
from pathlib import Path

# Use non-interactive backend and writable config dir when in sandbox
os.environ.setdefault("MPLBACKEND", "Agg")
_cache = Path(__file__).resolve().parent.parent / ".mpl_cache"
_cache.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache))

import matplotlib.pyplot as plt
import numpy as np


ACTION_KEYS = ["lr_scale", "momentum_coef", "grad_clip_threshold", "weight_decay_this_step"]
LABELS = ["LR scale", "Momentum", "Grad clip", "Weight decay"]


def load_action_log(path: Path) -> list[dict]:
    """Load one episode log. File can be: list of action dicts, or dict with 'action_log' key."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "action_log" in data:
        return data["action_log"]
    raise ValueError(f"Unknown format in {path}: expected list or dict with 'action_log'")


def collect_logs(paths: list[Path]) -> list[list[dict]]:
    """Load all episode logs from files and/or directories."""
    logs = []
    for p in paths:
        if p.is_file():
            logs.append(load_action_log(p))
        else:
            for f in sorted(p.glob("*.json")):
                logs.append(load_action_log(f))
    return logs


def logs_to_matrix(log: list[dict]) -> np.ndarray:
    """Shape (steps, 4) for the four action dimensions."""
    rows = []
    for entry in log:
        rows.append([entry.get(k, 0) for k in ACTION_KEYS])
    return np.array(rows) if rows else np.zeros((0, 4))


def plot_heatmap(matrix: np.ndarray, out_path: Path, title: str = "Learned optimizer actions over episode") -> None:
    """Single heatmap: steps (x) vs action dimension (y), color = value."""
    if matrix.size == 0:
        raise ValueError("Empty action log")
    steps, _ = matrix.shape
    fig, ax = plt.subplots(figsize=(max(8, steps * 0.15), 5))
    im = ax.imshow(matrix.T, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(4))
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Step")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Action value")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def summarize(log: list[dict]) -> str:
    """One-liner heuristic summary for the slide (e.g. 'LR high early, decays late')."""
    if len(log) < 2:
        return "Single step (no trend)"
    steps = np.arange(len(log))
    lr = np.array([e.get("lr_scale", 0) for e in log])
    mom = np.array([e.get("momentum_coef", 0) for e in log])
    half = len(log) // 2
    lr_early, lr_late = lr[:half].mean(), lr[half:].mean()
    mom_early, mom_late = mom[:half].mean(), mom[half:].mean()
    parts = []
    if lr_early > 1.5 * lr_late:
        parts.append("LR high early, decays late")
    elif lr_late > 1.5 * lr_early:
        parts.append("LR increases over episode")
    if mom_late > 1.2 * mom_early and mom_late > 0.1:
        parts.append("momentum rises mid-episode")
    elif mom_early > 1.2 * mom_late and mom_early > 0.1:
        parts.append("momentum high early then drops")
    if not parts:
        parts.append("No strong LR/momentum schedule detected")
    return "; ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize emergent optimizer behavior")
    parser.add_argument("paths", nargs="+", type=Path, help="JSON log file(s) or directory with *.json")
    parser.add_argument("-o", "--output", type=Path, default=Path("emergent_heatmap.png"), help="Output figure path")
    parser.add_argument("--episode", type=int, default=0, help="Which episode index to plot (if multiple)")
    parser.add_argument("--summary", action="store_true", help="Print one-liner emergent summary")
    args = parser.parse_args()

    logs = collect_logs(args.paths)
    if not logs:
        print("No episode logs found")
        return
    log = logs[args.episode % len(logs)]
    matrix = logs_to_matrix(log)
    if matrix.size == 0:
        print("Empty log")
        return
    plot_heatmap(matrix, args.output, title="Learned optimizer: actions over episode (emergent behavior)")
    print(f"Saved heatmap to {args.output}")
    if args.summary:
        print("Summary:", summarize(log))


if __name__ == "__main__":
    main()
