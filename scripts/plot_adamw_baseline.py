#!/usr/bin/env python3
"""
Run SGD baseline on an SLM task and save a graph of loss (and perplexity) vs step.
(Optimizer is SGD only for this graph; dashboard and other code still use AdamW where applicable.)

Usage:
  uv run --project my_env python scripts/plot_adamw_baseline.py
  uv run --project my_env python scripts/plot_adamw_baseline.py --task-id 50 --steps 100 --out sgd_baseline.png
"""

import argparse
import math
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from my_env.server.meta_optimizer_environment import run_sgd_baseline_slm


def main():
    parser = argparse.ArgumentParser(description="Run SGD baseline and plot loss curve")
    parser.add_argument("--task-id", type=int, default=0, help="SLM task ID (default 0)")
    parser.add_argument("--steps", type=int, default=100, help="Max training steps (default 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--out", type=str, default="sgd_baseline.png", help="Output graph path")
    args = parser.parse_args()

    result = run_sgd_baseline_slm(
        task_id=args.task_id,
        max_steps=args.steps,
        seed=args.seed,
        return_metrics=True,
    )

    loss_trajectory = result.get("loss_trajectory", [])
    if not loss_trajectory:
        print("No loss trajectory (error: %s)" % result.get("error", "unknown"), file=sys.stderr)
        sys.exit(1)

    steps = list(range(len(loss_trajectory)))
    perplexity = [math.exp(min(l, 20.0)) for l in loss_trajectory]
    final_loss = result.get("final_loss", loss_trajectory[-1])
    final_perp = result.get("perplexity", perplexity[-1])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax1.plot(steps, loss_trajectory, color="#007bff", label="Loss (cross-entropy)")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("SGD baseline on SLM task %d (seed=%d)" % (args.task_id, args.seed))
    ax1.text(0.02, 0.98, "Final loss: %.4f" % final_loss, transform=ax1.transAxes,
             fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax2.plot(steps, perplexity, color="#28a745", label="Perplexity")
    ax2.set_ylabel("Perplexity")
    ax2.set_xlabel("Step")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.98, "Final perplexity: %.4f" % final_perp, transform=ax2.transAxes,
             fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved graph to %s" % out_path)
    print(
        "Final loss: %.4f  Perplexity: %.4f  Success: %s"
        % (
            result.get("final_loss", 0),
            result.get("perplexity", 0),
            result.get("success", False),
        )
    )


if __name__ == "__main__":
    main()
