# Redemption: Meta-Optimizer RL for SLM Training

Train an RL agent (SAC) to act as an optimizer: at each step it chooses learning rate, momentum, gradient clipping, and weight decay to minimize the inner task loss. The inner task is either **sinusoidal regression** (MLP) or **next-token prediction** (tiny transformer SLM). Compare the learned meta-optimizer to baselines (AdamW for SLM, Adam/SGD for sinusoid) and view loss curves on the OpenEnv dashboard (e.g. on Hugging Face Spaces).

## Quick Start

```bash
# Install (from repo root)
cd my_env && uv sync

# Train the meta-optimizer (SAC) on 50 SLM tasks
uv run --project my_env python scripts/train_sac.py

# Run SGD baseline and save loss/perplexity graph
uv run --project my_env python scripts/plot_adamw_baseline.py --out sgd_baseline.png

# Compare SAC vs AdamW on SLM eval tasks
uv run --project my_env python scripts/compare_slm_baseline.py
```

## Project Structure

| Path | Purpose |
|------|--------|
| `my_env/server/meta_optimizer_environment.py` | Core env: inner SLM/sinusoid step, meta-optimizer action (lr, momentum, clip, wd), reward, baselines (AdamW, SGD, Adam). |
| `my_env/server/tasks.py` | Task specs: sinusoid (TaskSpec) and SLM (SLMTaskSpec), 50 train / 2 eval tasks, default corpus. |
| `my_env/server/slm_model.py` | Tiny decoder-only transformer (TinyLM) and corpus sampling for SLM. |
| `my_env/env_gym.py` | Gymnasium wrapper for SAC: obs (loss, step, grad_norm) → vector, action vector → MetaOptimizerAction. |
| `scripts/train_sac.py` | Train SAC on Distribution A; saves `saved_models/sac_meta_optimizer.zip`. |
| `scripts/plot_adamw_baseline.py` | Run SGD baseline on one SLM task, plot loss & perplexity → PNG. |
| `scripts/compare_slm_baseline.py` | Compare meta-optimizer (or SAC) vs AdamW on SLM eval tasks. |
| `scripts/compare_sac_adam.py` | Compare SAC vs SGD on sinusoid Distribution B. |
| `my_env/scripts/patch_openenv_web_interface*.py` | Patches for OpenEnv dashboard: number step, loss chart, Run baseline (AdamW). |

## RL Setup

- **Algorithm:** SAC (Stable-Baselines3, `MlpPolicy`, default 256×2 hidden).
- **Observation:** 3D vector — normalized log loss, step ratio, grad norm.
- **Action:** 4D continuous — lr scale, momentum, grad clip threshold, weight decay.
- **Reward:** Dense = `0.2 * (prev_loss - current_loss)`; at episode end add `-steps_to_threshold` so fewer steps to convergence is better.
- **Training:** 50k timesteps on 50 tasks (Distribution A); eval on held-out tasks (Distribution B / SLM eval).

## Running the Server and Dashboard

```bash
# From my_env (set ENABLE_WEB_INTERFACE for dashboard)
ENABLE_WEB_INTERFACE=true uv run --project . server --port 8000
```

Then open `http://localhost:8000/web`: Reset to start an episode (random SLM task), submit actions (lr, momentum, etc.); the Loss/Perplexity chart updates live. Use **Run baseline (AdamW)** to overlay the AdamW trajectory for the same task.

## Deploying to Hugging Face Spaces

Build and push with OpenEnv (or your HF Space Docker workflow). The Dockerfile in `my_env/server/` installs deps, applies the web-interface patches (chart + run-baseline), and runs `uvicorn server.app:app`. Set the Space to use the Dockerfile and enable the web interface so `/web` and the chart are available. See `README.spaces.md` for Space-specific options (e.g. `sdk: docker`, `app_port: 7860`).

## License

See LICENSE in the repository.
