# Deploy to Hugging Face Spaces

Run the meta-optimizer environment as a **Docker Space** so anyone can call the API (reset, step, WebSocket) from the browser or a client.

## 1. Create the Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Pick a **name** and **owner** (your user or org).
3. Choose **SDK: Docker**.
4. Set **Visibility** (Public or Private).  
5. Create the Space.

## 2. Configure the Space for Docker

In the Space repo, the **README** must tell HF to use Docker and which port the app uses. In the Space’s `README.md`, put this at the very top (YAML front matter):

```yaml
---
sdk: docker
app_port: 7860
---
```

Then add a short description of the Space below.

## 3. Push your code

From your **redemption** repo (this project):

**Option A – Push this repo into the Space (replace the Space’s content)**

```bash
# Add the Space as a remote (use your HF username and Space name)
git remote add spaces https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

# Push (use the Space’s branch, often "main")
git push spaces main
```

If the Space was just created, HF may have initialized it with a small README. You can force-push to overwrite:

```bash
git push spaces main --force
```

**Option B – Clone the Space and copy files**

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME
# Copy everything from redemption into this folder (or copy my_env, Dockerfile.spaces, .dockerignore, README with app_port)
cp -r /path/to/redemption/my_env .
cp /path/to/redemption/Dockerfile.spaces ./Dockerfile
cp /path/to/redemption/.dockerignore .
# Edit README.md to add the YAML block above
git add .
git commit -m "Add meta-optimizer env"
git push
```

## 4. Files the Space repo must have

- **Dockerfile** – Use the contents of `Dockerfile.spaces` from this repo, and name it `Dockerfile` in the Space (or rename `Dockerfile.spaces` to `Dockerfile` when you copy).
- **README.md** – Must start with the YAML block:
  - `sdk: docker`
  - `app_port: 7860`
- **my_env/** – The full `my_env` package (server, models, tasks, etc.).
- **.dockerignore** – So `.venv`, `__pycache__`, etc. are not sent to the Space (optional but recommended).

The Dockerfile in this repo is written to be run from the **redemption** root (it `COPY . .` and then installs from `my_env`). So the Space repo should look like **redemption** (same layout: `my_env/`, `Dockerfile`, etc.), not like a single folder with only `my_env`.

## 5. Build and run on HF

After you push, Hugging Face will build the Docker image and start the app. The log will show `uvicorn ... --port 7860`. When it’s running:

- **App URL:** `https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space` (or the URL shown on the Space page).
- **API base:** same URL, e.g. `https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space`
  - `POST /reset` – reset the env (optionally with `task_spec` in the body).
  - `POST /step` – send an action.
  - `GET /schema` – action/observation schemas.
  - **WebSocket:** `wss://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/ws`

## 6. Use the client against the Space

Point your client at the Space URL instead of localhost:

```python
from my_env import MetaOptimizerEnv, MetaOptimizerAction

BASE = "https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space"  # or your Space URL
with MetaOptimizerEnv(base_url=BASE) as client:
    result = client.reset(seed=42, task_spec={"type": "sinusoid", "amplitude": 1.0, "freq": 4.5, "phase": 0.0, "data_seed": 42, "arch_seed": 43})
    obs = result.observation
    while not obs.done:
        action = MetaOptimizerAction(lr_scale=0.02, momentum_coef=0.9, grad_clip_threshold=1.0, weight_decay_this_step=0.0)
        result = client.step(action)
        obs = result.observation
    print("Steps to threshold:", obs.steps_to_threshold)
```

## Troubleshooting

- **Build fails:** Check the Space’s **Logs** tab. Ensure the Dockerfile is at the repo root and that `my_env` and its `pyproject.toml` are present.
- **502 / app not responding:** The container must listen on **7860**. The provided Dockerfile already uses `--port 7860`.
- **CORS:** If you call the API from a browser on another domain, the OpenEnv/FastAPI app may need CORS middleware; add it in `server.app` if needed.
