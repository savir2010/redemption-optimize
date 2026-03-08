---
sdk: docker
app_port: 7860
---

# Meta-Optimizer Environment

RL meta-optimizer environment (sinusoidal regression). FastAPI + WebSocket API.

**Endpoints:** `POST /reset`, `POST /step`, `GET /schema`, `WS /ws`

Use with the OpenEnv client: point `base_url` at this Space's URL.
