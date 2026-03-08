#!/usr/bin/env python3
"""
Patch OpenEnv web_interface.py to add:
- Loss/perplexity chart and updateLossChart()
- POST /web/run-baseline and GET /web/current-task for baseline comparison
Idempotent: safe to run multiple times.
"""
import sys
from pathlib import Path


def _apply_routes_patch(text: str) -> str:
    """Add /web/run-baseline and /web/current-task routes."""
    old_routes = (
        '    @app.get("/web/state")\n'
        "    async def web_state():\n"
        '        """State endpoint for web interface."""\n'
        "        return web_manager.get_state()\n"
        "\n"
        "    return app"
    )
    new_routes = (
        '    @app.get("/web/state")\n'
        "    async def web_state():\n"
        '        """State endpoint for web interface."""\n'
        "        return web_manager.get_state()\n"
        "\n"
        '    @app.get("/web/current-task")\n'
        "    async def web_current_task():\n"
        '        """Current task spec for baseline comparison (if env supports it)."""\n'
        "        get_spec = getattr(web_manager.env, \"get_current_task_spec\", None)\n"
        "        if get_spec is None:\n"
        "            return {}\n"
        "        return get_spec() or {}\n"
        "\n"
        '    @app.post("/web/run-baseline")\n'
        "    async def web_run_baseline():\n"
        '        """Run baseline optimizer for current task; returns loss_trajectory and steps."""\n'
        "        run_bl = getattr(web_manager.env, \"run_baseline\", None)\n"
        "        if run_bl is None:\n"
        "            return {\"loss_trajectory\": [], \"steps\": [], \"error\": \"Env has no run_baseline\"}\n"
        "        return run_bl()\n"
        "\n"
        "    return app"
    )
    if "web/run-baseline" not in text and "web/state" in text and "return web_manager.get_state()" in text:
        text = text.replace(old_routes, new_routes, 1)
    return text


def main() -> None:
    if len(sys.argv) < 2:
        import openenv.core.env_server.web_interface as m
        path = Path(m.__file__).resolve()
    else:
        path = Path(sys.argv[1]).resolve()

    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    text = path.read_text()

    # 1) Add Chart.js script in head (after title)
    chart_script = '    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>\n'
    old_head = "<title>OpenEnv Web Interface</title>\n    <style>"
    new_head = "<title>OpenEnv Web Interface</title>\n" + chart_script + "    <style>"
    if chart_script not in text and old_head in text:
        text = text.replace(old_head, new_head, 1)

    # 2) Add chart container between Current Observation and Action Logs
    old_section = """                </div>
                </div>

                <!-- Action Logs -->
                <div class="logs-container">"""
    new_section = """                </div>
                </div>

                <!-- Loss chart -->
                <div class="state-display">
                    <h3>Loss / Perplexity</h3>
                    <div id="loss-chart-container" style="height:200px;"><canvas id="loss-chart"></canvas></div>
                    <button type="button" id="run-baseline-btn" class="btn btn-secondary" style="margin-top:8px;">Run baseline (AdamW)</button>
                </div>

                <!-- Action Logs -->
                <div class="logs-container">"""
    if "loss-chart-container" not in text and old_section in text:
        text = text.replace(old_section, new_section, 1)
    # If chart container exists but button does not, add button
    if "loss-chart-container" in text and "run-baseline-btn" not in text:
        text = text.replace(
            "<canvas id=\"loss-chart\"></canvas></div>\n                </div>",
            "<canvas id=\"loss-chart\"></canvas></div>\n                    <button type=\"button\" id=\"run-baseline-btn\" class=\"btn btn-secondary\" style=\"margin-top:8px;\">Run baseline (AdamW)</button>\n                </div>",
            1,
        )

    # 3) Add updateLossChart call and method before updateChatInterface
    old_update = """                }}
            }}

            updateChatInterface(episodeState) {{"""
    new_update = """                }}
                this.updateLossChart(episodeState);
            }}

            updateLossChart(episodeState) {{
                const container = document.getElementById('loss-chart-container');
                if (!container) return;
                const steps = [];
                const losses = [];
                const perplexities = [];
                if (episodeState.current_observation && typeof episodeState.current_observation.loss === 'number') {{
                    const o = episodeState.current_observation;
                    steps.push(o.step_count != null ? o.step_count : 0);
                    losses.push(o.loss);
                    if (typeof o.perplexity === 'number') perplexities.push(o.perplexity);
                }}
                (episodeState.action_logs || []).forEach(log => {{
                    if (log.observation && typeof log.observation.loss === 'number') {{
                        steps.push(log.observation.step_count != null ? log.observation.step_count : log.step_count);
                        losses.push(log.observation.loss);
                        if (typeof log.observation.perplexity === 'number') perplexities.push(log.observation.perplexity);
                    }}
                }});
                if (steps.length === 0) return;
                const ctx = document.getElementById('loss-chart');
                if (!ctx) return;
                if (this._lossChart) this._lossChart.destroy();
                this._lossChart = new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: steps,
                        datasets: [
                            {{ label: 'Loss', data: losses, borderColor: '#007bff', tension: 0.2, fill: false }}
                        ].concat(perplexities.length ? [{{ label: 'Perplexity', data: perplexities, borderColor: '#28a745', tension: 0.2, fill: false }}] : [])
                    }},
                    options: {{ responsive: true, maintainAspectRatio: false, scales: {{ x: {{ title: {{ display: true, text: 'Step' }} }} }} }}
                }});
            }}

            async runBaseline() {{
                const btn = document.getElementById('run-baseline-btn');
                if (btn) btn.disabled = true;
                try {{
                    const r = await fetch('/web/run-baseline', {{ method: 'POST' }});
                    const data = await r.json();
                    if (data.error || !data.loss_trajectory || !this._lossChart) {{ if (btn) btn.disabled = false; return; }}
                    const L = data.loss_trajectory.length;
                    const steps = data.steps && data.steps.length === L ? data.steps : Array.from({{ length: L }}, (_, i) => i);
                    const curLen = this._lossChart.data.labels.length;
                    const newLen = Math.max(curLen, steps.length);
                    const newLabels = Array.from({{ length: newLen }}, (_, i) => i);
                    this._lossChart.data.labels = newLabels;
                    this._lossChart.data.datasets.forEach(ds => {{
                        while (ds.data.length < newLen) ds.data.push(null);
                    }});
                    const baselineData = data.loss_trajectory.slice();
                    while (baselineData.length < newLen) baselineData.push(null);
                    this._lossChart.data.datasets.push({{ label: 'Baseline (AdamW)', data: baselineData, borderColor: '#dc3545', tension: 0.2, fill: false }});
                    this._lossChart.update();
                }} finally {{ if (btn) btn.disabled = false; }}
            }}

            updateChatInterface(episodeState) {{"""
    if "updateLossChart(episodeState)" not in text and old_update in text:
        text = text.replace(old_update, new_update, 1)

    # 3b) Add Run baseline button click listener
    old_listener = """                // State button
                document.getElementById('state-btn').addEventListener('click', () => {{
                    this.getState();
                }});
            }}"""
    new_listener = """                // State button
                document.getElementById('state-btn').addEventListener('click', () => {{
                    this.getState();
                }});

                const runBaselineBtn = document.getElementById('run-baseline-btn');
                if (runBaselineBtn) runBaselineBtn.addEventListener('click', () => this.runBaseline());
            }}"""
    if "run-baseline-btn" not in text or "runBaselineBtn.addEventListener" not in text:
        if old_listener in text:
            text = text.replace(old_listener, new_listener, 1)

    # 3c) If updateLossChart exists but runBaseline does not, insert runBaseline
    if "updateLossChart(episodeState)" in text and "async runBaseline()" not in text:
        run_baseline_method = """
            async runBaseline() {{
                const btn = document.getElementById('run-baseline-btn');
                if (btn) btn.disabled = true;
                try {{
                    const r = await fetch('/web/run-baseline', {{ method: 'POST' }});
                    const data = await r.json();
                    if (data.error || !data.loss_trajectory || !this._lossChart) {{ if (btn) btn.disabled = false; return; }}
                    const L = data.loss_trajectory.length;
                    const newLen = Math.max(this._lossChart.data.labels.length, L);
                    const newLabels = Array.from({{ length: newLen }}, (_, i) => i);
                    this._lossChart.data.labels = newLabels;
                    this._lossChart.data.datasets.forEach(ds => {{
                        while (ds.data.length < newLen) ds.data.push(null);
                    }});
                    const baselineData = data.loss_trajectory.slice();
                    while (baselineData.length < newLen) baselineData.push(null);
                    this._lossChart.data.datasets.push({{ label: 'Baseline (AdamW)', data: baselineData, borderColor: '#dc3545', tension: 0.2, fill: false }});
                    this._lossChart.update();
                }} finally {{ if (btn) btn.disabled = false; }}
            }}
"""
        text = text.replace(
            "                }});\n            }}\n\n            updateChatInterface(episodeState) {{",
            "                }});\n            }}\n" + run_baseline_method + "\n            updateChatInterface(episodeState) {{",
            1,
        )

    # 4) Add run-baseline and current-task routes
    text = _apply_routes_patch(text)

    path.write_text(text)
    print("Patched (chart + run-baseline):", path)


if __name__ == "__main__":
    main()
