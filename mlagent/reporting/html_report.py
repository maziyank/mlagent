"""HTML report generator."""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Template

from mlagent.pipeline.models import PipelineRun
from mlagent.reporting.markdown_report import generate_markdown_report

HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>ML Pipeline Report — {{ run_id }}</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
    h1 { color: #0f4c81; }
    .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.85rem; }
    .completed { background: #d4edda; color: #155724; }
    .failed { background: #f8d7da; color: #721c24; }
    pre { background: #f4f4f5; padding: 1rem; overflow-x: auto; border-radius: 6px; }
    table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
    th, td { border: 1px solid #ddd; padding: 0.5rem; text-align: left; }
    th { background: #eef2f7; }
  </style>
</head>
<body>
  <h1>ML Pipeline Report</h1>
  <p><span class="badge {{ status_class }}">{{ status }}</span></p>
  <table>
    <tr><th>Run ID</th><td>{{ run_id }}</td></tr>
    <tr><th>Dataset</th><td>{{ dataset }}</td></tr>
    <tr><th>Pipeline</th><td>{{ pipeline }}</td></tr>
    <tr><th>Created</th><td>{{ created_at }}</td></tr>
  </table>

  <h2>Final Metrics</h2>
  <pre>{{ metrics_json }}</pre>

  <h2>Stages</h2>
  <table>
    <tr><th>Stage</th><th>Status</th><th>Iterations</th></tr>
    {% for s in stages %}
    <tr><td>{{ s.name }}</td><td>{{ s.status }}</td><td>{{ s.iterations }}</td></tr>
    {% endfor %}
  </table>

  {% if eda_json %}
  <h2>EDA Summary</h2>
  <pre>{{ eda_json }}</pre>
  {% endif %}

  {% if eval_json %}
  <h2>Evaluation</h2>
  <pre>{{ eval_json }}</pre>
  {% endif %}

  <h2>Full Markdown</h2>
  <pre>{{ markdown }}</pre>
</body>
</html>
""")


def write_html_report(run: PipelineRun, workspace: Path) -> Path:
    eda = workspace / "data" / "eda_summary.json"
    ev = workspace / "data" / "evaluation_report.json"
    stages = [
        {
            "name": k,
            "status": v.status.value,
            "iterations": len(v.iterations),
        }
        for k, v in run.stages.items()
    ]
    html = HTML_TEMPLATE.render(
        run_id=run.run_id,
        dataset=run.dataset,
        pipeline=run.pipeline,
        status=run.status.value,
        status_class="completed" if run.status.value == "completed" else "failed",
        created_at=run.created_at.isoformat(),
        metrics_json=json.dumps(run.final_metrics, indent=2),
        stages=stages,
        eda_json=eda.read_text() if eda.exists() else None,
        eval_json=ev.read_text() if ev.exists() else None,
        markdown=generate_markdown_report(run, workspace),
    )
    out = workspace / "reports" / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out
