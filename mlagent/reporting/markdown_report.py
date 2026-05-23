"""Markdown report generator."""

from __future__ import annotations

import json
from pathlib import Path

from mlagent.pipeline.models import PipelineRun


def generate_markdown_report(run: PipelineRun, workspace: Path) -> str:
    lines = [
        f"# ML Pipeline Report — `{run.run_id}`",
        "",
        f"- **Dataset:** {run.dataset}",
        f"- **Pipeline:** {run.pipeline}",
        f"- **Status:** {run.status.value}",
        f"- **Created:** {run.created_at.isoformat()}",
        "",
        "## Final Metrics",
        "",
        "```json",
        json.dumps(run.final_metrics, indent=2),
        "```",
        "",
        "## Stage Summary",
        "",
    ]
    for name, stage in run.stages.items():
        lines.append(f"### {name}")
        lines.append(f"- Status: {stage.status.value}")
        lines.append(f"- Iterations: {len(stage.iterations)}")
        if stage.error:
            lines.append(f"- Error: {stage.error}")
        if stage.artifacts:
            lines.append("- Artifacts:")
            for a in stage.artifacts:
                lines.append(f"  - `{a.path}`")
        lines.append("")

    eda_path = workspace / "data" / "eda_summary.json"
    if eda_path.exists():
        lines.extend(["## EDA Insights", "", "```json", eda_path.read_text(), "```", ""])

    eval_path = workspace / "data" / "evaluation_report.json"
    if eval_path.exists():
        lines.extend(["## Evaluation", "", "```json", eval_path.read_text(), "```", ""])

    lines.extend(["## Execution Logs", ""])
    logs_dir = workspace / "logs"
    if logs_dir.exists():
        for log in sorted(logs_dir.rglob("*.json")):
            lines.append(f"- `{log.relative_to(workspace)}`")
    else:
        lines.append("_No logs recorded._")

    return "\n".join(lines)


def write_markdown_report(run: PipelineRun, workspace: Path) -> Path:
    content = generate_markdown_report(run, workspace)
    out = workspace / "reports" / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content)
    return out
