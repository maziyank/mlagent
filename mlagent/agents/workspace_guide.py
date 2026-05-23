"""Workspace layout and path conventions for ML pipeline agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlagent.pipeline.models import StageName
from mlagent.pipeline.validation import REQUIRED_ARTIFACTS

WORKSPACE_LAYOUT_FILENAME = "WORKSPACE_LAYOUT.md"

# Shared block injected into all stage/orchestrator prompts.
PATH_CONVENTIONS = """
## Workspace paths (read carefully)

Two path styles — do not mix them up:

| Tool / context | Path style | Example |
|----------------|------------|---------|
| Filesystem tools (read_file, write_file, ls, glob) | Virtual prefix `/workspace/` | `/workspace/data/raw.csv` |
| execute_sandbox_code, validate_stage_artifacts | Relative to run root (no prefix) | `code/data_understanding/run.py` |
| Python scripts executed in sandbox (cwd = run root) | Relative paths in print/output | `data/eda_summary.json` |

Directory layout (all under run root = `/workspace/` in filesystem tools):
- `data/` — datasets and stage outputs (raw.csv, eda_summary.json, train/test CSVs, evaluation_report.json)
- `code/<stage>/` — Python scripts you write (`run.py` per stage)
- `models/` — trained model.pkl
- `artifacts/` — agent handoff notes and optimization state
- `logs/<stage>/` — sandbox stdout/stderr per iteration (auto-written)
- `manifest.json` — run metadata at workspace root

## Exploration rules (avoid wasted iterations)

1. Call `get_workspace_guide` once at the start of a stage — do not rediscover layout with ls/glob.
2. Use `list_workspace_artifacts` to see existing files — avoid repeated empty directory listings.
3. Do not search paths outside `/workspace/`.
4. After writing code, run `execute_sandbox_code` then `validate_stage_artifacts` — do not glob for outputs.
"""

STAGE_OUTPUTS: dict[str, dict[str, Any]] = {
    "data_understanding": {
        "code_path": "code/data_understanding/run.py",
        "required_outputs": ["data/eda_summary.json"],
        "optional_outputs": ["artifacts/data_understanding_insights.txt"],
    },
    "data_preparation": {
        "code_path": "code/data_preparation/run.py",
        "required_outputs": [
            "data/X_train.csv",
            "data/X_test.csv",
            "data/y_train.csv",
            "data/y_test.csv",
        ],
        "inputs": ["data/raw.csv", "data/eda_summary.json"],
    },
    "modeling": {
        "code_path": "code/modeling/run.py",
        "required_outputs": ["models/model.pkl"],
        "inputs": ["data/X_train.csv", "data/y_train.csv"],
    },
    "evaluation": {
        "code_path": "code/evaluation/run.py",
        "required_outputs": ["data/evaluation_report.json", "data/metrics.json"],
        "inputs": ["models/model.pkl", "data/X_test.csv", "data/y_test.csv"],
    },
}


def artifact_location(name: str) -> str:
    """Return relative path for a required artifact name."""
    if name.endswith(".pkl"):
        return f"models/{name}"
    return f"data/{name}"


def build_workspace_layout_markdown() -> str:
    """Human-readable layout file written into each run directory."""
    lines = [
        "# ML Agent Run Workspace",
        "",
        PATH_CONVENTIONS.strip(),
        "",
        "## Per-stage checklist",
    ]
    for stage, spec in STAGE_OUTPUTS.items():
        lines.append(f"\n### {stage}")
        lines.append(f"- Write code: `{spec['code_path']}` (filesystem: `/workspace/{spec['code_path']}`)")
        lines.append("- Required outputs:")
        for out in spec["required_outputs"]:
            lines.append(f"  - `{out}`")
        if spec.get("inputs"):
            lines.append("- Reads from:")
            for inp in spec["inputs"]:
                lines.append(f"  - `{inp}`")
    lines.append("\n## Sandbox script conventions")
    lines.append("- Print `MLAGENT_METRIC:name=value` for metrics")
    lines.append("- Print `MLAGENT_ARTIFACT:relative/path` for produced files")
    return "\n".join(lines) + "\n"


def write_workspace_layout(workspace: Path) -> Path:
    """Write WORKSPACE_LAYOUT.md into a run directory."""
    path = workspace / WORKSPACE_LAYOUT_FILENAME
    path.write_text(build_workspace_layout_markdown())
    return path


def seed_stage_directories(workspace: Path) -> None:
    """Create code/<stage>/ with a short README so agents do not hunt empty dirs."""
    for stage in StageName:
        stage_dir = workspace / "code" / stage.value
        stage_dir.mkdir(parents=True, exist_ok=True)
        readme = stage_dir / "README.md"
        spec = STAGE_OUTPUTS.get(stage.value, {})
        readme.write_text(
            f"# {stage.value}\n\n"
            f"Write `run.py` here.\n"
            f"Sandbox path: `code/{stage.value}/run.py`\n"
            f"Filesystem path: `/workspace/code/{stage.value}/run.py`\n"
            f"Required outputs: {', '.join(spec.get('required_outputs', []))}\n"
        )


def build_workspace_context(
    workspace: Path,
    *,
    stage: str | None = None,
    dataset_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """JSON-serializable snapshot for agent tools and task messages."""
    listing: dict[str, list[str]] = {}
    for sub in ("data", "code", "models", "logs", "artifacts"):
        d = workspace / sub
        if d.exists():
            listing[sub] = sorted(
                str(p.relative_to(workspace))
                for p in d.rglob("*")
                if p.is_file() and p.name != "README.md"
            )

    ctx: dict[str, Any] = {
        "workspace_root": str(workspace.resolve()),
        "filesystem_prefix": "/workspace/",
        "sandbox_cwd": "workspace root (use relative paths, no /workspace/ prefix)",
        "layout_file": WORKSPACE_LAYOUT_FILENAME,
        "existing_files": listing,
        "stages": STAGE_OUTPUTS,
    }
    if dataset_meta:
        ctx["dataset"] = dataset_meta
    if stage:
        spec = STAGE_OUTPUTS.get(stage, {})
        required = REQUIRED_ARTIFACTS.get(StageName(stage), [])
        ctx["current_stage"] = {
            "name": stage,
            "code_path": spec.get("code_path", f"code/{stage}/run.py"),
            "filesystem_code_path": f"/workspace/code/{stage}/run.py",
            "required_outputs": [artifact_location(n) for n in required]
            or spec.get("required_outputs", []),
            "inputs": spec.get("inputs", []),
        }
    return ctx


def format_workspace_context_for_prompt(
    workspace: Path,
    *,
    stage: str,
    dataset_meta: dict[str, Any] | None = None,
) -> str:
    """Compact block appended to stage delegation messages."""
    ctx = build_workspace_context(workspace, stage=stage, dataset_meta=dataset_meta)
    return (
        "## Workspace context (authoritative — do not re-explore with ls/glob)\n"
        f"```json\n{json.dumps(ctx, indent=2)}\n```\n"
        f"Full layout: read `/workspace/{WORKSPACE_LAYOUT_FILENAME}` if needed.\n"
    )
