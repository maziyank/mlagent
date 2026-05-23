"""Custom tools for ML pipeline agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool

from mlagent.agents.optimization import OptimizationTracker
from mlagent.pipeline.models import StageName
from mlagent.pipeline.validation import ValidationError, validate_stage_outputs
from mlagent.sandbox.executor import SandboxExecutor


def make_pipeline_tools(workspace: Path, run_id: str):
    """Create tools bound to a specific pipeline run workspace."""

    @tool
    def execute_sandbox_code(
        script_relative_path: Annotated[str, "Path to script relative to workspace, e.g. code/data_understanding/run.py"],
        stage: Annotated[str, "Pipeline stage name"],
    ) -> str:
        """Execute Python code in the isolated sandbox and return JSON results."""
        executor = SandboxExecutor(workspace)
        script = workspace / script_relative_path
        result = executor.execute_script(script)
        return json.dumps(
            {
                "success": result.success,
                "exit_code": result.exit_code,
                "metrics": result.metrics,
                "artifacts": result.artifacts_produced,
                "stdout_tail": result.stdout[-2000:],
                "stderr_tail": result.stderr[-2000:],
                "duration_seconds": result.duration_seconds,
            },
            indent=2,
        )

    @tool
    def validate_stage_artifacts(
        stage: Annotated[str, "Stage to validate: data_understanding, data_preparation, modeling, evaluation"],
    ) -> str:
        """Cross-check that a stage produced required artifacts."""
        try:
            sn = StageName(stage)
            validate_stage_outputs(workspace, sn, strict=True)
            return json.dumps({"valid": True, "stage": stage})
        except ValidationError as e:
            return json.dumps({"valid": False, "error": str(e)})

    @tool
    def list_workspace_artifacts() -> str:
        """List files in workspace data/, code/, models/, logs/."""
        listing = {}
        for sub in ("data", "code", "models", "logs", "artifacts"):
            d = workspace / sub
            if d.exists():
                listing[sub] = [
                    str(p.relative_to(workspace))
                    for p in d.rglob("*")
                    if p.is_file()
                ]
        return json.dumps(listing, indent=2)

    @tool
    def read_run_manifest() -> str:
        """Read the pipeline run manifest JSON."""
        manifest = workspace / "manifest.json"
        if manifest.exists():
            return manifest.read_text()
        return json.dumps({"error": "manifest not found", "run_id": run_id})

    @tool
    def register_stage_insight(
        stage: Annotated[str, "Stage name"],
        insight: Annotated[str, "Key finding or handoff note for next agent"],
    ) -> str:
        """Record insights for inter-agent handoff."""
        path = workspace / "artifacts" / f"{stage}_insights.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(insight)
        return json.dumps({"saved": str(path.relative_to(workspace))})

    @tool
    def get_optimization_status(
        stage: Annotated[str, "Stage name (modeling or evaluation)"],
    ) -> str:
        """Return continuous optimization progress: best metric, stagnation, history."""
        status = OptimizationTracker.load(workspace, stage)
        if status is None:
            return json.dumps(
                {
                    "stage": stage,
                    "message": "No optimization data yet for this stage.",
                }
            )
        return json.dumps(status, indent=2)

    return [
        execute_sandbox_code,
        validate_stage_artifacts,
        list_workspace_artifacts,
        read_run_manifest,
        register_stage_insight,
        get_optimization_status,
    ]
