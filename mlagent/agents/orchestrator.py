"""Deep agent orchestrator factory."""

from __future__ import annotations

import os
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

from mlagent.agents.prompts import ORCHESTRATOR_PROMPT
from mlagent.agents.subagents import build_subagent_specs
from mlagent.agents.tools import make_pipeline_tools
from mlagent.agents.llm_log import console as llm_console
from mlagent.config import Settings


def create_orchestrator(
    workspace: Path,
    run_id: str,
    settings: Settings,
):
    """Build the main deep agent with stage sub-agents and workspace backend."""
    tools = make_pipeline_tools(workspace, run_id)
    subagents = build_subagent_specs(tools)

    # Pipeline files on disk; agent state in memory
    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/workspace/": FilesystemBackend(
                root_dir=str(workspace),
                virtual_mode=True,
            ),
        },
    )

    model = settings.mlagent_model
    if settings.openrouter_api_key:
        os.environ.setdefault("OPENROUTER_API_KEY", settings.openrouter_api_key)

    llm_console.print(
        f"[bold magenta]Creating orchestrator[/bold magenta] "
        f"model={model} run={run_id} subagents=4"
    )

    return create_deep_agent(
        model=model,
        tools=tools,
        subagents=subagents,
        system_prompt=ORCHESTRATOR_PROMPT,
        backend=backend,
        name="ml-pipeline-orchestrator",
    )


def build_stage_task_message(
    stage: str,
    dataset: str,
    pipeline: str,
    workspace: str,
    target_column: str,
    task_type: str,
    prior_insights: str = "",
) -> str:
    """User message for delegating a stage to a sub-agent."""
    return f"""Execute the '{stage}' stage for pipeline run.

Dataset: {dataset}
Pipeline config: {pipeline}
Workspace (absolute): {workspace}
Target column: {target_column}
Task type: {task_type}

Prior stage insights:
{prior_insights or "None — first stage or no insights recorded."}

Instructions:
1. Write production-ready Python to code/{stage}/run.py
2. Run via execute_sandbox_code and iterate until validation passes
3. Register key insights with register_stage_insight for the next agent
4. Required artifacts must exist before marking complete
"""
