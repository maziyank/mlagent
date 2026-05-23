"""End-to-end pipeline runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from rich.console import Console

from mlagent.agents.llm_log import invoke_agent_with_logging
from mlagent.agents.orchestrator import build_stage_task_message, create_orchestrator
from mlagent.config import Settings, get_settings
from mlagent.pipeline.datasets import load_dataset_config, materialize_dataset
from mlagent.pipeline.models import PipelineRun, RunStatus, StageName, StageStatus
from mlagent.pipeline.stages import StageRunner
from mlagent.storage.workspace import WorkspaceManager

console = Console()
STAGE_ORDER = list(StageName)


class PipelineRunner:
    """Coordinates full ML pipeline execution."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.workspace_mgr = WorkspaceManager(self.settings)
        self.config_dir = self.workspace_mgr.config_dir

    def run(
        self,
        dataset: str,
        pipeline: str | None = None,
        parameters: dict[str, Any] | None = None,
        run_id: str | None = None,
        on_progress: Callable[[str, str, str], None] | None = None,
        *,
        dataset_config: dict[str, Any] | None = None,
    ) -> PipelineRun:
        ds_cfg = dataset_config or load_dataset_config(dataset, self.config_dir)
        pipeline = pipeline or ds_cfg.get("default_pipeline", "binary_classification")
        pipelines = self.workspace_mgr.list_pipelines()
        if pipeline not in pipelines:
            raise ValueError(f"Unknown pipeline: {pipeline}")

        pipe_cfg = pipelines[pipeline]
        run = self.workspace_mgr.create_run(dataset, pipeline, parameters, run_id)
        run.status = RunStatus.RUNNING
        self.workspace_mgr.save_run(run)

        ws = Path(run.workspace_path)
        data_dir = ws / "data"
        materialize_dataset(dataset, data_dir, self.config_dir, cfg=ds_cfg)

        ctx = {
            "dataset": dataset,
            "target_column": ds_cfg.get("target_column", "target"),
            "task_type": ds_cfg.get("task_type", "classification"),
            "min_metric": parameters.get("min_metric") if parameters else None,
            "target_metric": pipe_cfg.get("target_metric", "accuracy"),
        }
        if ctx["min_metric"] is None:
            ctx["min_metric"] = pipe_cfg.get("min_metric", self.settings.min_accuracy)

        agent = None
        if self.settings.execution_mode == "agent":
            try:
                agent = create_orchestrator(ws, run.run_id, self.settings)
            except Exception as e:
                console.print(f"[yellow]Agent mode unavailable ({e}); using template mode[/yellow]")
                self.settings.execution_mode = "template"

        def agent_invoker(stage: str, prompt: str, *, purpose: str = "generate") -> str:
            if agent is None:
                from mlagent.pipeline.templates import get_stage_code
                return get_stage_code(StageName(stage), **{k: ctx[k] for k in ("target_column", "task_type") if k in ctx})
            msg = build_stage_task_message(
                stage=stage,
                dataset=dataset,
                pipeline=pipeline,
                workspace=str(ws),
                target_column=ctx["target_column"],
                task_type=ctx["task_type"],
            )
            input_state = {"messages": [{"role": "user", "content": msg + "\n\n" + prompt}]}
            result = invoke_agent_with_logging(
                agent,
                input_state,
                stage=stage,
                purpose=purpose,
                model=self.settings.mlagent_model,
                run_id=run.run_id,
                enabled=self.settings.llm_log_enabled,
            )
            # Extract last assistant text; agent writes files via tools
            messages = result.get("messages", [])
            for m in reversed(messages):
                content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
                if content and isinstance(content, str) and "import" in content:
                    return content
            # Fallback: read run.py if agent wrote it
            run_py = ws / "code" / stage / "run.py"
            if run_py.exists():
                return run_py.read_text()
            from mlagent.pipeline.templates import get_stage_code
            return get_stage_code(StageName(stage), target_column=ctx["target_column"], task_type=ctx["task_type"])

        stage_runner = StageRunner(
            ws,
            self.settings,
            self.config_dir,
            agent_invoker=agent_invoker if self.settings.execution_mode == "agent" else None,
        )

        prior = None
        for stage in STAGE_ORDER:
            if on_progress:
                on_progress(run.run_id, stage.value, "started")
            console.print(f"[cyan]▶ Stage: {stage.value}[/cyan]")

            state = run.stages[stage.value]
            state = stage_runner.run_stage(stage, state, ctx, prior, pipe_cfg)
            run.stages[stage.value] = state
            run.updated_at = run.updated_at
            self.workspace_mgr.save_run(run)

            if state.status == StageStatus.FAILED:
                run.status = RunStatus.FAILED
                self.workspace_mgr.save_run(run)
                if on_progress:
                    on_progress(run.run_id, stage.value, "failed")
                console.print(f"[red]✗ Stage failed: {stage.value} — {state.error}[/red]")
                return run

            prior = state
            if state.insights:
                (ws / "artifacts" / f"{stage.value}_handoff.json").write_text(
                    json.dumps({"insights": state.insights})
                )
            if on_progress:
                on_progress(run.run_id, stage.value, "completed")
            console.print(f"[green]✓ Completed: {stage.value}[/green]")

        # Final metrics from evaluation
        metrics_path = ws / "metrics.json"
        if metrics_path.exists():
            run.final_metrics = json.loads(metrics_path.read_text())

        run.status = RunStatus.COMPLETED
        self.workspace_mgr.save_run(run)
        console.print(f"[bold green]Pipeline complete — run {run.run_id}[/bold green]")
        return run

    def rerun(
        self,
        run_id: str,
        parameters: dict[str, Any] | None = None,
        from_stage: str | None = None,
    ) -> PipelineRun:
        old = self.workspace_mgr.load_run(run_id)
        params = {**old.parameters, **(parameters or {})}
        new_run = self.run(old.dataset, old.pipeline, params)
        return new_run
