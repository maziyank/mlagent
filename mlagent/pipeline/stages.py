"""Per-stage execution with sandbox iteration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from mlagent.agents.prompts import REFINEMENT_PROMPT
from mlagent.config import Settings
from mlagent.pipeline.datasets import load_dataset_config
from mlagent.pipeline.models import (
    CodeIteration,
    ExecutionResult,
    StageArtifact,
    StageName,
    StageState,
    StageStatus,
)
from mlagent.pipeline.templates import get_stage_code
from mlagent.pipeline.validation import ValidationError, validate_handoff, validate_stage_outputs
from mlagent.sandbox.executor import SandboxExecutor


class StageRunner:
    """Run a single pipeline stage with iterative sandbox refinement."""

    def __init__(
        self,
        workspace: Path,
        settings: Settings,
        config_dir: Path,
        *,
        code_generator: Callable[[StageName, dict], str] | None = None,
        agent_invoker: Callable[..., str] | None = None,
    ):
        self.workspace = workspace
        self.settings = settings
        self.config_dir = config_dir
        self.executor = SandboxExecutor(
            workspace, timeout_seconds=settings.sandbox_timeout_seconds
        )
        self.code_generator = code_generator or self._default_generator
        self.agent_invoker = agent_invoker

    def _default_generator(self, stage: StageName, ctx: dict) -> str:
        return get_stage_code(
            stage,
            target_column=ctx.get("target_column", "target"),
            task_type=ctx.get("task_type", "classification"),
        )

    def run_stage(
        self,
        stage: StageName,
        state: StageState,
        ctx: dict,
        prior: StageState | None,
        pipeline_cfg: dict,
    ) -> StageState:
        validate_handoff(prior, stage, self.workspace)
        state.status = StageStatus.RUNNING
        target_metric = pipeline_cfg.get("target_metric", "accuracy")
        min_metric = float(
            ctx.get("min_metric")
            or pipeline_cfg.get("min_metric")
            or self.settings.min_accuracy
        )

        for iteration in range(1, self.settings.max_iterations_per_stage + 1):
            state.status = StageStatus.ITERATING
            code = self._generate_code(stage, ctx, state, iteration)
            code_path = self._write_code(stage, iteration, code)
            result = self._execute_with_retries(code_path)

            ci = CodeIteration(
                iteration=iteration,
                code_path=str(code_path.relative_to(self.workspace)),
                result=result,
            )
            state.iterations.append(ci)
            state.current_code_path = ci.code_path

            self._log_iteration(stage, iteration, result)

            if result.success:
                try:
                    validate_stage_outputs(self.workspace, stage, strict=True)
                    if stage == StageName.EVALUATION and not self._meets_benchmark(
                        result, target_metric, min_metric
                    ):
                        ci.refinement_notes = (
                            f"Metric {target_metric} below threshold {min_metric}"
                        )
                        continue
                    state.status = StageStatus.COMPLETED
                    state.artifacts = self._collect_artifacts(stage)
                    return state
                except ValidationError as e:
                    ci.refinement_notes = str(e)
                    code = self._refine_code(stage, ctx, iteration, result, str(e), target_metric, min_metric)
                    self._write_code(stage, iteration + 1, code, suffix="_fix")
            else:
                code = self._refine_code(
                    stage, ctx, iteration, result,
                    result.stderr or "Execution failed",
                    target_metric, min_metric,
                )
                self._write_code(stage, iteration + 1, code, suffix="_fix")

        state.status = StageStatus.FAILED
        state.error = f"Max iterations ({self.settings.max_iterations_per_stage}) exceeded"
        return state

    def _generate_code(
        self, stage: StageName, ctx: dict, state: StageState, iteration: int
    ) -> str:
        if iteration > 1 and state.iterations:
            last = state.iterations[-1]
            return self._refine_code(
                stage, ctx, iteration - 1, last.result,
                last.refinement_notes or last.result.stderr,
                ctx.get("target_metric", "accuracy"),
                float(ctx.get("min_metric", self.settings.min_accuracy)),
            )
        if self.agent_invoker and self.settings.execution_mode == "agent":
            prompt = (
                f"Generate complete Python for stage {stage.value}. "
                f"Context: {json.dumps(ctx)}"
            )
            return self.agent_invoker(stage.value, prompt, purpose="generate")
        return self.code_generator(stage, ctx)

    def _refine_code(
        self,
        stage: StageName,
        ctx: dict,
        iteration: int,
        result: ExecutionResult,
        error: str,
        target_metric: str,
        min_metric: float,
    ) -> str:
        if self.agent_invoker and self.settings.execution_mode == "agent":
            prompt = REFINEMENT_PROMPT.format(
                stage=stage.value,
                iteration=iteration,
                exit_code=result.exit_code,
                metrics=result.metrics,
                target_metric=target_metric,
                min_metric=min_metric,
                stdout=result.stdout[-3000:],
                stderr=(error + "\n" + result.stderr)[-3000:],
            )
            return self.agent_invoker(stage.value, prompt, purpose="refine")
        # Template mode: return same template (deterministic pipelines should pass first try)
        return self.code_generator(stage, ctx)

    def _refine(self, *args, **kwargs):
        self._refine_code(*args[0:6])  # noqa — placeholder for agent path
        return args[1]  # state unchanged in template mode

    def _write_code(
        self, stage: StageName, iteration: int, code: str, suffix: str = ""
    ) -> Path:
        stage_dir = self.workspace / "code" / stage.value
        stage_dir.mkdir(parents=True, exist_ok=True)
        if iteration == 1 and not suffix:
            path = stage_dir / "run.py"
        else:
            path = stage_dir / f"run_{iteration:03d}{suffix}.py"
        path.write_text(code)
        # Always update main entrypoint
        (stage_dir / "run.py").write_text(code)
        return path

    def _execute_with_retries(self, code_path: Path) -> ExecutionResult:
        last: ExecutionResult | None = None
        for _ in range(self.settings.max_execution_retries):
            last = self.executor.execute_script(code_path)
            if last.success:
                return last
        return last  # type: ignore[return-value]

    def _log_iteration(self, stage: StageName, iteration: int, result: ExecutionResult) -> None:
        log_dir = self.workspace / "logs" / stage.value
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"iteration_{iteration:03d}.json").write_text(
            json.dumps(
                {
                    "iteration": iteration,
                    "success": result.success,
                    "exit_code": result.exit_code,
                    "metrics": result.metrics,
                    "stdout": result.stdout[-30000:],
                    "stderr": result.stderr[-30000:],
                },
                indent=2,
            )
        )

    def _meets_benchmark(
        self, result: ExecutionResult, target_metric: str, min_metric: float
    ) -> bool:
        val = result.metrics.get(target_metric)
        if val is None:
            metrics_file = self.workspace / "metrics.json"
            if metrics_file.exists():
                data = json.loads(metrics_file.read_text())
                val = data.get(target_metric)
        if val is None:
            return True  # no metric to check
        return float(val) >= min_metric

    def _collect_artifacts(self, stage: StageName) -> list[StageArtifact]:
        artifacts = []
        data_dir = self.workspace / "data"
        for name in data_dir.glob("*"):
            if name.is_file():
                artifacts.append(
                    StageArtifact(
                        name=name.name,
                        path=str(name.relative_to(self.workspace)),
                        stage=stage,
                        artifact_type="data",
                    )
                )
        if stage == StageName.MODELING:
            model = self.workspace / "models" / "model.pkl"
            if model.exists():
                artifacts.append(
                    StageArtifact(
                        name="model.pkl",
                        path=str(model.relative_to(self.workspace)),
                        stage=stage,
                        artifact_type="model",
                    )
                )
        return artifacts
