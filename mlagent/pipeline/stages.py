"""Per-stage execution with sandbox iteration and continuous optimization."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

from mlagent.agents.optimization import OptimizationTracker
from mlagent.agents.prompts import REFINEMENT_PROMPT
from mlagent.config import Settings
from mlagent.pipeline.datasets import load_dataset_config
from mlagent.pipeline.models import (
    CodeIteration,
    ExecutionResult,
    OptimizationSummary,
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

    def _optimization_enabled(self, stage: StageName) -> bool:
        # Continuous optimization requires agent-driven code changes between iterations.
        return (
            self.settings.optimization_enabled
            and self.settings.execution_mode == "agent"
            and self.agent_invoker is not None
            and stage.value in self.settings.optimization_stage_set()
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
        optimize = self._optimization_enabled(stage)
        tracker: OptimizationTracker | None = None
        if optimize:
            tracker = OptimizationTracker(
                stage=stage.value,
                target_metric=target_metric,
                min_metric=min_metric,
                patience=self.settings.optimization_patience,
                min_improvement=self.settings.optimization_min_improvement,
            )

        for iteration in range(1, self.settings.max_iterations_per_stage + 1):
            state.status = StageStatus.ITERATING
            code = self._generate_code(stage, ctx, state, iteration, tracker)
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
                    metric_val = None
                    if tracker:
                        metric_val = tracker.resolve_metric(result, self.workspace)
                        improved = tracker.update(
                            iteration,
                            metric_val,
                            ci.code_path,
                            success=True,
                        )
                        tracker.save(self.workspace)
                        ci.refinement_notes = self._optimization_notes(
                            tracker, improved, metric_val
                        )

                    if optimize and tracker:
                        if tracker.should_continue(
                            iteration, self.settings.max_iterations_per_stage
                        ):
                            code = self._refine_for_optimization(
                                stage,
                                ctx,
                                iteration,
                                result,
                                tracker,
                                target_metric,
                                min_metric,
                            )
                            self._write_code(stage, iteration + 1, code, suffix="_opt")
                            continue
                        return self._complete_with_best(
                            stage, state, tracker, ctx, target_metric, min_metric
                        )

                    # Non-optimization stages: complete on first valid run
                    state.status = StageStatus.COMPLETED
                    state.artifacts = self._collect_artifacts(stage)
                    if tracker:
                        state.optimization = self._optimization_summary(tracker)
                    return state

                except ValidationError as e:
                    ci.refinement_notes = str(e)
                    if tracker:
                        tracker.update(
                            iteration, None, ci.code_path, success=False
                        )
                        tracker.save(self.workspace)
                    code = self._refine_code(
                        stage,
                        ctx,
                        iteration,
                        result,
                        str(e),
                        target_metric,
                        min_metric,
                        tracker,
                    )
                    self._write_code(stage, iteration + 1, code, suffix="_fix")
            else:
                if tracker:
                    tracker.update(iteration, None, ci.code_path, success=False)
                    tracker.save(self.workspace)
                code = self._refine_code(
                    stage,
                    ctx,
                    iteration,
                    result,
                    result.stderr or "Execution failed",
                    target_metric,
                    min_metric,
                    tracker,
                )
                self._write_code(stage, iteration + 1, code, suffix="_fix")

        # Exhausted iterations — use best effort for optimization stages
        if optimize and tracker and tracker.best_code_path:
            return self._complete_with_best(
                stage, state, tracker, ctx, target_metric, min_metric, allow_below_target=True
            )

        state.status = StageStatus.FAILED
        state.error = f"Max iterations ({self.settings.max_iterations_per_stage}) exceeded"
        if tracker:
            state.optimization = self._optimization_summary(tracker)
        return state

    def _complete_with_best(
        self,
        stage: StageName,
        state: StageState,
        tracker: OptimizationTracker,
        ctx: dict,
        target_metric: str,
        min_metric: float,
        *,
        allow_below_target: bool = False,
    ) -> StageState:
        """Re-run the best iteration's code and finalize the stage."""
        best_path = (self.workspace / tracker.best_code_path).resolve()
        if best_path.exists():
            run_py = (self.workspace / "code" / stage.value / "run.py").resolve()
            run_py.parent.mkdir(parents=True, exist_ok=True)
            if best_path != run_py:
                shutil.copy2(best_path, run_py)
            result = self._execute_with_retries(run_py)
            if result.success:
                try:
                    validate_stage_outputs(self.workspace, stage, strict=True)
                    metric_val = tracker.resolve_metric(result, self.workspace)
                    if metric_val is not None:
                        tracker.best_value = metric_val
                    if allow_below_target or tracker.meets_target():
                        state.status = StageStatus.COMPLETED
                        state.artifacts = self._collect_artifacts(stage)
                        state.optimization = self._optimization_summary(tracker)
                        tracker.save(self.workspace)
                        return state
                except ValidationError:
                    pass

        if tracker.meets_target():
            state.status = StageStatus.COMPLETED
            state.artifacts = self._collect_artifacts(stage)
        else:
            state.status = StageStatus.FAILED
            state.error = (
                f"Metric {target_metric} best={tracker.best_value} "
                f"below threshold {min_metric}"
            )
        state.optimization = self._optimization_summary(tracker)
        tracker.save(self.workspace)
        return state

    def _optimization_summary(self, tracker: OptimizationTracker) -> OptimizationSummary:
        return OptimizationSummary(
            target_metric=tracker.target_metric,
            min_metric=tracker.min_metric,
            best_iteration=tracker.best_iteration,
            best_value=tracker.best_value,
            meets_target=tracker.meets_target(),
        )

    def _optimization_notes(
        self,
        tracker: OptimizationTracker,
        improved: bool,
        metric_val: float | None,
    ) -> str:
        parts = [
            f"metric={metric_val}",
            f"best={tracker.best_value} (iter {tracker.best_iteration})",
            f"target>={tracker.min_metric}",
            f"stagnation={tracker.stagnation_count}/{tracker.patience}",
        ]
        if improved:
            parts.append("new_best")
        return "; ".join(parts)

    def _refine_for_optimization(
        self,
        stage: StageName,
        ctx: dict,
        iteration: int,
        result: ExecutionResult,
        tracker: OptimizationTracker,
        target_metric: str,
        min_metric: float,
    ) -> str:
        hint = (
            f"Continuous optimization: improve {target_metric} beyond "
            f"best={tracker.best_value} (threshold {min_metric}). "
            "Try stronger models, hyperparameter tuning, feature engineering, "
            "or class balancing."
        )
        return self._refine_code(
            stage,
            ctx,
            iteration,
            result,
            hint,
            target_metric,
            min_metric,
            tracker,
        )

    def _generate_code(
        self,
        stage: StageName,
        ctx: dict,
        state: StageState,
        iteration: int,
        tracker: OptimizationTracker | None,
    ) -> str:
        if iteration > 1 and state.iterations:
            last = state.iterations[-1]
            return self._refine_code(
                stage,
                ctx,
                iteration - 1,
                last.result,
                last.refinement_notes or last.result.stderr,
                ctx.get("target_metric", "accuracy"),
                float(ctx.get("min_metric", self.settings.min_accuracy)),
                tracker,
            )
        if self.agent_invoker and self.settings.execution_mode == "agent":
            opt_hint = ""
            if tracker:
                opt_hint = (
                    f"\nContinuous optimization enabled for {stage.value}. "
                    f"Maximize {tracker.target_metric} (target >= {tracker.min_metric})."
                )
            prompt = (
                f"Generate complete Python for stage {stage.value}. "
                f"Context: {json.dumps(ctx)}{opt_hint}"
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
        tracker: OptimizationTracker | None = None,
    ) -> str:
        if self.agent_invoker and self.settings.execution_mode == "agent":
            best_context = ""
            if tracker:
                best_context = (
                    f"\nBest so far: {tracker.target_metric}={tracker.best_value} "
                    f"at iteration {tracker.best_iteration}. "
                    f"Stagnation: {tracker.stagnation_count}/{tracker.patience}."
                )
            prompt = REFINEMENT_PROMPT.format(
                stage=stage.value,
                iteration=iteration,
                exit_code=result.exit_code,
                metrics=result.metrics,
                target_metric=target_metric,
                min_metric=min_metric,
                stdout=result.stdout[-3000:],
                stderr=(error + "\n" + result.stderr + best_context)[-3000:],
            )
            return self.agent_invoker(stage.value, prompt, purpose="refine")
        return self.code_generator(stage, ctx)

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
