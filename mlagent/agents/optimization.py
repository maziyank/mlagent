"""Continuous metric optimization across sandbox iterations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlagent.pipeline.models import ExecutionResult, StageName

# Map pipeline target metrics to stage-specific stdout / metrics.json keys
METRIC_ALIASES: dict[str, dict[str, list[str]]] = {
    StageName.MODELING.value: {
        "accuracy": ["train_accuracy", "accuracy"],
        "r2": ["train_r2", "r2"],
    },
    StageName.EVALUATION.value: {
        "accuracy": ["accuracy", "test_accuracy"],
        "r2": ["r2", "test_r2"],
    },
}


@dataclass
class OptimizationRecord:
    iteration: int
    metric_value: float | None
    code_path: str
    success: bool


@dataclass
class OptimizationTracker:
    """Track best metric across iterations and decide when to stop optimizing."""

    stage: str
    target_metric: str
    min_metric: float
    patience: int = 2
    min_improvement: float = 0.001
    records: list[OptimizationRecord] = field(default_factory=list)
    best_iteration: int | None = None
    best_value: float | None = None
    best_code_path: str | None = None
    stagnation_count: int = 0

    def resolve_metric(self, result: ExecutionResult, workspace: Path) -> float | None:
        """Extract the primary metric from execution result or metrics.json."""
        aliases = METRIC_ALIASES.get(self.stage, {}).get(
            self.target_metric, [self.target_metric]
        )
        for key in aliases:
            if key in result.metrics:
                return float(result.metrics[key])
        metrics_file = workspace / "metrics.json"
        if metrics_file.exists():
            try:
                data = json.loads(metrics_file.read_text())
            except json.JSONDecodeError:
                data = {}
            for key in aliases:
                if key in data:
                    return float(data[key])
        if self.target_metric in result.metrics:
            return float(result.metrics[self.target_metric])
        return None

    def update(
        self,
        iteration: int,
        metric_value: float | None,
        code_path: str,
        *,
        success: bool,
    ) -> bool:
        """Record an iteration; return True if this is a new best."""
        self.records.append(
            OptimizationRecord(
                iteration=iteration,
                metric_value=metric_value,
                code_path=code_path,
                success=success,
            )
        )
        if metric_value is None:
            self.stagnation_count += 1
            return False

        improved = (
            self.best_value is None
            or metric_value > self.best_value + self.min_improvement
        )
        if improved:
            self.best_value = metric_value
            self.best_iteration = iteration
            self.best_code_path = code_path
            self.stagnation_count = 0
            return True

        self.stagnation_count += 1
        return False

    def meets_target(self) -> bool:
        if self.best_value is None:
            return False
        return self.best_value >= self.min_metric

    def should_continue(self, iteration: int, max_iterations: int) -> bool:
        """Whether to run another optimization iteration."""
        if iteration >= max_iterations:
            return False
        if self.best_value is None:
            return True
        if not self.meets_target():
            return True
        # Met threshold — keep improving until stagnation or cap
        return self.stagnation_count < self.patience

    def status(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "target_metric": self.target_metric,
            "min_metric": self.min_metric,
            "best_iteration": self.best_iteration,
            "best_value": self.best_value,
            "best_code_path": self.best_code_path,
            "meets_target": self.meets_target(),
            "stagnation_count": self.stagnation_count,
            "patience": self.patience,
            "history": [
                {
                    "iteration": r.iteration,
                    "metric": r.metric_value,
                    "success": r.success,
                    "code_path": r.code_path,
                }
                for r in self.records
            ],
        }

    def save(self, workspace: Path) -> Path:
        path = workspace / "artifacts" / "optimization_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except json.JSONDecodeError:
                existing = {}
        existing[self.stage] = self.status()
        path.write_text(json.dumps(existing, indent=2))
        return path

    @classmethod
    def load(cls, workspace: Path, stage: str) -> dict[str, Any] | None:
        path = workspace / "artifacts" / "optimization_state.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        return data.get(stage)
