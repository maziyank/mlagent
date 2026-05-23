"""Pipeline domain models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StageName(str, Enum):
    DATA_UNDERSTANDING = "data_understanding"
    DATA_PREPARATION = "data_preparation"
    MODELING = "modeling"
    EVALUATION = "evaluation"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    ITERATING = "iterating"
    COMPLETED = "completed"
    FAILED = "failed"


class ExecutionResult(BaseModel):
    success: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)
    artifacts_produced: list[str] = Field(default_factory=list)


class CodeIteration(BaseModel):
    iteration: int
    code_path: str
    result: ExecutionResult
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    refinement_notes: str = ""


class StageArtifact(BaseModel):
    name: str
    path: str
    stage: StageName
    artifact_type: str  # code, data, model, report, metrics
    metadata: dict[str, Any] = Field(default_factory=dict)


class OptimizationSummary(BaseModel):
    target_metric: str = ""
    min_metric: float = 0.0
    best_iteration: int | None = None
    best_value: float | None = None
    meets_target: bool = False


class StageState(BaseModel):
    name: StageName
    status: StageStatus = StageStatus.PENDING
    iterations: list[CodeIteration] = Field(default_factory=list)
    artifacts: list[StageArtifact] = Field(default_factory=list)
    current_code_path: str | None = None
    insights: str = ""
    error: str | None = None
    optimization: OptimizationSummary | None = None


class PipelineRun(BaseModel):
    run_id: str
    dataset: str
    pipeline: str
    status: RunStatus = RunStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    parameters: dict[str, Any] = Field(default_factory=dict)
    stages: dict[str, StageState] = Field(default_factory=dict)
    final_metrics: dict[str, float] = Field(default_factory=dict)
    workspace_path: str = ""
