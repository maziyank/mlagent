"""Continuous optimization tracker tests."""

from pathlib import Path

import pytest

from mlagent.agents.optimization import OptimizationTracker
from mlagent.pipeline.models import ExecutionResult


def test_tracker_records_improvements() -> None:
    tracker = OptimizationTracker(
        stage="modeling",
        target_metric="accuracy",
        min_metric=0.7,
        patience=2,
    )
    assert tracker.update(1, 0.75, "code/modeling/run.py", success=True)
    assert tracker.best_value == 0.75
    assert not tracker.update(2, 0.74, "code/modeling/run_002.py", success=True)
    assert tracker.best_value == 0.75
    assert tracker.stagnation_count == 1
    assert tracker.update(3, 0.82, "code/modeling/run_003.py", success=True)
    assert tracker.best_value == 0.82
    assert tracker.stagnation_count == 0


def test_should_continue_below_target() -> None:
    tracker = OptimizationTracker(
        stage="evaluation",
        target_metric="accuracy",
        min_metric=0.9,
        patience=2,
    )
    tracker.update(1, 0.8, "run.py", success=True)
    assert tracker.should_continue(1, max_iterations=5)
    assert not tracker.meets_target()


def test_should_stop_after_patience_when_above_target() -> None:
    tracker = OptimizationTracker(
        stage="evaluation",
        target_metric="accuracy",
        min_metric=0.7,
        patience=2,
    )
    tracker.update(1, 0.85, "run.py", success=True)
    tracker.update(2, 0.84, "run2.py", success=True)
    tracker.update(3, 0.84, "run3.py", success=True)
    assert tracker.meets_target()
    assert not tracker.should_continue(3, max_iterations=10)


def test_resolve_metric_aliases(tmp_path: Path) -> None:
    tracker = OptimizationTracker(
        stage="modeling",
        target_metric="accuracy",
        min_metric=0.7,
    )
    result = ExecutionResult(
        success=True,
        exit_code=0,
        metrics={"train_accuracy": 0.91},
    )
    assert tracker.resolve_metric(result, tmp_path) == 0.91


def test_save_and_load(tmp_path: Path) -> None:
    tracker = OptimizationTracker(
        stage="modeling",
        target_metric="r2",
        min_metric=0.4,
    )
    tracker.update(1, 0.55, "code/modeling/run.py", success=True)
    tracker.save(tmp_path)
    loaded = OptimizationTracker.load(tmp_path, "modeling")
    assert loaded is not None
    assert loaded["best_value"] == 0.55
