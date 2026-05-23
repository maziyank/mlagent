"""Cross-stage artifact validation."""

from __future__ import annotations

from pathlib import Path

from mlagent.pipeline.models import StageName, StageState


REQUIRED_ARTIFACTS: dict[StageName, list[str]] = {
    StageName.DATA_UNDERSTANDING: ["eda_summary.json"],
    StageName.DATA_PREPARATION: ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"],
    StageName.MODELING: ["model.pkl"],
    StageName.EVALUATION: ["evaluation_report.json"],
}


class ValidationError(Exception):
    """Raised when stage artifacts are incompatible or missing."""

    def __init__(self, stage: StageName, message: str):
        self.stage = stage
        super().__init__(f"[{stage.value}] {message}")


def validate_stage_outputs(
    workspace: Path,
    stage: StageName,
    *,
    strict: bool = True,
) -> list[str]:
    """Validate that a stage produced expected artifacts. Returns warnings."""
    data_dir = workspace / "data"
    warnings: list[str] = []
    required = REQUIRED_ARTIFACTS.get(stage, [])

    for name in required:
        if name.endswith(".pkl"):
            path = workspace / "models" / name
        else:
            path = data_dir / name
        if not path.exists():
            msg = f"Missing required artifact: {name}"
            if strict:
                raise ValidationError(stage, msg)
            warnings.append(msg)

    if stage == StageName.DATA_PREPARATION:
        _validate_csv_pair(data_dir / "X_train.csv", data_dir / "y_train.csv", stage)

    return warnings


def _validate_csv_pair(x_path: Path, y_path: Path, stage: StageName) -> None:
    if not x_path.exists() or not y_path.exists():
        return
    import pandas as pd

    x = pd.read_csv(x_path)
    y = pd.read_csv(y_path)
    if len(x) != len(y):
        raise ValidationError(
            stage,
            f"Row count mismatch: X_train={len(x)} vs y_train={len(y)}",
        )


def validate_handoff(
    prior_stage: StageState | None,
    current_stage: StageName,
    workspace: Path,
) -> None:
    """Ensure prior stage completed before current stage runs."""
    order = list(StageName)
    idx = order.index(current_stage)
    if idx == 0:
        return

    prev = order[idx - 1]
    if prior_stage is None:
        # Check filesystem for prior artifacts
        validate_stage_outputs(workspace, prev, strict=True)
        return

    from mlagent.pipeline.models import StageStatus

    if prior_stage.status != StageStatus.COMPLETED:
        raise ValidationError(
            current_stage,
            f"Prior stage '{prev.value}' not completed (status={prior_stage.status})",
        )
    validate_stage_outputs(workspace, prev, strict=True)
