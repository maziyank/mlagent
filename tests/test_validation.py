"""Artifact validation tests."""

from pathlib import Path

import pytest

from mlagent.pipeline.models import StageName
from mlagent.pipeline.validation import ValidationError, validate_stage_outputs


def test_missing_artifact_raises(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        validate_stage_outputs(tmp_path, StageName.DATA_UNDERSTANDING, strict=True)
