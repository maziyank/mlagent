"""LangSmith observability configuration tests."""

from __future__ import annotations

import os
from unittest.mock import patch

from mlagent.config import Settings
from mlagent.observability.langsmith import (
    build_agent_run_config,
    configure_langsmith,
    is_langsmith_active,
)


def test_build_agent_run_config_includes_run_context() -> None:
    config = build_agent_run_config(
        run_id="abc123",
        stage="modeling",
        purpose="refine",
        model="openrouter:openai/gpt-4o-mini",
        dataset="iris",
        pipeline="binary_classification",
    )
    assert config["run_name"] == "mlagent/modeling/refine"
    assert "run:abc123" in config["tags"]
    assert config["metadata"]["run_id"] == "abc123"
    assert config["metadata"]["dataset"] == "iris"
    assert config["metadata"]["pipeline"] == "binary_classification"


def test_configure_langsmith_sets_env_when_enabled() -> None:
    settings = Settings(
        langsmith_tracing=True,
        langsmith_api_key="test-key",
        langsmith_project="mlagent-test",
    )
    with patch.dict(os.environ, {}, clear=True):
        assert configure_langsmith(settings) is True
        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGSMITH_API_KEY"] == "test-key"
        assert os.environ["LANGSMITH_PROJECT"] == "mlagent-test"


def test_configure_langsmith_disabled_without_api_key() -> None:
    settings = Settings(langsmith_tracing=True, langsmith_api_key=None)
    with patch.dict(os.environ, {}, clear=True):
        assert configure_langsmith(settings) is False
        assert is_langsmith_active(settings) is False


def test_is_langsmith_active_reads_env_api_key() -> None:
    settings = Settings(langsmith_tracing=True, langsmith_api_key=None)
    with patch.dict(os.environ, {"LANGSMITH_API_KEY": "from-env"}, clear=True):
        assert is_langsmith_active(settings) is True
