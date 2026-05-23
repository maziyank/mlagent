"""LangSmith tracing for deep-agent pipeline runs."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from mlagent.config import Settings


def configure_langsmith(settings: Settings) -> bool:
    """Apply LangSmith environment variables from settings.

    Returns True when tracing is enabled and an API key is available.
    """
    if not settings.langsmith_tracing:
        return False

    api_key = settings.langsmith_api_key or os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key

    project = settings.langsmith_project
    if project:
        os.environ["LANGSMITH_PROJECT"] = project
        os.environ["LANGCHAIN_PROJECT"] = project

    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

    return True


def is_langsmith_active(settings: Settings) -> bool:
    """Whether LangSmith tracing is configured and will emit traces."""
    if not settings.langsmith_tracing:
        return False
    return bool(settings.langsmith_api_key or os.environ.get("LANGSMITH_API_KEY"))


def build_agent_run_config(
    *,
    run_id: str,
    stage: str,
    purpose: str,
    model: str,
    dataset: str | None = None,
    pipeline: str | None = None,
) -> dict[str, Any]:
    """RunnableConfig fields for a single agent invocation."""
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "stage": stage,
        "purpose": purpose,
        "model": model,
    }
    if dataset is not None:
        metadata["dataset"] = dataset
    if pipeline is not None:
        metadata["pipeline"] = pipeline

    tags = [
        "mlagent",
        f"stage:{stage}",
        f"purpose:{purpose}",
        f"run:{run_id}",
    ]
    if dataset:
        tags.append(f"dataset:{dataset}")
    if pipeline:
        tags.append(f"pipeline:{pipeline}")

    return {
        "run_name": f"mlagent/{stage}/{purpose}",
        "tags": tags,
        "metadata": metadata,
    }


@contextmanager
def agent_tracing_context(
    settings: Settings,
    *,
    run_id: str,
    dataset: str | None = None,
    pipeline: str | None = None,
) -> Iterator[None]:
    """Optional parent tracing context for an entire pipeline run."""
    if not is_langsmith_active(settings):
        yield
        return

    import langsmith as ls

    metadata: dict[str, Any] = {"run_id": run_id}
    tags = ["mlagent", f"run:{run_id}"]
    if dataset:
        metadata["dataset"] = dataset
        tags.append(f"dataset:{dataset}")
    if pipeline:
        metadata["pipeline"] = pipeline
        tags.append(f"pipeline:{pipeline}")

    with ls.tracing_context(
        project_name=settings.langsmith_project,
        enabled=True,
        tags=tags,
        metadata=metadata,
    ):
        yield
