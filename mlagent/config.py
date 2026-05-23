"""Application configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM (OpenRouter via deepagents provider profile)
    mlagent_model: str = Field(
        default="openrouter:openai/gpt-4o-mini",
        description="Model for deep agents (provider:model), e.g. openrouter:anthropic/claude-sonnet-4",
    )
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")

    # Pipeline defaults
    workspace_root: Path = Field(default=Path(".mlagent_runs"))
    max_iterations_per_stage: int = 5
    min_accuracy: float = 0.70
    sandbox_timeout_seconds: int = 300
    execution_mode: Literal["agent", "template"] = "template"
    llm_log_enabled: bool = True

    # Continuous optimization (modeling + evaluation)
    optimization_enabled: bool = Field(default=True, validation_alias="MLAGENT_OPTIMIZATION_ENABLED")
    optimization_patience: int = Field(default=2, validation_alias="MLAGENT_OPTIMIZATION_PATIENCE")
    optimization_min_improvement: float = Field(
        default=0.001, validation_alias="MLAGENT_OPTIMIZATION_MIN_IMPROVEMENT"
    )
    optimization_stages: str = Field(
        default="modeling,evaluation", validation_alias="MLAGENT_OPTIMIZATION_STAGES"
    )

    # Retry
    max_execution_retries: int = 3

    def optimization_stage_set(self) -> set[str]:
        return {s.strip() for s in self.optimization_stages.split(",") if s.strip()}


def get_settings() -> Settings:
    return Settings()
