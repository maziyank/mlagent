"""Deep agent sub-agent definitions for pipeline stages."""

from __future__ import annotations

from mlagent.agents.prompts import (
    DATA_PREPARATION_PROMPT,
    DATA_UNDERSTANDING_PROMPT,
    EVALUATION_PROMPT,
    MODELING_PROMPT,
)


def build_subagent_specs(tools: list) -> list[dict]:
    """Declarative sub-agent specs for create_deep_agent."""
    return [
        {
            "name": "data-understanding-agent",
            "description": (
                "Generates and refines exploratory data analysis (EDA) code. "
                "Use for initial data profiling, visualizations, and eda_summary.json."
            ),
            "system_prompt": DATA_UNDERSTANDING_PROMPT,
            "tools": tools,
        },
        {
            "name": "data-preparation-agent",
            "description": (
                "Handles data cleaning, preprocessing, and feature engineering. "
                "Use after EDA is complete."
            ),
            "system_prompt": DATA_PREPARATION_PROMPT,
            "tools": tools,
        },
        {
            "name": "modeling-agent",
            "description": (
                "Implements model selection, training, and serialization. "
                "Use after train/test splits exist."
            ),
            "system_prompt": MODELING_PROMPT,
            "tools": tools,
        },
        {
            "name": "evaluation-agent",
            "description": (
                "Produces performance validation, benchmarking, and final metrics. "
                "Use after model.pkl exists."
            ),
            "system_prompt": EVALUATION_PROMPT,
            "tools": tools,
        },
    ]
