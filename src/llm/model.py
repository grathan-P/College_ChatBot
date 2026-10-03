
"""LLM provider initialization."""

import os

from langchain_openrouter import ChatOpenRouter

from src.llm.config import LLMConfig


def create_chat_model(config: LLMConfig) -> ChatOpenRouter:
    """Create the configured chat model.

    Args:
        config: LLM configuration containing provider and model settings.

    Returns:
        Configured OpenRouter chat model.

    Raises:
        ValueError: If the provider is unsupported or the API key is missing.
    """
    if config.provider.lower() != "openrouter":
        raise ValueError(
            f"Unsupported LLM provider: {config.provider}"
        )

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not configured."
        )

    return ChatOpenRouter(
        model=config.model_name,
        api_key=api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
