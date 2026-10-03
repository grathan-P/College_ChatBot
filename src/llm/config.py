
"""Configuration for the LLM module."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    """Configuration values used by the LLM provider."""

    provider: str = "openrouter"
    model_name: str = "openrouter/free"
    temperature: float = 0.1
    max_tokens: int = 700

    @classmethod
    def from_environment(cls) -> "LLMConfig":
        """Create configuration using environment variables when provided."""
        return cls(
            provider=os.getenv("LLM_PROVIDER", "openrouter"),
            model_name=os.getenv("LLM_MODEL", "openrouter/free"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "700")),
        )
