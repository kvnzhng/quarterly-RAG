"""The one place that turns settings into an `LLM` (ADR-005).

Tracing (RAG-013) wraps the client here so every layer gets it for free.
"""

from __future__ import annotations

from quarterly_rag.config import Settings
from quarterly_rag.generation.anthropic_api import AnthropicLLM
from quarterly_rag.generation.base import LLM
from quarterly_rag.generation.openai_compatible import OpenAICompatibleLLM


def build_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "openai_compatible":
        return OpenAICompatibleLLM(
            settings.llm_base_url,
            settings.llm_api_key,
            settings.llm_model,
            timeout_s=settings.request_timeout_s,
        )
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(
            settings.llm_api_key, settings.llm_model, timeout_s=settings.request_timeout_s
        )
    raise ValueError(f"unknown LLM_PROVIDER {settings.llm_provider!r}")
