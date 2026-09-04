from __future__ import annotations

from quarterly_rag.config import Settings
from quarterly_rag.generation.anthropic_api import AnthropicLLM
from quarterly_rag.generation.llm import build_llm
from quarterly_rag.generation.openai_compatible import OpenAICompatibleLLM


def test_default_is_openai_compatible(settings: Settings) -> None:
    llm = build_llm(settings)
    assert isinstance(llm, OpenAICompatibleLLM)
    assert llm.label == "openai_compatible/llama3.1:8b"


def test_anthropic_provider(settings: Settings) -> None:
    configured = settings.model_copy(
        update={"llm_provider": "anthropic", "llm_api_key": "sk-test", "llm_model": "claude-opus-5"}
    )
    llm = build_llm(configured)
    assert isinstance(llm, AnthropicLLM)
    assert llm.label == "anthropic/claude-opus-5"
