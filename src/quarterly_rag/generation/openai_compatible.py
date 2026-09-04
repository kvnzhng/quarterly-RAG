"""`LLM` over the OpenAI-compatible chat completions endpoint."""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from quarterly_rag.errors import ModelServerError
from quarterly_rag.generation.base import ChatMessage, ChatResponse
from quarterly_rag.openai_compatible import OpenAICompatibleClient


class OpenAICompatibleLLM:
    provider = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout_s: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self._client = OpenAICompatibleClient(
            base_url, api_key, timeout_s=timeout_s, transport=transport
        )

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = self._client.post_json("/chat/completions", payload)
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelServerError(f"unexpected chat response shape: {str(data)[:200]}") from exc
        usage = data.get("usage") or {}
        return ChatResponse(
            text=text,
            model=str(data.get("model") or self.model),
            stop_reason=choice.get("finish_reason"),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    def list_models(self) -> list[str]:
        return self._client.list_models()
