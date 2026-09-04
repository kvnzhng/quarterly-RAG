"""`LLM` over the Anthropic Messages API, through the official SDK.

The SDK owns retries, typed errors and keeps pace with API changes (adaptive thinking,
removed sampling parameters); a hand-rolled client would need re-verifying on every one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import anthropic

from quarterly_rag.errors import ModelServerError
from quarterly_rag.generation.base import ChatMessage, ChatResponse


class AnthropicLLM:
    provider = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout_s: float = 120.0,
        client: Any | None = None,
    ) -> None:
        self.model = model
        if client is None:
            try:
                # An empty key lets the SDK resolve ANTHROPIC_API_KEY or an `ant auth` profile.
                client = anthropic.Anthropic(api_key=api_key or None, timeout=timeout_s)
            except anthropic.AnthropicError as exc:
                raise ModelServerError(f"anthropic client: {exc}") from exc
        self._client = client

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
        # `temperature` is accepted for protocol parity and deliberately not sent: current
        # Claude models reject sampling parameters and think adaptively by default.
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system or anthropic.NOT_GIVEN,
                messages=turns,
            )
        except anthropic.APIStatusError as exc:
            raise ModelServerError(
                f"anthropic: HTTP {exc.status_code}: {exc.message}", status_code=exc.status_code
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ModelServerError(f"anthropic: {exc.__class__.__name__}: {exc}") from exc
        text = "".join(block.text for block in response.content if block.type == "text")
        return ChatResponse(
            text=text,
            model=response.model,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def list_models(self) -> list[str]:
        try:
            return sorted(model.id for model in self._client.models.list())
        except anthropic.APIStatusError as exc:
            raise ModelServerError(
                f"anthropic: HTTP {exc.status_code}: {exc.message}", status_code=exc.status_code
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ModelServerError(f"anthropic: {exc.__class__.__name__}: {exc}") from exc
