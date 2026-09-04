"""Chat-model interface. Providers implement `LLM`; everything else imports only this."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str


class ChatResponse(BaseModel):
    text: str
    model: str
    stop_reason: str | None = None
    """Provider-reported reason (`stop`, `end_turn`, `max_tokens`, `refusal`, ...).

    A provider-side refusal is not the same thing as this project's refusal gate
    (RAG-011); it is carried through so the two can be told apart.
    """
    input_tokens: int | None = None
    output_tokens: int | None = None


@runtime_checkable
class LLM(Protocol):
    @property
    def label(self) -> str:
        """`provider/model`, recorded with every eval number (ADR-005)."""
        ...

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ChatResponse: ...

    def list_models(self) -> list[str]:
        """Model ids the endpoint serves. Raises `ModelServerError` if it cannot say."""
        ...
