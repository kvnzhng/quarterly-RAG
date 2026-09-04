"""`Embedder` over the OpenAI-compatible embeddings endpoint."""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from quarterly_rag.errors import ModelServerError
from quarterly_rag.openai_compatible import OpenAICompatibleClient


class OpenAICompatibleEmbedder:
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

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        data = self._client.post_json("/embeddings", {"model": self.model, "input": list(texts)})
        try:
            items = sorted(data["data"], key=lambda item: item["index"])
            vectors = [[float(x) for x in item["embedding"]] for item in items]
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelServerError(
                f"unexpected embeddings response shape: {str(data)[:200]}"
            ) from exc
        if len(vectors) != len(texts):
            raise ModelServerError(f"asked for {len(texts)} embeddings, got {len(vectors)}")
        return vectors

    def list_models(self) -> list[str]:
        return self._client.list_models()
