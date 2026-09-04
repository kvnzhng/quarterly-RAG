"""The one place that turns settings into an `Embedder` (ADR-005)."""

from __future__ import annotations

from quarterly_rag.config import Settings
from quarterly_rag.indexing.base import Embedder
from quarterly_rag.indexing.openai_compatible import OpenAICompatibleEmbedder


def build_embedder(settings: Settings) -> Embedder:
    if settings.embed_provider == "openai_compatible":
        return OpenAICompatibleEmbedder(
            settings.embed_base_url,
            settings.embed_api_key,
            settings.embed_model,
            timeout_s=settings.request_timeout_s,
        )
    if settings.embed_provider == "sentence_transformers":
        raise NotImplementedError(
            "EMBED_PROVIDER=sentence_transformers arrives with RAG-006; "
            "use openai_compatible until then"
        )
    raise ValueError(f"unknown EMBED_PROVIDER {settings.embed_provider!r}")
