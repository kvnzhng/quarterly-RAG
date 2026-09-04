from __future__ import annotations

import pytest

from quarterly_rag.config import Settings
from quarterly_rag.indexing.embedder import build_embedder
from quarterly_rag.indexing.openai_compatible import OpenAICompatibleEmbedder


def test_default_is_openai_compatible(settings: Settings) -> None:
    embedder = build_embedder(settings)
    assert isinstance(embedder, OpenAICompatibleEmbedder)
    assert embedder.label == "openai_compatible/nomic-embed-text"


def test_sentence_transformers_not_yet_implemented(settings: Settings) -> None:
    configured = settings.model_copy(update={"embed_provider": "sentence_transformers"})
    with pytest.raises(NotImplementedError, match="RAG-006"):
        build_embedder(configured)
