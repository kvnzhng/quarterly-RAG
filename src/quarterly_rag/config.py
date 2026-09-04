"""Single source of configuration. Loaded from environment and `.env`.

The model provider is the user's choice. Defaults keep everything local and free
(Ollama on this machine), but any OpenAI-compatible server on the network or a
hosted API with a token works by changing `.env` only. See ADR-005.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["openai_compatible", "anthropic"]
EmbedProvider = Literal["openai_compatible", "sentence_transformers"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Chat model -------------------------------------------------------------
    # `openai_compatible` covers Ollama, vLLM, LM Studio, llama.cpp server, OpenRouter,
    # OpenAI and friends: one client per wire protocol, not per vendor.
    llm_provider: LLMProvider = "openai_compatible"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = Field(
        default="ollama", description="Ollama ignores it; hosted APIs need one."
    )
    llm_model: str = "llama3.1:8b"

    # --- Embeddings -------------------------------------------------------------
    # Separate from the chat model: Anthropic serves no embeddings, and pairing a hosted
    # chat model with local embeddings is the sensible default (the index is rebuilt often).
    embed_provider: EmbedProvider = "openai_compatible"
    embed_base_url: str = "http://localhost:11434/v1"
    embed_api_key: str = "ollama"
    embed_model: str = "nomic-embed-text"

    # --- Requests -----------------------------------------------------------------
    request_timeout_s: float = Field(
        default=120.0,
        description="Per-request timeout. A cold local model can take a minute to load.",
    )

    # --- Chunking ---------------------------------------------------------------
    # Sizes are whitespace words, not model tokens: a word here averages 6.4 characters on
    # this corpus and a BPE tokenizer splits figures like `$109,417` into several tokens.
    chunk_words: int = 350
    chunk_overlap_words: int = 60

    # --- Vector store -----------------------------------------------------------
    vector_store: Literal["chroma", "faiss"] = "chroma"

    # --- SEC EDGAR requires a descriptive User-Agent with a contact address ---------
    edgar_user_agent: str = Field(
        default="quarterly-RAG unknown unknown@example.com",
        description="Format: '<app> <name> <email>'.",
    )

    # --- Observability ----------------------------------------------------------
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # --- Paths ------------------------------------------------------------------
    data_dir: Path = Path("data")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"

    @property
    def chunk_dir(self) -> Path:
        return self.data_dir / "chunks"

    def model_label(self) -> str:
        """Provider + model, for eval reports and traces (e.g. 'openai_compatible/llama3.1:8b')."""
        return f"{self.llm_provider}/{self.llm_model}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
