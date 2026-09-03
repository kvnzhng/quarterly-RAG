from __future__ import annotations

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from quarterly_rag import __version__
from quarterly_rag.cli import app
from quarterly_rag.config import Settings


def test_defaults_are_local_and_free(settings: Settings) -> None:
    assert settings.llm_provider == "openai_compatible"
    assert settings.llm_base_url.startswith("http://localhost")
    assert settings.embed_provider == "openai_compatible"
    assert settings.vector_store == "chroma"
    assert settings.raw_dir == settings.data_dir / "raw"
    assert settings.index_dir.name == "indexes"


def test_provider_from_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-opus-5")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("EMBED_BASE_URL", "http://ai-server.local:11434/v1")
    s = Settings(_env_file=None)
    assert s.llm_provider == "anthropic"
    assert s.model_label() == "anthropic/claude-opus-5"
    assert s.embed_base_url == "http://ai-server.local:11434/v1"


def test_unknown_provider_rejected(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "carrier-pigeon")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_cli_config_redacts_keys(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-very-secret")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "lf-very-secret")
    result = CliRunner().invoke(app, ["config"])
    assert result.exit_code == 0
    assert "very-secret" not in result.stdout
