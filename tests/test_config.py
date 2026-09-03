from __future__ import annotations

from typer.testing import CliRunner

from quarterly_rag import __version__
from quarterly_rag.cli import app
from quarterly_rag.config import Settings


def test_settings_defaults(settings: Settings) -> None:
    assert settings.vector_store == "chroma"
    assert settings.raw_dir == settings.data_dir / "raw"
    assert settings.index_dir.name == "indexes"


def test_settings_from_env(monkeypatch) -> None:
    monkeypatch.setenv("VECTOR_STORE", "faiss")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b")
    s = Settings(_env_file=None)
    assert s.vector_store == "faiss"
    assert s.llm_model == "qwen2.5:7b"


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout
