from __future__ import annotations

import os
from collections.abc import Sequence

import pytest
from typer.testing import CliRunner

from quarterly_rag import doctor as doctor_module
from quarterly_rag.cli import app
from quarterly_rag.config import Settings
from quarterly_rag.doctor import CheckResult, run_doctor
from quarterly_rag.errors import ModelServerError
from quarterly_rag.generation.base import ChatMessage, ChatResponse


class FakeLLM:
    label = "fake/llm"

    def __init__(
        self,
        models: Sequence[str] = ("llama3.1:8b", "nomic-embed-text:latest"),
        reply: str = "pong",
        list_error: Exception | None = None,
        chat_error: Exception | None = None,
    ) -> None:
        self.models = list(models)
        self.reply = reply
        self.list_error = list_error
        self.chat_error = chat_error

    def chat(self, messages: Sequence[ChatMessage], *, temperature=0.0, max_tokens=1024):
        if self.chat_error is not None:
            raise self.chat_error
        return ChatResponse(text=self.reply, model="fake", stop_reason="stop")

    def list_models(self) -> list[str]:
        if self.list_error is not None:
            raise self.list_error
        return self.models


class FakeEmbedder:
    label = "fake/embedder"

    def __init__(self, models: Sequence[str] = ("nomic-embed-text:latest",), dim: int = 8) -> None:
        self.models = list(models)
        self.dim = dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1] * self.dim for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * self.dim

    def list_models(self) -> list[str]:
        return self.models


def by_name(results: list[CheckResult]) -> dict[str, CheckResult]:
    return {r.name: r for r in results}


def run(settings: Settings, llm: FakeLLM | None = None, embedder: FakeEmbedder | None = None):
    return run_doctor(
        settings,
        llm_factory=lambda s: llm or FakeLLM(),
        embedder_factory=lambda s: embedder or FakeEmbedder(),
    )


def test_all_checks_pass(settings: Settings) -> None:
    results = by_name(run(settings))
    assert list(results) == [
        "data dirs writable",
        "chat model listed",
        "chat round-trip",
        "embedding model listed",
        "embedding round-trip",
    ]
    assert {r.status for r in results.values()} == {"ok"}
    assert "8-dim" in results["embedding round-trip"].detail
    assert results["chat round-trip"].latency_ms is not None
    assert (settings.data_dir / "indexes").is_dir()


def test_missing_model_fails_with_ollama_pull_hint(settings: Settings) -> None:
    results = by_name(run(settings, llm=FakeLLM(models=["something-else"])))
    listed = results["chat model listed"]
    assert listed.status == "fail"
    assert "llama3.1:8b not in server list" in listed.detail
    assert "ollama pull llama3.1:8b" in listed.detail
    assert results["chat round-trip"].status == "ok"  # still attempted


def test_untagged_model_matches_latest(settings: Settings) -> None:
    configured = settings.model_copy(update={"llm_model": "llama3.1"})
    results = by_name(run(configured, llm=FakeLLM(models=["llama3.1:latest"])))
    assert results["chat model listed"].status == "ok"


def test_server_without_listing_is_a_warning(settings: Settings) -> None:
    llm = FakeLLM(list_error=ModelServerError("GET /models -> HTTP 404", status_code=404))
    results = by_name(run(settings, llm=llm))
    assert results["chat model listed"].status == "warn"
    assert results["chat round-trip"].status == "ok"


def test_404_without_v1_suffix_gets_a_path_hint(settings: Settings) -> None:
    configured = settings.model_copy(update={"llm_base_url": "http://ai-server.local:11434/"})
    error = ModelServerError(
        "POST http://ai-server.local:11434/chat/completions -> HTTP 404", status_code=404
    )
    results = by_name(run(configured, llm=FakeLLM(list_error=error, chat_error=error)))
    assert results["chat model listed"].status == "warn"
    assert results["chat round-trip"].status == "fail"
    assert "http://ai-server.local:11434/v1" in results["chat round-trip"].detail
    assert "LLM_BASE_URL" in results["chat round-trip"].detail
    # With the suffix present, a 404 is reported as is.
    results = by_name(run(settings, llm=FakeLLM(chat_error=error)))
    assert "Hint" not in results["chat round-trip"].detail


def test_unreachable_server_fails_listing_and_chat(settings: Settings) -> None:
    error = ModelServerError("ConnectError: connection refused")
    results = by_name(run(settings, llm=FakeLLM(list_error=error, chat_error=error)))
    assert results["chat model listed"].status == "fail"
    assert results["chat round-trip"].status == "fail"
    assert "connection refused" in results["chat round-trip"].detail


def test_empty_reply_fails(settings: Settings) -> None:
    results = by_name(run(settings, llm=FakeLLM(reply="   ")))
    assert results["chat round-trip"].status == "fail"


def test_factory_error_becomes_failed_check(settings: Settings) -> None:
    def broken(_: Settings):
        raise NotImplementedError("sentence_transformers arrives with RAG-006")

    results = by_name(
        run_doctor(settings, llm_factory=lambda s: FakeLLM(), embedder_factory=broken)
    )
    assert results["embedding client"].status == "fail"
    assert "RAG-006" in results["embedding client"].detail
    assert results["chat round-trip"].status == "ok"
    assert "embedding round-trip" not in results


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
def test_unwritable_data_dir_fails(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True)
    settings.data_dir.chmod(0o500)
    try:
        results = by_name(run(settings))
    finally:
        settings.data_dir.chmod(0o700)
    assert results["data dirs writable"].status == "fail"


def test_cli_doctor_exit_codes(monkeypatch, settings: Settings) -> None:
    monkeypatch.setattr("quarterly_rag.cli.get_settings", lambda: settings)
    monkeypatch.setattr(doctor_module, "build_embedder", lambda s: FakeEmbedder())

    monkeypatch.setattr(doctor_module, "build_llm", lambda s: FakeLLM())
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0, result.stdout
    assert "all checks passed" in result.stdout
    assert "fake/llm" in result.stdout

    monkeypatch.setattr(doctor_module, "build_llm", lambda s: FakeLLM(models=["other"]))
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "1 check(s) failed" in result.stdout
