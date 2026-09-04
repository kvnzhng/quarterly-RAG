"""`rag doctor`: can the configured setup run the pipeline?

Pure functions over the `LLM` and `Embedder` protocols so every check is unit-testable
with fakes. The CLI only renders the results.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from quarterly_rag.config import Settings
from quarterly_rag.errors import ModelServerError
from quarterly_rag.generation.base import LLM, ChatMessage
from quarterly_rag.generation.llm import build_llm
from quarterly_rag.indexing.base import Embedder
from quarterly_rag.indexing.embedder import build_embedder

Status = Literal["ok", "warn", "fail"]

# HTTP statuses that mean "this server has no model listing", not "this server is broken".
_NO_LISTING = {404, 405, 501}


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    detail: str
    latency_ms: float | None = None


def _timed[T](fn: Callable[[], T]) -> tuple[T, float]:
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1000


def check_data_dirs(settings: Settings) -> CheckResult:
    for directory in (
        settings.raw_dir,
        settings.processed_dir,
        settings.index_dir,
        settings.eval_dir,
    ):
        probe = directory / f".doctor-{uuid.uuid4().hex}"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe.write_text("ok")
            probe.unlink()
        except OSError as exc:
            return CheckResult("data dirs writable", "fail", f"{directory}: {exc}")
    return CheckResult(
        "data dirs writable",
        "ok",
        f"{settings.data_dir.resolve()}: raw, processed, indexes, eval",
    )


def check_model_listed(
    name: str, list_models: Callable[[], list[str]], model: str, *, hint: str
) -> CheckResult:
    """Three outcomes: listed (ok), server cannot list (warn), listed but absent (fail).

    The listing is a hint; the round-trip checks are what prove the model works.
    """
    try:
        models, ms = _timed(list_models)
    except ModelServerError as exc:
        if exc.status_code in _NO_LISTING:
            return CheckResult(name, "warn", f"server does not list models, skipped ({exc})")
        return CheckResult(name, "fail", f"cannot list models: {exc}")
    if not models:
        return CheckResult(name, "warn", "server lists no models", ms)
    # Ollama stores an untagged pull as `<model>:latest`.
    if model in models or f"{model}:latest" in models:
        return CheckResult(name, "ok", f"{model} listed ({len(models)} models available)", ms)
    preview = ", ".join(models[:5]) + (", ..." if len(models) > 5 else "")
    return CheckResult(name, "fail", f"{model} not in server list [{preview}]. {hint}", ms)


def check_chat(llm: LLM) -> CheckResult:
    prompt = [ChatMessage(role="user", content="Reply with exactly one word: pong")]
    # Generous cap: on thinking-mode models (current Claude, qwen3) reasoning tokens count
    # against max_tokens, and a tight cap would fail a healthy endpoint with an empty reply.
    try:
        response, ms = _timed(lambda: llm.chat(prompt, max_tokens=256))
    except ModelServerError as exc:
        return CheckResult("chat round-trip", "fail", str(exc))
    reply = response.text.strip()
    if not reply:
        return CheckResult(
            "chat round-trip", "fail", f"empty reply (stop_reason={response.stop_reason})", ms
        )
    return CheckResult("chat round-trip", "ok", f"{llm.label} replied {reply[:40]!r}", ms)


def check_embed(embedder: Embedder) -> CheckResult:
    try:
        vectors, ms = _timed(lambda: embedder.embed(["Apple reported quarterly revenue."]))
    except ModelServerError as exc:
        return CheckResult("embedding round-trip", "fail", str(exc))
    if len(vectors) != 1 or not vectors[0]:
        return CheckResult("embedding round-trip", "fail", "no vector returned", ms)
    return CheckResult(
        "embedding round-trip",
        "ok",
        f"{embedder.label} returned a {len(vectors[0])}-dim vector",
        ms,
    )


def _with_path_hint(result: CheckResult, provider: str, base_url: str, env_var: str) -> CheckResult:
    """A 404 from an OpenAI-compatible URL without `/v1` is almost always the missing suffix."""
    if (
        result.status == "ok"
        or provider != "openai_compatible"
        or "HTTP 404" not in result.detail
        or base_url.rstrip("/").endswith("/v1")
    ):
        return result
    hint = (
        f"Hint: {env_var} has no /v1 suffix; OpenAI-compatible endpoints usually live at "
        f"{base_url.rstrip('/')}/v1 (Ollama's own API is at the root)."
    )
    return CheckResult(result.name, result.status, f"{result.detail} {hint}", result.latency_ms)


def _pull_hint(provider: str, base_url: str, model: str, env_var: str) -> str:
    if provider == "openai_compatible" and ":11434" in base_url:
        return f"Run `ollama pull {model}` on the server (or `make models` with OLLAMA_HOST set)."
    return f"Check {env_var} against the server's model list."


def run_doctor(
    settings: Settings,
    *,
    llm_factory: Callable[[Settings], LLM] | None = None,
    embedder_factory: Callable[[Settings], Embedder] | None = None,
) -> list[CheckResult]:
    # Resolved at call time so tests can monkeypatch the module-level factories.
    llm_factory = llm_factory or build_llm
    embedder_factory = embedder_factory or build_embedder
    results = [check_data_dirs(settings)]

    try:
        llm = llm_factory(settings)
    except (ModelServerError, NotImplementedError, ValueError) as exc:
        results.append(CheckResult("chat model client", "fail", str(exc)))
    else:
        hint = _pull_hint(
            settings.llm_provider, settings.llm_base_url, settings.llm_model, "LLM_MODEL"
        )
        for result in (
            check_model_listed("chat model listed", llm.list_models, settings.llm_model, hint=hint),
            check_chat(llm),
        ):
            results.append(
                _with_path_hint(
                    result, settings.llm_provider, settings.llm_base_url, "LLM_BASE_URL"
                )
            )

    try:
        embedder = embedder_factory(settings)
    except (ModelServerError, NotImplementedError, ValueError) as exc:
        results.append(CheckResult("embedding client", "fail", str(exc)))
    else:
        hint = _pull_hint(
            settings.embed_provider, settings.embed_base_url, settings.embed_model, "EMBED_MODEL"
        )
        for result in (
            check_model_listed(
                "embedding model listed", embedder.list_models, settings.embed_model, hint=hint
            ),
            check_embed(embedder),
        ):
            results.append(
                _with_path_hint(
                    result, settings.embed_provider, settings.embed_base_url, "EMBED_BASE_URL"
                )
            )
    return results


def failed(results: list[CheckResult]) -> list[CheckResult]:
    return [r for r in results if r.status == "fail"]
