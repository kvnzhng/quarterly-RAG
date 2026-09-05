"""Tracing a question through the pipeline, or not tracing it at all (RAG-013).

Two rules shape this module.

**Tracing must never change an answer.** Every call into the Langfuse SDK is wrapped: a
server that is down, keys that are wrong, a value that will not serialise, none of them may
turn a working `rag ask` into a traceback. The cost of that promise is that a broken tracer
is quiet, so `rag doctor` checks Langfuse explicitly rather than leaving it to be noticed.

**Nothing here knows what a `Chunk` or an `Answer` is.** Spans take strings, numbers and
plain dicts, so `observability` imports from no other layer and the pipeline decides what is
worth recording. That also makes the whole thing testable with a fake and no server.

When Langfuse is not configured, `build_tracer` returns `NullTracer`, whose spans are real
objects that do nothing. The pipeline is then free of `if tracing:` branches, and the code
path with tracing off is the one the 365 unit tests exercise.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

from quarterly_rag.config import Settings

log = logging.getLogger(__name__)

SPAN = "span"
GENERATION = "generation"
RETRIEVER = "retriever"
"""Observation kinds this project uses. Langfuse accepts more; these three are the ones the
pipeline has, and naming them here keeps the string literals out of the pipeline."""

HEALTH_TIMEOUT_S = 2.0
"""How long to wait for the server before deciding it is not there.

Measured: with `LANGFUSE_HOST` pointing at a closed port, `rag ask` still answered but took
13.9 s against a 3.0 s baseline, because the exporter retries with backoff inside `flush()`.
One cheap probe up front turns eleven seconds of retrying into a tracer that does nothing."""

NUMERIC = "NUMERIC"
BOOLEAN = "BOOLEAN"
CATEGORICAL = "CATEGORICAL"


@runtime_checkable
class Span(Protocol):
    """One timed step. `trace_id` is empty when nothing is being recorded."""

    @property
    def trace_id(self) -> str: ...

    def update(self, **attributes: Any) -> None:
        """Record what the step produced. Called after the work, before the span closes."""
        ...


@runtime_checkable
class Tracer(Protocol):
    @property
    def enabled(self) -> bool: ...

    def span(self, name: str, *, kind: str = SPAN, **attributes: Any) -> Any:
        """A context manager yielding a `Span`. Nested calls nest in the trace."""
        ...

    def score(
        self,
        trace_id: str,
        name: str,
        value: float | bool | str,
        *,
        data_type: str = NUMERIC,
        comment: str | None = None,
    ) -> None:
        """Attach an eval result to a finished trace."""
        ...

    def flush(self) -> None:
        """Send anything buffered. A CLI process exits long before the SDK's own timer."""
        ...


class NullSpan:
    """A span that records nothing, so callers never branch on whether tracing is on."""

    trace_id = ""

    def update(self, **attributes: Any) -> None:
        return None


class NullTracer:
    """The default. Chosen whenever Langfuse is not fully configured."""

    enabled = False

    @contextmanager
    def span(self, name: str, *, kind: str = SPAN, **attributes: Any) -> Iterator[NullSpan]:
        yield NullSpan()

    def score(
        self,
        trace_id: str,
        name: str,
        value: float | bool | str,
        *,
        data_type: str = NUMERIC,
        comment: str | None = None,
    ) -> None:
        return None

    def flush(self) -> None:
        return None


class _LangfuseSpan:
    """Adapts one Langfuse observation to `Span`, swallowing anything it raises."""

    def __init__(self, observation: Any) -> None:
        self._observation = observation

    @property
    def trace_id(self) -> str:
        return str(getattr(self._observation, "trace_id", "") or "")

    def update(self, **attributes: Any) -> None:
        try:
            self._observation.update(**attributes)
        except Exception:  # pragma: no cover - defensive; the SDK is not supposed to raise
            log.debug("langfuse: span update failed", exc_info=True)


def _close(manager: Any, name: str, exc: BaseException | None) -> None:
    """End a span, whether the body succeeded or not. Failing to close is the tracer's
    problem and nobody else's, so it is the one thing swallowed here."""
    try:
        if exc is None:
            manager.__exit__(None, None, None)
        else:
            manager.__exit__(type(exc), exc, exc.__traceback__)
    except Exception:
        log.debug("langfuse: span %s failed to close", name, exc_info=True)


class LangfuseTracer:
    """Records to a Langfuse server. Never raises at the call site."""

    enabled = True

    def __init__(self, client: Any, *, tags: list[str] | None = None) -> None:
        self._client = client
        self._tags = tags or []

    @contextmanager
    def span(self, name: str, *, kind: str = SPAN, **attributes: Any) -> Iterator[Any]:
        try:
            manager = self._client.start_as_current_observation(
                name=name, as_type=kind, **attributes
            )
        except Exception:
            log.debug("langfuse: could not start span %s", name, exc_info=True)
            yield NullSpan()
            return
        try:
            observation = manager.__enter__()
        except Exception:
            log.debug("langfuse: could not enter span %s", name, exc_info=True)
            yield NullSpan()
            return
        # The body's exception must not be caught here. A `@contextmanager` that catches what
        # is thrown into its `yield` and does not re-raise swallows it for the caller, which
        # turned a model-server error into a silent `None` from `Pipeline.ask`. Only closing
        # the span is guarded, because only that is the tracer's own failure to have.
        try:
            yield _LangfuseSpan(observation)
        except BaseException as exc:
            _close(manager, name, exc)
            raise
        _close(manager, name, None)

    def score(
        self,
        trace_id: str,
        name: str,
        value: float | bool | str,
        *,
        data_type: str = NUMERIC,
        comment: str | None = None,
    ) -> None:
        if not trace_id:
            return
        try:
            self._client.create_score(
                trace_id=trace_id,
                name=name,
                value=value,
                data_type=data_type,
                comment=comment,
            )
        except Exception:
            log.debug("langfuse: could not score %s on %s", name, trace_id, exc_info=True)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception:
            log.debug("langfuse: flush failed", exc_info=True)


def configured(settings: Settings) -> bool:
    """Tracing is on only when there is somewhere to send it and something to send with."""
    return bool(
        settings.langfuse_host and settings.langfuse_public_key and settings.langfuse_secret_key
    )


def reachable(settings: Settings, timeout: float = HEALTH_TIMEOUT_S) -> bool:
    """Whether the configured server answers, cheaply and once."""
    import httpx

    try:
        response = httpx.get(
            f"{settings.langfuse_host.rstrip('/')}/api/public/health", timeout=timeout
        )
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def build_tracer(settings: Settings, *, tags: list[str] | None = None) -> Tracer:
    """The configured tracer, or one that does nothing. Never raises.

    The SDK is imported here rather than at module scope: it pulls in OpenTelemetry, and a
    `rag --help` that never traces anything should not pay for it.
    """
    if not configured(settings):
        return NullTracer()
    if not reachable(settings):
        log.debug("langfuse: %s did not answer; tracing is off", settings.langfuse_host)
        return NullTracer()
    try:
        from langfuse import Langfuse
    except ImportError:
        log.debug("langfuse: the SDK is not installed; tracing is off")
        return NullTracer()
    try:
        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_host,
            environment=settings.langfuse_environment,
        )
    except Exception:
        log.debug("langfuse: client could not be built; tracing is off", exc_info=True)
        return NullTracer()
    return LangfuseTracer(client, tags=tags)


def trace_metadata(settings: Settings, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The run facts every trace should carry, so a trace explains itself a month later.

    The model endpoint is deliberately absent: `llm_model` names what answered, and the
    address of the machine it answered on is nobody's business but the operator's.
    """
    metadata: dict[str, Any] = {
        "llm_model": settings.llm_model,
        "embed_model": settings.embed_model,
        "prompt_version": settings.answer_prompt_version,
        "chunk_strategy": settings.chunk_strategy,
        "retrieval_strategy": settings.retrieval_strategy,
        "vector_store": settings.vector_store,
    }
    if extra:
        metadata.update(extra)
    return metadata
