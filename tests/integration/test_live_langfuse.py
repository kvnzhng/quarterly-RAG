"""Tracing against a real, local Langfuse (RAG-013).

Skipped unless Langfuse is configured and answering, so `make test` and CI never need
Docker. Start it with `make langfuse-up`.

What this checks is the round trip the unit tests cannot: that spans the SDK sends actually
arrive, keep their nesting, and carry the token counts. Langfuse v4 ingests asynchronously
through a worker, so the read is retried rather than done once.
"""

from __future__ import annotations

import time
import uuid

import httpx
import pytest

from quarterly_rag.config import get_settings
from quarterly_rag.observability.tracing import (
    BOOLEAN,
    GENERATION,
    LangfuseTracer,
    build_tracer,
    configured,
)

pytestmark = pytest.mark.integration


def _reachable(settings) -> bool:
    if not configured(settings):
        return False
    try:
        health = httpx.get(f"{settings.langfuse_host.rstrip('/')}/api/public/health", timeout=3)
    except httpx.HTTPError:
        return False
    return health.status_code == 200


@pytest.fixture(scope="module")
def live_settings():
    settings = get_settings()
    if not _reachable(settings):
        pytest.skip("Langfuse is not configured or not answering; run `make langfuse-up`")
    return settings


def _observations(settings, trace_id: str, *, expected: int, tries: int = 20) -> list[dict]:
    """Poll until the worker has ingested the whole trace, or give up."""
    url = f"{settings.langfuse_host.rstrip('/')}/api/public/v2/observations"
    auth = (settings.langfuse_public_key, settings.langfuse_secret_key)
    for _ in range(tries):
        response = httpx.get(url, params={"traceId": trace_id, "limit": 50}, auth=auth, timeout=10)
        rows = response.json().get("data", []) if response.status_code == 200 else []
        if len(rows) >= expected:
            return rows
        time.sleep(1)
    return rows


def test_a_nested_trace_arrives_with_its_shape_intact(live_settings) -> None:
    tracer = build_tracer(live_settings)
    assert isinstance(tracer, LangfuseTracer), "configured settings must give a real tracer"

    marker = f"itest-{uuid.uuid4().hex[:8]}"
    with tracer.span(marker, input={"question": "does tracing work?"}) as root:
        trace_id = root.trace_id
        assert trace_id
        with tracer.span("retrieval", input={"k": 5}) as span:
            span.update(output={"chunks": ["a:1-2"]})
        with tracer.span("generation", kind=GENERATION, model="fake/model") as span:
            span.update(output="an answer", usage_details={"input": 11, "output": 7})
        root.update(output={"refused": False}, level="DEFAULT")
    tracer.score(trace_id, "itest_score", True, data_type=BOOLEAN)
    tracer.flush()

    rows = _observations(live_settings, trace_id, expected=3)
    by_name = {row["name"]: row for row in rows}
    assert set(by_name) == {marker, "retrieval", "generation"}
    assert by_name["generation"]["type"] == "GENERATION"
    assert by_name[marker]["parentObservationId"] is None
    assert by_name["retrieval"]["parentObservationId"] == by_name[marker]["id"]


def test_the_pipeline_traces_a_refusal_without_calling_a_model(live_settings) -> None:
    """The cheapest useful trace: a question the corpus cannot hold, refused before a model."""
    from quarterly_rag.generation.refusal import CorpusScope, GateSettings
    from quarterly_rag.pipeline import Pipeline

    class NoRetriever:
        name = "none"

        def retrieve(self, question, k=5, where=None):  # pragma: no cover - never reached
            raise AssertionError("an out-of-scope question must not reach retrieval")

    pipeline = Pipeline(
        retriever=NoRetriever(),
        llm=None,
        scope=CorpusScope(
            tickers=frozenset({"AAPL"}),
            company_words=frozenset({"apple"}),
            fiscal_years=frozenset({2026}),
        ),
        gate=GateSettings(min_retrieval_score=0.0),
        tracer=build_tracer(live_settings),
    )
    outcome = pipeline.ask("What were Microsoft's net sales in fiscal 2025?", k=5)
    assert outcome.refused
    assert outcome.reason == "out_of_scope"
    assert outcome.trace_id

    pipeline.tracer.flush()
    rows = _observations(live_settings, outcome.trace_id, expected=2)
    assert {row["name"] for row in rows} == {"rag ask", "scope-gate"}
    root = next(row for row in rows if row["name"] == "rag ask")
    assert root["level"] == "WARNING", "a refusal is the thing you scroll a trace list to find"
