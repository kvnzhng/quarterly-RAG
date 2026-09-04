"""Thin httpx wrapper for the OpenAI-compatible wire protocol.

Ollama, vLLM, LM Studio, llama.cpp server, OpenRouter, Groq and OpenAI itself all speak
it, so chat (generation layer) and embeddings (indexing layer) share this client: one
client per wire protocol, not per vendor (ADR-005).
"""

from __future__ import annotations

from typing import Any

import httpx

from quarterly_rag.errors import ModelServerError

CONNECT_TIMEOUT_S = 10.0


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        *,
        timeout_s: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=CONNECT_TIMEOUT_S),
            transport=transport,
        )

    def get_json(self, path: str) -> Any:
        return self._request("GET", path)

    def post_json(self, path: str, payload: dict[str, Any]) -> Any:
        return self._request("POST", path, json=payload)

    def list_models(self) -> list[str]:
        """Model ids from `GET /models`: pulled tags on Ollama, the catalogue on hosted APIs."""
        data = self.get_json("/models")
        items = data.get("data", []) if isinstance(data, dict) else []
        return sorted(str(item["id"]) for item in items if isinstance(item, dict) and "id" in item)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:  # connect refused, DNS, timeout, protocol errors
            raise ModelServerError(f"{method} {url}: {exc.__class__.__name__}: {exc}") from exc
        if response.is_error:
            snippet = response.text[:200].replace("\n", " ")
            raise ModelServerError(
                f"{method} {url} -> HTTP {response.status_code}: {snippet}",
                status_code=response.status_code,
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ModelServerError(f"{method} {url}: response is not JSON") from exc
