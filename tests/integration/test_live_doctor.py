"""Runs `rag doctor` against the endpoints in `.env`. Deselected by default (`make test-all`)."""

from __future__ import annotations

import pytest

from quarterly_rag.config import get_settings
from quarterly_rag.doctor import failed, run_doctor

pytestmark = pytest.mark.integration


def test_live_doctor_passes() -> None:
    results = run_doctor(get_settings())
    failures = failed(results)
    assert not failures, "\n".join(f"{r.name}: {r.detail}" for r in failures)
