from __future__ import annotations

import pytest

from quarterly_rag.indexing.embed_text import CONTEXT, RAW, context_header, embed_text


def test_raw_is_the_chunk_untouched(make_chunk) -> None:
    chunk = make_chunk("a:1-2", "Total net sales | 109,417 | 94,036")
    assert embed_text(chunk, RAW) == chunk.text
    assert embed_text(chunk) == chunk.text  # raw is the default


def test_context_prepends_company_period_and_section(make_chunk) -> None:
    chunk = make_chunk("a:1-2", "Total net sales | 109,417 | 94,036")
    text = embed_text(chunk, CONTEXT)

    # The chunk's own text names neither the company nor the quarter; the header does.
    assert "Apple" not in chunk.text and "FY2026 Q3" not in chunk.text
    assert text.startswith(context_header(chunk))
    assert text.endswith(chunk.text)
    for expected in ("Apple Inc.", "AAPL", "10-Q", "FY2026 Q3", "Part I.Item 1"):
        assert expected in text


def test_unknown_variant_is_rejected(make_chunk) -> None:
    with pytest.raises(ValueError, match="unknown embed variant"):
        embed_text(make_chunk("a:1-2", "x"), "semantic")
