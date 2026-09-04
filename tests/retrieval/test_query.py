from __future__ import annotations

import pytest

from quarterly_rag.retrieval.query import QueryFacets, expand, parse_facets


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "What were Apple's total net sales in the third quarter of fiscal 2026?",
            QueryFacets(("AAPL",), 2026, 3),
        ),
        ("What was Nvidia's revenue in Q2 FY2027?", QueryFacets(("NVDA",), 2027, 2)),
        ("Apple's effective tax rate in fiscal 2025?", QueryFacets(("AAPL",), 2025, None)),
        ("How much did NVDA spend on R and D in FY2026?", QueryFacets(("NVDA",), 2026, None)),
        ("How does Apple compare with Nvidia?", QueryFacets(("AAPL", "NVDA"), None, None)),
        ("What is the weather today?", QueryFacets()),
    ],
)
def test_a_question_is_read_for_company_and_period(question, expected) -> None:
    assert parse_facets(question) == expected


def test_period_label_matches_how_the_corpus_spells_it() -> None:
    assert parse_facets("third quarter of fiscal 2026").period_label == "FY2026 Q3"
    assert parse_facets("fiscal 2025").period_label == "FY2025"
    assert parse_facets("no period here").period_label is None


def test_a_filter_is_only_offered_for_one_named_company() -> None:
    assert parse_facets("Apple's net sales?").as_filter() == {"ticker": "AAPL"}
    # Two companies: the answer may need both, so no filter.
    assert parse_facets("Apple versus Nvidia").as_filter() is None
    assert parse_facets("total revenue?").as_filter() is None


def test_a_period_never_becomes_a_filter() -> None:
    # A filing quotes prior years for comparison, so filtering on the period would
    # discard the passage that answers a cross-period question.
    facets = parse_facets("Apple's net sales in fiscal 2025?")
    assert facets.fiscal_year == 2025
    assert facets.as_filter() == {"ticker": "AAPL"}


def test_expansion_adds_the_corpus_spelling_to_the_question() -> None:
    expanded = expand("What were Apple's total net sales in the third quarter of fiscal 2026?")
    assert "FY2026" in expanded and "Q3" in expanded and "AAPL" in expanded
    assert expanded.startswith("What were Apple's")
    assert expand("What is the weather today?") == "What is the weather today?"
