from __future__ import annotations

import pytest

from quarterly_rag.retrieval.base import Retriever
from quarterly_rag.retrieval.bm25 import BM25Retriever, tokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Currency, parentheses and percent signs all fall away, so the same figure
        # written three ways becomes the same token.
        ("$109,417 million", ["109,417", "million"]),
        ("(109,417)", ["109,417"]),
        ("109,417", ["109,417"]),
        ("46.9%", ["46.9"]),
        # A period label must survive as one token or a question can never match it.
        ("Q3 FY2026", ["q3", "fy2026"]),
        ("Total net sales | 109,417 | 94,036", ["total", "net", "sales", "109,417", "94,036"]),
    ],
)
def test_the_tokeniser_keeps_figures_and_period_labels_whole(text, expected) -> None:
    assert tokenize(text) == expected


@pytest.fixture
def corpus(make_chunk):
    """Six documents. BM25 weights a term by how rare it is, and on a corpus of two or
    three every term is either in half the documents or all of them, which makes every
    weight zero. A test on a toy corpus would measure the formula's edge case."""
    return [
        make_chunk("a:1-2", "(In millions)\nTotal net sales | 109,417 | 94,036"),
        make_chunk("b:1-2", "Americas net sales increased during the third quarter."),
        make_chunk("d:1-2", "Research and development | 34,550 | 31,370"),
        make_chunk("e:1-2", "The Company had approximately 166,000 employees."),
        make_chunk(
            "c:1-2",
            "Revenue | 96,221 | 46,743",
            ticker="NVDA",
            company="NVIDIA CORP",
            period_label="FY2027 Q2",
        ),
        make_chunk(
            "f:1-2",
            "Data Center | 193,737 | 115,186",
            ticker="NVDA",
            company="NVIDIA CORP",
            period_label="FY2026",
        ),
    ]


def test_an_exact_figure_finds_its_passage(corpus) -> None:
    retriever = BM25Retriever(corpus)
    assert isinstance(retriever, Retriever)
    assert retriever.name == "bm25"
    (top, *_) = retriever.retrieve("What were total net sales of 109,417?", k=3)
    assert top.chunk.chunk_id == "a:1-2"
    assert top.retriever == "bm25"
    assert top.rank == 1
    assert 0 < top.score <= 1.0


def test_the_period_label_is_searchable_because_the_header_is_indexed(corpus) -> None:
    # The chunk text says nothing about FY2027; the provenance header does.
    results = BM25Retriever(corpus).retrieve(
        "Nvidia revenue in the second quarter of fiscal 2027", k=1
    )
    assert results[0].chunk.ticker == "NVDA"


def test_query_expansion_can_be_switched_off(corpus) -> None:
    # Without expansion the question never produces the token FY2027 that the header holds.
    plain = BM25Retriever(corpus, expand_query=False).retrieve("second quarter of fiscal 2027", k=3)
    expanded = BM25Retriever(corpus).retrieve("second quarter of fiscal 2027", k=3)
    assert not plain or plain[0].chunk.period_label != "FY2027 Q2"
    assert expanded[0].chunk.period_label == "FY2027 Q2"


def test_a_filter_removes_candidates(corpus) -> None:
    results = BM25Retriever(corpus).retrieve("net sales revenue", k=5, where={"ticker": "NVDA"})
    assert {r.chunk.ticker for r in results} == {"NVDA"}


def test_zero_scoring_passages_are_not_returned(corpus) -> None:
    assert BM25Retriever(corpus).retrieve("zebra xylophone", k=5) == []
    assert BM25Retriever(corpus).retrieve("   ") == []
    assert BM25Retriever([]).retrieve("anything") == []
