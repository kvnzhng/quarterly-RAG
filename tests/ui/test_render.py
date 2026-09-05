"""What the page shows, without starting Streamlit (RAG-014).

The highlighting is the part worth testing: it decides which numbers a reader is told were
checked, and it writes HTML into a page rendered with `unsafe_allow_html`.
"""

from __future__ import annotations

from quarterly_rag.ui.render import (
    MARK_OPEN,
    as_markdown,
    citation_label,
    figures_to_mark,
    highlight,
    refusal_headline,
    trace_url,
    verdict,
)

TABLE = "(In millions)\nTotal net sales | 109,417 | 94,036"


def test_only_figures_the_passage_states_are_marked() -> None:
    """The highlight runs the verifier's own check, so it means "found here", not "looks alike"."""
    answer = "Total net sales were $109,417 million, up from $94,036 million [c1]."
    assert figures_to_mark(answer, TABLE) == ["109,417", "94,036"]


def test_a_figure_the_passage_does_not_state_is_not_marked() -> None:
    answer = "Net sales rose $15,381 million [c1]."
    assert figures_to_mark(answer, TABLE) == []


def test_a_figure_written_at_another_scale_is_still_matched() -> None:
    """`$109.4 billion` and a table of millions are the same number."""
    assert figures_to_mark("Sales were $109.4 billion [c1].", TABLE) == ["109.4"]


def test_the_passage_is_escaped_before_it_is_marked() -> None:
    """A filing is full of `<` and `&`, and this goes into a page with unsafe_allow_html."""
    passage = "Net sales <b>109,417</b> & rising"
    rendered = highlight(passage, ["109,417"])
    assert "<b>" not in rendered
    assert "&lt;b&gt;" in rendered
    assert "&amp; rising" in rendered
    assert f"{MARK_OPEN}109,417</mark>" in rendered


def test_a_marked_figure_is_not_marked_again_inside_a_longer_one() -> None:
    rendered = highlight("109,417 and 417", ["109,417", "417"])
    assert rendered.count(MARK_OPEN) == 2
    assert f"{MARK_OPEN}109,417</mark>" in rendered


def test_nothing_to_mark_still_escapes() -> None:
    assert highlight("a < b", []) == "a &lt; b"


def test_the_verdict_leads_with_the_worst_thing_wrong() -> None:
    unresolvable = {"invalid_tags": ["c9"], "unsupported_sentences": ["x"], "fully_grounded": False}
    level, message = verdict(unresolvable)
    assert level == "error"
    assert "c9" in message

    uncited = {"invalid_tags": [], "unsupported_sentences": ["Net sales rose."]}
    assert verdict(uncited)[0] == "error"

    unchecked = {"unverified_derived": ["$15,381 million"]}
    level, message = verdict(unchecked)
    assert level == "warning"
    assert "$15,381 million" in message


def test_a_clean_answer_says_what_was_checked() -> None:
    level, message = verdict({"fully_grounded": True})
    assert level == "success"
    assert "every figure is in the one it cites" in message

    level, message = verdict({"fully_grounded": True, "verified_derived": ["26.2%"]})
    assert level == "success"
    assert "26.2% recomputed" in message


def test_a_truncated_answer_is_flagged_over_a_clean_one() -> None:
    assert verdict({"truncated": True})[0] == "warning"


def test_a_refusal_reason_is_shown_in_words() -> None:
    assert refusal_headline({"reason": "out_of_scope"}) == "That is not in this corpus"
    assert refusal_headline({"reason": "insufficient_evidence"}).startswith("The filings do not")
    assert refusal_headline({"reason": "something new"}) == "Cannot answer"


def test_a_citation_label_names_the_filing_a_reader_would_open() -> None:
    label = citation_label(
        {
            "tag": "c1",
            "ticker": "AAPL",
            "form": "10-Q",
            "period_label": "FY2026 Q3",
            "section": "Part I.Item 1",
        }
    )
    assert label == "[c1] AAPL 10-Q FY2026 Q3, Part I.Item 1"


def test_the_trace_link_points_at_the_configured_langfuse() -> None:
    assert trace_url("http://localhost:3000/", "abc") == (
        "http://localhost:3000/project/quarterly-rag/traces/abc"
    )


def test_dollar_amounts_are_not_rendered_as_maths() -> None:
    """Streamlit reads `$...$` as LaTeX, and an answer about filings is all dollar amounts.

    Two amounts in one sentence rendered everything between them as an equation. Caught on
    the first screenshot of the page.
    """
    marked = as_markdown("Services net sales of $109,158 million of $416,161 million.")
    assert marked.count("\\$") == 2
    assert "$1" not in marked.replace("\\$1", "")


def test_a_citation_label_is_not_a_figure_to_highlight() -> None:
    """`[c1]` parses as the number 1, and every filing contains a 1.

    Left in, it highlighted the footnote markers in Apple's product table, which makes the
    highlight mean nothing. The verifier strips tags before checking figures too.
    """
    answer = "Services net sales were $109,158 million [c1]."
    passage = "Services (1) | 109,158 | 14% | 96,169"
    assert figures_to_mark(answer, passage) == ["109,158"]
