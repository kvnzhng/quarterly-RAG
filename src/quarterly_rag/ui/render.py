"""Turning an API response into something readable (RAG-014).

Pure functions, no Streamlit, so the part worth testing is testable. `app.py` next door is
the thin Streamlit script that calls these.

The highlighting is the point of this module. A citation is only worth showing if the reader
can see *why* it is the citation, so the figures the verifier matched in that passage are
marked in it. The same check the verifier used decides what gets marked, which means a
highlight is not decoration: it is the evidence, shown.
"""

from __future__ import annotations

import html
import re

from quarterly_rag.generation.citations import strip_tags
from quarterly_rag.generation.numbers import figure_supported, parse_figures

MARK_OPEN = '<mark style="background:#fde68a;padding:0 2px;border-radius:2px">'
MARK_CLOSE = "</mark>"


def figures_to_mark(answer_text: str, passage: str) -> list[str]:
    """The figures the answer states that this passage really contains.

    Runs the verifier's own presence check, so a marked number is one that was checked here
    and found, not one that merely looks similar.
    """
    marked: list[str] = []
    # The citation labels go first. `[c1]` parses as the figure 1, and a filing has a `1`
    # in it somewhere, so leaving them in highlighted the footnote markers in the table.
    # The verifier strips them for the same reason before it checks anything.
    for figure in parse_figures(strip_tags(answer_text)):
        if not figure_supported(figure, passage):
            continue
        digits = _digits(figure.raw)
        if digits and digits not in marked:
            marked.append(digits)
    return marked


def _digits(raw: str) -> str:
    """`$109,417 million` -> `109,417`, which is how the filing prints it."""
    match = re.search(r"\d[\d,]*(?:\.\d+)?", raw)
    return match.group(0) if match else ""


def highlight(passage: str, figures: list[str]) -> str:
    """The passage as HTML, with those figures marked.

    The text is escaped first and marked second. A filing is full of `<` and `&`, and a
    passage rendered with `unsafe_allow_html` that was not escaped is an injection waiting
    for the right table.
    """
    escaped = html.escape(passage)
    if not figures:
        return escaped
    # Longest first, so marking `109,417` does not leave `417` to be marked inside it.
    pattern = "|".join(re.escape(f) for f in sorted(figures, key=len, reverse=True))
    return re.sub(
        rf"(?<![\d,.]){pattern}(?![\d,.])",
        lambda m: f"{MARK_OPEN}{m.group(0)}{MARK_CLOSE}",
        escaped,
    )


def as_markdown(text: str) -> str:
    """Text safe to hand to Streamlit's markdown, which reads `$` as maths.

    An answer about filings is full of dollar amounts, and two of them in one sentence made
    Streamlit render everything between them as LaTeX. Seen on the first screenshot.
    """
    return text.replace("$", "\\$")


def citation_label(citation: dict) -> str:
    """`[c1] AAPL 10-Q FY2026 Q3, Part I.Item 1`, the handle a reader recognises."""
    return (
        f"[{citation['tag']}] {citation['ticker']} {citation['form']} "
        f"{citation['period_label']}, {citation['section']}"
    )


def verdict(answer: dict) -> tuple[str, str]:
    """A one-line judgement on the answer, and which Streamlit box to put it in.

    Ordered by how much it should worry the reader: a citation that does not resolve is
    worse than a sentence with no citation, which is worse than a figure nobody could check.
    """
    if answer.get("invalid_tags"):
        cited = ", ".join(answer["invalid_tags"])
        return "error", f"The answer cited {cited}, which it was never given."
    if answer.get("unsupported_sentences"):
        count = len(answer["unsupported_sentences"])
        return "error", f"{count} sentence(s) carry no citation that resolves."
    if answer.get("unverified_derived"):
        listed = ", ".join(answer["unverified_derived"])
        return "warning", f"Not checked against any passage: {listed}."
    if answer.get("truncated"):
        return "warning", "The token budget cut this answer off; raise ANSWER_MAX_TOKENS."
    if answer.get("verified_derived"):
        listed = ", ".join(answer["verified_derived"])
        return "success", f"Every sentence cited, and {listed} recomputed from the passages."
    return "success", "Every sentence cites a passage, and every figure is in the one it cites."


def refusal_headline(refusal: dict) -> str:
    """A refusal reason in words rather than in an enum."""
    reasons = {
        "out_of_scope": "That is not in this corpus",
        "low_confidence": "Nothing retrieved was close enough",
        "insufficient_evidence": "The filings do not answer that",
        "verification_failed": "Nothing in the answer survived checking",
    }
    return reasons.get(refusal.get("reason", ""), "Cannot answer")


def trace_url(host: str, trace_id: str) -> str:
    return f"{host.rstrip('/')}/project/quarterly-rag/traces/{trace_id}"
