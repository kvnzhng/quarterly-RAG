"""The `[c1]` labels that bind a claim to the passage it came from (RAG-010).

Split out of `answer.py` so the calculation verifier (RAG-021) can read the same tags
without importing the module that imports it.

Models bracket citations differently: `[c1]` and the full-width `\u3010c1\u3011` both appear in
practice, so both are accepted rather than scored as a missing citation.
"""

from __future__ import annotations

import re

OPEN = "\\[\u3010"
CLOSE = "\\]\u3011"

TAG = re.compile(rf"[{OPEN}]\s*c\s*(\d+)((?:\s*[,;]?\s*c?\s*\d+)*)\s*[{CLOSE}]", re.I)
"""One citation bracket, which may hold several labels: `[c1]`, `[c1, c3]`, `[c1;c3]`."""

_EXTRA_TAG = re.compile(r"c?\s*(\d+)", re.I)


def tag_for(index: int) -> str:
    return f"c{index}"


def parse_tags(text: str) -> list[str]:
    """Every passage label the text cites, in order, deduplicated.

    Tolerates the shapes an 8B model actually writes: `[c1]`, `[c1][c3]`, `[c1, c3]`.
    """
    found: list[str] = []
    for match in TAG.finditer(text):
        numbers = [match.group(1), *_EXTRA_TAG.findall(match.group(2) or "")]
        for number in numbers:
            tag = f"c{int(number)}"
            if tag not in found:
                found.append(tag)
    return found


def strip_tags(text: str) -> str:
    """The text without its citation brackets, so `[c1]` cannot be read as the figure 1."""
    return TAG.sub(" ", text)
