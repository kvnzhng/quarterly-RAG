"""Filing HTML -> normalized text plus one record per SEC Item, with offsets (RAG-004).

Both companies file inline XBRL produced by Workiva: thousands of nested `div`/`span`
elements with no semantic headings. What survives is layout, so the parser flattens the
document at block boundaries and finds Item headings as lines that *begin* with the item
number and carry a title. That separates real headings from the two lookalikes: table of
contents rows (the number and the title sit in different cells, so the line is bare
`Item 1.`) and inline cross-references (`... described in Item 1A of this Form 10-K`,
which never starts a line).

Offsets index into the normalized text, which is written next to the records, so a
section, a chunk (RAG-005), a gold evidence span (RAG-019), and a citation (RAG-010) all
address the same string.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

BLOCK_TAGS = {
    "address", "article", "blockquote", "br", "caption", "div", "dl", "dt", "dd",
    "fieldset", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6",
    "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section", "table",
    "td", "th", "tr", "ul",
}  # fmt: skip
PARSER_VERSION = "1"
"""Bumped when the heading or table rules change, so a stored offset can be dated."""

TABLE_OPEN = "[TABLE]"
TABLE_CLOSE = "[/TABLE]"
HEADER_PREFIX = "header: "

_ITEM_HEADING = re.compile(
    r"^item\s*(\d{1,2})\s*([a-c])?\s*[.:\u2013\u2014-]?\s+(?P<title>\S.*)$", re.I
)
_PART_LINE = re.compile(r"^part\s+(i{1,3}|iv)\b\s*(?P<rest>.*)$", re.I)
_SIGNATURES = re.compile(r"^signatures?$", re.I)
_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
ROMAN_NUMERALS = {1: "I", 2: "II", 3: "III", 4: "IV"}
# What may follow a part number in a real heading: "PART II - OTHER INFORMATION".
_PART_SEPARATORS = " \u2013\u2014\u2010-:."
_PART_SEPARATOR_STRIP = re.compile(f"^[{re.escape(_PART_SEPARATORS)}]+")
# A heading line is short; a paragraph that happens to open with "Item 5. ..." is not.
MAX_HEADING_WORDS = 25

# Items whose absence means the parse failed, per form. Others are informational:
# a company with no legal proceedings simply omits that item.
CRITICAL_ITEMS: dict[str, tuple[tuple[int, str], ...]] = {
    "10-K": ((1, "1"), (1, "1A"), (2, "7"), (2, "7A"), (2, "8")),
    "10-Q": ((1, "1"), (1, "2"), (2, "1A")),
}
# (part, item). A 10-K runs Part I through Part IV; a 10-Q has two parts and repeats
# item numbers across them, which is why the part belongs in the key.
_TENK_ITEMS = (
    (1, "1"),
    (1, "1A"),
    (1, "1B"),
    (1, "1C"),
    (1, "2"),
    (1, "3"),
    (1, "4"),
    (2, "5"),
    (2, "6"),
    (2, "7"),
    (2, "7A"),
    (2, "8"),
    (2, "9"),
    (2, "9A"),
    (2, "9B"),
    (2, "9C"),
    (3, "10"),
    (3, "11"),
    (3, "12"),
    (3, "13"),
    (3, "14"),
    (4, "15"),
    (4, "16"),
)
_TENQ_ITEMS = (
    (1, "1"),
    (1, "2"),
    (1, "3"),
    (1, "4"),
    (2, "1"),
    (2, "1A"),
    (2, "2"),
    (2, "3"),
    (2, "4"),
    (2, "5"),
    (2, "6"),
)
EXPECTED_ITEMS: dict[str, tuple[tuple[int, str], ...]] = {"10-K": _TENK_ITEMS, "10-Q": _TENQ_ITEMS}


@dataclass(frozen=True)
class Section:
    part: int
    """1 or 2 for a 10-Q; 0 when the filing does not mark parts around this item."""
    item: str
    """`1`, `1A`, `7A`, ... uppercased, without the word "Item"."""
    title: str
    text: str
    char_start: int
    char_end: int

    @property
    def key(self) -> str:
        if not self.part:
            return f"Item {self.item}"
        return f"Part {ROMAN_NUMERALS[self.part]}.Item {self.item}"


@dataclass(frozen=True)
class ParsedFiling:
    text: str
    """Normalized full text. Every offset in `sections` indexes into this string."""
    sections: list[Section]

    def coverage(self, form: str) -> Coverage:
        found = {(s.part, s.item) for s in self.sections}
        expected = EXPECTED_ITEMS.get(form.upper(), ())
        critical = CRITICAL_ITEMS.get(form.upper(), ())
        return Coverage(
            form=form,
            found=sorted(found),
            missing=[key for key in expected if key not in found],
            missing_critical=[key for key in critical if key not in found],
            unexpected=sorted(key for key in found if expected and key not in expected),
        )


@dataclass(frozen=True)
class Coverage:
    form: str
    found: list[tuple[int, str]]
    missing: list[tuple[int, str]]
    missing_critical: list[tuple[int, str]]
    unexpected: list[tuple[int, str]]

    @property
    def ok(self) -> bool:
        return not self.missing_critical


# --- HTML -> text ----------------------------------------------------------------


def _cell_text(cell) -> str:
    return re.sub(r"\s+", " ", cell.get_text(separator=" ")).replace("\xa0", " ").strip()


def _tidy_numbers(value: str) -> str:
    """`( 21,507 )` -> `(21,507)`, `$ 78,678` -> `$78,678`: RAG-010 matches numbers verbatim."""
    value = re.sub(r"\(\s+", "(", value)
    value = re.sub(r"\s+\)", ")", value)
    return re.sub(r"([$(])\s+(?=[\d.,])", r"\1", value)


def render_table(table) -> list[str]:
    """A table as pipe-delimited lines wrapped in markers, header row labelled.

    Filers pad tables with empty spacer cells and put currency symbols in their own
    column; both are dropped or merged so a row reads like a row.
    """
    rows: list[str] = []
    for tr in table.find_all("tr"):
        cells = [_cell_text(c) for c in tr.find_all(["td", "th"])]
        merged: list[str] = []
        for cell in cells:
            if not cell:
                continue
            leading = bool(merged) and merged[-1] in {"$", "(", "%"}
            trailing = bool(merged) and cell in {")", ")%", "%"}
            if leading or trailing:
                merged[-1] += cell
            else:
                merged.append(cell)
        if merged:
            rows.append(_tidy_numbers(" | ".join(merged)))
    if not rows:
        return []
    return [TABLE_OPEN, HEADER_PREFIX + rows[0], *rows[1:], TABLE_CLOSE]


def to_text(html: str) -> str:
    """Flatten a filing to text, one block element per line, tables pipe-delimited."""
    with warnings.catch_warnings():
        # The filings are XHTML; the HTML parser handles them and says so every time.
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    for junk in soup(["script", "style", "ix:header"]):
        junk.decompose()
    for table in soup.find_all("table"):
        table.replace_with("\n" + "\n".join(render_table(table)) + "\n")

    pieces: list[str] = []
    for node in soup.descendants:
        name = getattr(node, "name", None)
        if name in BLOCK_TAGS:
            pieces.append("\n")
        elif isinstance(node, str):
            pieces.append(node)
    text = "".join(pieces).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# --- text -> sections ------------------------------------------------------------


@dataclass(frozen=True)
class _Heading:
    part: int
    item: str
    title: str
    start: int
    """Offset of the heading line in the normalized text."""
    body_start: int


def _is_part_marker(match: re.Match[str]) -> bool:
    """True for a real part heading, false for a sentence that opens with a cross-reference.

    `Part I, Item 1A of the 2025 Form 10-K describes ...` starts a paragraph inside Part II;
    letting it set the part would misattribute every heading after it. A real part heading is
    either bare (`Part II`) or continues with a separator (`PART II - OTHER INFORMATION`).
    """
    rest = match.group("rest").strip()
    return not rest or rest[0] in _PART_SEPARATORS


def _find_headings(text: str) -> tuple[list[_Heading], int]:
    """Item headings in document order, plus the offset where the signature block starts."""
    headings: list[_Heading] = []
    part = 0
    offset = 0
    in_table = False
    signatures_at = len(text)
    for line in text.split("\n"):
        stripped = line.strip()
        end = offset + len(line)
        if stripped == TABLE_OPEN:
            in_table = True
        elif stripped == TABLE_CLOSE:
            in_table = False
        if in_table:
            offset = end + 1
            continue
        if _SIGNATURES.match(stripped) and headings:
            signatures_at = min(signatures_at, offset)
        if (match := _PART_LINE.match(stripped)) and _is_part_marker(match):
            # A bare "Part II" row in the table of contents also matches; harmless, because
            # the body's own part line resets the state before the first real heading.
            part = _ROMAN[match.group(1).lower()]
            rest = _PART_SEPARATOR_STRIP.sub("", match.group("rest"))
            if not rest or not _ITEM_HEADING.match(rest):
                offset = end + 1
                continue
            stripped = rest
        if (match := _ITEM_HEADING.match(stripped)) and len(stripped.split()) <= MAX_HEADING_WORDS:
            title = match.group("title").strip()
            headings.append(
                _Heading(
                    part=part,
                    item=f"{match.group(1)}{(match.group(2) or '').upper()}",
                    title=title,
                    start=offset,
                    body_start=end + 1,
                )
            )
        offset = end + 1
    return headings, signatures_at


def parse_filing(html: str) -> ParsedFiling:
    text = to_text(html)
    headings, signatures_at = _find_headings(text)
    sections: list[Section] = []
    for index, heading in enumerate(headings):
        next_start = headings[index + 1].start if index + 1 < len(headings) else signatures_at
        end = max(heading.start, min(next_start, len(text)))
        if index + 1 == len(headings):
            end = max(end, heading.body_start) if signatures_at > heading.start else end
        body = text[heading.start : end].rstrip()
        sections.append(
            Section(
                part=heading.part,
                item=heading.item,
                title=heading.title,
                text=body,
                char_start=heading.start,
                char_end=heading.start + len(body),
            )
        )
    return ParsedFiling(text=text, sections=sections)
