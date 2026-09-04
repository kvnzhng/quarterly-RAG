# Filing parsing: custom HTML parser vs sec-parser vs edgartools vs unstructured

**Status:** decided (ADR-007; measured on the 16-filing corpus, 2026-09-04)
**Ticket:** RAG-004

## Question

How do inline-XBRL filings become sectioned text with character offsets? Everything downstream depends on this: chunk boundaries (RAG-005), gold evidence spans (RAG-019), and the passages a citation quotes (RAG-010). A parser that mislabels a section poisons all three, and a parser that reflows text invalidates every stored offset.

## Candidates

| Candidate | One-line description | License | Runs locally? |
|---|---|---|---|
| Custom BeautifulSoup + lxml (chosen) | Flatten at block boundaries, detect Item headings as line starts, render tables pipe-delimited | ours | yes, no network |
| `sec-parser` | Purpose-built SEC semantic element parser | MIT | yes |
| `edgartools` | Full EDGAR client with parsing, XBRL facts, and its own data model | Apache-2.0 | yes, downloads too |
| `unstructured` | General document partitioner (HTML, PDF, ...) | Apache-2.0 | yes, heavy deps |

## Criteria

| Criterion | How measured | Weight |
|---|---|---|
| Section detection accuracy | critical Items found on all 16 filings; false headings (contents rows, cross-references) counted | high |
| Stable character offsets | `text[char_start:char_end] == section.text` for every record | high |
| Table fidelity | financial tables readable as rows, numbers unmangled (RAG-010 matches numbers verbatim) | high |
| Dependency weight | install size and transitive deps | medium |
| Network independence | CI and tests must never call the SEC | medium |
| Control over the output | can we change the text format when chunking needs it? | medium |

## Results

Measured with `rag ingest parse --ticker AAPL --ticker NVDA` on the 16-filing corpus (Apple FY2024 10-K to FY2026 Q3, Nvidia FY2025 Q3 to FY2027 Q2).

| Metric | Custom parser |
|---|---|
| Filings parsed | 16 / 16 |
| Filings missing a critical Item | 0 |
| Unexpected sections (false headings) | 0 |
| 10-K sections found | 23 / 23 expected, on all four |
| 10-Q sections found | Apple 11 / 11; Nvidia 9 (Items 3 and 4 of Part II genuinely absent) |
| Offset invariant holds | every record, both companies |
| Runtime dependencies added | `beautifulsoup4`, `lxml` |

Run record: commit `d031820`, corpus manifests `data/raw/{AAPL,NVDA}/manifest.json`, parser `src/quarterly_rag/ingestion/parse.py`, no model involved.

The alternatives were not benchmarked head to head. The custom parser reached full coverage on the first corpus, and the criteria that would justify a library (accuracy, offsets, table fidelity) are already met. Re-open this page if a third company's filings break heading detection: that is the failure mode a maintained library would fix.

## Decision

**Custom parser.** Both companies file inline XBRL from the same preparer: thousands of nested `div` and `span` elements, no semantic headings, layout as the only structure. So the parser flattens at block boundaries and treats a line that *begins* with an Item number and carries a title as a heading. That one rule separates real headings from both lookalikes: contents rows put the number and title in different table cells, and cross-references never start a line.

Two rules earn their keep:

- **Headings are never read inside a rendered table.** The contents table renders as `Item 1. | Financial Statements | 1`, which passes the heading test until you notice where it is.
- **The section key carries the Part.** A 10-Q's Item 1 is Financial Statements in Part I and Legal Proceedings in Part II.

`edgartools` wins if the project ever needs XBRL facts rather than narrative text; that is a different question from this one. `unstructured` is the choice for mixed PDF corpora, which ADR-004 put out of scope.

## Interview one-liner

Filings have no semantic headings, only layout, so the parser flattens the HTML at block boundaries and finds Item headings as lines that start with the item number, which is also what tells a real heading apart from a table-of-contents row or a cross-reference.
