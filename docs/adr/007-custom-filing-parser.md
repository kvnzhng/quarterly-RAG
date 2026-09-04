# ADR-007: Filings are parsed by a custom block-boundary parser, not a library

**Date:** 2026-09-04
**Status:** accepted
**Ticket:** RAG-004

## Context

Everything downstream indexes into the parser's output: chunk boundaries (RAG-005), gold evidence spans (RAG-019), and the passages a citation quotes (RAG-010). A parser that mislabels a section poisons all three, and one that reflows text invalidates every stored offset. Apple and Nvidia both file inline XBRL from the same preparer: 1 to 2 MB of nested `div` and `span` elements, no semantic headings, layout as the only structure. `sec-parser`, `edgartools`, and `unstructured` all offer to do this.

## Decision

A custom parser built on BeautifulSoup and lxml, both added as runtime dependencies.

- **The text is flattened at block boundaries**, one block element per line, and written once per filing as `<accession>.txt`. Every offset anywhere in the project indexes into that string.
- **A heading is a line that begins with an Item number and carries a title.** That one rule separates real headings from both lookalikes: table-of-contents rows put the number and the title in different cells, so the line is bare; cross-references never start a line.
- **Headings are never read inside a rendered table.** The contents table renders as `Item 1. | Financial Statements | 1`, which passes the heading test until you notice where it is.
- **A part marker must be bare or continue with a separator.** `PART II - OTHER INFORMATION` sets the part; `Part I, Item 1A of the 2025 Form 10-K describes ...` is a sentence and must not.
- **The section key carries the part**, because a 10-Q's Item 1 is Financial Statements in Part I and Legal Proceedings in Part II.
- **Tables are pipe-delimited** between `[TABLE]` and `[/TABLE]` with the header row labelled, spacer cells dropped, and currency symbols and parentheses re-joined so numbers survive verbatim for the RAG-010 verifier.

The alternatives were not benchmarked head to head. The custom parser reached full coverage on the first corpus (16 of 16 filings, no missing critical item, no false heading), so the criteria that would justify a library were already met. `docs/tradeoffs/parsing.md` holds the comparison and the numbers.

## Consequences

- Heading detection is tuned to two filers. A third company whose filings break it is the signal to revisit this decision, and the coverage report is what will say so: a missing critical item or an unexpected section.
- The text format is ours to change. When chunking needs a different table rendering, no upstream library has to agree.
- `edgartools` remains the right answer if the project ever needs XBRL facts rather than narrative text. That is a different question from this one, and ADR-004 keeps PDFs out of scope, which is where `unstructured` would earn its weight.
- Two more runtime dependencies, both small and widely used.
