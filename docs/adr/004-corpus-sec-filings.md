# ADR-004: Corpus is SEC 10-Q and 10-K filings from EDGAR

**Date:** 2026-09-03
**Status:** accepted
**Ticket:** RAG-001

## Context

A RAG corpus for learning should be free, public, well-structured, large enough to make retrieval non-trivial, and full of facts that can be verified exactly (so hallucinations are detectable). Quarterly reports fit: long documents, repeated structure across periods (ideal for testing metadata filtering and chunk boundaries), and precise numbers.

## Decision

- Source: **SEC EDGAR** (`https://www.sec.gov/`), forms **10-Q** and **10-K**, starting with **Apple (AAPL)** and **Nvidia (NVDA)**, last 8 quarters. Adding a ticker is a config change.
- EDGAR access rules are followed: a declared `User-Agent` with contact email, at most 10 requests per second, and downloaded filings cached under `data/raw/` (gitignored).
- Filings are parsed into **sections keyed by SEC Item** (Risk Factors, MD&A, Financial Statements, ...). Section is a first-class metadata field used by chunking, filtering, and citations.
- Investor-relations PDFs and earnings-call transcripts are out of scope for now; they can be added later as additional document types.

## Consequences

- Raw data is not committed; the downloader must be idempotent and reproducible from the manifest.
- Questions in the eval sets are anchored to a company and a fiscal period, which makes "period not in corpus" a concrete refusal case.
- Fiscal-year vs calendar-year mismatches (Apple's FY ends in September, Nvidia's in January) are a deliberate source of hard questions.
