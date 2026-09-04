"""Draft the eval set (RAG-019). Run: `uv run python scripts/draft_eval_questions.py`.

Offsets are never typed by hand: each span is located by searching the parsed filing text,
so a re-parse plus a re-run regenerates correct offsets. Questions and gold answers are
drafted here and then human-verified against the filing; `rag eval check` is the gate.
"""

import re
import sys

from quarterly_rag.config import Settings
from quarterly_rag.evaluation.questions import (
    EvalQuestion,
    EvidenceSpan,
    questions_path,
    save_questions,
)
from quarterly_rag.ingestion.records import load_records

settings = Settings()


def span(ticker, accession, pattern, occurrence=0):
    text = (settings.processed_dir / ticker / f"{accession}.txt").read_text()
    recs = load_records(settings.processed_dir / ticker / f"{accession}.jsonl")
    matches = list(re.finditer(pattern, text))
    if not matches:
        sys.exit(f"NO MATCH for {pattern!r} in {accession}")
    m = matches[occurrence]
    sec = next((r for r in recs if r.char_start <= m.start() < r.char_end), None)
    if sec is None:
        sys.exit(f"span outside any section: {pattern!r}")
    return EvidenceSpan(
        accession=accession,
        section=sec.section,
        char_start=m.start(),
        char_end=m.end(),
        quote=m.group(0),
    )


def table_span(ticker, accession, row_pattern, occurrence=0, lead_lines=2):
    """A row plus the table it sits in: the header carries the period and the unit, and a
    row of bare numbers means nothing without them."""
    text = (settings.processed_dir / ticker / f"{accession}.txt").read_text()
    recs = load_records(settings.processed_dir / ticker / f"{accession}.jsonl")
    matches = list(re.finditer(row_pattern, text))
    if not matches:
        sys.exit(f"NO MATCH for {row_pattern!r} in {accession}")
    m = matches[occurrence]
    open_at = text.rfind("[TABLE]", 0, m.start())
    if open_at == -1:
        sys.exit(f"no enclosing table for {row_pattern!r}")
    start = open_at
    for _ in range(lead_lines):  # the caption above the table names the years
        previous = text.rfind("\n", 0, start - 1)
        if previous == -1:
            break
        start = previous + 1
    sec = next((r for r in recs if r.char_start <= m.start() < r.char_end), None)
    if sec is None:
        sys.exit(f"span outside any section: {row_pattern!r}")
    return EvidenceSpan(
        accession=accession,
        section=sec.section,
        char_start=start,
        char_end=m.end(),
        quote=text[start : m.end()],
    )


A_10K25 = "0000320193-25-000079"  # Apple FY2025 10-K
A_10K24 = "0000320193-24-000123"  # Apple FY2024 10-K
A_Q326 = "0000320193-26-000020"  # Apple FY2026 Q3 10-Q
N_10K26 = "0001045810-26-000021"  # Nvidia FY2026 10-K
N_Q227 = "0001045810-26-000075"  # Nvidia FY2027 Q2 10-Q

questions = [
    EvalQuestion(
        id="q001",
        ticker="AAPL",
        type="lookup",
        question="What were Apple's total net sales in the third quarter of fiscal 2026?",
        gold_answer="$109,417 million",
        evidence=[
            table_span(
                "AAPL", A_Q326, r"Total net sales \| 109,417 \| 94,036 \| 364,357 \| 313,695"
            )
        ],
        note="Straight table lookup. The row holds four numbers: three-month and nine-month, "
        "current and prior year, so retrieval must return the row and generation must pick "
        "the right column.",
    ),
    EvalQuestion(
        id="q002",
        ticker="AAPL",
        type="lookup",
        question=(
            "How many full-time equivalent employees did Apple have at the end of fiscal 2025?"
        ),
        gold_answer="approximately 166,000",
        evidence=[
            span(
                "AAPL",
                A_10K25,
                r"the Company had approximately 166,000 full-time equivalent employees",
            )
        ],
        note=(
            "Prose lookup in Item 1, not a table. Tests that the corpus is more than "
            "financial tables."
        ),
    ),
    EvalQuestion(
        id="q003",
        ticker="AAPL",
        type="lookup",
        question="What was Apple's total gross margin percentage in fiscal 2025?",
        gold_answer="46.9%",
        evidence=[
            table_span(
                "AAPL",
                A_10K25,
                r"Total gross margin percentage \| 46\.9% \| 46\.2% \| 44\.1%",
                lead_lines=6,
            )
        ],
        note="Three fiscal years sit on one row; the answer is the first column. A model that "
        "reads the wrong column still produces a number that appears in the source, which "
        "is why RAG-010 verifies more than presence.",
    ),
    EvalQuestion(
        id="q004",
        ticker="AAPL",
        type="lookup",
        question="What was Apple's quarterly cash dividend per share as of June 27, 2026?",
        gold_answer="$0.27 per share",
        evidence=[
            span(
                "AAPL",
                A_Q326,
                "the Company\u2019s quarterly cash dividend was \\$0\\.27 per share",
            )
        ],
        note="Stated in MD&A prose and again in the equity statement, so more than one span "
        "supports it; the label keeps the MD&A one.",
    ),
    EvalQuestion(
        id="q005",
        ticker="NVDA",
        type="lookup",
        question="What was Nvidia's revenue in the second quarter of fiscal 2027?",
        gold_answer="$96,221 million",
        evidence=[
            table_span("NVDA", N_Q227, r"Revenue \| \$96,221 \| \$46,743 \| \$177,837 \| \$90,805")
        ],
        note="Nvidia's fiscal year ends in January, so its Q2 FY2027 covers mid-2026. Period "
        "filtering has to use the fiscal label, not the calendar.",
    ),
    EvalQuestion(
        id="q006",
        ticker="NVDA",
        type="lookup",
        question="How much revenue did Nvidia's Data Center market generate in fiscal 2026?",
        gold_answer="$193,737 million",
        evidence=[
            table_span("NVDA", N_10K26, r"Data Center \| \$193,737 \| \$115,186 \| \$47,525")
        ],
        note="Nvidia files its consolidated financial statements under Part IV Item 15, not "
        "Item 8. A section filter built on the 10-K convention alone would miss this.",
    ),
    EvalQuestion(
        id="q007",
        ticker="NVDA",
        type="derived",
        question=(
            "By what percentage did Nvidia's revenue grow year over year in the second "
            "quarter of fiscal 2027?"
        ),
        gold_answer="about 106% (from $46,743 million to $96,221 million)",
        evidence=[
            table_span("NVDA", N_Q227, r"Revenue \| \$96,221 \| \$46,743 \| \$177,837 \| \$90,805")
        ],
        note="Both operands are on one row; the growth rate is on none. A verbatim number check "
        "cannot confirm this answer, which is what RAG-021 addresses.",
    ),
    EvalQuestion(
        id="q008",
        ticker="AAPL",
        type="derived",
        question="What was Apple's gross margin percentage in the third quarter of fiscal 2026?",
        gold_answer="about 50.1% ($54,770 million gross margin on $109,417 million of net sales)",
        evidence=[
            table_span("AAPL", A_Q326, r"Gross margin \| 54,770 \| 43,718 \| 178,782 \| 146,860")
        ],
        note="A 10-Q states no quarterly gross margin percentage; it must be computed from two "
        "rows. Two evidence spans, one answer.",
    ),
    EvalQuestion(
        id="q009",
        ticker="AAPL",
        type="cross_period",
        question="How much did Apple's total net sales change from fiscal 2024 to fiscal 2025?",
        gold_answer="up about $25.1 billion, from $391,035 million to $416,161 million (6%)",
        evidence=[
            table_span(
                "AAPL",
                A_10K25,
                r"Total net sales \| \$416,161 \| 6% \| \$391,035 \| 2% \| \$383,285",
            )
        ],
        note="The FY2025 10-K carries both years, but the label also points at the FY2024 filing: "
        "retrieval that finds either span is on the right track.",
    ),
    EvalQuestion(
        id="q010",
        ticker="TSLA",
        type="unanswerable",
        question="What was Tesla's total revenue in fiscal 2025?",
        gold_answer="Not in the filings: the corpus holds only Apple and Nvidia.",
        refusal_reason="out_of_scope",
        note="Company outside the corpus. The system must refuse rather than answer from "
        "pretraining, and the reason must say which company is missing.",
    ),
]

path = questions_path(settings)
save_questions(path, questions)
print(f"wrote {len(questions)} questions -> {path}")
