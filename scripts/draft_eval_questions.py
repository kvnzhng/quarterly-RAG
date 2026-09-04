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
A_Q125 = "0000320193-25-000008"  # Apple FY2025 Q1 10-Q
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
    EvalQuestion(
        id="q011",
        ticker="AAPL",
        type="lookup",
        question="What are Apple's reportable segments?",
        gold_answer="the Americas, Europe, Greater China, Japan and Rest of Asia Pacific",
        evidence=[
            span(
                "AAPL",
                A_10K25,
                "The Company manages its business primarily on a geographic basis\\. "
                "The Company\u2019s reportable segments consist of the Americas, Europe, "
                "Greater China, Japan and Rest of Asia Pacific\\.",
            )
        ],
        note=(
            "Prose in Item 1, and the answer is a list rather than a number. Apple reports by "
            "geography while Nvidia reports by product line, so a single segment filter cannot "
            "serve both."
        ),
    ),
    EvalQuestion(
        id="q012",
        ticker="AAPL",
        type="lookup",
        question="How does Apple define its fiscal year?",
        gold_answer="the 52- or 53-week period that ends on the last Saturday of September",
        evidence=[
            span(
                "AAPL",
                A_10K25,
                "The Company\u2019s fiscal year is the 52- or 53-week period that ends on "
                "the last Saturday of September\\.",
            )
        ],
        note=(
            "The fact that explains why Apple's FY2025 ended on 27 September and FY2024 on 28 "
            "September. A system that answers period questions should be able to state the rule."
        ),
    ),
    EvalQuestion(
        id="q013",
        ticker="AAPL",
        type="lookup",
        question="What was Apple's effective tax rate in fiscal 2025?",
        gold_answer="15.6%",
        evidence=[
            table_span(
                "AAPL", A_10K25, r"Effective tax rate \| 15\.6% \| 24\.1% \| 14\.7%", lead_lines=2
            )
        ],
        note=(
            "The row below it holds the statutory rate of 21%, a plausible wrong answer that is "
            "also present in the source."
        ),
    ),
    EvalQuestion(
        id="q014",
        ticker="AAPL",
        type="lookup",
        question="How many shares did Apple repurchase in fiscal 2025, and for how much?",
        gold_answer="402 million shares for $89.3 billion",
        evidence=[
            span(
                "AAPL",
                A_10K25,
                r"During 2025, the Company repurchased 402 million shares of its common stock for "
                r"\$89\.3 billion\.",
            )
        ],
        note=(
            "Two numbers in one sentence, in the notes to the financial statements rather than "
            "MD&A. An answer that reports one without the other is incomplete."
        ),
    ),
    EvalQuestion(
        id="q015",
        ticker="AAPL",
        type="lookup",
        question="What were Apple's total assets at the end of fiscal 2025?",
        gold_answer="$359,241 million",
        evidence=[table_span("AAPL", A_10K25, r"Total assets \| \$359,241 \| \$364,980")],
        note=(
            "A balance sheet figure, so the period is a date rather than a range. The prior year "
            "is larger, which makes a column mix-up look like growth."
        ),
    ),
    EvalQuestion(
        id="q016",
        ticker="AAPL",
        type="lookup",
        question="How much did Apple spend on research and development in fiscal 2025?",
        gold_answer="$34,550 million",
        evidence=[
            table_span(
                "AAPL",
                A_10K25,
                r"Research and development \| \$34,550 \| 10% \| \$31,370 \| 5% \| \$29,915",
                lead_lines=2,
            )
        ],
        note=(
            "The same figure appears in MD&A with growth percentages and again in the income "
            "statement without them; the label points at the MD&A table."
        ),
    ),
    EvalQuestion(
        id="q017",
        ticker="AAPL",
        type="lookup",
        question=(
            "How much did Apple owe Ireland under the State Aid Decision as of September 28, 2024?"
        ),
        gold_answer="€14.2 billion, or $15.8 billion",
        evidence=[
            span(
                "AAPL",
                A_10K24,
                r"As of September 28, 2024, the Company had an obligation to pay €14\.2 billion "
                r"or \$15\.8 billion to Ireland in connection with the State Aid Decision",
            )
        ],
        note=(
            "Two currencies for one obligation. A verifier that matches numbers must accept either "
            "and must not treat the euro figure as unsupported."
        ),
    ),
    EvalQuestion(
        id="q018",
        ticker="NVDA",
        type="lookup",
        question="What are Nvidia's two reportable segments?",
        gold_answer="the Compute & Networking segment and the Graphics segment",
        evidence=[
            span(
                "NVDA",
                N_10K26,
                r"We report our business results in two segments\.\n"
                r"The Compute & Networking segment includes[^\n]*\n[^\n]*",
            )
        ],
        note=(
            "The counterpart to q011. Apple splits by geography, Nvidia by product line, so a "
            "question about 'segments' means different things per company. The answer repeats the "
            "word 'segment' on purpose: the ampersand inside 'Compute & Networking' makes a bare "
            "list read as three items rather than two."
        ),
    ),
    EvalQuestion(
        id="q019",
        ticker="NVDA",
        type="lookup",
        question="How much did Nvidia spend on research and development in fiscal 2026?",
        gold_answer="$18,497 million",
        evidence=[
            table_span(
                "NVDA",
                N_10K26,
                r"Research and development \| \$18,497 \| \$12,914 \| \$5,583 \| 43%",
                lead_lines=2,
            )
        ],
        note=(
            "Directly comparable to q016 for Apple, which is what makes cross-company questions "
            "possible later."
        ),
    ),
    EvalQuestion(
        id="q020",
        ticker="NVDA",
        type="lookup",
        question="What were Nvidia's total assets at the end of fiscal 2026?",
        gold_answer="$206,803 million",
        evidence=[table_span("NVDA", N_10K26, r"Total assets \| \$206,803 \| \$111,601")],
        note=(
            "In Part IV Item 15 like the rest of Nvidia's statements, unlike Apple's, which sit "
            "under Item 8. Section filtering has to be learned per filer."
        ),
    ),
    EvalQuestion(
        id="q021",
        ticker="NVDA",
        type="lookup",
        question="How many employees did Nvidia have at the end of fiscal 2026?",
        gold_answer="approximately 42,000 employees in 38 countries",
        evidence=[
            span(
                "NVDA",
                N_10K26,
                r"As of the end of fiscal year 2026, we had approximately 42,000 employees in 38 "
                r"countries",
            )
        ],
        note="The counterpart to q002 for Apple, and prose rather than a table in both cases.",
    ),
    EvalQuestion(
        id="q022",
        ticker="NVDA",
        type="lookup",
        question="What was the revenue of Nvidia's Compute & Networking segment in fiscal 2026?",
        gold_answer="$193,479 million",
        evidence=[
            table_span(
                "NVDA",
                N_10K26,
                r"Compute & Networking \| \$193,479 \| \$116,193 \| \$77,286 \| 67%",
                lead_lines=2,
            )
        ],
        note=(
            "Close to the Data Center end-market figure in q006 ($193,737 million) but not equal: "
            "segments and end markets are different cuts. Retrieval that confuses them produces a "
            "number that exists in the filing and still answers the wrong question."
        ),
    ),
    EvalQuestion(
        id="q023",
        ticker="NVDA",
        type="lookup",
        question="What was Nvidia's gross margin percentage in fiscal 2026?",
        gold_answer="71.1%",
        evidence=[
            table_span(
                "NVDA", N_10K26, r"Gross margin \| 71\.1% \| 75\.0% \| -3\.9 pts", lead_lines=1
            )
        ],
        note=(
            "Stated outright in the 10-K, unlike the quarterly figure in q008 which has to be "
            "computed. The same question is a lookup in one filing and a calculation in another."
        ),
    ),
    EvalQuestion(
        id="q024",
        ticker="AAPL",
        type="lookup",
        question="What were Apple's total net sales in the first quarter of fiscal 2025?",
        gold_answer="$124,300 million",
        evidence=[
            table_span("AAPL", A_Q125, r"Total net sales \| \$124,300 \| \$119,575", lead_lines=2),
            span("AAPL", A_Q125, r"During the first quarter of 2025, the Company announced"),
        ],
        note=(
            "Apple's first fiscal quarter ends in late December, so this covers the 2024 holiday "
            "season and a calendar-year filter would look in the wrong filing. The second span is "
            "the filing naming its own quarter, which is what ties 'December 28, 2024' in the "
            "table header to 'first quarter of fiscal 2025' in the question."
        ),
    ),
    EvalQuestion(
        id="q025",
        ticker="AAPL",
        type="lookup",
        question="Why did Apple's Greater China net sales decrease in fiscal 2025?",
        gold_answer="lower net sales of iPhone",
        evidence=[
            span(
                "AAPL",
                A_10K25,
                r"Greater China net sales decreased during 2025 compared to 2024 primarily due to "
                r"lower net sales of iPhone[^.]*\.",
            )
        ],
        note=(
            "An explanation rather than a figure. The answer cannot be assembled from a table, so "
            "a table-biased retriever fails it."
        ),
    ),
    EvalQuestion(
        id="q026",
        ticker="NVDA",
        type="lookup",
        question="How does Nvidia describe its own business in its fiscal 2026 annual report?",
        gold_answer="a data center scale AI infrastructure company reshaping all industries",
        evidence=[
            span(
                "NVDA",
                N_10K26,
                r"NVIDIA pioneered accelerated computing to help solve the most challenging "
                r"computational problems\. NVIDIA is now a data center scale AI infrastructure "
                r"company reshaping all industries\.",
            )
        ],
        note=(
            "No numbers at all. It checks that retrieval reaches the narrative opening of Item 1, "
            "and that the verifier does not treat a numberless answer as unsupported."
        ),
    ),
    EvalQuestion(
        id="q027",
        ticker="AAPL",
        type="lookup",
        question="What was Apple's total gross margin in fiscal 2025?",
        gold_answer="$195,201 million",
        evidence=[
            table_span(
                "AAPL",
                A_10K25,
                r"Total gross margin \| \$195,201 \| \$180,683 \| \$169,148",
                lead_lines=3,
            )
        ],
        note=(
            "The dollar amount, where q003 asks for the percentage from the table immediately "
            "below it. Adjacent tables, different answers."
        ),
    ),
    EvalQuestion(
        id="q028",
        ticker="NVDA",
        type="derived",
        question=(
            "How much of Nvidia's fiscal 2026 revenue came from the Graphics segment, "
            "as a share of the total?"
        ),
        gold_answer="about 10.4% ($22,459 million of $215,938 million)",
        evidence=[
            table_span(
                "NVDA",
                N_10K26,
                r"Compute & Networking \| \$193,479 \| \$116,193 \| \$77,286 \| 67%\n"
                r"Graphics \| 22,459 \| 14,304 \| 8,155 \| 57%\n"
                r"Total \| \$215,938 \| \$130,497 \| \$85,441 \| 65%",
                lead_lines=2,
            )
        ],
        note=(
            "Both operands are in the same table and the ratio is in none of it. The share is "
            "small, so a plausible-looking wrong answer is easy to produce."
        ),
    ),
    EvalQuestion(
        id="q029",
        ticker="AAPL",
        type="derived",
        question="What share of Apple's fiscal 2025 total net sales came from Services?",
        gold_answer="about 26.2% ($109,158 million of $416,161 million)",
        evidence=[
            table_span(
                "AAPL",
                A_10K25,
                r"Services \(1\) \| 109,158 \| 14% \| 96,169 \| 13% \| 85,200\nTotal net sales \| "
                r"\$416,161 \| 6% \| \$391,035 \| 2% \| \$383,285",
                lead_lines=2,
            )
        ],
        note=(
            "Apple states the growth rate but never this share. Both operands sit in one table, "
            "one row apart."
        ),
    ),
    EvalQuestion(
        id="q030",
        ticker="AAPL",
        type="derived",
        question="What was Apple's operating income in the third quarter of fiscal 2026?",
        gold_answer=(
            "$35,695 million (gross margin of $54,770 million less $19,075 million "
            "of operating expenses)"
        ),
        evidence=[
            table_span(
                "AAPL",
                A_Q326,
                r"Total operating expenses \| 19,075 \| 15,516 \| 56,350 \| 46,237\n"
                r"Operating income \| 35,695 \| 28,202 \| 122,432 \| 100,623",
                lead_lines=2,
            ),
            span("AAPL", A_Q326, r"During the third quarter of 2026, the Company announced"),
        ],
        note=(
            "The figure is stated outright, so the arithmetic in the gold answer checks the answer "
            "rather than being the only route to it. The first column is the three months ended "
            "June 27, 2026; Apple's fiscal year ends in late September, so that quarter is fiscal "
            "Q3 and not calendar Q2. The second span is the filing saying so in its own words, "
            "which is what makes the label verifiable without knowing Apple's calendar."
        ),
    ),
    EvalQuestion(
        id="q031",
        ticker="NVDA",
        type="cross_period",
        question=(
            "How much did Nvidia's research and development spending grow from "
            "fiscal 2025 to fiscal 2026?"
        ),
        gold_answer="up $5,583 million, from $12,914 million to $18,497 million (43%)",
        evidence=[
            table_span(
                "NVDA",
                N_10K26,
                r"Research and development \| \$18,497 \| \$12,914 \| \$5,583 \| 43%",
                lead_lines=2,
            )
        ],
        note=(
            "Nvidia's MD&A states the change and the percentage, so this is a cross-period "
            "question whose answer needs no arithmetic. Compare with q009, where Apple does the "
            "same, and q007, where neither is given."
        ),
    ),
    EvalQuestion(
        id="q032",
        ticker="AAPL",
        type="cross_period",
        question="How did Apple's effective tax rate change from fiscal 2024 to fiscal 2025?",
        gold_answer="down from 24.1% to 15.6%",
        evidence=[
            table_span(
                "AAPL", A_10K25, r"Effective tax rate \| 15\.6% \| 24\.1% \| 14\.7%", lead_lines=2
            )
        ],
        note=(
            "One row, two periods, and the answer is a direction as well as two numbers. Reading "
            "the columns in the wrong order reverses the meaning while quoting real figures."
        ),
    ),
    EvalQuestion(
        id="q033",
        ticker="NVDA",
        type="cross_period",
        question="How did Nvidia's total assets change from fiscal 2025 to fiscal 2026?",
        gold_answer="up from $111,601 million to $206,803 million",
        evidence=[table_span("NVDA", N_10K26, r"Total assets \| \$206,803 \| \$111,601")],
        note=(
            "A balance sheet comparison inside one filing, with no change column to lean on. The "
            "column order has to be read from the header."
        ),
    ),
    EvalQuestion(
        id="q034",
        ticker="AAPL",
        type="cross_period",
        question=(
            "How did Apple's research and development spending change from fiscal 2023 "
            "to fiscal 2025?"
        ),
        gold_answer="up from $29,915 million to $34,550 million",
        evidence=[
            table_span(
                "AAPL",
                A_10K25,
                r"Research and development \| \$34,550 \| 10% \| \$31,370 \| 5% \| \$29,915",
                lead_lines=2,
            )
        ],
        note=(
            "Three fiscal years on one row and the question skips the middle one, so the answer "
            "must use the first and last columns rather than the adjacent pair."
        ),
    ),
    EvalQuestion(
        id="q035",
        ticker="MSFT",
        type="unanswerable",
        question="What was Microsoft's cloud revenue in fiscal 2025?",
        gold_answer="Not in the filings: the corpus holds only Apple and Nvidia.",
        refusal_reason="out_of_scope",
        note=(
            "A second out-of-scope company, and one a model is very likely to have memorised. The "
            "refusal has to hold even when the answer is easy to guess."
        ),
    ),
    EvalQuestion(
        id="q036",
        ticker="AAPL",
        type="unanswerable",
        question="What were Apple's total net sales in fiscal 2019?",
        gold_answer="Not in the filings: the corpus starts with Apple's fiscal 2024 annual report.",
        refusal_reason="out_of_scope",
        note=(
            "The right company, the wrong period. Harder than an out-of-scope company because "
            "retrieval will return plausible Apple revenue passages from other years."
        ),
    ),
    EvalQuestion(
        id="q037",
        ticker="AAPL",
        type="unanswerable",
        question="How many iPhone units did Apple sell in fiscal 2025?",
        gold_answer=(
            "Not in the filings: Apple reports iPhone net sales in dollars, not unit volumes."
        ),
        refusal_reason="insufficient_evidence",
        note=(
            "Right company, right period, and the filings simply do not contain it. Apple stopped "
            "disclosing unit sales in 2018. iPhone revenue is in the corpus, so a system that "
            "conflates units with dollars will answer confidently and wrongly."
        ),
    ),
    EvalQuestion(
        id="q038",
        ticker="NVDA",
        type="unanswerable",
        question="What is Nvidia's revenue guidance for the fourth quarter of fiscal 2027?",
        gold_answer="Not in the filings: forward-looking guidance is not part of a 10-Q or 10-K.",
        refusal_reason="insufficient_evidence",
        note=(
            "Forward-looking. The filings discuss future risks at length, so retrieval will find "
            "confident-sounding passages that do not answer the question."
        ),
    ),
    EvalQuestion(
        id="q039",
        ticker="AAPL",
        type="unanswerable",
        question="What was Tim Cook's total compensation in fiscal 2025?",
        gold_answer=(
            "Not in the filings: executive compensation is disclosed in the proxy statement, which "
            "the 10-K incorporates by reference."
        ),
        refusal_reason="insufficient_evidence",
        note=(
            "The 10-K's Item 11 names executive compensation and then points elsewhere for it. A "
            "system that scores section-title matches highly will retrieve that pointer and may "
            "treat it as evidence."
        ),
    ),
    EvalQuestion(
        id="q041",
        ticker="AAPL",
        type="unanswerable",
        question="How many employees does Apple have in each country?",
        gold_answer=(
            "Not in the filings: Apple reports one worldwide headcount, with no country breakdown."
        ),
        refusal_reason="insufficient_evidence",
        note=(
            "A near miss for q002, which the corpus does answer. The refusal has to survive "
            "retrieval landing on the right paragraph with the wrong granularity."
        ),
    ),
    EvalQuestion(
        id="q042",
        ticker="NVDA",
        type="unanswerable",
        question="Is Nvidia stock a good investment right now?",
        gold_answer=(
            "Not a question the filings answer: they report results and risks, not investment "
            "advice."
        ),
        refusal_reason="out_of_scope",
        note=(
            "Asks for an opinion rather than a fact. Risk Factors is full of language that a "
            "retriever will score as relevant, which makes this a realistic failure mode."
        ),
    ),
    EvalQuestion(
        id="q043",
        ticker="AAPL",
        type="unanswerable",
        question="What was Apple's revenue from the Vision Pro in fiscal 2025?",
        gold_answer=(
            "Not in the filings: Vision Pro sits inside Wearables, Home and Accessories and is "
            "not reported separately."
        ),
        refusal_reason="insufficient_evidence",
        note=(
            "The product is named in Item 1 and the category is in the revenue tables, so both "
            "halves of the question are present and the answer still is not."
        ),
    ),
    EvalQuestion(
        id="q040",
        ticker="NVDA",
        type="unanswerable",
        question="What is Nvidia's current share price?",
        gold_answer=(
            "Not in the filings: a filing is a point-in-time document and carries no "
            "live market data."
        ),
        refusal_reason="out_of_scope",
        note=(
            "Not a filing question at all. It checks that the refusal reason distinguishes 'not in "
            "these documents' from 'not in this kind of document'."
        ),
    ),
]

path = questions_path(settings)
save_questions(path, questions)
print(f"wrote {len(questions)} questions -> {path}")
