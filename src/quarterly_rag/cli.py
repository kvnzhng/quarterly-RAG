"""`rag` command line entry point. Subcommands grow with the tickets."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from quarterly_rag import __version__
from quarterly_rag.config import get_settings
from quarterly_rag.doctor import failed, run_doctor
from quarterly_rag.evaluation.questions import (
    check_gold_answers,
    check_spans,
    counts_by_type,
    load_questions,
    questions_path,
)
from quarterly_rag.ingestion.download import DEFAULT_FORMS, download_filings
from quarterly_rag.ingestion.edgar import EdgarClient, EdgarError
from quarterly_rag.ingestion.records import parse_ticker

app = typer.Typer(help="Local RAG over SEC 10-Q/10-K filings.", no_args_is_help=True)
ingest = typer.Typer(help="Fetch and parse SEC filings.", no_args_is_help=True)
app.add_typer(ingest, name="ingest")
evaluate = typer.Typer(help="Evaluation sets and metrics.", no_args_is_help=True)
app.add_typer(evaluate, name="eval")
console = Console()


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"quarterly-RAG {__version__}")


@app.command()
def config() -> None:
    """Show the effective configuration (secrets redacted)."""
    settings = get_settings()
    table = Table(title="quarterly-RAG settings")
    table.add_column("key")
    table.add_column("value")
    for key, value in settings.model_dump().items():
        shown = "***" if key.endswith("_key") and value else str(value)
        table.add_row(key, shown)
    console.print(table)


@app.command()
def doctor() -> None:
    """Check the configured model endpoints, models, and data directories."""
    settings = get_settings()
    llm_where = (
        "api.anthropic.com" if settings.llm_provider == "anthropic" else settings.llm_base_url
    )
    console.print(f"chat model:  {settings.llm_provider} / {settings.llm_model} at {llm_where}")
    console.print(
        f"embeddings:  {settings.embed_provider} / {settings.embed_model} "
        f"at {settings.embed_base_url}"
    )
    results = run_doctor(settings)

    table = Table(title="rag doctor")
    table.add_column("check")
    table.add_column("status")
    table.add_column("latency", justify="right")
    table.add_column("detail", overflow="fold")
    styles = {"ok": "green", "warn": "yellow", "fail": "red"}
    for result in results:
        latency = f"{result.latency_ms:.0f} ms" if result.latency_ms is not None else ""
        table.add_row(
            result.name,
            f"[{styles[result.status]}]{result.status}[/]",
            latency,
            escape(result.detail),
        )
    console.print(table)

    if failures := failed(results):
        console.print(f"[red]{len(failures)} check(s) failed[/red]")
        raise typer.Exit(code=1)
    console.print("[green]all checks passed[/green]")


@ingest.command("download")
def ingest_download(
    ticker: Annotated[
        list[str], typer.Option("--ticker", "-t", help="Ticker symbol; repeat for several.")
    ],
    forms: Annotated[str, typer.Option(help="Comma-separated form types.")] = ",".join(
        DEFAULT_FORMS
    ),
    since: Annotated[
        str | None,
        typer.Option(
            help="Earliest filing date, YYYY-MM-DD. Default: two years ago (about eight quarters)."
        ),
    ] = None,
) -> None:
    """Download 10-Q/10-K primary documents from EDGAR into data/raw/<TICKER>/ with a manifest."""
    settings = get_settings()
    since_date = date.fromisoformat(since) if since else date.today() - timedelta(days=730)
    form_list = tuple(f.strip() for f in forms.split(",") if f.strip())
    try:
        client = EdgarClient(settings.edgar_user_agent, timeout_s=settings.request_timeout_s)
    except EdgarError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from None

    failures = 0
    for symbol in ticker:
        try:
            report = download_filings(settings, client, symbol, forms=form_list, since=since_date)
        except EdgarError as exc:
            console.print(f"[red]{symbol.upper()}: {escape(str(exc))}[/red]")
            failures += 1
            continue
        table = Table(title=f"{report.ticker} ({report.company}), filings since {since_date}")
        table.add_column("form")
        table.add_column("period")
        table.add_column("filed")
        table.add_column("accession")
        table.add_column("status")
        table.add_column("size", justify="right")
        styles = {"new": "green", "cached": "dim", "failed": "red"}
        for item in report.items:
            detail = item.status if not item.error else f"{item.status}: {escape(item.error)}"
            table.add_row(
                item.form,
                item.period_label,
                item.filing_date.isoformat(),
                item.accession,
                f"[{styles[item.status]}]{detail}[/]",
                f"{item.size_bytes / 1e6:.1f} MB" if item.size_bytes else "",
            )
        console.print(table)
        console.print(
            f"{report.count('new')} new, {report.count('cached')} cached, "
            f"{report.count('failed')} failed; manifest "
            f"{'written' if report.manifest_written else 'unchanged'}: {report.manifest_path}"
        )
        failures += report.count("failed")
    if failures:
        raise typer.Exit(code=1)


@ingest.command("parse")
def ingest_parse(
    ticker: Annotated[
        list[str], typer.Option("--ticker", "-t", help="Ticker symbol; repeat for several.")
    ],
) -> None:
    """Parse downloaded filings into sectioned JSONL with provenance and offsets."""
    settings = get_settings()
    failures = 0
    for symbol in ticker:
        try:
            report = parse_ticker(settings, symbol)
        except FileNotFoundError as exc:
            console.print(f"[red]{escape(str(exc))}[/red]")
            failures += 1
            continue
        table = Table(title=f"{report.ticker}: parsed filings")
        table.add_column("form")
        table.add_column("period")
        table.add_column("sections", justify="right")
        table.add_column("chars", justify="right")
        table.add_column("coverage")
        for result in report.results:
            missing = result.coverage.missing
            if not result.ok:
                coverage = (
                    "[red]missing "
                    + ", ".join(f"{p}.{i}" for p, i in result.coverage.missing_critical)
                    + "[/red]"
                )
            elif missing:
                coverage = (
                    "[yellow]all critical; absent: "
                    + ", ".join(f"{p}.{i}" for p, i in missing)
                    + "[/yellow]"
                )
            else:
                coverage = "[green]complete[/green]"
            table.add_row(
                result.form,
                result.period_label,
                str(result.sections),
                f"{result.chars:,}",
                coverage,
            )
        for accession, message in report.errors:
            table.add_row("", accession, "", "", f"[red]{escape(message)}[/red]")
        console.print(table)
        written = sum(1 for r in report.results if r.written)
        console.print(
            f"{len(report.results)} filings parsed, {written} written, "
            f"{report.failures} failed -> {settings.processed_dir / report.ticker}"
        )
        failures += report.failures
    if failures:
        raise typer.Exit(code=1)


@evaluate.command("check")
def eval_check() -> None:
    """Verify every gold evidence span still resolves in the parsed filings."""
    settings = get_settings()
    path = questions_path(settings)
    if not path.exists():
        console.print(f"[red]no eval set at {path}[/red]")
        raise typer.Exit(code=2)
    questions = load_questions(path)
    checks = check_spans(settings, questions)
    answer_problems = check_gold_answers(questions)

    counts = counts_by_type(questions)
    console.print(
        f"{len(questions)} questions: "
        + ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items()))
    )
    failures = [c for c in checks if not c.ok]
    if failures:
        table = Table(title="spans that no longer resolve")
        table.add_column("question")
        table.add_column("accession")
        table.add_column("section")
        table.add_column("offsets")
        table.add_column("problem")
        for check in failures:
            table.add_row(
                check.question_id,
                check.span.accession,
                check.span.section,
                f"{check.span.char_start}:{check.span.char_end}",
                escape(check.detail),
            )
        console.print(table)
    for question_id, problem in answer_problems:
        console.print(f"[red]{question_id}: {escape(problem)}[/red]")

    console.print(f"{len(checks) - len(failures)}/{len(checks)} spans resolve")
    if failures or answer_problems:
        raise typer.Exit(code=1)
    console.print("[green]every span resolves and every lookup answer is in its evidence[/green]")


# Planned subcommands (see project/tickets.md):
#   rag index build       RAG-006  chunk + embed + store
#   rag ask "..."         RAG-010  grounded answer or refusal
#   rag eval retrieval    RAG-008  recall@k / MRR / nDCG
#   rag eval all          RAG-012  retrieval + faithfulness + abstention


if __name__ == "__main__":
    app()
