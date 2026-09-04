"""`rag` command line entry point. Subcommands grow with the tickets."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from quarterly_rag import __version__
from quarterly_rag.config import get_settings
from quarterly_rag.doctor import failed, run_doctor

app = typer.Typer(help="Local RAG over SEC 10-Q/10-K filings.", no_args_is_help=True)
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


# Planned subcommands (see project/tickets.md):
#   rag ingest download   RAG-003  fetch filings from EDGAR
#   rag ingest parse      RAG-004  filings -> sectioned text
#   rag eval check        RAG-019  every gold evidence span resolves into the parsed filings
#   rag index build       RAG-006  chunk + embed + store
#   rag ask "..."         RAG-010  grounded answer or refusal
#   rag eval retrieval    RAG-008  recall@k / MRR / nDCG
#   rag eval all          RAG-012  retrieval + faithfulness + abstention


if __name__ == "__main__":
    app()
