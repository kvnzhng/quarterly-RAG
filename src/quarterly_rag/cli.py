"""`rag` command line entry point. Subcommands grow with the tickets."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from quarterly_rag import __version__
from quarterly_rag.config import get_settings

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
        shown = "***" if "secret" in key and value else str(value)
        table.add_row(key, shown)
    console.print(table)


# Planned subcommands (see project/tickets.md):
#   rag doctor            RAG-002  check Ollama, models, data dirs
#   rag ingest download   RAG-003  fetch filings from EDGAR
#   rag ingest parse      RAG-004  filings -> sectioned text
#   rag index build       RAG-006  chunk + embed + store
#   rag ask "..."         RAG-010  grounded answer or refusal
#   rag eval retrieval    RAG-008  recall@k / MRR / nDCG
#   rag eval all          RAG-012  retrieval + faithfulness + abstention


if __name__ == "__main__":
    app()
