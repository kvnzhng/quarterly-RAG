"""Run the retrieval eval and write a report with a run record (RAG-008).

Every number this produces is traceable: the report names the commit, the corpus, the
chunker, the embedding model and variant, the store, and the overlap rule that decided
relevance. A number without that record is a draft (project conventions).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from quarterly_rag.chunking.base import Chunk
from quarterly_rag.chunking.build import iter_chunks
from quarterly_rag.config import Settings
from quarterly_rag.evaluation.metrics import (
    DEFAULT_KS,
    QuestionResult,
    evaluate_question,
    group_by,
    near_miss_rates,
    summarise,
)
from quarterly_rag.evaluation.questions import iter_answerable, load_questions, questions_path
from quarterly_rag.evaluation.relevance import DEFAULT_RULE, OverlapRule, is_relevant
from quarterly_rag.indexing.build import load_manifest
from quarterly_rag.ingestion.manifest import Manifest
from quarterly_rag.ingestion.parse import PARSER_VERSION
from quarterly_rag.retrieval.base import Retriever

REPORTS_DIRNAME = "reports"
TICKERS = ("AAPL", "NVDA")


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def _sha256(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


@dataclass
class RunRecord:
    """Everything needed to reproduce a number, or to explain why two differ."""

    timestamp: str
    git_commit: str
    git_dirty: bool
    corpus_hash: str
    eval_set_hash: str
    parser_version: str
    chunk_strategy: str
    chunk_words: int
    chunk_overlap_words: int
    embed_variant: str
    embedder: str
    embed_query_prefix: str
    embed_document_prefix: str
    vector_store: str
    indexed_chunks: int | None
    retriever: str
    k_values: list[int]
    overlap_rule: str
    filters: dict[str, object] | None
    question_count: int
    prompt_version: str | None = None
    """Set once generation is scored (RAG-012); retrieval uses no prompt."""


def build_run_record(
    settings: Settings,
    *,
    retriever_name: str,
    store: str,
    strategy: str,
    variant: str,
    ks: Sequence[int],
    rule: OverlapRule,
    question_count: int,
    filters: dict[str, object] | None = None,
) -> RunRecord:
    # The index manifest, not Settings, describes what was actually queried.
    manifest = load_manifest(settings, store, strategy, variant) or {}
    return RunRecord(
        timestamp=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        git_commit=_git("rev-parse", "--short", "HEAD"),
        git_dirty=bool(_git("status", "--porcelain")),
        corpus_hash=_sha256(*(Manifest.path_for(settings.raw_dir, t) for t in TICKERS)),
        eval_set_hash=_sha256(questions_path(settings)),
        parser_version=PARSER_VERSION,
        chunk_strategy=manifest.get("chunk_strategy", strategy),
        chunk_words=settings.chunk_words,
        chunk_overlap_words=settings.chunk_overlap_words,
        embed_variant=manifest.get("embed_variant", variant),
        embedder=manifest.get("embedder", ""),
        embed_query_prefix=settings.embed_query_prefix,
        embed_document_prefix=settings.embed_document_prefix,
        vector_store=manifest.get("store", store),
        indexed_chunks=manifest.get("chunks"),
        retriever=retriever_name,
        k_values=list(ks),
        overlap_rule=rule.describe(),
        filters=filters,
        question_count=question_count,
    )


@dataclass
class RetrievalReport:
    run: RunRecord
    results: list[QuestionResult] = field(default_factory=list)
    all_ranks: dict[str, list[int]] = field(default_factory=dict)
    ks: tuple[int, ...] = DEFAULT_KS
    skipped_unanswerable: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "run_record": asdict(self.run),
            "overall": summarise(self.results, self.all_ranks, self.ks).as_dict(),
            "by_ticker": {n: m.as_dict() for n, m in self._grouped("ticker").items()},
            "by_form": {n: m.as_dict() for n, m in self._grouped("form").items()},
            "by_type": {n: m.as_dict() for n, m in self._grouped("question_type").items()},
            "by_section": {n: m.as_dict() for n, m in self._grouped("section").items()},
            "near_miss": {
                f"@{cutoff}": {
                    name: round(value, 4)
                    for name, value in near_miss_rates(self.results, cutoff).items()
                }
                for cutoff in self.ks
            },
            "excluded_unanswerable": self.skipped_unanswerable,
            "per_question": [asdict(r) for r in self.results],
        }

    def _grouped(self, key: str):
        return group_by(self.results, key, self.all_ranks, self.ks)

    def write(self, settings: Settings) -> Path:
        stamp = self.run.timestamp.replace(":", "").replace("-", "")
        path = (
            Path(settings.data_dir).parent
            / REPORTS_DIRNAME
            / f"retrieval-{self.run.embed_variant}-{stamp}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n")
        return path


def run_retrieval_eval(
    settings: Settings,
    retriever: Retriever,
    *,
    store: str = "chroma",
    strategy: str = "fixed",
    variant: str = "raw",
    ks: Sequence[int] = DEFAULT_KS,
    rule: OverlapRule = DEFAULT_RULE,
    filters: dict[str, object] | None = None,
) -> RetrievalReport:
    questions = load_questions(questions_path(settings))
    answerable = list(iter_answerable(questions))
    corpus: list[Chunk] = [c for t in TICKERS for c in iter_chunks(settings, t, strategy)]
    if not corpus:
        raise FileNotFoundError(f"no chunks for strategy {strategy!r}; run `rag chunk build` first")
    forms = {c.accession: c.form for c in corpus}

    report = RetrievalReport(
        run=build_run_record(
            settings,
            retriever_name=retriever.name,
            store=store,
            strategy=strategy,
            variant=variant,
            ks=ks,
            rule=rule,
            question_count=len(answerable),
            filters=filters,
        ),
        ks=tuple(ks),
        skipped_unanswerable=len(questions) - len(answerable),
    )

    top_k = max(ks)
    for question in answerable:
        results = retriever.retrieve(question.question, k=top_k, where=filters)
        relevant_total = sum(1 for c in corpus if is_relevant(c, question, rule))
        report.all_ranks[question.id] = [
            r.rank for r in results if is_relevant(r.chunk, question, rule)
        ]
        report.results.append(
            evaluate_question(
                question,
                results,
                relevant_total=relevant_total,
                form=forms.get(question.evidence[0].accession, "") if question.evidence else "",
                rule=rule,
            )
        )
    return report
