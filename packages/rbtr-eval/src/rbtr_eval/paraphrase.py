"""`rbtr-eval paraphrase` subcommand.

Reads sampled queries from extract, loads symbol content
from the index, and generates LLM-paraphrased concept queries
via pydantic-ai.  Output is a parquet file of `QueryRow` rows
with `provenance="concept"`.

The model string comes from `--model` (CLI / DVC var).
Endpoint and API key come from environment variables — no code
changes needed to switch providers.  pydantic-ai's
`OpenAIProvider` reads `OPENAI_BASE_URL` and `OPENAI_API_KEY`
automatically; any OpenAI-compatible endpoint (OpenRouter,
Together, local vLLM, etc.) works.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import dataframely as dy
import minijinja
import polars as pl
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import AgentRunError, ModelHTTPError
from pydantic_ai.models import Model

from rbtr.cli.output import ProgressCallback, progress_reporter
from rbtr.git import read_head
from rbtr.index.frames import ChunkContentRow
from rbtr.index.models import CODE_KINDS, ChunkKind
from rbtr.index.store import IndexStore
from rbtr_eval.formatting import heading_label, md_table
from rbtr_eval.schemas import ConceptQuery, QueryRow

log = logging.getLogger(__name__)

_ALL_CHUNK_CONTENT_SQL = (
    resources.files("rbtr_eval.sql").joinpath("all_chunk_content.sql").read_text()
)


# ── Deps ─────────────────────────────────────────────────────────────


@dataclass
class SymbolContext:
    """Per-symbol data passed as pydantic-ai deps."""

    language: str
    symbol_kind: ChunkKind
    excluded_identifiers: list[str]


# ── Agent ────────────────────────────────────────────────────────────

paraphrase_agent: Agent[SymbolContext, ConceptQuery] = Agent(
    output_type=ConceptQuery,
    deps_type=SymbolContext,
    output_retries=3,
)


@paraphrase_agent.instructions
def _build_instructions(ctx: RunContext[SymbolContext]) -> str:
    excluded = ", ".join(f"`{e}`" for e in ctx.deps.excluded_identifiers)
    if ctx.deps.symbol_kind in CODE_KINDS:
        return f"""\
You are a code search query generator. Given a source code \
symbol, write ONE sentence that a developer would type into \
a search box to find this code.

Rules:
- Do NOT use any of these identifiers: {excluded}
- Describe purpose and intent, not implementation details.
- Write a natural search query, not a docstring or comment.

Example input:

    def _merge_dicts(a, b):
        merged = a.copy()
        merged.update(b)
        return merged

Output: combine two dictionaries into one

Example input:

    class UserAuthenticator:
        def __init__(self, store):
            self.store = store
        def authenticate(self, username, password):
            ...

Output: verify user login credentials"""
    return f"""\
You are a code search query generator. Given a documentation or \
configuration section, write ONE sentence that a developer would \
type into a search box to find this content.

Rules:
- Do NOT use any of these identifiers: {excluded}
- Describe the topic and purpose, not the heading text.
- Write a natural search query, not a summary or comment.

Example input:

    # Environment Variable Precedence

    When .env files and shell variables define
    the same key, the shell variable takes precedence.

Output: how do env vars override config files

Example input:

    [build]
    output-dir = "dist"
    target = "wheels"

Output: where to set the build output path"""


@paraphrase_agent.output_validator
def _check_excluded(ctx: RunContext[SymbolContext], data: ConceptQuery) -> ConceptQuery:
    lower = data.text.lower()
    for ident in ctx.deps.excluded_identifiers:
        if ident.lower() in lower:
            raise ModelRetry(ident)
    return data


# ── Symbol content from index ─────────────────────────────────────────


def _load_symbol_content(
    store: IndexStore,
    repo_path: str,
    repo_id: int,
) -> dy.DataFrame[ChunkContentRow]:
    """Load symbol content from the index.

    Returns all chunks at HEAD as a content-only frame.
    The caller (``paraphrase_symbols``) joins against
    ``QueryRow`` on primary-key columns, so only chunks that
    were sampled by extract survive.
    """
    sha = read_head(repo_path)
    if sha is None:
        msg = f"no HEAD in {repo_path}"
        raise SystemExit(msg)
    return store.get_chunks_frame(sha, repo_id=repo_id)


# ── Excluded identifiers ─────────────────────────────────────────────


def _excluded_identifiers(name: str, scope: str, file_path: str) -> list[str]:
    """Build the list of identifiers the LLM must not use.

    Includes the symbol name, scope parts, and path segments
    (stems ≥ 3 chars) so the paraphrase describes intent
    without leaking the symbol's identity.
    """
    excluded = {name}
    if scope:
        excluded.add(scope)
        for part in scope.split("."):
            if part:
                excluded.add(part)
    for segment in Path(file_path).parts:
        stem = Path(segment).stem
        if stem and len(stem) >= 3:
            excluded.add(stem)
    return sorted(excluded)


# ── Paraphrasing ─────────────────────────────────────────────────────


async def _paraphrase_one(
    results: list[dict[str, str | int]],
    completed: list[int],
    total: int,
    on_progress: ProgressCallback | None,
    slug: str,
    file_path: str,
    scope: str,
    name: str,
    symbol_kind: ChunkKind,
    line_start: int,
    language: str,
    content: str,
    model: str | Model,
    semaphore: asyncio.Semaphore,
) -> None:
    """Call the LLM for one symbol; append a QueryRow dict on success."""
    excluded = _excluded_identifiers(name, scope, file_path)
    deps = SymbolContext(language=language, symbol_kind=symbol_kind, excluded_identifiers=excluded)
    async with semaphore:
        try:
            result = await paraphrase_agent.run(
                f"```{language}\n{content}\n```",
                deps=deps,
                model=model,
            )

        except ModelHTTPError as exc:
            log.warning(
                "HTTP %d from %s for %s: %s",
                exc.status_code,
                exc.model_name,
                name,
                exc.body,
            )
            completed.append(1)
            if on_progress is not None:
                on_progress(len(completed), total)
            return
        except (AgentRunError, ValueError, KeyError) as exc:
            log.warning("LLM call failed for %s: %s", name, exc)
            completed.append(1)
            if on_progress is not None:
                on_progress(len(completed), total)
            return
    results.append(
        {
            "slug": slug,
            "file_path": file_path,
            "scope": scope,
            "name": name,
            "symbol_kind": symbol_kind.value,
            "line_start": line_start,
            "language": language,
            "provenance": "concept",
            "text": result.output.text,
        }
    )
    completed.append(1)
    if on_progress is not None:
        on_progress(len(completed), total)


def paraphrase_symbols(
    queries: dy.DataFrame[QueryRow],
    store: IndexStore,
    repo_path: str,
    repo_id: int,
    model: str | Model,
    *,
    concurrency: int = 10,
    on_progress: ProgressCallback | None = None,
) -> dy.DataFrame[QueryRow]:
    """Generate concept queries for each unique symbol in *queries*.

    Loads symbol content from the index, calls the LLM
    concurrently for each symbol, and returns validated
    concept QueryRows.
    """
    symbols = _load_symbol_content(store, repo_path, repo_id)
    join_keys = ["file_path", "scope", "name", "line_start"]

    joined = (
        queries.select("slug", *join_keys, "symbol_kind")
        .unique(subset=join_keys)
        .join(symbols.select(*join_keys, "language", "content"), on=join_keys, how="inner")
    )

    total = joined.height
    log.info("Paraphrasing %d symbols with concurrency %d", total, concurrency)
    semaphore = asyncio.Semaphore(concurrency)

    async def _run_all() -> list[dict[str, str | int]]:
        results: list[dict[str, str | int]] = []
        completed: list[int] = []

        async with asyncio.TaskGroup() as tg:
            for row in joined.iter_rows(named=True):
                tg.create_task(
                    _paraphrase_one(
                        results,
                        completed,
                        total,
                        on_progress,
                        row["slug"],
                        row["file_path"],
                        row["scope"],
                        row["name"],
                        ChunkKind(row["symbol_kind"]),
                        row["line_start"],
                        row["language"],
                        row["content"],
                        model,
                        semaphore,
                    )
                )

        return results

    rows = asyncio.run(_run_all())
    log.info("Generated %d concept queries from %d symbols", len(rows), total)
    if not rows:
        return QueryRow.create_empty()
    return pl.DataFrame(rows).pipe(QueryRow.validate, cast=True)


# ── Report ───────────────────────────────────────────────────────────


def _render_paraphrase_report(
    concepts: dy.DataFrame[QueryRow],
    model: str,
    store: IndexStore,
) -> str:
    """Render PARAPHRASE.md from concept + index data."""
    repo_table = concepts.group_by("slug").agg(pl.len().alias("n")).sort("slug")

    lang_table = concepts.group_by("language").agg(pl.len().alias("n")).sort("n", descending=True)

    # Look up source content for sampled examples from the index.
    sampled = concepts.sample(10, seed=42)
    content_frame = store._cursor.execute(_ALL_CHUNK_CONTENT_SQL).pl()
    examples_df = sampled.join(
        content_frame,
        on=["file_path", "name", "line_start"],
        how="left",
    )
    examples = [
        {
            "slug": r["slug"],
            "name": r["name"],
            "heading": heading_label(r["name"]),
            "language": r["language"],
            "content": r["content"] or "",
            "concept": r["text"],
        }
        for r in examples_df.iter_rows(named=True)
    ]

    template = resources.files("rbtr_eval.templates").joinpath("paraphrase.md.j2").read_text()
    return minijinja.Environment().render_str(
        template,
        model=model,
        total=concepts.height,
        repo_table=md_table(repo_table),
        lang_table=md_table(lang_table),
        examples=examples,
    )


# ── CLI subcommands ──────────────────────────────────────────────────


class ParaphraseCmd(BaseModel):
    """Generate LLM-paraphrased concept queries for sampled symbols."""

    slug: str = Field(description="Repo slug.")
    symbols: Path = Field(description="Path to the subsampled queries parquet from extract.")
    repo_path: Path = Field(description="Path to the checked-out repository.")
    data_dir: Path = Field(description="Directory for the DuckDB index.")
    out: Path = Field(description="Output path for concept parquet.")
    model: str = Field(
        description="pydantic-ai model string (e.g. openai:anthropic/claude-haiku).",
    )
    concurrency: int = Field(10, description="Max concurrent LLM requests.")

    def cli_cmd(self) -> None:
        query_df = pl.read_parquet(self.symbols).pipe(QueryRow.validate, cast=True)
        store = IndexStore(str(self.data_dir / "index.duckdb"))
        repo_path = str(Path(self.repo_path).resolve())
        repo_id = store.get_repo_id(repo_path)
        if repo_id is None:
            msg = f"Repo not indexed: {repo_path}"
            raise SystemExit(msg)

        with progress_reporter("paraphrase") as (on_progress,):
            result = paraphrase_symbols(
                query_df,
                store,
                repo_path,
                repo_id,
                self.model,
                concurrency=self.concurrency,
                on_progress=on_progress,
            )

        self.out.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(self.out)
        log.info("Wrote %d rows to %s", result.height, self.out)


class ParaphraseReportCmd(BaseModel):
    """Generate PARAPHRASE.md from concept + extract data."""

    concept_dir: Path = Field(description="Directory holding concept parquet files.")
    data_dir: Path = Field(description="Directory for the DuckDB index.")
    model: str = Field(description="Model string used for paraphrasing.")
    report: Path = Field(description="Output path for PARAPHRASE.md.")

    def cli_cmd(self) -> None:
        concepts = pl.concat(
            [pl.read_parquet(f) for f in sorted(self.concept_dir.glob("*.parquet"))]
        ).pipe(QueryRow.validate, cast=True)
        store = IndexStore(str(self.data_dir / "index.duckdb"))

        self.report.parent.mkdir(parents=True, exist_ok=True)
        self.report.write_text(
            _render_paraphrase_report(concepts, self.model, store),
            encoding="utf-8",
        )
