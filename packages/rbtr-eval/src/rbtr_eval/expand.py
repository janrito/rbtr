"""`rbtr-eval expand` subcommand.

Reads sampled queries from extract + concept parquets,
classifies each query, and generates LLM-produced keywords
and variants for every query via pydantic-ai.  Output is a
single parquet file of `ExpansionRow` rows consumed by
`measure --expansion-dir`.

All query kinds get both keywords and variants so the
downstream ablation in `measure` can isolate the effect of
each expansion channel per kind.  The prompt is tailored per
kind, but neither channel is mandatory: if the model returns
no variants for a query, that query simply has none.

The model string comes from `--model` (CLI / DVC var).
Endpoint and API key come from environment variables —
pydantic-ai's `OpenAIProvider` reads `OPENAI_BASE_URL` and
`OPENAI_API_KEY` automatically.
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
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import AgentRunError, ModelHTTPError
from pydantic_ai.models import Model

from rbtr.cli.output import ProgressCallback, progress_reporter
from rbtr.index.classify import classify_query
from rbtr.index.models import QueryKind
from rbtr_eval.formatting import heading_label, md_table
from rbtr_eval.queries import load_all_queries
from rbtr_eval.schemas import ExpansionRow, QueryRow

log = logging.getLogger(__name__)


# ── Deps ─────────────────────────────────────────────────────────────


@dataclass
class ExpansionContext:
    """Per-query data passed as pydantic-ai deps."""

    kind: QueryKind
    query: str


# ── Agent output ─────────────────────────────────────────────────────


class ExpansionResult(BaseModel):
    """LLM output: keywords and optional variant rephrases."""

    keywords: list[str] = Field(
        min_length=1,
        max_length=7,
        description="3-5 synonym identifiers or alternative search terms.",
    )
    variants: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="1-2 semantically diverse rephrasings of the query.",
    )

    @field_validator("keywords")
    @classmethod
    def _keywords_non_empty(cls, v: list[str]) -> list[str]:
        cleaned = [k.strip() for k in v if k.strip()]
        if not cleaned:
            msg = "keywords must contain at least one non-empty string"
            raise ValueError(msg)
        return cleaned


# ── Agent ────────────────────────────────────────────────────────────

expand_agent: Agent[ExpansionContext, ExpansionResult] = Agent(
    output_type=ExpansionResult,
    deps_type=ExpansionContext,
    retries={"output": 3},
)


@expand_agent.instructions
def _build_instructions(ctx: RunContext[ExpansionContext]) -> str:
    if ctx.deps.kind == QueryKind.CONCEPT:
        return """\
You are a code search query expansion engine. Given a \
natural-language search query, produce synonym keywords and \
variant rephrasings that a developer might use to find the same code.

Rules:
- `keywords`: 3-5 synonym identifiers or alternative terms. \
Use names a developer would actually type: function names, \
class names, config keys, CLI flags.
- `variants`: 1-2 semantically diverse rephrasings using \
different vocabulary. Each variant must stand alone as a \
valid search query.
- Do NOT repeat the original query in keywords or variants.

Example input: "how does idle unload work"
keywords: ["timeout", "evict", "unload_model", "release_gpu"]
variants: ["when does the model get released from memory"]

Example input: "where are embeddings stored"
keywords: ["vector_store", "embedding_table", "persist", "save_vectors"]
variants: ["how does the index persist embedding vectors"]

Example input: "how to configure logging level"
keywords: ["log_level", "set_verbosity", "debug", "LOG_CONFIG"]
variants: ["change the debug output verbosity"]"""

    if ctx.deps.kind == QueryKind.CODE:
        return """\
You are a code search query expansion engine. Given a short \
code fragment (a signature, statement, or expression), produce \
keywords and variant phrasings that help locate the same code.

Rules:
- `keywords`: 3-5 identifiers or terms drawn from the fragment \
or its likely surroundings: function and parameter names, types, \
called methods, related identifiers.
- `variants`: 1-2 alternative phrasings of the fragment — either \
the same logic written differently, or a short natural-language \
description of what the fragment does. Each variant must stand \
alone as a valid search query.
- Do NOT repeat the original query verbatim in keywords or variants.

Example input: "def fuse_scores(candidates, query, *, alpha"
keywords: ["fuse_scores", "alpha", "candidates", "weighted_fusion", "rank"]
variants: ["combine candidate scores with alpha weighting"]

Example input: "for item in self._cache"
keywords: ["_cache", "iterate", "cache_entries", "items"]
variants: ["loop over the cache entries"]

Example input: "raise IndexNotBuiltError(msg)"
keywords: ["IndexNotBuiltError", "raise", "index_missing", "not_built"]
variants: ["signal that the index has not been built yet"]"""

    return """\
You are a code search query expansion engine. Given a \
symbol name or short identifier query, produce alternative \
names the symbol might have in different codebases or \
naming conventions, plus a short description of what it does.

Rules:
- `keywords`: 3-5 alternative names. Consider snake_case, \
camelCase, abbreviations, and semantic synonyms.
- `variants`: 1-2 short natural-language descriptions of what \
the symbol likely does. Each variant must stand alone as a \
valid search query.
- Do NOT repeat the original query in keywords or variants.

Example input: "fuse_scores"
keywords: ["merge_scores", "combine_results", "blend_scores", "aggregate_scores"]
variants: ["combine multiple ranking signals into one score"]

Example input: "HttpClient"
keywords: ["http_client", "RequestSession", "WebClient", "ApiClient"]
variants: ["send HTTP requests to a remote server"]

Example input: "retry_backoff"
keywords: ["exponential_backoff", "retry_delay", "backoff_strategy", "wait_retry"]
variants: ["wait progressively longer between retry attempts"]"""


@expand_agent.output_validator
def _check_expansion(ctx: RunContext[ExpansionContext], data: ExpansionResult) -> ExpansionResult:
    query = ctx.deps.query.lower()
    for kw in data.keywords:
        if kw.lower() == query:
            msg = f"keyword {kw!r} is identical to the query"
            raise ModelRetry(msg)
    return data


# ── Expansion ────────────────────────────────────────────────────────


async def _expand_one(
    results: list[dict[str, str | int | list[str]]],
    completed: list[int],
    total: int,
    on_progress: ProgressCallback | None,
    slug: str,
    file_path: str,
    scope: str,
    name: str,
    line_start: int,
    provenance: str,
    text: str,
    kind: QueryKind,
    model: str | Model,
    semaphore: asyncio.Semaphore,
) -> None:
    """Call the LLM for one query; append an ExpansionRow dict on success."""
    async with semaphore:
        try:
            result = await expand_agent.run(
                text,
                deps=ExpansionContext(kind=kind, query=text),
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
    output = result.output
    results.append(
        {
            "slug": slug,
            "file_path": file_path,
            "scope": scope,
            "name": name,
            "line_start": line_start,
            "provenance": provenance,
            "query_kind": kind.value,
            "keywords": output.keywords,
            "variants": output.variants,
        },
    )
    completed.append(1)
    if on_progress is not None:
        on_progress(len(completed), total)


def expand_queries(
    queries: dy.DataFrame[QueryRow],
    model: str | Model,
    *,
    concurrency: int = 10,
    on_progress: ProgressCallback | None = None,
) -> dy.DataFrame[ExpansionRow]:
    """Generate keywords and variants for each query.

    All query kinds get both keywords and variants; the prompt
    is tailored per kind.  Neither channel is mandatory — a
    query may end up with no variants if the model returns
    none.
    """
    # Classify all queries in polars.
    classified = queries.with_columns(
        pl.col("text")
        .map_elements(lambda q: classify_query(q).value, return_dtype=pl.String)
        .alias("query_kind"),
    )

    total = classified.height
    log.info("Expanding %d queries", total)

    semaphore = asyncio.Semaphore(concurrency)

    async def _run_all() -> list[dict[str, str | int | list[str]]]:
        results: list[dict[str, str | int | list[str]]] = []
        completed: list[int] = []

        async with asyncio.TaskGroup() as tg:
            for row in classified.iter_rows(named=True):
                tg.create_task(
                    _expand_one(
                        results,
                        completed,
                        total,
                        on_progress,
                        row["slug"],
                        row["file_path"],
                        row["scope"],
                        row["name"],
                        row["line_start"],
                        row["provenance"],
                        row["text"],
                        QueryKind(row["query_kind"]),
                        model,
                        semaphore,
                    ),
                )

        return results

    rows = asyncio.run(_run_all())
    log.info("Expanded %d of %d queries", len(rows), total)

    if rows:
        return pl.DataFrame(rows).pipe(ExpansionRow.validate, cast=True)
    return ExpansionRow.create_empty()


# ── CLI subcommand ───────────────────────────────────────────────────


class ExpandCmd(BaseModel):
    """Generate LLM-produced keywords and variants for eval queries."""

    per_repo_dir: Path = Field(description="Directory with per-repo query parquets.")
    concept_dir: Path = Field(description="Directory with concept parquets.")
    out: Path = Field(description="Output path for expansion parquet.")
    model: str = Field(
        description="pydantic-ai model string (e.g. openai-chat:zai-org/GLM-5.1).",
    )
    concurrency: int = Field(10, description="Max concurrent LLM requests.")
    report: Path = Field(description="Output path for EXPANSION.md report.")

    def cli_cmd(self) -> None:
        queries = load_all_queries(self.per_repo_dir, self.concept_dir)

        with progress_reporter("expand") as (on_progress,):
            result = expand_queries(
                queries,
                self.model,
                concurrency=self.concurrency,
                on_progress=on_progress,
            )

        self.out.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(self.out)
        log.info("Wrote %d rows to %s", result.height, self.out)

        self.report.parent.mkdir(parents=True, exist_ok=True)
        self.report.write_text(
            _render_expansion_report(result, queries, self.model),
            encoding="utf-8",
        )


# ── Report ───────────────────────────────────────────────────────────


def _render_expansion_report(
    expansions: dy.DataFrame[ExpansionRow],
    queries: dy.DataFrame[QueryRow],
    model: str,
) -> str:
    """Render EXPANSION.md from expansion parquet data.

    *queries* supplies the original query `text` and `language`
    (not carried on `ExpansionRow`) so examples can show the
    full query — e.g. the code fragment the model actually
    expanded — rather than just the symbol name.
    """
    total = expansions.height
    expanded = expansions.filter(
        pl.col("keywords").list.len() > 0,
    ).join(
        queries.select(*ExpansionRow.primary_key(), "text", "language"),
        on=ExpansionRow.primary_key(),
        how="left",
    )
    n_expanded = expanded.height

    # Summary table.
    summary = pl.DataFrame(
        {
            "field": ["model", "total queries", "expanded"],
            "value": [
                f"`{model}`",
                str(total),
                f"{n_expanded} / {total} ({n_expanded * 100 // total}%)" if total else "0",
            ],
        },
    )

    # Per-kind breakdown.
    kind_df = (
        expansions.group_by("query_kind")
        .agg(
            pl.len().alias("n"),
            pl.col("keywords").list.len().mean().round(1).alias("avg_keywords"),
            pl.col("variants").list.len().mean().round(1).alias("avg_variants"),
        )
        .sort("query_kind")
    )

    # Per-repo breakdown.
    repo_df = (
        expansions.group_by("slug")
        .agg(
            pl.len().alias("total"),
            (pl.col("keywords").list.len() > 0).sum().alias("expanded"),
        )
        .with_columns(
            pl.format(
                "{}%",
                pl.col("expanded") * 100 // pl.col("total"),
            ).alias("rate"),
        )
        .sort("slug")
    )

    # Per-provenance breakdown.
    prov_df = (
        expansions.group_by("provenance")
        .agg(
            pl.len().alias("total"),
            (pl.col("keywords").list.len() > 0).sum().alias("expanded"),
        )
        .with_columns(
            pl.format(
                "{}%",
                pl.col("expanded") * 100 // pl.col("total"),
            ).alias("rate"),
        )
        .sort("provenance")
    )

    # Sampled examples: up to 5 per kind. Show the actual query
    # text (the fragment the model expanded), not the name.
    examples: list[dict[str, str]] = []
    for kind in (QueryKind.CONCEPT, QueryKind.IDENTIFIER, QueryKind.CODE):
        # Unique by name so the sample never repeats a symbol
        # (e.g. many GitHub workflows share the name `on`).
        subset = expanded.filter(pl.col("query_kind") == kind.value).unique(
            subset="name", keep="first"
        )
        if subset.height > 0:
            sample = subset.sample(min(5, subset.height), seed=42)
            for row in sample.iter_rows(named=True):
                examples.append(
                    {
                        "kind": row["query_kind"],
                        "name": row["name"],
                        "heading": heading_label(row["name"]),
                        "slug": row["slug"],
                        "language": row["language"] or "",
                        "query": row["text"] or row["name"],
                        "keywords": ", ".join(row["keywords"]),
                        "variants": ", ".join(row["variants"]) if row["variants"] else "—",
                    },
                )

    template = resources.files("rbtr_eval.templates").joinpath("expansion.md.j2").read_text()
    return minijinja.Environment().render_str(
        template,
        summary_table=md_table(summary),
        kind_table=md_table(kind_df),
        repo_table=md_table(repo_df),
        provenance_table=md_table(prov_df),
        examples=examples,
    )
