"""`rbtr-eval extract` subcommand.

Reads every measurable chunk kind from the index (all of
`ChunkKind` except the eval's `EXCLUDED_KINDS`), generates
name / body / docstring queries — each strategy self-gating on
the chunk's content — subsamples per `(slug, language,
provenance)` to a target count, and writes one queries parquet
plus a header.
"""

from __future__ import annotations

import re
from pathlib import Path

import dataframely as dy
import polars as pl
from pydantic import BaseModel, Field

from rbtr.index.identity import SCOPE_SEPARATOR
from rbtr.index.store import IndexStore
from rbtr.languages.manager import get_manager
from rbtr.languages.registration import QueryExtraction
from rbtr.languages.treesitter import extract_doc_spans
from rbtr_eval.kinds import EXCLUDED_KINDS
from rbtr_eval.queries import subsample
from rbtr_eval.schemas import QueryRow, RepoHeader

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n\s*\n")
_QUERY_MIN_LEN = 15
_QUERY_MAX_LEN = 200
_REJECT_LOOKAHEAD_RE = re.compile(r"^[\s/#*\-!\"'`]+")
_REJECT_PREFIXES = (
    "todo",
    "fixme",
    "xxx",
    "hack",
    "deprecated",
    "@param",
    "@return",
    "@returns",
    "@throws",
    "@type",
    "@see",
    "@link",
)


def first_sentence(doc_text: str) -> str | None:
    """Return the first sentence of *doc_text*, or None if unusable.

    Turns a raw docstring (comment markers and all) into a
    search-friendly query string.  Comment markers -- line
    comments, triple-quotes, block-comment delimiters, `*`
    gutters -- are kept in the returned text: stripping them
    means re-implementing per-language comment syntax, which
    is the smell rbtr-eval is built to avoid.  The indexer
    already saw `///` / `\"\"\"` / `#` inside chunk content;
    the bench treats them the same way.

    Splits on a sentence terminator followed by whitespace,
    or on a blank-line paragraph break.  Truncates at
    `_QUERY_MAX_LEN` (too long makes a poor query) if no
    terminator appears in the first paragraph.  Returns
    `None` when the candidate is shorter than `_QUERY_MIN_LEN`
    (stub comments like `hi`, `wip`) or starts with a
    boilerplate / scaffolding prefix (`todo`, `@param`, etc.)
    after skipping leading comment-marker noise.
    """
    cleaned = doc_text.strip()
    if not cleaned:
        return None
    parts = _SENTENCE_SPLIT_RE.split(cleaned, maxsplit=1)
    candidate = parts[0].strip() if parts else cleaned
    if len(candidate) > _QUERY_MAX_LEN:
        candidate = candidate[:_QUERY_MAX_LEN].rstrip()
    if len(candidate) < _QUERY_MIN_LEN:
        return None
    post_noise = _REJECT_LOOKAHEAD_RE.sub("", candidate).lower()
    if any(post_noise.startswith(p) for p in _REJECT_PREFIXES):
        return None
    return candidate


_BODY_MIN_LEN = 15
_BODY_MAX_LEN = 200
_BODY_LINE_RE = re.compile(r"[.!?]\s+|\n\s*\n")


def _body_sentence(body_text: str) -> str | None:
    """Return the first substantial sentence from *body_text*, or None.

    Strips leading / trailing whitespace and comment markers, then
    picks the first sentence that is at least `_BODY_MIN_LEN`
    characters long and doesn't start with a boilerplate prefix.
    """
    cleaned = body_text.strip()
    if not cleaned:
        return None
    parts = _BODY_LINE_RE.split(cleaned, maxsplit=1)
    candidate = parts[0].strip() if parts else cleaned
    if len(candidate) > _BODY_MAX_LEN:
        candidate = candidate[:_BODY_MAX_LEN].rstrip()
    if len(candidate) < _BODY_MIN_LEN:
        return None
    post_noise = _REJECT_LOOKAHEAD_RE.sub("", candidate).lower()
    if any(post_noise.startswith(p) for p in _REJECT_PREFIXES):
        return None
    return candidate


def queries_for_symbol(
    *,
    slug: str,
    file_path: str,
    scope: str,
    name: str,
    symbol_kind: str,
    line_start: int,
    language: str,
    content: str,
) -> list[dict[str, str | int]]:
    """Generate query-row dicts for one symbol.

    Always yields a `name` query.  Yields `body` if
    `first_sentence` produces a usable string from the
    content.  Yields `docstring` if `extract_doc_spans`
    finds a doc comment in the content.
    """
    queries: list[dict[str, str | int]] = []

    def _row(provenance: str, text: str) -> dict[str, str | int]:
        return {
            "slug": slug,
            "file_path": file_path,
            "scope": scope,
            "name": name,
            "symbol_kind": symbol_kind,
            "line_start": line_start,
            "language": language,
            "provenance": provenance,
            "text": text,
        }

    # Name query for chunks with a real name; anonymous chunks
    # (comments, raw chunks) and heading-less paragraphs yield
    # body / docstring only.
    if name and name != "<anonymous>":
        name_text = f"{scope}{SCOPE_SEPARATOR}{name}" if scope else name
        queries.append(_row("name", name_text))

    # Body query: first sentence of the full content.
    body_text = _body_sentence(content)
    if body_text is not None:
        queries.append(_row("body", body_text))

    # Docstring query: parse with tree-sitter to find doc spans.
    mgr = get_manager()
    reg = mgr.get_registration(language)
    extraction = reg.extraction if reg is not None else None
    if isinstance(extraction, QueryExtraction):
        grammar = mgr.grammar(language)
        if grammar is not None:
            content_bytes = content.encode("utf-8")
            for span in extract_doc_spans(
                content_bytes,
                grammar,
                extraction.query,
                scope_types=extraction.scope_types,
                class_scope_types=extraction.class_scope_types,
            ):
                doc_bytes = b"\n".join(content_bytes[s:e] for s, e in span.ranges)
                doc_text = first_sentence(doc_bytes.decode("utf-8", errors="replace"))
                if doc_text is not None:
                    queries.append(_row("docstring", doc_text))
                break  # one docstring per symbol

    return queries


def resolve_repo(store: IndexStore, slug: str) -> tuple[int, str]:
    """Find a repo in the store by slug; return `(repo_id, sha)`."""
    for rid, path in store.list_repos():
        if Path(path).name == slug:
            commits = store.list_indexed_commits(rid)
            if commits:
                return rid, commits[0][0]
    msg = f"repo {slug} not found or not indexed"
    raise SystemExit(msg)


def extract_queries(
    store: IndexStore,
    slug: str,
    repo_id: int,
    sha: str,
    *,
    min_per_language: int = 50,
) -> tuple[dy.DataFrame[QueryRow], int, list[dict[str, str | int]]]:
    """Generate queries for every measurable chunk.

    Chunks of any `ChunkKind` except `EXCLUDED_KINDS` are
    used; each of the name / body / docstring strategies emits a
    query only when the chunk supports it (a real name, a usable
    first sentence, a doc span).  Languages with fewer than
    *min_per_language* measurable chunks are dropped (too few for
    a stable MRR).

    Returns `(queries_frame, n_measurable_chunks, dropped_languages)`,
    where `dropped_languages` lists the skipped languages and their
    chunk counts.
    """
    chunks = store.get_chunks(sha, repo_id=repo_id)
    symbols = [c for c in chunks if c.kind not in EXCLUDED_KINDS]

    lang_counts = (
        pl.DataFrame({"language": [c.language for c in symbols]}).group_by("language").len()
    )
    keep_langs = set(
        lang_counts.filter(pl.col("len") >= min_per_language).get_column("language").to_list()
    )
    dropped = (
        lang_counts.filter(pl.col("len") < min_per_language)
        .rename({"len": "n_chunks"})
        .sort("language")
        .to_dicts()
    )

    rows: list[dict[str, str | int]] = []
    for c in symbols:
        if c.language not in keep_langs:
            continue
        rows.extend(
            queries_for_symbol(
                slug=slug,
                file_path=c.file_path,
                scope=c.scope,
                name=c.name,
                symbol_kind=c.kind.value,
                line_start=c.line_start,
                language=c.language,
                content=c.content,
            )
        )

    return pl.DataFrame(rows).pipe(QueryRow.validate, cast=True), len(symbols), dropped


class ExtractCmd(BaseModel):
    """Generate and subsample queries from indexed symbols."""

    slug: str = Field(description="Repo slug.")
    data_dir: Path = Field(description="Directory for the DuckDB index.")
    out_dir: Path = Field(description="Directory receiving queries parquet.")
    headers_dir: Path = Field(description="Directory receiving header parquet.")
    seed: int = Field(0, description="Deterministic-sampling RNG seed.")
    queries_per_cell: int = Field(
        50,
        description="Target queries per (slug, language, symbol_kind, provenance) cell.",
    )
    min_per_language: int = Field(
        50,
        description="Languages with fewer symbols than this are excluded.",
    )

    def cli_cmd(self) -> None:
        store = IndexStore(str(self.data_dir / "index.duckdb"), writable=True)
        repo_id, sha = resolve_repo(store, self.slug)

        all_queries, n_symbols, dropped = extract_queries(
            store,
            self.slug,
            repo_id,
            sha,
            min_per_language=self.min_per_language,
        )
        sampled = subsample(
            all_queries,
            queries_per_cell=self.queries_per_cell,
            seed=self.seed,
            strat_keys=("slug", "language", "symbol_kind", "provenance"),
        )

        header = pl.DataFrame(
            {
                "slug": [self.slug],
                "sha": [sha],
                "seed": [self.seed],
                "queries_per_cell": [self.queries_per_cell],
                "n_documented": [n_symbols],
                "n_queries": [sampled.height],
                "dropped_languages": [dropped],
            },
            schema_overrides={
                "dropped_languages": pl.List(
                    pl.Struct({"language": pl.String, "n_chunks": pl.UInt32})
                )
            },
        ).pipe(RepoHeader.validate, cast=True)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        sampled.write_parquet(self.out_dir / f"{self.slug}.parquet")
        self.headers_dir.mkdir(parents=True, exist_ok=True)
        header.write_parquet(self.headers_dir / f"{self.slug}.parquet")
