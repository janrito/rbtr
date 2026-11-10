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

from rbtr.config import config
from rbtr.domain.identity import SCOPE_SEPARATOR
from rbtr.domain.models import SnapshotRef
from rbtr.index.index import Index
from rbtr.languages.manager import get_manager
from rbtr.languages.registration import QueryExtraction
from rbtr.languages.treesitter import extract_doc_spans
from rbtr_eval.corpus import corpus_ref
from rbtr_eval.kinds import EXCLUDED_KINDS
from rbtr_eval.queries import subsample
from rbtr_eval.shared_schemas import QueryRow, RepoHeader

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


def _doc_sentence(chunk: dict[str, str]) -> str | None:
    """The first sentence of the doc comment *chunk*'s `content` holds, or None.

    Reads the content with the query of *chunk*'s `language`; a language
    with no query has no docstrings to find.
    """
    mgr = get_manager()
    reg = mgr.get_registration(chunk["language"])
    extraction = reg.extraction if reg is not None else None
    grammar = mgr.grammar(chunk["language"])
    if not isinstance(extraction, QueryExtraction) or grammar is None:
        return None
    content = chunk["content"].encode("utf-8")
    for span in extract_doc_spans(
        content,
        grammar,
        extraction.query,
        scope_types=extraction.scope_types,
        class_scope_types=extraction.class_scope_types,
    ):
        # One docstring per symbol: the first.
        doc = b"\n".join(content[s:e] for s, e in span.ranges)
        return first_sentence(doc.decode("utf-8", errors="replace"))
    return None


def symbol_queries(symbols: pl.DataFrame, slug: str) -> dy.DataFrame[QueryRow]:
    """The queries each of *symbols* yields: by its name, its body and its docstring.

    *symbols* holds chunk rows (`file_path`, `scope`, `name`, `kind`,
    `line_start`, `line_end`, `language`, `content`). A symbol with a
    name yields a `name` query, its scope-qualified name; one whose
    content opens with a usable sentence a `body` query; one with a doc
    comment a `docstring` query. Unnamed chunks (comments, raw slices,
    heading-less paragraphs) yield body and docstring queries only.
    """
    targets = symbols.with_columns(slug=pl.lit(slug), symbol_kind=pl.col("kind"))
    by_name = targets.filter(pl.col("name") != "").with_columns(
        provenance=pl.lit("name"),
        text=pl.when(pl.col("scope") != "")
        .then(pl.concat_str("scope", pl.lit(SCOPE_SEPARATOR), "name"))
        .otherwise("name"),
    )
    by_body = targets.with_columns(
        provenance=pl.lit("body"),
        text=pl.col("content").map_elements(_body_sentence, return_dtype=pl.String),
    )
    by_docstring = targets.with_columns(
        provenance=pl.lit("docstring"),
        text=pl.struct("content", "language").map_elements(_doc_sentence, return_dtype=pl.String),
    )
    return (
        pl.concat([by_name, by_body, by_docstring])
        .drop_nulls("text")
        .pipe(QueryRow.validate, cast=True)
    )


def extract_queries(
    index: Index,
    slug: str,
    at: SnapshotRef,
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
    symbols = index.chunks(at=at).filter(~pl.col("kind").is_in(list(EXCLUDED_KINDS)))

    lang_counts = symbols.group_by("language").len()
    dropped = (
        lang_counts.filter(pl.col("len") < min_per_language)
        .rename({"len": "n_chunks"})
        .sort("language")
        .to_dicts()
    )

    measured = symbols.join(
        lang_counts.filter(pl.col("len") >= min_per_language), on="language", how="semi"
    )
    return symbol_queries(measured, slug), symbols.height, dropped


class ExtractCmd(BaseModel):
    """Generate and subsample queries from indexed symbols."""

    slug: str = Field(description="Repo slug.")
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
        index = Index(str(config.db_path), writable=True)
        ref = corpus_ref(index, self.slug)

        all_queries, n_symbols, dropped = extract_queries(
            index,
            self.slug,
            ref,
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
                "sha": [ref.snapshot_sha],
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
