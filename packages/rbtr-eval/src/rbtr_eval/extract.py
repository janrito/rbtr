"""`rbtr-eval extract` subcommand.

Walks one repo, builds a polars frame of documented symbols,
deterministically samples up to `sample_cap`, and writes:

* `<out_dir>/<slug>.queries.parquet` - one row per sampled
  query (validated against `QueryRow`).
* `<out_dir>/<slug>.header.parquet`  - one-row repo header
  (validated against `RepoHeader`).

This module is the only place rbtr-eval touches `tree-sitter`
/ rbtr's language plugins; re-implementing per-language
docstring recognition would mean rolling our own comment
parsing -- the smell rbtr-eval is built to avoid.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import dataframely as dy
import polars as pl
import pygit2
from pydantic import BaseModel, Field

from rbtr.config import config
from rbtr.git import list_files, read_head
from rbtr.index.models import ChunkKind
from rbtr.index.treesitter import extract_doc_spans
from rbtr.languages import get_manager
from rbtr.rbtrignore import load_ignore
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


# ── Symbol walk ──────────────────────────────────────────────────────────────


def iter_queries(repo_path: Path, slug: str) -> Iterator[dict[str, str | int]]:
    """Walk *repo_path*, yielding one query-row dict per documented symbol.

    Delegates docstring identification to `extract_doc_spans`
    so it matches the indexer exactly.  Only function / class /
    method kinds are kept; anonymous names are dropped; symbols
    whose docstring produces no usable first sentence are
    skipped.  Row dicts match `QueryRow`; validation happens
    when the caller builds the frame.
    """
    repo = pygit2.Repository(str(repo_path))
    mgr = get_manager()
    repo_root = Path(repo.workdir).resolve() if repo.workdir else Path.cwd()
    ignore = load_ignore(repo_root)

    for entry in list_files(repo, "HEAD", max_file_size=config.max_file_size, ignore=ignore):
        lang_id = mgr.detect_language(entry.path)
        if lang_id is None:
            continue
        reg = mgr.get_registration(lang_id)
        if reg is None or reg.query is None:
            continue
        grammar = mgr.load_grammar(lang_id)
        if grammar is None:
            continue

        for span in extract_doc_spans(
            entry.content,
            grammar,
            reg.query,
            scope_types=reg.scope_types,
            doc_comment_node_types=reg.doc_comment_node_types,
        ):
            if span.kind not in (ChunkKind.FUNCTION, ChunkKind.CLASS, ChunkKind.METHOD):
                continue
            if not span.name or span.name == "<anonymous>":
                continue
            doc_bytes = b"\n".join(entry.content[s:e] for s, e in span.ranges)
            text = first_sentence(doc_bytes.decode("utf-8", errors="replace"))
            if text is None:
                continue
            yield {
                "slug": slug,
                "file_path": entry.path,
                "scope": span.scope,
                "name": span.name,
                "symbol_kind": span.kind.value,
                "line_start": span.line_start,
                "language": lang_id,
                "text": text,
            }


def sample_queries(
    candidates: dy.DataFrame[QueryRow], *, seed: int, sample_cap: int
) -> dy.DataFrame[QueryRow]:
    """Deterministically sample up to *sample_cap* queries per repo.

    Stratifies by `slug` so a multi-repo candidate frame would
    be sampled independently per repo (extract is per-repo
    today, but the idiom is the right one).  Results are
    sorted on the primary-key columns so the parquet file is
    byte-stable across runs.
    """
    sampled = candidates.group_by("slug").map_groups(
        lambda g: g.sample(n=min(len(g), sample_cap), seed=seed, shuffle=False)
    )
    return sampled.sort(["slug", "file_path", "line_start", "scope", "name"]).pipe(
        QueryRow.validate, cast=True
    )


# ── CLI subcommand ───────────────────────────────────────────────────────────


class ExtractCmd(BaseModel):
    """Sample docstring-derived queries from one repo.

    Writes two parquet files: `<out_dir>/<slug>.queries.parquet`
    and `<out_dir>/<slug>.header.parquet`.
    """

    slug: str = Field(description="Repo slug (used for file naming and as a row column).")
    repo_path: Path = Field(description="Path to the checked-out repository.")
    out_dir: Path = Field(description="Directory receiving `<slug>.queries.parquet` + header.")
    seed: int = Field(0, description="Deterministic-sampling RNG seed.")
    sample_cap: int = Field(300, description="Maximum number of queries to keep per repo.")

    def cli_cmd(self) -> None:
        sha = read_head(str(self.repo_path))
        if sha is None:
            msg = f"no HEAD in {self.repo_path}"
            raise SystemExit(msg)

        candidates = pl.DataFrame(list(iter_queries(self.repo_path, self.slug))).pipe(
            QueryRow.validate, cast=True
        )
        sampled = sample_queries(candidates, seed=self.seed, sample_cap=self.sample_cap)

        header = pl.DataFrame(
            {
                "slug": [self.slug],
                "sha": [sha],
                "seed": [self.seed],
                "sample_cap": [self.sample_cap],
                "n_documented": [candidates.height],
                "n_sampled": [sampled.height],
            },
        ).pipe(RepoHeader.validate, cast=True)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        sampled.write_parquet(self.out_dir / f"{self.slug}.queries.parquet")
        header.write_parquet(self.out_dir / f"{self.slug}.header.parquet")
