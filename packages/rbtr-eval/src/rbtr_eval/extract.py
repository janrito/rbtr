"""`rbtr-eval extract` and `rbtr-eval merge-dataset` subcommands.

`extract` walks one repo, samples docstring-derived queries,
writes a per-repo JSONL.  `merge-dataset` concatenates per-repo
JSONLs into one dataset.

This module is the *only* place rbtr-eval imports from `rbtr` —
it uses several rbtr APIs (see the imports below) to identify
documented symbols using exactly the indexer's view.  Re-
implementing any of that would mean rolling our own per-language
comment / symbol parsing, which is the smell rbtr-eval is
built to avoid.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pygit2
from pydantic import BaseModel, Field

from rbtr.config import config
from rbtr.git import FileEntry, list_files
from rbtr.index.treesitter import extract_doc_spans
from rbtr.languages import get_manager
from rbtr.rbtrignore import load_ignore

# ── Query projection ───────────────────────────────────────────────────────────
#
# `first_sentence` turns a raw docstring (comment markers and
# all) into a search-friendly query string.  We deliberately do
# *not* strip comment markers — doing so means re-implementing
# per-language comment syntax, which is the smell rbtr-eval is
# built to avoid.  The indexer already saw `///` / `"""` / `#`
# inside chunk content; the bench treats them the same way.
#
# YAGNI: if marker noise turns out to materially hurt search
# quality asymmetrically (default vs --strip-docstrings), we
# revisit.  Empirical answer beats premature normalisation.

# A sentence ends at `.`, `!`, or `?` followed by whitespace,
# or at a blank line (paragraph break).  Pure English
# punctuation, not language-specific.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n\s*\n")

# Length band for usable queries.  Shorter than this is
# usually a stub comment (`hi`, `wip`); longer than this makes
# a poor query.
_QUERY_MIN_LEN = 15
_QUERY_MAX_LEN = 200

# Leading-character noise the boilerplate check skips before
# comparing against `_REJECT_PREFIXES`.  Used only for the
# rejection decision — not applied to the returned query text.
_REJECT_LOOKAHEAD_RE = re.compile(r"^[\s/#*\-!\"'`]+")

# Boilerplate / scaffolding markers; queries starting with any
# of these (after skipping marker noise) are dropped.
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

    Splits on a sentence terminator followed by whitespace, or
    on a blank line.  Truncates at `_QUERY_MAX_LEN` if no
    terminator appears in the first paragraph.  Returns `None`
    when the candidate is shorter than `_QUERY_MIN_LEN` or
    starts with a boilerplate prefix (after skipping leading
    comment-marker noise).

    Comment markers (line comments, triple-quotes, block-
    comment delimiters, `*` gutters) are *kept* in the
    returned text - see the module-level note.
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


# ── Dataset record types ───────────────────────────────────────────────────────


SymbolKind = Literal["function", "class", "method"]


class Header(BaseModel, frozen=True):
    """First line of a per-repo JSONL dataset."""

    kind: Literal["header"] = "header"
    slug: str
    sha: str
    seed: int
    sample_cap: int
    n_documented: int
    n_sampled: int


class Query(BaseModel, frozen=True):
    """One sampled query, one per line after the header.

    `scope` is the parent-symbol path (empty string for
    top-level symbols).  Required in the match key because
    `(file_path, name)` alone collides on overloaded methods,
    multiple `__init__`, and same-named test methods across
    classes.

    No `slug` field; the slug is implicit in the file name
    (`<slug>.jsonl`).  Consumers that need it carry it
    alongside the loaded queries.
    """

    kind: Literal["query"] = "query"
    file_path: str
    scope: str
    name: str
    symbol_kind: SymbolKind
    line_start: int
    language: str
    text: str


# ── Symbol walk ────────────────────────────────────────────────────────────────


_KIND_TO_SYMBOL: dict[str, SymbolKind] = {
    "function": "function",
    "class": "class",
    "method": "method",
}


def _iter_file_entries(repo: pygit2.Repository, ref: str) -> Iterator[FileEntry]:
    repo_root = Path(repo.workdir).resolve() if repo.workdir else Path.cwd()
    ignore = load_ignore(repo_root)
    yield from list_files(
        repo,
        ref,
        max_file_size=config.max_file_size,
        ignore=ignore,
    )


def iter_queries(repo_path: Path) -> Iterator[Query]:
    """Walk *repo_path*, yielding one `Query` per documented symbol.

    Delegates docstring identification to `extract_doc_spans`
    so it matches the indexer exactly.  Non-(function, class,
    method) kinds are skipped.  Symbols without a usable
    first-sentence (empty / trivia-only comments) are skipped.
    """
    repo = pygit2.Repository(str(repo_path))
    mgr = get_manager()

    for entry in _iter_file_entries(repo, "HEAD"):
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
            symbol_kind = _KIND_TO_SYMBOL.get(str(span.kind))
            if symbol_kind is None:
                continue
            if not span.name or span.name == "<anonymous>":
                continue
            doc_bytes = b"\n".join(entry.content[s:e] for s, e in span.ranges)
            text = first_sentence(doc_bytes.decode("utf-8", errors="replace"))
            if text is None:
                continue
            yield Query(
                file_path=entry.path,
                scope=span.scope,
                name=span.name,
                symbol_kind=symbol_kind,
                line_start=span.line_start,
                language=lang_id,
                text=text,
            )


def _sort_key(q: Query) -> tuple[str, int, str, str]:
    return (q.file_path, q.line_start, q.scope, q.name)


def sample_queries(
    queries: list[Query],
    *,
    seed: int,
    sample_cap: int,
) -> list[Query]:
    """Deterministically sample up to *sample_cap* queries.

    Sorts on `(file_path, line_start, scope, name)` before the
    RNG draw so the input to `random.Random.sample` is stable,
    then re-sorts the sample so the JSONL is byte-stable
    across runs.
    """
    sorted_queries = sorted(queries, key=_sort_key)
    if len(sorted_queries) > sample_cap:
        sorted_queries = random.Random(seed).sample(  # noqa: S311 — deterministic, not crypto
            sorted_queries, sample_cap
        )
    return sorted(sorted_queries, key=_sort_key)


def _resolve_head_sha(repo_path: Path) -> str:
    """Return the resolved HEAD SHA for *repo_path* via pygit2.

    pygit2 is already a transitive dep through rbtr; using it
    here keeps git interaction in one library and avoids
    spawning a `git` subprocess.
    """
    repo = pygit2.Repository(str(repo_path))
    return str(repo.head.target)


# ── CLI subcommands ────────────────────────────────────────────────────────────


class ExtractCmd(BaseModel):
    """Sample docstring-derived queries from one repo.

    Reads a checked-out repo at *repo-path*; writes a JSONL
    artefact with one header line plus one query per line.
    """

    slug: str = Field(description="Repo slug (used for the JSONL header).")
    repo_path: Path = Field(description="Path to the checked-out repository.")
    output: Path = Field(description="Per-repo JSONL output path.")
    seed: int = Field(0, description="Deterministic-sampling RNG seed.")
    sample_cap: int = Field(300, description="Maximum number of queries to keep per repo.")

    def cli_cmd(self) -> None:
        sha = _resolve_head_sha(self.repo_path)
        candidates = list(iter_queries(self.repo_path))
        sampled = sample_queries(candidates, seed=self.seed, sample_cap=self.sample_cap)

        header = Header(
            slug=self.slug,
            sha=sha,
            seed=self.seed,
            sample_cap=self.sample_cap,
            n_documented=len(candidates),
            n_sampled=len(sampled),
        )

        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("w", encoding="utf-8") as fh:
            fh.write(header.model_dump_json() + "\n")
            for q in sampled:
                fh.write(q.model_dump_json() + "\n")


def load_per_repo(path: Path) -> tuple[Header, list[Query]]:
    """Read one per-repo JSONL: the header line and all queries.

    Used by `measure` and `tune` to consume per-repo files
    directly — there is no merged dataset.
    """
    header: Header | None = None
    queries: list[Query] = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if header is None:
                header = Header.model_validate_json(line)
                continue
            queries.append(Query.model_validate_json(line))
    if header is None:
        msg = f"{path}: no header line"
        raise SystemExit(msg)
    return header, queries
