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
import subprocess
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

    `slug` is empty in per-repo JSONLs (implicit from the file
    name) and populated by `merge-dataset` in the combined
    dataset.  Downstream consumers can therefore read either
    format through the same model, dispatching on `slug`
    presence if needed.
    """

    kind: Literal["query"] = "query"
    slug: str = ""
    file_path: str
    scope: str
    name: str
    symbol_kind: SymbolKind
    line_start: int
    language: str
    text: str


# ── Symbol walk ────────────────────────────────────────────────────────────────


class _DocumentedSymbol(BaseModel, frozen=True):
    """One (symbol, docstring) pair found by the walk.

    Pre-filter record; `first_sentence` may still reject it.
    """

    file_path: str
    scope: str
    name: str
    symbol_kind: SymbolKind
    line_start: int
    language: str
    doc_text: str


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


def iter_documented_symbols(repo_path: Path) -> Iterator[_DocumentedSymbol]:
    """Walk *repo_path*, yielding documented function / class / method symbols.

    Delegates docstring identification to `extract_doc_spans`
    so it matches the indexer exactly.  Non-(function, class,
    method) kinds are skipped (we don't benchmark variables or
    imports).  Symbols without any doc bytes are skipped.
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
            yield _DocumentedSymbol(
                file_path=entry.path,
                scope=span.scope,
                name=span.name,
                symbol_kind=symbol_kind,
                line_start=span.line_start,
                language=lang_id,
                doc_text=doc_bytes.decode("utf-8", errors="replace"),
            )


def _sort_key(item: _DocumentedSymbol | Query) -> tuple[str, int, str, str]:
    return (item.file_path, item.line_start, item.scope, item.name)


def sample_queries(
    symbols: list[_DocumentedSymbol],
    *,
    seed: int,
    sample_cap: int,
) -> list[Query]:
    """Project to `Query`s, filter, and deterministically sample.

    Steps:

    1. Sort candidates by `(file_path, line_start, scope, name)`
       so the input to `random.Random.sample` is stable.
    2. Drop any whose `first_sentence` returns None.
    3. Sample up to *sample_cap* via `random.Random(seed).sample`.
    4. Re-sort the sample the same way so the JSONL is
       byte-stable across runs.
    """
    sorted_symbols = sorted(symbols, key=_sort_key)
    queries: list[Query] = []
    for sym in sorted_symbols:
        text = first_sentence(sym.doc_text)
        if text is None:
            continue
        queries.append(
            Query(
                file_path=sym.file_path,
                scope=sym.scope,
                name=sym.name,
                symbol_kind=sym.symbol_kind,
                line_start=sym.line_start,
                language=sym.language,
                text=text,
            )
        )

    if len(queries) > sample_cap:
        rng = random.Random(seed)  # noqa: S311 — deterministic sampling, not crypto
        queries = rng.sample(queries, sample_cap)

    return sorted(queries, key=_sort_key)


def _resolve_head_sha(repo_path: Path) -> str:
    """Return the resolved HEAD SHA for *repo_path*.

    Pure subprocess; no pygit2.  Matches the plan's "no pygit2
    in our own code" rule (rbtr.git is a carve-out).
    """
    result = subprocess.run(  # noqa: S603 — trusted args
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],  # noqa: S607 — `git` on PATH
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


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
        symbols = list(iter_documented_symbols(self.repo_path))
        queries = sample_queries(symbols, seed=self.seed, sample_cap=self.sample_cap)

        header = Header(
            slug=self.slug,
            sha=sha,
            seed=self.seed,
            sample_cap=self.sample_cap,
            n_documented=len(symbols),
            n_sampled=len(queries),
        )

        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("w", encoding="utf-8") as fh:
            fh.write(header.model_dump_json() + "\n")
            for q in queries:
                fh.write(q.model_dump_json() + "\n")


class MergedHeader(BaseModel, frozen=True):
    """First line of a merged dataset JSONL.

    Carries every per-repo `Header` so `measure` and `tune`
    can attribute queries back to their source repo after the
    merge (SHAs, per-repo sampling parameters, counts).
    """

    kind: Literal["merged_header"] = "merged_header"
    seed: int
    sample_cap: int
    per_repo: list[Header]


class MergeDatasetCmd(BaseModel):
    """Concatenate per-repo JSONL files into one dataset.

    Validates each per-repo file round-trips through `Header` /
    `Query`, emits one `MergedHeader` carrying every per-repo
    header, then emits each query with `slug` injected from
    the owning header.  Input files are processed in alphabetical
    order to make the output byte-stable.

    Refuses to run if per-repo `seed` or `sample_cap` disagree;
    those are global knobs and a mismatch means the inputs came
    from inconsistent pipeline runs.
    """

    inputs: Path = Field(description="Directory holding per-repo JSONL files.")
    output: Path = Field(description="Combined dataset output path.")

    def cli_cmd(self) -> None:
        files = sorted(self.inputs.glob("*.jsonl"))
        if not files:
            msg = f"no JSONL files under {self.inputs}"
            raise SystemExit(msg)

        per_repo_headers: list[Header] = []
        queries_by_slug: list[tuple[str, list[Query]]] = []

        for path in files:
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
            per_repo_headers.append(header)
            queries_by_slug.append((header.slug, queries))

        seeds = {h.seed for h in per_repo_headers}
        caps = {h.sample_cap for h in per_repo_headers}
        if len(seeds) != 1 or len(caps) != 1:
            msg = (
                "per-repo headers disagree on seed / sample_cap; "
                f"seeds={sorted(seeds)} caps={sorted(caps)}"
            )
            raise SystemExit(msg)

        merged = MergedHeader(
            seed=per_repo_headers[0].seed,
            sample_cap=per_repo_headers[0].sample_cap,
            per_repo=per_repo_headers,
        )

        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("w", encoding="utf-8") as fh:
            fh.write(merged.model_dump_json() + "\n")
            for slug, queries in queries_by_slug:
                for q in queries:
                    fh.write(q.model_copy(update={"slug": slug}).model_dump_json() + "\n")
