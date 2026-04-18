"""Benchmark helper: extract `(symbol, docstring)` pairs from a repo.

Walks every tree-sittered source file in a provisioned repo and
yields one `DocSymbol` per documented function / class / method.

No per-language parsing here.  The engine's `extract_doc_spans`
returns the exact byte ranges the indexer would blank under
`--strip-docstrings`, so docstring identification stays in
lockstep with the measured path.  Raw docstring bytes are the
concatenation of those ranges - comment markers included (`///`,
triple-quote, `#`, `/* */`, whatever the grammar tagged).

The query sampler in `bench_docstrings.py` then projects those
bytes into search strings.  Stripping comment markers there is
done via the engine's view of the ranges, not by re-parsing
comment syntax in this file.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path

import pygit2

from rbtr.config import config
from rbtr.git import FileEntry, list_files
from rbtr.index.treesitter import extract_doc_spans
from rbtr.languages import get_manager
from rbtr.rbtrignore import load_ignore


@dataclasses.dataclass(frozen=True)
class DocSymbol:
    """A documented symbol paired with its raw docstring bytes.

    `doc_text` is the exact byte content of the docstring as
    the indexer sees it - comment markers are part of the text.
    Downstream code strips them via tree-sitter-aware helpers,
    never by re-parsing the language by hand.
    """

    repo_slug: str
    file_path: str
    name: str
    line_start: int
    language: str
    doc_text: str


def _iter_file_entries(repo: pygit2.Repository, ref: str) -> Iterator[FileEntry]:
    """Yield indexable files for *ref*, honouring the same filters rbtr uses."""
    repo_root = Path(repo.workdir).resolve() if repo.workdir else Path.cwd()
    ignore = load_ignore(repo_root)
    yield from list_files(
        repo,
        ref,
        max_file_size=config.max_file_size,
        ignore=ignore,
    )


def iter_doc_symbols(repo_path: Path, repo_slug: str) -> Iterator[DocSymbol]:
    """Walk *repo_path*, yielding one `DocSymbol` per documented symbol.

    Delegates docstring identification to `extract_doc_spans`
    so it matches the indexer's definition exactly.  Symbols
    without any doc bytes are skipped; callers apply further
    quality filters (min/max query length, boilerplate prefixes)
    downstream.
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
            doc_bytes = b"\n".join(entry.content[s:e] for s, e in span.ranges)
            yield DocSymbol(
                repo_slug=repo_slug,
                file_path=entry.path,
                name=span.name,
                line_start=span.line_start,
                language=lang_id,
                doc_text=doc_bytes.decode("utf-8", errors="replace"),
            )
