"""Read an index `indexed` built as an agent reads one: a file's outline, a symbol's definitions.

The index holds one repository, the one `indexed` committed the files to;
each reader finds it.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pygit2

from rbtr.daemon.dto import SymbolOut, SymbolRefOut
from rbtr.daemon.handlers import handle_list_symbols, handle_read_symbol
from rbtr.daemon.messages import ListSymbolsRequest, ReadSymbolRequest
from rbtr.index.index import Index


def repo_path(index: Index) -> str:
    """The path of the one repository *index* holds."""
    [repo] = index.repos()
    return repo.repo_path


def committed_files(index: Index) -> dict[str, str]:
    """Each file the commit `indexed` made holds, as `{path: text}`, in path order."""
    repo = pygit2.Repository(repo_path(index))
    files: dict[str, str] = {}
    pending = [("", repo.head.peel(pygit2.Commit).tree)]
    while pending:
        prefix, tree = pending.pop()
        for entry in tree:
            if isinstance(entry, pygit2.Tree):
                pending.append((f"{prefix}{entry.name}/", entry))
            elif isinstance(entry, pygit2.Blob):
                files[f"{prefix}{entry.name}"] = entry.data.decode()
    return dict(sorted(files.items(), key=lambda item: Path(item[0])))


def outline(index: Index, paths: Iterable[str] | None = None) -> list[SymbolRefOut]:
    """The symbols `list-symbols` gives for each of *paths*, file by file; for every committed file when none are given."""
    root = repo_path(index)
    if paths is None:
        paths = committed_files(index)
    return [
        symbol
        for path in paths
        for symbol in handle_list_symbols(
            ListSymbolsRequest(repo_path=root, file_path=path), index
        ).chunks
    ]


def read(index: Index, symbol: str, path: str | None = None) -> list[SymbolOut]:
    """The definitions `read-symbol` gives for *symbol*, with their text; only *path*'s when given."""
    return handle_read_symbol(
        ReadSymbolRequest(
            repo_path=repo_path(index),
            symbol=symbol,
            file_paths=None if path is None else [path],
        ),
        index,
    ).chunks
