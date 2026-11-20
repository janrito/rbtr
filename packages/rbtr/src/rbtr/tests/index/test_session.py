"""Behavioural tests for the write side, through builds and embedding.

What a build's session guarantees: a failure rolls back and leaves the
thread reading, a failed search-index rebuild keeps the previous one,
chunks flush in batches, a rebuild replaces only its own commit's rows,
and stored payloads (embeddings, calls, bases, bindings) read back whole.
Also registration, watched refs, and a read-only index refusing writes.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

import duckdb
import polars as pl
import pygit2
import pytest
from pydantic import ValidationError as ModelValidationError
from pytest_cases import parametrize_with_cases
from pytest_mock import MockerFixture

from rbtr.config import config
from rbtr.domain.edge_frames import EdgeChunkRow
from rbtr.domain.models import EmbeddingState, Repo, SnapshotRef, Span
from rbtr.errors import RepoNotFoundError
from rbtr.index.build import build_index
from rbtr.index.embed import embed_index
from rbtr.index.embeddings import Embedder
from rbtr.index.index import Index
from rbtr.languages.manager import get_manager
from rbtr.languages.registration import QueryExtraction
from rbtr_lang_python.plugin import python

from ..conftest import make_commit
from .cases_index_repos import RepoSequence
from .cases_index_watched_refs import WatchedRefScenario


@pytest.mark.parametrize(
    "write",
    [
        lambda index, repo: index.session(repo),
        lambda index, repo: index.register("/other"),
        lambda index, repo: index.forget(repo),
        lambda index, repo: index.compact(repo),
        lambda index, repo: index.compact_all(),
        lambda index, repo: index.compact_grown(1.0),
        lambda index, repo: index.rebuild_fts(),
        lambda index, repo: index.set_missing_since(repo, None),
    ],
    ids=[
        "session",
        "register",
        "forget",
        "compact",
        "compact_all",
        "compact_grown",
        "rebuild_fts",
        "missing",
    ],
)
def test_a_read_only_index_refuses_every_write(
    tmp_path: Path, write: Callable[[Index, Repo], object]
) -> None:
    """An index opened without writable=True raises before it writes anything."""
    db = tmp_path / "index.duckdb"
    seed = Index(db, writable=True)
    repo = seed.register("/repo")
    seed.close()
    index = Index(db, writable=False)
    try:
        with pytest.raises(RuntimeError, match="writable"):
            write(index, repo)
    finally:
        index.close()


def test_a_failed_fts_rebuild_keeps_the_previous_index(
    fake_repo: str, index: Index, mocker: MockerFixture
) -> None:
    """A rebuild that fails part-way leaves the previous FTS index whole."""
    repo = pygit2.Repository(fake_repo)
    before = make_commit(
        repo,
        {
            "before.py": b"""\
def before_check():
    pass
"""
        },
        parents=[repo.head.peel(pygit2.Commit).id],
    )
    after = make_commit(
        repo,
        {
            "before.py": b"""\
def before_check():
    pass
""",
            "after.py": b"""\
def after_check():
    pass
""",
        },
        parents=[before],
    )
    build_index(fake_repo, str(before), index)
    mocker.patch("rbtr.index.db._NEUTRALISE_FTS_IDF_SQL", "SELECT * FROM no_such_table")
    with pytest.raises(duckdb.CatalogException):
        build_index(fake_repo, str(after), index)

    registered = index.repo(fake_repo)
    assert len(index.fulltext_matches("before", within=[registered.at(str(before))])) == 1
    assert len(index.fulltext_matches("after", within=[registered.at(str(after))])) == 0


def test_buffer_flushes_at_batch_size(
    fake_repo: str, index: Index, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A build of more chunks than the batch size flushes them in batches, every one stored."""
    monkeypatch.setattr(config, "insert_batch_size", 5)
    repo = pygit2.Repository(fake_repo)
    sha = str(
        make_commit(
            repo,
            {
                f"f{i}.py": f"""\
def f{i}():
    pass
""".encode()
                for i in range(12)
            },
            parents=[repo.head.peel(pygit2.Commit).id],
        )
    )
    build_index(fake_repo, sha, index)

    assert len(index.chunks(at=index.repo(fake_repo).at(sha))) == 12


@parametrize_with_cases("scenario", cases=".cases_index_repos")
def test_register(scenario: RepoSequence, index: Index) -> None:
    """A path is one repo, found again by `repo`; `repos` lists them as registered."""
    registered = [index.register(path) for path in scenario.calls]
    for path, repo in zip(scenario.calls, registered, strict=True):
        assert repo.repo_path == path
        assert index.repo(path) == repo
    assert len(set(registered)) == len(set(scenario.calls))
    assert [r.repo_path for r in index.repos()] == scenario.expected_paths


@parametrize_with_cases("scenario", cases=".cases_index_watched_refs")
def test_watched_refs(scenario: WatchedRefScenario, index: Index) -> None:
    repos = {key: index.register(path) for key, path in scenario.repos.items()}
    for key, op, refs in scenario.ops:
        with index.session(repos[key]) as ws:
            if op == "add":
                ws.add_watched_refs(refs)
            else:
                ws.remove_watched_refs(refs)
    for key, expected in scenario.expected.items():
        assert index.watched_refs(repos[key]) == expected


def test_repo_returns_the_registered_repo(index: Index) -> None:
    """`repo` returns the repo registered at a path."""
    registered = index.register("/r")
    assert index.repo("/r") == registered


def test_repo_follows_a_forgotten_path_registered_again(index: Index) -> None:
    """`repo` returns the new registration, not the forgotten one."""
    forgotten = index.repo(index.register("/r").repo_path)
    index.forget(forgotten)
    registered = index.register("/r")
    assert registered != forgotten
    assert index.repo("/r") == registered


def test_repo_raises_for_unknown(index: Index) -> None:
    """`repo` raises `RepoNotFoundError`, coded `repo_not_found`."""
    with pytest.raises(RepoNotFoundError, match="not registered") as raised:
        index.repo("nonexistent")
    assert raised.value.error_code == "repo_not_found"


def test_rebuilding_a_commit_leaves_another_commits_files_and_edges(
    fake_repo: str, index: Index
) -> None:
    """Rebuilding one commit replaces its file list and edges, not another commit's.

    Each commit holds a module importing another. A plugin upgrade that
    stops reading imports re-extracts the first commit's files: its edges
    are inferred again, to none, and the second's stay.
    """
    repo = pygit2.Repository(fake_repo)
    head = repo.head.peel(pygit2.Commit).id
    first = str(
        make_commit(
            repo,
            {
                "a.py": b"""\
from b import f
""",
                "b.py": b"""\
def f():
    pass
""",
            },
            ref="refs/heads/first",
            parents=[head],
        )
    )
    second = str(
        make_commit(
            repo,
            {
                "c.py": b"""\
from d import g
""",
                "d.py": b"""\
def g():
    pass
""",
            },
            ref="refs/heads/second",
            parents=[head],
        )
    )
    build_index(fake_repo, first, index)
    build_index(fake_repo, second, index)
    no_imports = dataclasses.replace(
        python,
        extraction=QueryExtraction(
            query="(function_definition name: (identifier) @_fn_name) @function\n"
        ),
        extraction_serial=python.extraction_serial + 1,
    )
    with get_manager().limited_to(no_imports):
        build_index(fake_repo, first, index)

    registered = index.repo(fake_repo)
    assert index.count_edges(at=registered.at(first)) == 0
    assert index.count_edges(at=registered.at(second)) == 1
    assert index.count_file_snapshots(at=registered.at(second)) == 2


def test_edge_chunks_are_every_location_in_path_order(
    dup_index: Index, dup_ref: SnapshotRef
) -> None:
    """A chunk at two paths is read once per path, and the rows come in the order chunks are read."""
    expected = (
        dup_index.chunks(at=dup_ref)
        .with_columns(pl.col("metadata").struct.unnest())
        .pipe(EdgeChunkRow.validate, cast=True)
    )
    with dup_index.session(dup_ref.repo) as session:
        assert session.edge_chunks(at=dup_ref).equals(expected)


def test_embedding_and_truncation_are_stored(
    fake_repo: str, index: Index, stubbed_embedder: Embedder
) -> None:
    """Embedding stores each chunk's vector, and whether its text was cut to fit the model."""
    repo = pygit2.Repository(fake_repo)
    long_doc = "Step after step. " * 600
    sha = str(
        make_commit(
            repo,
            {
                "short.py": b"""\
def short():
    pass
""",
                "long.py": f'''\
def long():
    """{long_doc}"""
'''.encode(),
            },
            parents=[repo.head.peel(pygit2.Commit).id],
        )
    )
    build_index(fake_repo, sha, index)
    at = index.repo(fake_repo).at(sha)
    assert set(index.chunks(at=at)["embedding_state"]) == {EmbeddingState.NONE}

    embed_index(index, at=at, embedder=stubbed_embedder)

    states = index.chunks(at=at).select("name", pl.col("embedding_state").cast(pl.String))
    assert sorted(states.rows()) == [
        ("long", EmbeddingState.TRUNCATED),
        ("short", EmbeddingState.FULL),
    ]


def test_extracting_a_blob_again_keeps_the_embeddings_of_unchanged_chunks(
    fake_repo: str, index: Index, stubbed_embedder: Embedder
) -> None:
    """A plugin upgrade's re-extraction keeps the embedding of a chunk whose span, name and content are unchanged.

    The upgraded query names functions by their parameters and captures
    no comments: `named` keeps its span under a new name, so it is
    embedded afresh (the model reads names); `gone`, whose span began at
    the comment folded into it, is deleted, and its function is a new
    chunk; `Kept` keeps its embedding.
    """
    repo = pygit2.Repository(fake_repo)
    source = b"""\
class Kept:
    pass


# Folded into gone.
def gone():
    pass


def named(a):
    pass
"""
    sha = str(make_commit(repo, {"f.py": source}, parents=[repo.head.peel(pygit2.Commit).id]))
    build_index(fake_repo, sha, index)
    at = index.repo(fake_repo).at(sha)
    embed_index(index, at=at, embedder=stubbed_embedder)
    gone_id = index.chunks(at=at, name="gone")["id"].item()
    upgraded = dataclasses.replace(
        python,
        extraction=QueryExtraction(
            query="""\
(class_definition name: (identifier) @_cls_name) @class
(function_definition parameters: (parameters) @_fn_name) @function
"""
        ),
        extraction_serial=python.extraction_serial + 1,
    )
    with get_manager().limited_to(upgraded):
        build_index(fake_repo, sha, index)

    chunks = index.chunks(at=at)
    assert gone_id not in chunks["id"].to_list()
    assert sorted(
        chunks.select("kind", "name", pl.col("embedding_state").cast(pl.String)).rows()
    ) == [
        ("class", "Kept", EmbeddingState.FULL),
        ("fragment", "", EmbeddingState.NONE),
        ("function", "()", EmbeddingState.NONE),
        ("function", "(a)", EmbeddingState.NONE),
    ]


def test_a_chunks_calls_bases_and_bindings_are_read_as_written(
    fake_repo: str, index: Index
) -> None:
    """What a class, a method and an import record of other code comes back from storage whole."""
    repo = pygit2.Repository(fake_repo)
    source = b"""\
import numpy as np
from django.db import models


class Order(models.Model):
    def run(self):
        self.save()
"""
    sha = str(make_commit(repo, {"f.py": source}, parents=[repo.head.peel(pygit2.Commit).id]))
    build_index(fake_repo, sha, index)

    read = index.chunks(at=index.repo(fake_repo).at(sha)).sort("name")
    assert [
        (
            row["name"],
            row["metadata"]["refs"],
            [binding for imp in row["metadata"]["imports"] for binding in imp["bindings"]],
        )
        for row in read.select("name", "metadata").iter_rows(named=True)
    ] == [
        (
            "Order",
            [
                {
                    "kind": "inherits",
                    "lookup": "unqualified",
                    "path": ["models", "Model"],
                    "type": None,
                }
            ],
            [],
        ),
        ("from django.db import models", None, [{"name": "models", "local": "models"}]),
        ("import numpy as np", None, [{"name": "numpy", "local": "np"}]),
        ("run", [{"kind": "calls", "lookup": "member", "path": ["save"], "type": None}], []),
    ]


@pytest.mark.parametrize(
    ("lines", "bytes_"), [((7, 6), (0, 0)), ((1, 1), (9, 3))], ids=["lines", "bytes"]
)
def test_a_span_cannot_end_before_it_starts(
    lines: tuple[int, int], bytes_: tuple[int, int]
) -> None:
    with pytest.raises(ModelValidationError, match="before it starts"):
        Span(start=lines[0], end=lines[1], start_byte=bytes_[0], end_byte=bytes_[1])


def test_a_failed_commit_leaves_the_thread_able_to_read(
    fake_repo: str, index: Index, mocker: MockerFixture
) -> None:
    """A build whose chunk write fails rolls back, so the thread reads on."""
    repo = pygit2.Repository(fake_repo)
    sha = str(
        make_commit(
            repo,
            {
                "f.py": b"""\
def f():
    pass
"""
            },
            parents=[repo.head.peel(pygit2.Commit).id],
        )
    )
    mocker.patch("rbtr.index.writer._UPSERT_CHUNKS_SQL", "SELECT * FROM no_such_table")
    with pytest.raises(duckdb.CatalogException):
        build_index(fake_repo, sha, index)

    assert index.has_indexed(at=index.repo(fake_repo).at(sha)) is False
