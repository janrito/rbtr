"""Thread-safety tests for Index and WriteSession, through builds on threads."""

from __future__ import annotations

import threading
from pathlib import Path

import pygit2

from rbtr.index.build import build_index
from rbtr.index.embed import embed_index
from rbtr.index.embeddings import Embedder
from rbtr.index.index import Index

from ..conftest import make_commit


def test_commit_makes_writes_visible_during_concurrent_work(
    fake_repo: str, stubbed_embedder: Embedder
) -> None:
    """A build on a worker thread is visible to another thread once it commits, while the worker goes on to embed."""
    repo = pygit2.Repository(fake_repo)
    sha = str(
        make_commit(
            repo,
            {
                "maths.py": b"""\
def add(a, b):
    return a + b
""",
                "http.py": b"from maths import add\n",
                "strings.py": b"""\
def shout(s):
    return s.upper()
""",
            },
            parents=[repo.head.peel(pygit2.Commit).id],
        )
    )
    index = Index(writable=True)
    first_done = threading.Event()

    def writer() -> None:
        build_index(fake_repo, sha, index)
        first_done.set()
        embed_index(index, at=index.repo(fake_repo).at(sha), embedder=stubbed_embedder)

    t = threading.Thread(target=writer)
    t.start()

    first_done.wait(timeout=30)
    written = index.repo(fake_repo).at(sha)
    assert sorted(index.chunks(at=written)["name"]) == ["add", "from maths import add", "shout"]
    assert len(index.edges(within=[written])) == 1

    t.join()
    index.close()


def test_concurrent_batch_and_search(fake_repo: str, tmp_path: Path) -> None:
    """A reader gets results while builds of later commits run on another thread."""
    repo = pygit2.Repository(fake_repo)
    files = {
        f"src/f{i}.py": f"""\
def func_{i}():
    common_term
""".encode()
        for i in range(50)
    }
    head = make_commit(repo, files, parents=[repo.head.peel(pygit2.Commit).id])
    index = Index(tmp_path / "index.duckdb", writable=True)
    build_index(fake_repo, str(head), index)
    at = index.repo(fake_repo).at(str(head))

    errors: list[str] = []
    good_reads = 0
    stop = threading.Event()

    def writer() -> None:
        parent = head
        for i in range(5):
            files[f"src/extra_{i}.py"] = f"""\
def extra_{i}():
    common_term
""".encode()
            parent = make_commit(repo, files, parents=[parent])
            build_index(fake_repo, str(parent), index)
        stop.set()

    def reader() -> None:
        nonlocal good_reads
        while not stop.is_set():
            try:
                if not index.fulltext_matches("common_term", within=[at], top_k=5).is_empty():
                    good_reads += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))

    t_w = threading.Thread(target=writer)
    t_r = threading.Thread(target=reader)
    t_w.start()
    t_r.start()
    t_w.join()
    t_r.join()

    assert not errors, f"reader errors: {errors}"
    assert good_reads > 0, "reader never got results"

    index.close()


def test_index_survives_threads_that_write_and_exit(fake_repo: str) -> None:
    """A thread may build and exit; another thread still reads the store.

    The build worker is such a thread. On duckdb 1.5.5, without pyarrow
    imported before threading, the writing thread's exit leaves the
    connection corrupt and this segfaults instead of failing; on 1.5.6
    it passes with or without the import.
    """
    repo = pygit2.Repository(fake_repo)
    one = make_commit(
        repo,
        {
            "maths.py": b"""\
def add(a, b):
    return a + b
"""
        },
        parents=[repo.head.peel(pygit2.Commit).id],
    )
    two = make_commit(
        repo,
        {
            "maths.py": b"""\
def add(a, b):
    return a + b
""",
            "strings.py": b"""\
def shout(s):
    return s.upper()
""",
        },
        parents=[one],
    )
    index = Index(writable=True)
    seen: list[int] = []

    for sha in (str(one), str(two)):
        for work in (
            lambda sha=sha: build_index(fake_repo, sha, index),
            lambda sha=sha: seen.append(len(index.chunks(at=index.repo(fake_repo).at(sha)))),
        ):
            thread = threading.Thread(target=work)
            thread.start()
            thread.join()

    assert seen == [1, 2]
    index.close()
