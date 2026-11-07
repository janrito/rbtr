"""Thread-safety tests for Index and WriteSession."""

from __future__ import annotations

import threading
from pathlib import Path

import polars as pl

from rbtr.domain.models import EdgeKind, FileSnapshot
from rbtr.index.index import Index
from rbtr.index.staging import EmbeddingStagingRow, TokenisedChunk
from rbtr.languages.edges import InferredEdgeRow

from .conftest import make_chunk


def test_commit_makes_writes_visible_during_concurrent_work(
    math_func: TokenisedChunk,
    http_func: TokenisedChunk,
    string_func: TokenisedChunk,
) -> None:
    """A repo registered and written on a worker thread is visible to
    another thread once the session commits, while the worker goes on
    to update embeddings."""
    index = Index(writable=True)
    first_done = threading.Event()

    def writer() -> None:
        repo = index.register("/repo")
        with index.session(repo) as session:
            for c in [math_func, http_func, string_func]:
                session.add_chunk(c)
            session.replace_snapshots(
                [
                    FileSnapshot(
                        snapshot_sha="head",
                        file_path=c.file_path,
                        blob_sha=c.blob_sha,
                    )
                    for c in [math_func, http_func, string_func]
                ],
                snapshot_sha="head",
            )
            session.replace_edges(
                InferredEdgeRow.validate(
                    pl.DataFrame(
                        [
                            {
                                "source_id": math_func.id,
                                "target_id": http_func.id,
                                "kind": EdgeKind.IMPORTS,
                                "source_path": math_func.file_path,
                                "target_path": http_func.file_path,
                            }
                        ]
                    ),
                    cast=True,
                ),
                snapshot_sha="head",
            )
        first_done.set()
        with index.session(repo) as session:
            session.update_embeddings(
                EmbeddingStagingRow.validate(
                    pl.DataFrame(
                        {"id": [math_func.id], "embedding": [[0.1, 0.2, 0.3]]}
                    ).with_columns(embedding_truncated=False),
                    cast=True,
                )
            )
            session.update_embeddings(
                EmbeddingStagingRow.validate(
                    pl.DataFrame(
                        {"id": [http_func.id], "embedding": [[0.3, 0.4, 0.5]]}
                    ).with_columns(embedding_truncated=False),
                    cast=True,
                )
            )
            session.update_embeddings(
                EmbeddingStagingRow.validate(
                    pl.DataFrame(
                        {"id": [string_func.id], "embedding": [[0.5, 0.6, 0.7]]}
                    ).with_columns(embedding_truncated=False),
                    cast=True,
                )
            )

    t = threading.Thread(target=writer)
    t.start()

    first_done.wait(timeout=5)
    written = index.repo("/repo").at("head")
    assert len(index.chunks(at=written)) == 3
    assert len(index.edges(within=[written])) == 1

    t.join()
    index.close()


def test_concurrent_batch_and_search(tmp_path: Path) -> None:
    """Reader gets results while writer runs batches on another thread."""
    db_path = tmp_path / "index.duckdb"
    index = Index(db_path, writable=True)
    repo = index.register("/repo")

    chunks = [
        make_chunk(
            f"c{i}", name=f"func_{i}", content=f"def func_{i}(): common_term", path=f"src/f{i}.py"
        )
        for i in range(200)
    ]
    with index.session(repo) as session:
        for c in chunks:
            session.add_chunk(c)
        session.replace_snapshots(
            [
                FileSnapshot(snapshot_sha="head", file_path=c.file_path, blob_sha=c.blob_sha)
                for c in chunks
            ],
            snapshot_sha="head",
        )

    errors: list[str] = []
    good_reads = 0
    stop = threading.Event()

    def writer() -> None:
        for i in range(5):
            extra = make_chunk(
                f"extra_{i}",
                name=f"extra_{i}",
                content=f"def extra_{i}(): common_term",
                path=f"src/extra_{i}.py",
            )
            # Each batch is a snapshot of its own, as each build is.
            batch = f"batch_{i}"
            with index.session(repo) as session:
                session.add_chunk(extra)
                session.replace_snapshots(
                    [
                        FileSnapshot(
                            snapshot_sha=batch, file_path=extra.file_path, blob_sha=extra.blob_sha
                        )
                    ],
                    snapshot_sha=batch,
                )
        stop.set()

    def reader() -> None:
        nonlocal good_reads
        while not stop.is_set():
            try:
                results = index.fulltext_matches("common_term", within=[repo.at("head")], top_k=5)
                if len(results) > 0:
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


def test_index_survives_threads_that_write_and_exit(
    math_func: TokenisedChunk, http_func: TokenisedChunk
) -> None:
    """A thread may write and exit; another thread still reads the store.

    The build worker is such a thread. On duckdb 1.5.5, without pyarrow
    imported before threading, the writing thread's exit leaves the
    connection corrupt and this segfaults instead of failing; on 1.5.6
    it passes with or without the import.
    """
    index = Index(writable=True)
    repo = index.register("/repo")
    seen: list[int] = []

    def write(chunks: list[TokenisedChunk]) -> None:
        """Write `head` as a build does: the snapshot's every file, replacing the last."""
        with index.session(repo) as session:
            for chunk in chunks:
                session.add_chunk(chunk)
            session.replace_snapshots(
                [
                    FileSnapshot(snapshot_sha="head", file_path=c.file_path, blob_sha=c.blob_sha)
                    for c in chunks
                ],
                snapshot_sha="head",
            )

    def read() -> None:
        seen.append(len(index.chunks(at=repo.at("head"))))

    for chunks in ([math_func], [math_func, http_func]):
        for work in (lambda c=chunks: write(c), read):
            thread = threading.Thread(target=work)
            thread.start()
            thread.join()

    assert seen == [1, 2]
    index.close()
