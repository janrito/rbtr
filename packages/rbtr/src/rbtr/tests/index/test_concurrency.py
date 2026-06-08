"""Thread-safety tests for IndexStore and WriteSession."""

from __future__ import annotations

import threading
from pathlib import Path

from rbtr.index.models import Edge, EdgeKind, Snapshot, TokenisedChunk
from rbtr.index.store import IndexStore

from .conftest import make_chunk


def test_concurrent_write_then_read(
    math_func: TokenisedChunk, http_func: TokenisedChunk, string_func: TokenisedChunk
) -> None:
    """Writes from a background thread are visible after join."""
    store = IndexStore(writable=True)

    def writer() -> None:
        with store.session() as ws:
            ws.add_chunk(math_func)
            ws.add_chunk(http_func)
            ws.add_chunk(string_func)
            ws.insert_snapshots(
                [
                    Snapshot(
                        commit_sha="head",
                        file_path=c.file_path,
                        blob_sha=c.blob_sha,
                    )
                    for c in [math_func, http_func, string_func]
                ],
                repo_id=1,
            )
            ws.insert_edges(
                [Edge(source_id=math_func.id, target_id=http_func.id, kind=EdgeKind.IMPORTS)],
                "head",
                repo_id=1,
            )

    t = threading.Thread(target=writer)
    t.start()
    t.join()

    assert len(store.get_chunks("head", repo_id=1)) == 3
    assert len(store.get_edges("head", repo_id=1)) == 1
    store.close()


def test_commit_makes_writes_visible_during_concurrent_work(
    math_func: TokenisedChunk,
    http_func: TokenisedChunk,
    string_func: TokenisedChunk,
) -> None:
    """After session exit, a second thread sees rows while the
    writer continues with embedding updates."""
    store = IndexStore(writable=True)
    first_done = threading.Event()

    def writer() -> None:
        with store.session() as session:
            for c in [math_func, http_func, string_func]:
                session.add_chunk(c)
            session.insert_snapshots(
                [
                    Snapshot(
                        commit_sha="head",
                        file_path=c.file_path,
                        blob_sha=c.blob_sha,
                    )
                    for c in [math_func, http_func, string_func]
                ],
                repo_id=1,
            )
        first_done.set()
        with store.session() as session:
            session.update_embeddings([math_func.id], [[0.1, 0.2, 0.3]], repo_id=1)
            session.update_embeddings([http_func.id], [[0.3, 0.4, 0.5]], repo_id=1)
            session.update_embeddings([string_func.id], [[0.5, 0.6, 0.7]], repo_id=1)

    t = threading.Thread(target=writer)
    t.start()

    first_done.wait(timeout=5)
    assert len(store.get_chunks("head", repo_id=1)) == 3

    t.join()
    store.close()


def test_concurrent_batch_and_search(tmp_path: Path) -> None:
    """Reader gets results while writer runs batches on another thread."""
    db_path = tmp_path / "index.duckdb"
    store = IndexStore(db_path, writable=True)

    chunks = [
        make_chunk(
            f"c{i}", name=f"func_{i}", content=f"def func_{i}(): common_term", path=f"src/f{i}.py"
        )
        for i in range(200)
    ]
    with store.session() as session:
        for c in chunks:
            session.add_chunk(c)
        session.insert_snapshots(
            [
                Snapshot(commit_sha="head", file_path=c.file_path, blob_sha=c.blob_sha)
                for c in chunks
            ],
            repo_id=1,
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
            with store.session() as session:
                session.add_chunk(extra)
                session.insert_snapshots(
                    [
                        Snapshot(
                            commit_sha="head", file_path=extra.file_path, blob_sha=extra.blob_sha
                        )
                    ],
                    repo_id=1,
                )
        stop.set()

    def reader() -> None:
        nonlocal good_reads
        while not stop.is_set():
            try:
                results = store.match_fulltext("head", "common_term", top_k=5, repo_id=1)
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

    store.close()
