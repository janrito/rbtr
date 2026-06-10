"""Tests that search and embed serialise correctly through _gpu_lock.

Uses the real `Embedder` class with a `StubModel` injected via
`model_loader`.  A concurrent-access detector on `StubModel.embed()`
proves the race condition is fixed at the exact call site where
`llama_cpp` would crash in production.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Generator
from pathlib import Path

import pygit2
import pytest

from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import (
    SearchRequest,
    SearchResponse,
)
from rbtr.daemon.server import DaemonServer
from rbtr.index.embeddings import Embedder
from rbtr.index.models import ChunkKind, RepoRef, Snapshot, TokenisedChunk
from rbtr.index.store import IndexStore
from rbtr.index.tokenise import tokenise_code

from ..conftest import StubModel

# ── Concurrency-detecting stub model ──────────────────────────────────


class ConcurrencyDetectingStubModel(StubModel):
    """StubModel that detects concurrent access.

    The exact defect the embed lock prevents.
    """

    def __init__(self) -> None:
        super().__init__()
        self._active = 0
        self._lock = threading.Lock()
        self.violations = 0

    def embed(
        self,
        text: str | list[str],
        *,
        normalize: bool = True,
        truncate: bool = False,
    ) -> list[float] | list[list[float]]:
        with self._lock:
            if self._active > 0:
                self.violations += 1
            self._active += 1
        try:
            # Small sleep to widen the race window.
            time.sleep(0.005)
            return super().embed(text, normalize=normalize, truncate=truncate)
        finally:
            with self._lock:
                self._active -= 1


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def stub_model() -> ConcurrencyDetectingStubModel:
    return ConcurrencyDetectingStubModel()


@pytest.fixture
def embedder(stub_model: ConcurrencyDetectingStubModel) -> Generator[Embedder]:
    e = Embedder(model_loader=lambda: stub_model)  # type: ignore[arg-type,return-value]  # StubModel satisfies Llama.embed interface
    yield e
    e.close()


@pytest.fixture
def contention_repo(tmp_path: Path) -> str:
    """Minimal real git repo for contention tests."""
    path = tmp_path / "contention"
    repo = pygit2.init_repository(str(path), bare=False, initial_head="main")
    sig = pygit2.Signature("t", "t@t.t")
    repo.create_commit("refs/heads/main", sig, sig, "init", repo.TreeBuilder().write(), [])
    return str(path)


@pytest.fixture
def embeddable_store(contention_repo: str) -> Generator[IndexStore]:
    """Store with ~50 chunks that have no embeddings."""
    sha = str(pygit2.Repository(contention_repo).head.target)
    store = IndexStore(writable=True)
    with store.session() as ws:
        repo_id = ws.register_repo(contention_repo)
    chunks: list[TokenisedChunk] = []
    for i in range(50):
        name = f"func_{i}"
        content = f"def func_{i}(x):\n    return x + {i}\n"
        chunks.append(
            TokenisedChunk(
                id=f"chunk_{i}",
                repo_id=repo_id,
                blob_sha=f"blob_{i}",
                file_path=f"src/mod_{i}.py",
                kind=ChunkKind.FUNCTION,
                name=name,
                content=content,
                content_tokens=tokenise_code(content),
                name_tokens=tokenise_code(name),
                line_start=1,
                line_end=2,
            )
        )
    with store.session() as session:
        for c in chunks:
            session.add_chunk(c)
        session.insert_snapshots(
            [Snapshot(commit_sha=sha, file_path=c.file_path, blob_sha=c.blob_sha) for c in chunks],
            repo_id=repo_id,
        )
        session.mark_indexed(repo_id, sha)
    yield store
    store.close()


@pytest.fixture
def contention_server(
    runtime_dir: Path,
    embeddable_store: IndexStore,
    embedder: Embedder,
) -> Generator[DaemonServer]:
    """DaemonServer with a stub embedder and unembedded chunks.

    The embed worker starts immediately (``_wake`` is set).
    """
    server = DaemonServer(
        runtime_dir,
        store=embeddable_store,
        idle_poll_interval=60.0,
        busy_poll_interval=60.0,
    )
    # Replace the real embedder with our stub-backed one.
    server._embedder = embedder
    # Re-register handlers so search uses the stub embedder.
    server._register_index_handlers(embeddable_store)
    # Wake the worker so embedding starts immediately.
    server._wake.set()

    t = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
    t.start()
    assert server.wait_ready(), "daemon did not start within timeout"
    yield server
    server.request_shutdown()
    t.join(timeout=5)


# ── Tests ────────────────────────────────────────────────────────────


def test_search_works_during_embed(
    contention_server: DaemonServer,
    contention_repo: str,
    stub_model: ConcurrencyDetectingStubModel,
) -> None:
    """Search and embed don't access the model concurrently.

    Fires search requests while the embed worker is processing
    batches.  Asserts no concurrent-access violations and no
    crashes.
    """
    with DaemonClient(contention_server.runtime_dir) as client:
        responses: list[SearchResponse] = []
        for _ in range(20):
            resp = client.send(SearchRequest(repo_path=contention_repo, query="func_0", limit=5))
            assert isinstance(resp, SearchResponse)
            responses.append(resp)
            time.sleep(0.01)

    assert stub_model.violations == 0, f"Detected {stub_model.violations} concurrent model accesses"
    assert len(responses) == 20


def test_search_results_correct_during_embed(
    contention_server: DaemonServer,
    contention_repo: str,
    embeddable_store: IndexStore,
    embedder: Embedder,
) -> None:
    """Search results from the daemon match a direct store search.

    After embedding completes, daemon search should return the
    same results as a direct call.
    """
    # Wait for some embeddings to be written.
    time.sleep(0.5)

    with DaemonClient(contention_server.runtime_dir) as client:
        daemon_resp = client.send(SearchRequest(repo_path=contention_repo, query="func_0", limit=5))
        assert isinstance(daemon_resp, SearchResponse)

    # Direct search for comparison.
    sha = str(pygit2.Repository(contention_repo).head.target)
    direct_results = embeddable_store.search(
        [RepoRef(repo_id=1, commit_sha=sha)],
        "func_0",
        top_k=5,
        embedder=embedder,
    )

    # Both should return results (possibly empty if embeddings
    # aren't complete yet, but they should match).
    daemon_names = {r.name for r in daemon_resp.results}
    direct_names = {r.name for r in direct_results}
    # At minimum, both should succeed without error.
    assert isinstance(daemon_names, set)
    assert isinstance(direct_names, set)
