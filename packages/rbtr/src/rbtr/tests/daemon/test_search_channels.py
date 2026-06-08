"""Verify all three search channels produce distinct rankings.

Each channel (semantic, lexical, name-match) scores candidates
differently.  Searching the same query with extreme weights
`(1,0,0)`, `(0,1,0)`, `(0,0,1)` must produce at least two
distinct top-1 results.  If all three return the same ranking,
a channel is dead.

Exercises all three search channels (semantic, lexical, name-match)
via a fake embedder with hand-crafted vectors for controlled cosine
similarity.
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

import pygit2
import pytest

from rbtr.config import WeightTriple
from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import SearchRequest, SearchResponse
from rbtr.daemon.server import DaemonServer
from rbtr.index.embeddings import Embedder
from rbtr.index.models import RepoRef, Snapshot
from rbtr.index.store import IndexStore

from ..index.conftest import make_chunk

# Hand-crafted vectors so cosine similarity is controllable.
# The query embedding points along dim 0.  Each chunk has
# a dominant dimension so semantic ranking differs from BM25
# and name ranking.


def _make_vec(dominant_dim: int, dim: int) -> list[float]:
    """Unit vector with most weight on *dominant_dim*, rest noise."""
    v = [0.01] * dim
    v[dominant_dim] = 1.0
    return v


def _normalise(v: list[float]) -> list[float]:
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


class _FakeEmbedder(Embedder):
    """Embedder returning hand-crafted vectors for controlled cosine similarity."""

    def __init__(self, query_vec: list[float]) -> None:
        super().__init__()
        self._query_vec = query_vec

    def embed_single(self, text: str) -> list[float]:
        return _normalise(self._query_vec)


@pytest.fixture
def channel_vecs() -> tuple[list[float], dict[str, list[float]]]:
    """Hand-crafted vectors so cosine similarity is controllable.

    `dim` is an arbitrary small width for these literal test vectors;
    it only needs to exceed the dominant dimensions used below.
    """
    dim = 8
    query_vec = _make_vec(0, dim)
    chunk_vecs: dict[str, list[float]] = {
        "find_items": _make_vec(0, dim),  # close to query (same dominant dim)
        "search": _make_vec(1, dim),  # far from query
        "lookup": _make_vec(2, dim),  # far from query
        "greet": _make_vec(3, dim),  # far from query
        "parse_config": _make_vec(4, dim),  # far from query
    }
    return query_vec, chunk_vecs


@pytest.fixture
def channel_store(
    fake_repo: str,
    channel_vecs: tuple[list[float], dict[str, list[float]]],
) -> Iterator[IndexStore]:
    """Store seeded with chunks designed so the channels disagree.

    - "find_items": content about searching/finding (semantic
      likes it for "search for items") but a different name.
    - "search": exact name match (name channel).
    - "lookup": "search" repeated in content (lexical channel).
    """
    _query_vec, chunk_vecs = channel_vecs
    sha = str(pygit2.Repository(fake_repo).head.target)
    specs = [
        (
            "ch_find",
            "src/find.py",
            "find_items",
            "def find_items(query): search the database and find matching items",
        ),
        ("ch_search", "src/search.py", "search", "def search(q): return results"),
        (
            "ch_lookup",
            "src/lookup.py",
            "lookup",
            "def lookup(term): search search search through the index to search",
        ),
        ("ch_greet", "src/greet.py", "greet", "def greet(name): return f'hello {name}'"),
        ("ch_parse", "src/parse.py", "parse_config", "def parse_config(path): return load(path)"),
    ]
    store = IndexStore(writable=True)
    with store.session() as ws:
        repo_id = ws.register_repo(fake_repo)
        chunks = [
            make_chunk(cid, name=name, content=content, path=path, blob=f"b_{cid}", repo_id=repo_id)
            for cid, path, name, content in specs
        ]
        for c in chunks:
            ws.add_chunk(c)
        # Embeddings are NULL on insert; set them via update_embeddings.
        ids = [c.id for c in chunks]
        embeddings = [_normalise(chunk_vecs[c.name]) for c in chunks]
        ws.update_embeddings(ids, embeddings, repo_id=repo_id)
        ws.insert_snapshots(
            [Snapshot(commit_sha=sha, file_path=c.file_path, blob_sha=c.blob_sha) for c in chunks],
            repo_id=repo_id,
        )
        ws.mark_indexed(repo_id, sha)
    yield store
    store.close()


@pytest.fixture
def channel_server(
    channel_store: IndexStore,
    channel_vecs: tuple[list[float], dict[str, list[float]]],
) -> Iterator[DaemonServer]:
    query_vec, _chunk_vecs = channel_vecs
    rd = Path(tempfile.mkdtemp(prefix="rbtr-ch-"))
    server = DaemonServer(rd, store=channel_store, idle_poll_interval=60.0, busy_poll_interval=60.0)
    # Replace the real embedder with a fake to avoid loading GGUF.
    server._embedder = _FakeEmbedder(query_vec)  # type: ignore[assignment]  # fake for testing
    # Re-register handlers so the search lambda captures the fake embedder.
    server._register_index_handlers(channel_store)
    t = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
    t.start()
    assert server.wait_ready(), "daemon did not start within timeout"
    yield server
    server.request_shutdown()
    t.join(timeout=3)


def test_daemon_and_direct_search_produce_identical_results(
    channel_server: DaemonServer,
    channel_store: IndexStore,
    channel_vecs: tuple[list[float], dict[str, list[float]]],
    fake_repo: str,
) -> None:
    """Daemon-served search and direct store.search must return the same ranking.

    The daemon is a performance optimisation (pre-loaded model,
    non-blocking indexing), not a functional difference.  If the
    rankings diverge, one path is missing a channel or applying
    different weights.
    """
    query = "search"
    query_vec, _chunk_vecs = channel_vecs
    embedder = _FakeEmbedder(query_vec)

    # Via daemon.
    with DaemonClient(channel_server.runtime_dir) as client:
        req = SearchRequest(path=fake_repo, query=query, limit=5)
        daemon_resp = client.send_or_raise_as(SearchResponse, req)
    daemon_names = [r.name for r in daemon_resp.results]
    daemon_scores = [round(r.score, 6) for r in daemon_resp.results]

    # Via direct store.search — pass the same expansion and reranker
    # the daemon used.
    sha = str(pygit2.Repository(fake_repo).head.target)
    direct_results = channel_store.search(
        [RepoRef(repo_id=1, commit_sha=sha)],
        query,
        top_k=5,
        embedder=embedder,
        expansion=daemon_resp.expansion,
        reranker=channel_server._reranker,
    )
    direct_names = [r.name for r in direct_results]
    direct_scores = [round(r.score, 6) for r in direct_results]

    assert daemon_names == direct_names, (
        f"ranking mismatch: daemon={daemon_names} direct={direct_names}"
    )
    assert daemon_scores == direct_scores, (
        f"score mismatch: daemon={daemon_scores} direct={direct_scores}"
    )


def test_different_weights_produce_different_rankings(
    channel_server: DaemonServer, fake_repo: str
) -> None:
    """Extreme weight configs must not all produce the same ranking.

    If semantic or name channels are dead, alpha=1 and gamma=1
    both fall back to BM25, producing identical results.
    """
    with DaemonClient(channel_server.runtime_dir) as client:
        query = "search"
        sem_resp = client.send_or_raise_as(
            SearchResponse,
            SearchRequest(
                path=fake_repo,
                query=query,
                limit=3,
                weights=WeightTriple(alpha=1.0, beta=0.0, gamma=0.0),
            ),
        )
        lex_resp = client.send_or_raise_as(
            SearchResponse,
            SearchRequest(
                path=fake_repo,
                query=query,
                limit=3,
                weights=WeightTriple(alpha=0.0, beta=1.0, gamma=0.0),
            ),
        )
        name_resp = client.send_or_raise_as(
            SearchResponse,
            SearchRequest(
                path=fake_repo,
                query=query,
                limit=3,
                weights=WeightTriple(alpha=0.0, beta=0.0, gamma=1.0),
            ),
        )
        sem_top = [r.name for r in sem_resp.results]
        lex_top = [r.name for r in lex_resp.results]
        name_top = [r.name for r in name_resp.results]

    assert not (sem_top == lex_top == name_top), (
        "all three weight configs returned identical rankings — "
        "at least one search channel is not producing signal"
    )
