"""Verify all three search channels produce distinct rankings.

Each channel (semantic, lexical, name-match) scores candidates
differently.  Searching the same query with extreme weights
`(1,0,0)`, `(0,1,0)`, `(0,0,1)` must produce at least two
distinct top-1 results.  If all three return the same ranking,
a channel is dead.

Exercises all three search channels (semantic, lexical, name-match)
with a fixed query vector at the Embedder boundary for controlled cosine
similarity.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pygit2
import pytest
from pytest_mock import MockerFixture

from rbtr.config import WeightTriple, config
from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import SearchRequest, SearchResponse
from rbtr.daemon.server import Daemon
from rbtr.index.build import build_index
from rbtr.index.embed import embed_index
from rbtr.index.embeddings import Embedder
from rbtr.index.index import Index
from rbtr.index.reranker import Reranker
from rbtr.index.search import search

from ..conftest import make_commit
from .conftest import serving

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


@pytest.fixture
def embedder() -> Generator[Embedder]:
    instance = Embedder()
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture
def reranker() -> Generator[Reranker | None]:
    instance = Reranker() if config.reranker_model else None
    try:
        yield instance
    finally:
        if instance is not None:
            instance.close()


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
def stub_vectors(
    channel_vecs: tuple[list[float], dict[str, list[float]]],
) -> dict[str, list[float]]:
    """Each channel function's hand-crafted vector, normalised, for `StubModel`."""
    _query_vec, chunk_vecs = channel_vecs
    return {name: _normalise(vec) for name, vec in chunk_vecs.items()}


@pytest.fixture
def channel_index(fake_repo: str, index: Index, stubbed_embedder: Embedder) -> Index:
    """`fake_repo` with HEAD built and embedded, its functions designed so the channels disagree.

    - "find_items": content about searching/finding (semantic
      likes it for "search for items") but a different name.
    - "search": exact name match (name channel).
    - "lookup": "search" repeated in content (lexical channel).
    """
    find = b"""\
def find_items(query):
    '''Search the database and find matching items.'''
"""
    search_py = b"""\
def search(q):
    return results
"""
    lookup = b"""\
def lookup(term):
    '''Search search search through the index to search.'''
"""
    greet = b"""\
def greet(name):
    return f"hello {name}"
"""
    parse = b"""\
def parse_config(path):
    return load(path)
"""
    files = {
        "src/find.py": find,
        "src/search.py": search_py,
        "src/lookup.py": lookup,
        "src/greet.py": greet,
        "src/parse.py": parse,
    }
    repo = pygit2.Repository(fake_repo)
    sha = str(make_commit(repo, files, parents=[repo.head.peel(pygit2.Commit).id]))
    build_index(fake_repo, sha, index)
    embed_index(index, at=index.register(fake_repo).at(sha), embedder=stubbed_embedder)
    return index


@pytest.fixture
def running_daemon(
    runtime_dir: Path,
    channel_index: Index,
    channel_vecs: tuple[list[float], dict[str, list[float]]],
    stub_embedding_model: None,
    mocker: MockerFixture,
) -> Generator[Daemon]:
    """A served daemon whose embedder returns the fixture's query vector."""
    query_vec, _chunk_vecs = channel_vecs
    mocker.patch.object(Embedder, "embed_single", return_value=_normalise(query_vec))
    server = Daemon(
        runtime_dir, index=channel_index, idle_poll_interval=60.0, busy_poll_interval=60.0
    )
    with serving(server):
        yield server


def test_daemon_and_direct_search_produce_identical_results(
    running_daemon: Daemon,
    channel_index: Index,
    embedder: Embedder,
    reranker: Reranker | None,
    fake_repo: str,
) -> None:
    """Daemon-served search and direct store.search must return the same ranking.

    The daemon is a performance optimisation (pre-loaded model,
    non-blocking indexing), not a functional difference.  If the
    rankings diverge, one path is missing a channel or applying
    different weights.
    """
    query = "search"

    # Via daemon.
    with DaemonClient(running_daemon.runtime_dir) as client:
        req = SearchRequest(repo_path=fake_repo, query=query, limit=5)
        daemon_resp = client.send_or_raise_as(SearchResponse, req)
    daemon_names = [r.name for r in daemon_resp.results]
    daemon_scores = [round(r.score, 6) for r in daemon_resp.results]

    # Via direct store.search — the request sent no keywords/variants, so
    # the daemon applied no expansion; mirror that with the reranker it used.
    sha = str(pygit2.Repository(fake_repo).head.target)
    [repo] = channel_index.repos()
    direct_results = search(
        channel_index,
        query,
        within=[repo.at(sha)],
        top_k=5,
        embedder=embedder,
        reranker=reranker,
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
    running_daemon: Daemon, fake_repo: str
) -> None:
    """Extreme weight configs must not all produce the same ranking.

    If semantic or name channels are dead, alpha=1 and gamma=1
    both fall back to BM25, producing identical results.
    """
    with DaemonClient(running_daemon.runtime_dir) as client:
        query = "search"
        sem_resp = client.send_or_raise_as(
            SearchResponse,
            SearchRequest(
                repo_path=fake_repo,
                query=query,
                limit=3,
                weights=WeightTriple(alpha=1.0, beta=0.0, gamma=0.0),
            ),
        )
        lex_resp = client.send_or_raise_as(
            SearchResponse,
            SearchRequest(
                repo_path=fake_repo,
                query=query,
                limit=3,
                weights=WeightTriple(alpha=0.0, beta=1.0, gamma=0.0),
            ),
        )
        name_resp = client.send_or_raise_as(
            SearchResponse,
            SearchRequest(
                repo_path=fake_repo,
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
