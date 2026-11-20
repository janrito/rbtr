"""Cross-repo handlers end-to-end through the daemon RPC socket.

Serves a real `Daemon` over two indexed repos to verify scope
behaviour and non-search isolation over the wire, not just via direct
handler calls.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pygit2
import pytest

from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import (
    ReadSymbolRequest,
    ReadSymbolResponse,
    SearchAllRequest,
    SearchRequest,
    SearchResponse,
    StatusAllRequest,
    StatusRequest,
    StatusResponse,
)
from rbtr.daemon.server import Daemon
from rbtr.index.build import build_index
from rbtr.index.index import Index

from ..conftest import make_commit
from .conftest import serving


@pytest.fixture
def shared_symbol_index(fake_repo: str, second_repo: str, index: Index) -> Index:
    """Both repos built at a new HEAD, each holding a unique symbol plus `shared_fn`.

    The shared name is what makes cross-repo merge, workspace isolation
    and non-search isolation observable: a leak shows as the other
    repo's symbol, or as a second chunk for the same name.  Each repo is
    built at its HEAD, because the handlers resolve HEAD through git.
    """
    shared = b"""\
def shared_fn():
    pass
"""
    alpha = b"""\
def alpha_fn():
    pass
"""
    beta = b"""\
def beta_fn():
    pass
"""
    for path, files in (
        (fake_repo, {"alpha.py": alpha, "shared.py": shared}),
        (second_repo, {"beta.py": beta, "shared.py": shared}),
    ):
        repo = pygit2.Repository(path)
        sha = str(make_commit(repo, files, parents=[repo.head.peel(pygit2.Commit).id]))
        repo.checkout_head(strategy=pygit2.GIT_CHECKOUT_FORCE)  # type: ignore[no-untyped-call]  # pygit2 untyped
        build_index(path, sha, index)
    return index


@pytest.fixture
def running_daemon(
    runtime_dir: Path,
    shared_symbol_index: Index,
    stub_embedding_model: None,
) -> Generator[Daemon]:
    """A served daemon over both repos (stub embedder: routing, not vectors)."""
    with serving(
        Daemon(
            runtime_dir, index=shared_symbol_index, idle_poll_interval=60.0, busy_poll_interval=60.0
        )
    ) as server:
        yield server


def test_search_scope_all_merges_repos(
    running_daemon: Daemon, fake_repo: str, second_repo: str
) -> None:
    """`scope=all` over the socket returns both repos, attributed."""
    with DaemonClient(running_daemon.runtime_dir) as client:
        resp = client.send(SearchAllRequest(query="shared_fn"))
    assert isinstance(resp, SearchResponse)
    # Both repos hold a `shared_fn`; the hit carries repo_path attribution
    # rather than an id, so the merge shows as both repo_paths present.
    assert "shared_fn" in {r.name for r in resp.results}
    assert {r.repo_path for r in resp.results} == {
        fake_repo,
        second_repo,
    }


def test_search_workspace_excludes_other_repo(running_daemon: Daemon, second_repo: str) -> None:
    """A workspace search over the socket stays in the path's repo."""
    with DaemonClient(running_daemon.runtime_dir) as client:
        resp = client.send(SearchRequest(repo_path=second_repo, query="shared_fn"))
    assert isinstance(resp, SearchResponse)
    names = {r.name for r in resp.results}
    assert "shared_fn" in names
    # Repo A's unique symbol must not leak into a repo-B workspace search.
    assert "alpha_fn" not in names
    assert all(r.repo_path is None for r in resp.results)


def test_status_scope_all_lists_both_repos(
    running_daemon: Daemon, fake_repo: str, second_repo: str
) -> None:
    """`status --scope all` over the socket reports both repos."""
    with DaemonClient(running_daemon.runtime_dir) as client:
        resp = client.send(StatusAllRequest())
    assert isinstance(resp, StatusResponse)
    repo_paths = {ref.repo_path for ref in resp.indexed_refs}
    assert repo_paths == {fake_repo, second_repo}


def test_status_workspace_single_repo(running_daemon: Daemon, fake_repo: str) -> None:
    """Workspace status over the socket reports only the path's repo."""
    with DaemonClient(running_daemon.runtime_dir) as client:
        resp = client.send(StatusRequest(repo_path=fake_repo))
    assert isinstance(resp, StatusResponse)
    assert resp.indexed_refs
    assert all(ref.repo_path == fake_repo for ref in resp.indexed_refs)


def test_read_symbol_isolated_to_repo(running_daemon: Daemon, fake_repo: str) -> None:
    """read_symbol for a colliding name returns only the path's repo."""
    with DaemonClient(running_daemon.runtime_dir) as client:
        resp = client.send(ReadSymbolRequest(repo_path=fake_repo, symbol="shared_fn"))
    assert isinstance(resp, ReadSymbolResponse)
    # Both repos hold a `shared_fn` in `shared.py`; the DTO carries no id,
    # so isolation shows as exactly one chunk (a leak would return two).
    assert len(resp.chunks) == 1
    assert resp.chunks[0].name == "shared_fn"
