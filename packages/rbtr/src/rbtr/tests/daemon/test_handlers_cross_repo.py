"""Cross-repo handlers end-to-end through the daemon RPC socket.

Uses the `two_repo_server` fixture (real `DaemonServer` over two
indexed repos) to verify scope behaviour and non-search isolation
over the wire, not just via direct handler calls.
"""

from __future__ import annotations

from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import (
    ReadSymbolRequest,
    ReadSymbolResponse,
    Scope,
    SearchRequest,
    SearchResponse,
    StatusRequest,
    StatusResponse,
)

from .conftest import TwoRepoServer


def test_search_scope_all_merges_repos(two_repo_server: TwoRepoServer) -> None:
    """`scope=all` over the socket returns both repos, attributed."""
    with DaemonClient(two_repo_server.server.runtime_dir) as client:
        resp = client.send(
            SearchRequest(repo_path=two_repo_server.path_a, query="shared_fn", scope=Scope.ALL)
        )
    assert isinstance(resp, SearchResponse)
    # Both repos hold a `shared_fn`; the hit carries repo_path attribution
    # rather than an id, so the merge shows as both repo_paths present.
    assert "shared_fn" in {r.name for r in resp.results}
    assert {r.repo_path for r in resp.results} == {
        two_repo_server.path_a,
        two_repo_server.path_b,
    }


def test_search_workspace_excludes_other_repo(two_repo_server: TwoRepoServer) -> None:
    """A workspace search over the socket stays in the path's repo."""
    with DaemonClient(two_repo_server.server.runtime_dir) as client:
        resp = client.send(SearchRequest(repo_path=two_repo_server.path_b, query="shared_fn"))
    assert isinstance(resp, SearchResponse)
    names = {r.name for r in resp.results}
    assert "shared_fn" in names
    # Repo A's unique symbol must not leak into a repo-B workspace search.
    assert "alpha_fn" not in names
    assert all(r.repo_path is None for r in resp.results)


def test_status_scope_all_lists_both_repos(two_repo_server: TwoRepoServer) -> None:
    """`status --scope all` over the socket reports both repos."""
    with DaemonClient(two_repo_server.server.runtime_dir) as client:
        resp = client.send(StatusRequest(repo_path=two_repo_server.path_a, scope=Scope.ALL))
    assert isinstance(resp, StatusResponse)
    repo_paths = {ref.repo_path for ref in resp.indexed_refs}
    assert repo_paths == {two_repo_server.path_a, two_repo_server.path_b}


def test_status_workspace_single_repo(two_repo_server: TwoRepoServer) -> None:
    """Workspace status over the socket reports only the path's repo."""
    with DaemonClient(two_repo_server.server.runtime_dir) as client:
        resp = client.send(StatusRequest(repo_path=two_repo_server.path_a))
    assert isinstance(resp, StatusResponse)
    assert resp.indexed_refs
    assert all(ref.repo_path == two_repo_server.path_a for ref in resp.indexed_refs)


def test_read_symbol_isolated_to_repo(two_repo_server: TwoRepoServer) -> None:
    """read_symbol for a colliding name returns only the path's repo."""
    with DaemonClient(two_repo_server.server.runtime_dir) as client:
        resp = client.send(ReadSymbolRequest(repo_path=two_repo_server.path_a, symbol="shared_fn"))
    assert isinstance(resp, ReadSymbolResponse)
    # Both repos hold a `shared_fn` in `shared.py`; the DTO carries no id,
    # so isolation shows as exactly one chunk (a leak would return two).
    assert len(resp.chunks) == 1
    assert resp.chunks[0].name == "shared_fn"
