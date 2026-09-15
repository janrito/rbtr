"""Cross-repo handlers end-to-end through the daemon RPC socket.

Serves a real `DaemonServer` over two indexed repos to verify scope
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
    Scope,
    SearchRequest,
    SearchResponse,
    StatusRequest,
    StatusResponse,
)
from rbtr.daemon.server import DaemonServer
from rbtr.domain.models import FileSnapshot, SnapshotRef
from rbtr.index.store import IndexStore

from ..index.conftest import make_chunk
from .conftest import serving


@pytest.fixture
def shared_symbol_store(fake_repo: str, second_repo: str, store: IndexStore) -> IndexStore:
    """Index both repos, each holding a unique symbol plus `shared_fn`.

    The shared name is what makes cross-repo merge, workspace isolation
    and non-search isolation observable: a leak shows as the other
    repo's symbol, or as a second chunk for the same name.  Chunks are
    seeded under each repo's real id and HEAD sha, because the handlers
    resolve HEAD through git.
    """
    for path, uniq in ((fake_repo, "alpha"), (second_repo, "beta")):
        head = str(pygit2.Repository(path).head.target)
        with store.session() as ws:
            repo_id = ws.register_repo(path)
            ws.add_chunk(make_chunk(f"{uniq}_id", name=f"{uniq}_fn", path=f"{uniq}.py"))
            ws.add_chunk(make_chunk(f"shared_{uniq}", name="shared_fn", path="shared.py"))
            ws.insert_snapshots(
                [
                    FileSnapshot(
                        snapshot_sha=head, file_path=f"{uniq}.py", blob_sha=f"blob_{uniq}_id"
                    ),
                    FileSnapshot(
                        snapshot_sha=head, file_path="shared.py", blob_sha=f"blob_shared_{uniq}"
                    ),
                ],
                repo_id=repo_id,
            )
            ws.mark_indexed(at=SnapshotRef(repo_id=repo_id, snapshot_sha=head))
    return store


@pytest.fixture
def running_daemon(
    runtime_dir: Path,
    shared_symbol_store: IndexStore,
    stub_embedding_model: None,
) -> Generator[DaemonServer]:
    """A served daemon over both repos (stub embedder: routing, not vectors)."""
    with serving(
        DaemonServer(
            runtime_dir, store=shared_symbol_store, idle_poll_interval=60.0, busy_poll_interval=60.0
        )
    ) as server:
        yield server


def test_search_scope_all_merges_repos(
    running_daemon: DaemonServer, fake_repo: str, second_repo: str
) -> None:
    """`scope=all` over the socket returns both repos, attributed."""
    with DaemonClient(running_daemon.runtime_dir) as client:
        resp = client.send(SearchRequest(repo_path=fake_repo, query="shared_fn", scope=Scope.ALL))
    assert isinstance(resp, SearchResponse)
    # Both repos hold a `shared_fn`; the hit carries repo_path attribution
    # rather than an id, so the merge shows as both repo_paths present.
    assert "shared_fn" in {r.name for r in resp.results}
    assert {r.repo_path for r in resp.results} == {
        fake_repo,
        second_repo,
    }


def test_search_workspace_excludes_other_repo(
    running_daemon: DaemonServer, second_repo: str
) -> None:
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
    running_daemon: DaemonServer, fake_repo: str, second_repo: str
) -> None:
    """`status --scope all` over the socket reports both repos."""
    with DaemonClient(running_daemon.runtime_dir) as client:
        resp = client.send(StatusRequest(repo_path=fake_repo, scope=Scope.ALL))
    assert isinstance(resp, StatusResponse)
    repo_paths = {ref.repo_path for ref in resp.indexed_refs}
    assert repo_paths == {fake_repo, second_repo}


def test_status_workspace_single_repo(running_daemon: DaemonServer, fake_repo: str) -> None:
    """Workspace status over the socket reports only the path's repo."""
    with DaemonClient(running_daemon.runtime_dir) as client:
        resp = client.send(StatusRequest(repo_path=fake_repo))
    assert isinstance(resp, StatusResponse)
    assert resp.indexed_refs
    assert all(ref.repo_path == fake_repo for ref in resp.indexed_refs)


def test_read_symbol_isolated_to_repo(running_daemon: DaemonServer, fake_repo: str) -> None:
    """read_symbol for a colliding name returns only the path's repo."""
    with DaemonClient(running_daemon.runtime_dir) as client:
        resp = client.send(ReadSymbolRequest(repo_path=fake_repo, symbol="shared_fn"))
    assert isinstance(resp, ReadSymbolResponse)
    # Both repos hold a `shared_fn` in `shared.py`; the DTO carries no id,
    # so isolation shows as exactly one chunk (a leak would return two).
    assert len(resp.chunks) == 1
    assert resp.chunks[0].name == "shared_fn"
