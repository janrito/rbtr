"""Tests for daemon index handlers — end-to-end via ZMQ.

Uses `running_server_with_index` fixture which starts a real
server backed by an in-memory IndexStore seeded with test data.
Tests verify typed responses via the actual socket path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import (
    ErrorResponse,
    FindRefsRequest,
    FindRefsResponse,
    ListSymbolsRequest,
    ListSymbolsResponse,
    ReadSymbolRequest,
    ReadSymbolResponse,
    SearchRequest,
    SearchResponse,
    StatusRequest,
    StatusResponse,
)
from rbtr.daemon.server import DaemonServer


@pytest.fixture
def unindexed_ref() -> str:
    """A ref that resolves but was never indexed.

    A 40-char hex SHA resolves to itself without repo access, and the
    seeded store never indexed it — so it exercises the "resolved but
    not indexed" path rather than "cannot resolve".
    """
    return "0" * 40


# ── Search ───────────────────────────────────────────────────────────


def test_search_returns_results(running_server_with_index: DaemonServer, fake_repo: str) -> None:
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        resp = client.send(SearchRequest(repo_path=fake_repo, query="load_config"))
    assert isinstance(resp, SearchResponse)
    assert len(resp.results) > 0
    names = {r.name for r in resp.results}
    assert "load_config" in names


def test_search_respects_limit(running_server_with_index: DaemonServer, fake_repo: str) -> None:
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        resp = client.send(SearchRequest(repo_path=fake_repo, query="config", limit=1))
    assert isinstance(resp, SearchResponse)
    assert len(resp.results) <= 1


def test_search_with_client_keywords(
    running_server_with_index: DaemonServer,
    fake_repo: str,
) -> None:
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        resp = client.send(
            SearchRequest(
                repo_path=fake_repo,
                query="load_config",
                keywords=["settings", "configuration"],
            ),
        )
    assert isinstance(resp, SearchResponse)
    assert resp.expansion is not None
    assert resp.expansion.keywords == ["settings", "configuration"]


def test_search_with_client_variants(
    running_server_with_index: DaemonServer,
    fake_repo: str,
) -> None:
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        resp = client.send(
            SearchRequest(
                repo_path=fake_repo,
                query="load_config",
                variants=["read configuration from file"],
            ),
        )
    assert isinstance(resp, SearchResponse)
    assert resp.expansion is not None
    assert resp.expansion.variants == ["read configuration from file"]


# ── Read symbol ──────────────────────────────────────────────────────


def test_read_symbol(running_server_with_index: DaemonServer, fake_repo: str) -> None:
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        resp = client.send(ReadSymbolRequest(repo_path=fake_repo, symbol="load_config"))
    assert isinstance(resp, ReadSymbolResponse)
    assert len(resp.chunks) >= 1
    names = {c.name for c in resp.chunks}
    assert "load_config" in names


def test_read_symbol_not_found(running_server_with_index: DaemonServer, fake_repo: str) -> None:
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        resp = client.send(ReadSymbolRequest(repo_path=fake_repo, symbol="nonexistent_xyz"))
    assert isinstance(resp, ReadSymbolResponse)
    assert len(resp.chunks) == 0


def test_read_symbol_unindexed_ref_errors(
    running_server_with_index: DaemonServer, fake_repo: str, unindexed_ref: str
) -> None:
    """An explicit ref that isn't indexed is an error, not an empty result."""
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        resp = client.send(
            ReadSymbolRequest(repo_path=fake_repo, symbol="load_config", ref=unindexed_ref)
        )
    assert isinstance(resp, ErrorResponse)
    assert "not indexed" in resp.message


def test_read_symbol_with_file_paths(
    running_server_with_index: DaemonServer, fake_repo: str
) -> None:
    """`file_paths` narrows the lookup to chunks from the listed files.

    The seeded `load_config` matches both the function (`src/config.py`)
    and the import line (`src/app.py`, name `from config import
    load_config`), so scoping must partition by file.
    """
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        unscoped = client.send(ReadSymbolRequest(repo_path=fake_repo, symbol="load_config"))
        scoped = client.send(
            ReadSymbolRequest(
                repo_path=fake_repo, symbol="load_config", file_paths=["src/config.py"]
            )
        )
        missing = client.send(
            ReadSymbolRequest(
                repo_path=fake_repo, symbol="load_config", file_paths=["nonexistent.py"]
            )
        )
    assert isinstance(unscoped, ReadSymbolResponse)
    unscoped_files = {c.file_path for c in unscoped.chunks}
    assert {"src/config.py", "src/app.py"} <= unscoped_files
    assert isinstance(scoped, ReadSymbolResponse)
    assert len(scoped.chunks) >= 1
    assert all(c.file_path == "src/config.py" for c in scoped.chunks)
    assert isinstance(missing, ReadSymbolResponse)
    assert len(missing.chunks) == 0


# ── List symbols ─────────────────────────────────────────────────────


def test_list_symbols(running_server_with_index: DaemonServer, fake_repo: str) -> None:
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        resp = client.send(ListSymbolsRequest(repo_path=fake_repo, file_path="src/config.py"))
    assert isinstance(resp, ListSymbolsResponse)
    assert len(resp.chunks) >= 1
    names = {c.name for c in resp.chunks}
    assert "load_config" in names


def test_list_symbols_empty_file(running_server_with_index: DaemonServer, fake_repo: str) -> None:
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        resp = client.send(ListSymbolsRequest(repo_path=fake_repo, file_path="nonexistent.py"))
    assert isinstance(resp, ListSymbolsResponse)
    assert len(resp.chunks) == 0


def test_list_symbols_unindexed_ref_errors(
    running_server_with_index: DaemonServer, fake_repo: str, unindexed_ref: str
) -> None:
    """An explicit ref that isn't indexed is an error, not an empty outline."""
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        resp = client.send(
            ListSymbolsRequest(repo_path=fake_repo, file_path="src/config.py", ref=unindexed_ref)
        )
    assert isinstance(resp, ErrorResponse)
    assert "not indexed" in resp.message


# ── Find refs ────────────────────────────────────────────────────────


def test_find_refs(running_server_with_index: DaemonServer, fake_repo: str) -> None:
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        resp = client.send(FindRefsRequest(repo_path=fake_repo, symbol="load_config"))
    assert isinstance(resp, FindRefsResponse)
    assert len(resp.edges) >= 1
    assert resp.edges[0].target_id == "fn_config"


def test_find_refs_unindexed_ref_errors(
    running_server_with_index: DaemonServer, fake_repo: str, unindexed_ref: str
) -> None:
    """An explicit ref that isn't indexed is an error, not empty edges."""
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        resp = client.send(
            FindRefsRequest(repo_path=fake_repo, symbol="load_config", ref=unindexed_ref)
        )
    assert isinstance(resp, ErrorResponse)
    assert "not indexed" in resp.message


def test_find_refs_with_file_paths(running_server_with_index: DaemonServer, fake_repo: str) -> None:
    """`file_paths` narrows name resolution before edges are queried."""
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        scoped = client.send(
            FindRefsRequest(repo_path=fake_repo, symbol="load_config", file_paths=["src/config.py"])
        )
        elsewhere = client.send(
            FindRefsRequest(repo_path=fake_repo, symbol="load_config", file_paths=["src/app.py"])
        )
    assert isinstance(scoped, FindRefsResponse)
    assert len(scoped.edges) >= 1
    assert isinstance(elsewhere, FindRefsResponse)
    assert len(elsewhere.edges) == 0


# ── Status ───────────────────────────────────────────────────────────


def test_status_with_index(
    running_server_with_index: DaemonServer,
    fake_repo: str,
    daemon_commit: str,
) -> None:
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        resp = client.send(StatusRequest(repo_path=fake_repo))
    assert isinstance(resp, StatusResponse)
    assert len(resp.indexed_refs) == 1
    assert resp.indexed_refs[0].sha == daemon_commit
    assert resp.indexed_refs[0].total > 0
    assert resp.indexed_refs[0].embedded == 0  # no embedder configured


def test_status_unknown_repo(
    running_server_with_index: DaemonServer, fake_repo: str, tmp_path: Path
) -> None:
    """Status for an unregistered repo returns empty response."""
    # A real git repo that isn't registered in the store.
    other = tmp_path / "other"
    import pygit2

    r = pygit2.init_repository(str(other), bare=False, initial_head="main")
    sig = pygit2.Signature("t", "t@t.t")
    r.create_commit("refs/heads/main", sig, sig, "init", r.TreeBuilder().write(), [])
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        resp = client.send(StatusRequest(repo_path=str(other)))
    assert isinstance(resp, StatusResponse)
    assert resp.indexed_refs == []
