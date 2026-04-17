"""Tests for daemon index handlers — end-to-end via ZMQ.

Uses `running_server_with_index` fixture which starts a real
server backed by an in-memory IndexStore seeded with test data.
Tests verify typed responses via the actual socket path.
"""

from __future__ import annotations

from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import (
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

# ── Search ───────────────────────────────────────────────────────────


def test_search_returns_results(running_server_with_index: DaemonServer) -> None:
    with DaemonClient(running_server_with_index.sock_dir) as client:
        resp = client.send(SearchRequest(repo="/test/repo", query="load_config"))
    assert isinstance(resp, SearchResponse)
    assert len(resp.results) > 0
    names = {r.chunk.name for r in resp.results}
    assert "load_config" in names


def test_search_respects_limit(running_server_with_index: DaemonServer) -> None:
    with DaemonClient(running_server_with_index.sock_dir) as client:
        resp = client.send(SearchRequest(repo="/test/repo", query="config", limit=1))
    assert isinstance(resp, SearchResponse)
    assert len(resp.results) <= 1


# ── Read symbol ──────────────────────────────────────────────────────


def test_read_symbol(running_server_with_index: DaemonServer) -> None:
    with DaemonClient(running_server_with_index.sock_dir) as client:
        resp = client.send(ReadSymbolRequest(repo="/test/repo", name="load_config"))
    assert isinstance(resp, ReadSymbolResponse)
    assert len(resp.chunks) >= 1
    names = {c.name for c in resp.chunks}
    assert "load_config" in names


def test_read_symbol_not_found(running_server_with_index: DaemonServer) -> None:
    with DaemonClient(running_server_with_index.sock_dir) as client:
        resp = client.send(ReadSymbolRequest(repo="/test/repo", name="nonexistent_xyz"))
    assert isinstance(resp, ReadSymbolResponse)
    assert len(resp.chunks) == 0


# ── List symbols ─────────────────────────────────────────────────────


def test_list_symbols(running_server_with_index: DaemonServer) -> None:
    with DaemonClient(running_server_with_index.sock_dir) as client:
        resp = client.send(ListSymbolsRequest(repo="/test/repo", file_path="src/config.py"))
    assert isinstance(resp, ListSymbolsResponse)
    assert len(resp.chunks) >= 1
    names = {c.name for c in resp.chunks}
    assert "load_config" in names


def test_list_symbols_empty_file(running_server_with_index: DaemonServer) -> None:
    with DaemonClient(running_server_with_index.sock_dir) as client:
        resp = client.send(ListSymbolsRequest(repo="/test/repo", file_path="nonexistent.py"))
    assert isinstance(resp, ListSymbolsResponse)
    assert len(resp.chunks) == 0


# ── Find refs ────────────────────────────────────────────────────────


def test_find_refs(running_server_with_index: DaemonServer) -> None:
    with DaemonClient(running_server_with_index.sock_dir) as client:
        resp = client.send(FindRefsRequest(repo="/test/repo", symbol="fn_config"))
    assert isinstance(resp, FindRefsResponse)
    assert len(resp.edges) >= 1
    assert resp.edges[0].target_id == "fn_config"


# ── Status ───────────────────────────────────────────────────────────


def test_status_with_index(
    running_server_with_index: DaemonServer,
    daemon_commit: str,
) -> None:
    with DaemonClient(running_server_with_index.sock_dir) as client:
        resp = client.send(StatusRequest(repo="/test/repo"))
    assert isinstance(resp, StatusResponse)
    assert resp.exists is True
    assert resp.total_chunks is not None
    assert resp.total_chunks > 0
    assert resp.indexed_refs == [daemon_commit]


def test_status_unknown_repo(running_server_with_index: DaemonServer) -> None:
    with DaemonClient(running_server_with_index.sock_dir) as client:
        resp = client.send(StatusRequest(repo="/unknown/repo"))
    assert isinstance(resp, StatusResponse)
    assert resp.total_chunks == 0
    assert resp.indexed_refs == []
