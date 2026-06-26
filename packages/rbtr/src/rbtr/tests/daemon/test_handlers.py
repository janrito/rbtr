"""Tests for daemon index handlers — end-to-end via ZMQ.

Uses `running_server_with_index` fixture which starts a real
server backed by an in-memory IndexStore seeded with test data.
Tests verify typed responses via the actual socket path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import structlog

from rbtr.daemon.client import DaemonClient
from rbtr.daemon.handlers import handle_build_index, handle_gc, handle_status
from rbtr.daemon.messages import (
    ActiveJob,
    BuildIndexRequest,
    ErrorResponse,
    FindRefsRequest,
    FindRefsResponse,
    GcMode,
    GcRequest,
    ListSymbolsRequest,
    ListSymbolsResponse,
    OkResponse,
    ReadSymbolRequest,
    ReadSymbolResponse,
    SearchRequest,
    SearchResponse,
    StatusRequest,
    StatusResponse,
)
from rbtr.daemon.server import DaemonServer
from rbtr.errors import RbtrError
from rbtr.index.models import EdgeKind, QueryKind
from rbtr.index.store import IndexStore


@pytest.fixture
def unindexed_ref() -> str:
    """A ref that resolves but was never indexed.

    A 40-char hex SHA resolves to itself without repo access, and the
    seeded store never indexed it — so it exercises the "resolved but
    not indexed" path rather than "cannot resolve".
    """
    return "0" * 40


@pytest.mark.parametrize(
    "mode",
    [GcMode.WATCHED_ONLY, GcMode.HEAD_ONLY, GcMode.KEEP, GcMode.ORPHANS],
)
def test_handle_gc_global_rejects_non_watched_mode(mode: GcMode, store: IndexStore) -> None:
    """A global request (no repo_path) is restricted to the safe default
    reclamation; any other mode is rejected before touching the store."""
    with pytest.raises(RbtrError, match="default"):
        handle_gc(GcRequest(repo_path=None, mode=mode), store)


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


@pytest.mark.parametrize(
    ("keywords", "variants"),
    [
        (["settings", "configuration"], None),
        (None, ["read configuration from file"]),
    ],
)
def test_search_accepts_expansion_inputs(
    running_server_with_index: DaemonServer,
    fake_repo: str,
    keywords: list[str] | None,
    variants: list[str] | None,
) -> None:
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        resp = client.send(
            SearchRequest(
                repo_path=fake_repo,
                query="load_config",
                keywords=keywords,
                variants=variants,
            ),
        )
    assert isinstance(resp, SearchResponse)
    assert len(resp.results) > 0


@pytest.mark.parametrize("explain", [True, False])
def test_search_query_kind_only_under_explain(
    running_server_with_index: DaemonServer,
    fake_repo: str,
    explain: bool,
) -> None:
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        resp = client.send(
            SearchRequest(repo_path=fake_repo, query="load_config", explain=explain),
        )
    assert isinstance(resp, SearchResponse)
    assert (resp.query_kind is not None) == explain


def test_search_query_kind_override_without_expansion(
    running_server_with_index: DaemonServer,
    fake_repo: str,
) -> None:
    """An explicit query_kind is honoured with no keywords/variants present."""
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        resp = client.send(
            SearchRequest(
                repo_path=fake_repo, query="load_config", query_kind="code", explain=True
            ),
        )
    assert isinstance(resp, SearchResponse)
    assert resp.query_kind == QueryKind.CODE


# ── Read symbol ──────────────────────────────────────────────────────


def test_read_symbol(running_server_with_index: DaemonServer, fake_repo: str) -> None:
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        resp = client.send(ReadSymbolRequest(repo_path=fake_repo, symbol="load_config"))
    assert isinstance(resp, ReadSymbolResponse)
    assert len(resp.chunks) >= 1
    names = {c.name for c in resp.chunks}
    assert "load_config" in names


def test_read_symbol_returns_variable(
    running_server_with_index: DaemonServer, fake_repo: str
) -> None:
    """Module-level VARIABLE chunks are readable like any other symbol."""
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        resp = client.send(ReadSymbolRequest(repo_path=fake_repo, symbol="MAX_SIZE"))
    assert isinstance(resp, ReadSymbolResponse)
    names = {c.name for c in resp.chunks}
    assert "MAX_SIZE" in names


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


def test_read_symbol_unindexed_ref_reports_building_when_active(
    running_server_with_index: DaemonServer, fake_repo: str, unindexed_ref: str
) -> None:
    """While a build is active, an unindexed ref reports 'building'.

    Build-awareness lives only in the daemon's `_dispatch`: the handler
    raises a plain not-indexed error, which `_dispatch` upgrades when a
    build is running.
    """
    server = running_server_with_index
    server._active_build = ActiveJob(
        repo_path=fake_repo, ref="x" * 40, phase="parsing", current=1, total=2, elapsed_seconds=0.0
    )
    try:
        with DaemonClient(server.runtime_dir) as client:
            resp = client.send(
                ReadSymbolRequest(repo_path=fake_repo, symbol="load_config", ref=unindexed_ref)
            )
    finally:
        server._active_build = None
    assert isinstance(resp, ErrorResponse)
    assert "building" in resp.message.lower()


def test_read_symbol_with_file_paths(
    running_server_with_index: DaemonServer, fake_repo: str
) -> None:
    """`file_paths` narrows the lookup to chunks from the listed files.

    Tiered name resolution means `load_config` exact-matches only
    the function definition in `src/config.py`.  The import line
    in `src/app.py` (name `from config import load_config`) is a
    substring match at a lower tier and is excluded.
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
    assert "src/config.py" in unscoped_files
    assert "src/app.py" not in unscoped_files  # substring match excluded
    assert isinstance(scoped, ReadSymbolResponse)
    assert len(scoped.chunks) >= 1
    assert all(c.file_path == "src/config.py" for c in scoped.chunks)
    assert isinstance(missing, ReadSymbolResponse)
    assert len(missing.chunks) == 0


def test_read_symbol_file_paths_absolute_end_to_end(
    running_server_with_index: DaemonServer, fake_repo: str
) -> None:
    """An absolute scoping path resolves through the full wire path.

    Path-form normalisation itself is covered at the model level
    (`cases_messages.py`); this guards that the normalised request is
    what the daemon deserialises and scopes against stored chunks.
    """
    abs_path = str(Path(fake_repo) / "src/config.py")
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        resp = client.send(
            ReadSymbolRequest(repo_path=fake_repo, symbol="load_config", file_paths=[abs_path])
        )
    assert isinstance(resp, ReadSymbolResponse)
    assert len(resp.chunks) >= 1
    assert all(c.file_path == "src/config.py" for c in resp.chunks)


def test_read_symbol_implicit_falls_back_when_head_unindexed(
    running_server_with_index: DaemonServer, fake_repo: str, daemon_commit: str
) -> None:
    """A moved-but-unindexed HEAD resolves to the latest indexed commit.

    Reproduces the session-start race (HEAD advanced or still building)
    where the resolved ref isn't indexed: the implicit lookup should
    fall back rather than report the symbol missing.
    """
    import pygit2

    repo = pygit2.Repository(fake_repo)
    sig = pygit2.Signature("t", "t@t.t")
    tree = repo.head.peel(pygit2.Commit).tree.id
    repo.create_commit("HEAD", sig, sig, "second", tree, [repo.head.target])
    assert str(repo.head.target) != daemon_commit

    with DaemonClient(running_server_with_index.runtime_dir) as client:
        resp = client.send(ReadSymbolRequest(repo_path=fake_repo, symbol="load_config"))
    assert isinstance(resp, ReadSymbolResponse)
    assert len(resp.chunks) >= 1


# ── List symbols ─────────────────────────────────────────────────────


def test_list_symbols(running_server_with_index: DaemonServer, fake_repo: str) -> None:
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        resp = client.send(ListSymbolsRequest(repo_path=fake_repo, file_path="src/config.py"))
    assert isinstance(resp, ListSymbolsResponse)
    assert len(resp.chunks) >= 1
    names = {c.name for c in resp.chunks}
    assert "load_config" in names
    assert "MAX_SIZE" in names


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
    assert len(resp.refs) >= 1
    # The reference resolves to the importing chunk in src/app.py.
    assert resp.refs[0].edge == EdgeKind.IMPORTS
    assert resp.refs[0].file_path == "src/app.py"


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
    assert len(scoped.refs) >= 1
    assert isinstance(elsewhere, FindRefsResponse)
    assert len(elsewhere.refs) == 0


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
    # The backfilled HEAD watch is surfaced and reported indexed.
    assert any(w.ref == "HEAD" and w.indexed for w in resp.watched)


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


# ── Index (watch_refs) ──────────────────────────────────────────
#
# Single-threaded: drive `handle_build_index` against a non-served
# server (no worker thread) so the watch-set write is observable
# without socket round-trips or build/embed background activity.


@pytest.fixture
def index_server(runtime_dir: Path, seeded_store: IndexStore) -> DaemonServer:
    """A server over the seeded store, not serving (no worker thread)."""
    return DaemonServer(
        runtime_dir, store=seeded_store, idle_poll_interval=60.0, busy_poll_interval=60.0
    )


def test_index_always_watches_head(
    index_server: DaemonServer, seeded_store: IndexStore, tmp_path: Path
) -> None:
    """A repo first seen via `index <ref>` (not startup backfill) still
    watches HEAD — the invariant that HEAD is always watched."""
    other = str(tmp_path / "other")
    handle_build_index(BuildIndexRequest(repo_path=other, refs=["main"]), index_server.watch_refs)
    watched = seeded_store.list_watched_refs(seeded_store.resolve_repo(other))
    assert "HEAD" in watched
    assert "main" in watched


def test_remove_on_unregistered_repo_is_noop(
    index_server: DaemonServer, seeded_store: IndexStore, tmp_path: Path
) -> None:
    """Removing from a repo that was never indexed is a no-op — and must not
    spuriously register the repo."""
    other = str(tmp_path / "unregistered")
    resp = handle_build_index(
        BuildIndexRequest(repo_path=other, refs=["main"], remove=True), index_server.watch_refs
    )
    assert isinstance(resp, OkResponse)
    assert seeded_store.get_repo_id(other) is None


def test_index_add_then_remove(
    index_server: DaemonServer, seeded_store: IndexStore, fake_repo: str
) -> None:
    """`index` records a ref in the watch set; `--remove` drops it."""
    repo_id = seeded_store.resolve_repo(fake_repo)
    added = handle_build_index(
        BuildIndexRequest(repo_path=fake_repo, refs=["main"]), index_server.watch_refs
    )
    assert isinstance(added, OkResponse)
    assert "main" in seeded_store.list_watched_refs(repo_id)

    removed = handle_build_index(
        BuildIndexRequest(repo_path=fake_repo, refs=["main"], remove=True),
        index_server.watch_refs,
    )
    assert isinstance(removed, OkResponse)
    assert "main" not in seeded_store.list_watched_refs(repo_id)


def test_index_remove_head_rejected_atomically(
    index_server: DaemonServer, seeded_store: IndexStore, fake_repo: str
) -> None:
    """`--remove HEAD` fails wholesale: no co-listed ref is deleted."""
    repo_id = seeded_store.resolve_repo(fake_repo)
    handle_build_index(
        BuildIndexRequest(repo_path=fake_repo, refs=["main"]), index_server.watch_refs
    )
    with pytest.raises(RbtrError, match="HEAD"):
        handle_build_index(
            BuildIndexRequest(repo_path=fake_repo, refs=["main", "HEAD"], remove=True),
            index_server.watch_refs,
        )
    watched = seeded_store.list_watched_refs(repo_id)
    assert "HEAD" in watched
    assert "main" in watched


def test_status_reports_watch_set_states(seeded_store: IndexStore, fake_repo: str) -> None:
    """Each watched ref is reported as indexed, pending, or unresolvable."""
    repo_id = seeded_store.resolve_repo(fake_repo)
    pending_sha = "0" * 40  # a bare SHA: resolves to itself, never indexed
    with seeded_store.session() as ws:
        ws.add_watched_refs(repo_id, ["main", pending_sha, "nonexistent-branch"])
    resp = handle_status(StatusRequest(repo_path=fake_repo), seeded_store)
    watched = {w.ref: w for w in resp.watched}
    assert watched["main"].indexed is True  # resolves to the indexed HEAD
    assert watched[pending_sha].indexed is False
    assert watched[pending_sha].sha == pending_sha
    assert watched["nonexistent-branch"].sha is None  # unresolvable


def test_watch_refs_logs_intent(
    index_server: DaemonServer,
    fake_repo: str,
    log_output: structlog.testing.LogCapture,
) -> None:
    """Add and remove each emit a correlated intent event."""
    handle_build_index(
        BuildIndexRequest(repo_path=fake_repo, refs=["main"]), index_server.watch_refs
    )
    handle_build_index(
        BuildIndexRequest(repo_path=fake_repo, refs=["main"], remove=True),
        index_server.watch_refs,
    )
    events = [e["event"] for e in log_output.entries]
    assert "watched_refs_added" in events
    assert "watched_refs_removed" in events
