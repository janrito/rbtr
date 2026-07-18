"""Tests for index compaction — `WriteSession.compact` and the gc path.

Compaction is physical (reclaiming disk), distinct from the logical GC
modes in `test_gc.py` (which rows drop). Those use an in-memory store,
which cannot be rewritten, so these build a *file-backed* store (via
the `isolated_db` fixture) over a real repo (`fake_repo`) and churn it
so a rewrite has free space to reclaim. No mocking except the injected
I/O failure.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pytest
from pytest_cases import parametrize_with_cases
from pytest_mock import MockerFixture

from rbtr.daemon.handlers import handle_gc
from rbtr.daemon.messages import GcMode, GcRequest, GcResponse
from rbtr.git import head_sha
from rbtr.index.store import IndexStore

from .cases_gc_compaction import CompactionScenario
from .conftest import make_chunk, make_snap


@dataclass(frozen=True)
class ChurnedIndex:
    """A file-backed store bloated with reclaimable free space, plus the
    handles a gc or search needs against it."""

    store: IndexStore
    repo_path: str
    repo_id: int
    commit_sha: str
    query: str


@pytest.fixture
def churned_index(fake_repo: str, isolated_db: Path) -> Generator[ChurnedIndex]:
    """A file-backed store over `fake_repo`, churned to leave free space.

    All churn is claimed by the repo's single HEAD commit, so a default
    gc drops nothing and only compaction can change the file size.
    Re-seeding the same chunks across several committed, checkpointed
    passes leaves stale row versions in the file — the bloat a
    re-indexed store accumulates.
    """
    head = head_sha(fake_repo)
    repo_id = 0
    # 800 chunks across 3 passes inflates the file past DuckDB's block
    # granularity; a rewrite then shrinks it. Reopen the store per pass
    # so each pass's stale row-versions are consolidated onto disk (as
    # across daemon restarts), accumulating the bloat a rewrite reclaims.
    for pass_no in range(3):
        store = IndexStore.from_config(writable=True)
        with store.session() as ws:
            if pass_no == 0:
                repo_id = ws.register_repo(fake_repo)
            snaps = []
            for i in range(800):
                content = (
                    f"def calculate_retry_backoff_{i}_v{pass_no}(attempt): return 2 ** attempt"
                )
                ws.add_chunk(
                    make_chunk(
                        f"chunk-{i}",
                        name=f"calculate_retry_backoff_{i}",
                        content=content,
                        blob=f"blob-{i}",
                        path=f"m{i}.py",
                    )
                )
                snaps.append(make_snap(head, f"m{i}.py", f"blob-{i}"))
            ws.insert_snapshots(snaps, repo_id=repo_id)
            ws.mark_indexed(repo_id, head)
        store.close()
    store = IndexStore.from_config(writable=True)
    yield ChurnedIndex(
        store=store,
        repo_path=fake_repo,
        repo_id=repo_id,
        commit_sha=head,
        query="retry backoff",
    )
    store.close()


@parametrize_with_cases("scenario", cases=".cases_gc_compaction")
def test_gc_compaction_respects_flag(
    churned_index: ChurnedIndex,
    scenario: CompactionScenario,
) -> None:
    """A real gc compacts by default and skips it under `--no-compact`.

    Either way the row count is unchanged — compaction rewrites the same
    data, it does not drop any.
    """
    ci = churned_index
    before_size = ci.store.data_size_bytes()
    before_count = ci.store.count_chunks(ci.commit_sha, ci.repo_id)

    request = GcRequest(repo_path=ci.repo_path, mode=GcMode.WATCHED, compact=scenario.compact)
    handle_gc(request, ci.store, allow_compact=True)

    assert ci.store.count_chunks(ci.commit_sha, ci.repo_id) == before_count
    if scenario.expect_shrink:
        assert ci.store.data_size_bytes() < before_size
    else:
        assert ci.store.data_size_bytes() == before_size


def test_gc_reports_size_change_when_compacting(churned_index: ChurnedIndex) -> None:
    """A compacting gc reports the on-disk footprint before and after."""
    ci = churned_index
    response = handle_gc(
        GcRequest(repo_path=ci.repo_path, mode=GcMode.WATCHED, compact=True),
        ci.store,
        allow_compact=True,
    )
    assert response.size_before_bytes > 0
    # Block rounding can round the byte delta to zero at this scale, so the
    # reclaim is not strictly less; it is never larger.
    assert response.size_after_bytes <= response.size_before_bytes


def test_disk_size_bytes_reports_the_footprint(churned_index: ChurnedIndex) -> None:
    """`disk_size_bytes` counts the real files, so it is non-zero on disk.

    Unlike `data_size_bytes` (in-file pages, zero until the WAL is
    consolidated), this reads the database file's actual size.
    """
    assert churned_index.store.disk_size_bytes() > 0


def test_fts_index_survives_compaction(churned_index: ChurnedIndex) -> None:
    """Search still works after the rewrite, with no FTS rebuild.

    `COPY FROM DATABASE` carries the FTS index across; running a search
    is the point of this test, guarding against a regression where the
    rewrite drops it.
    """
    ci = churned_index
    request = GcRequest(repo_path=ci.repo_path, mode=GcMode.WATCHED, compact=True)
    handle_gc(request, ci.store, allow_compact=True)

    hits = ci.store.match_fulltext(ci.commit_sha, ci.query, top_k=5, repo_id=ci.repo_id)
    assert hits, "search returned nothing after compaction"


def test_handle_gc_does_not_compact_without_opt_in(churned_index: ChurnedIndex) -> None:
    """`handle_gc` rewrites only when the caller passes `allow_compact`.

    The flag guards callers that do not own the connection exclusively.
    """
    ci = churned_index
    before_size = ci.store.data_size_bytes()
    request = GcRequest(repo_path=ci.repo_path, mode=GcMode.WATCHED, compact=True)
    handle_gc(request, ci.store)  # allow_compact defaults False
    assert ci.store.data_size_bytes() == before_size


def test_gc_compaction_failure_is_non_fatal(
    churned_index: ChurnedIndex,
    mocker: MockerFixture,
) -> None:
    """A failed rewrite leaves the index intact and gc still succeeds.

    The logical deletes commit before compaction, so an I/O error during
    the file swap must not lose data: the original file is unchanged, the
    temp copy is cleaned up, and gc returns its counts.
    """
    ci = churned_index
    before_size = ci.store.data_size_bytes()
    before_count = ci.store.count_chunks(ci.commit_sha, ci.repo_id)
    mocker.patch("rbtr.index.writer.os.replace", side_effect=OSError("disk full"))

    request = GcRequest(repo_path=ci.repo_path, mode=GcMode.WATCHED, compact=True)
    response = handle_gc(request, ci.store, allow_compact=True)

    assert isinstance(response, GcResponse)  # swallowed, not raised
    assert ci.store.data_size_bytes() == before_size  # rewrite did not take effect
    assert ci.store.count_chunks(ci.commit_sha, ci.repo_id) == before_count
    db_path = Path(ci.store.db_path or "")
    # RCU names each temp `.compact-<uuid>`; none may linger after a fail.
    assert not list(db_path.parent.glob(f"{db_path.name}.compact-*"))


def test_search_survives_concurrent_compaction(churned_index: ChurnedIndex) -> None:
    """Concurrent searches keep working while a gc compacts — no errors.

    Compaction publishes a fresh connection (RCU) instead of closing the
    live one, so a reader is never cut off mid-query: it finishes on the
    old connection and rebinds on its next call. Two register-pattern
    read shapes run here (`match_fulltext` and `match_by_name`, both via
    `_view`) so the torn-cursor path — a read spanning a swap — is
    covered by more than one query. Any error (notably
    `IndexNotBuiltError`/`CatalogException` from a torn cursor) fails the
    test.
    """
    ci = churned_index
    errors: list[Exception] = []
    stop = threading.Event()

    def reader() -> None:
        try:
            while not stop.is_set():
                assert ci.store.match_fulltext(ci.commit_sha, ci.query, top_k=5, repo_id=ci.repo_id)
                assert ci.store.match_by_name(
                    ci.commit_sha, "calculate_retry_backoff_1", repo_id=ci.repo_id
                )
        except Exception as exc:  # noqa: BLE001 - re-asserted on main thread
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    try:
        for _ in range(3):
            request = GcRequest(repo_path=ci.repo_path, mode=GcMode.WATCHED, compact=True)
            handle_gc(request, ci.store, allow_compact=True)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=5)

    assert not any(t.is_alive() for t in threads), "a reader hung"
    assert not errors, f"readers errored during compaction: {errors}"


def test_compact_in_memory_is_noop() -> None:
    """An in-memory store has nothing on disk; compact is a silent no-op."""
    store = IndexStore(writable=True)
    with store.session() as ws:
        ws.compact()  # must not raise
    store.close()


def test_compact_requires_a_write_session(tmp_path: Path) -> None:
    """Compaction is a write, so a read-only store cannot reach it."""
    db = tmp_path / "index.duckdb"
    IndexStore(db, writable=True).close()  # create the file
    store = IndexStore(db, writable=False)
    with pytest.raises(RuntimeError, match="writable"):
        store.session()
    store.close()
