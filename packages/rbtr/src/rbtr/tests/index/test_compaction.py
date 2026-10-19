"""Tests for index compaction — `Index.compact` and the gc path.

Compaction is physical (reclaiming disk), distinct from the logical GC
modes in `test_gc.py` (which rows drop). A rewrite only reclaims space
on disk, so these build a *file-backed* index (via the `isolated_db`
fixture) over a real repo (`fake_repo`) and churn it so a rewrite has
free space to reclaim. No mocking except the injected I/O failure.
"""

from __future__ import annotations

import multiprocessing
import os
import threading
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pytest
from pytest_cases import parametrize_with_cases
from pytest_mock import MockerFixture

from rbtr.config import config
from rbtr.daemon.handlers import handle_gc
from rbtr.daemon.messages import GcRequest, GcResponse
from rbtr.domain.models import GcMode, SnapshotRef
from rbtr.git import head_sha
from rbtr.index.index import Index

from .cases_gc_compaction import CompactionScenario
from .conftest import make_chunk, make_snap


@dataclass(frozen=True)
class ChurnedIndex:
    """A file-backed store bloated with reclaimable free space, plus the
    handles a gc or search needs against it."""

    index: Index
    repo_path: str
    ref: SnapshotRef
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
    # 800 chunks across 3 passes inflates the file past DuckDB's block
    # granularity; a rewrite then shrinks it. Reopen the store per pass
    # so each pass's stale row-versions are consolidated onto disk (as
    # across daemon restarts), accumulating the bloat a rewrite reclaims.
    for pass_no in range(3):
        index = Index.from_config(writable=True)
        repo = index.register(fake_repo)
        with index.session(repo) as ws:
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
            ws.insert_snapshots(snaps)
            ws.mark_indexed(snapshot_sha=head)
        index.close()
    index = Index.from_config(writable=True)
    yield ChurnedIndex(
        index=index,
        repo_path=fake_repo,
        ref=repo.at(head),
        query="retry backoff",
    )
    index.close()


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
    before_size = ci.index.disk_size_bytes()
    before_count = ci.index.chunk_counts_for_snapshot(at=ci.ref).total

    request = GcRequest(repo_path=ci.repo_path, mode=GcMode.WATCHED, compact=scenario.compact)
    handle_gc(request, ci.index, allow_compact=True)

    assert ci.index.chunk_counts_for_snapshot(at=ci.ref).total == before_count
    if scenario.expect_shrink:
        assert ci.index.disk_size_bytes() < before_size
    else:
        assert ci.index.disk_size_bytes() == before_size


def test_gc_reports_size_change_when_compacting(churned_index: ChurnedIndex) -> None:
    """A compacting gc reports the on-disk footprint before and after."""
    ci = churned_index
    response = handle_gc(
        GcRequest(repo_path=ci.repo_path, mode=GcMode.WATCHED, compact=True),
        ci.index,
        allow_compact=True,
    )
    assert response.size_before_bytes > 0
    # Block rounding can round the byte delta to zero at this scale, so the
    # reclaim is not strictly less; it is never larger.
    assert response.size_after_bytes <= response.size_before_bytes


def test_disk_size_bytes_reports_the_footprint(churned_index: ChurnedIndex) -> None:
    """`disk_size_bytes` counts the real files, so it is non-zero on disk."""
    assert churned_index.index.disk_size_bytes() > 0


def test_fts_index_survives_compaction(churned_index: ChurnedIndex) -> None:
    """Search still works after the rewrite, with no FTS rebuild.

    `COPY FROM DATABASE` carries the FTS index across; running a search
    is the point of this test, guarding against a regression where the
    rewrite drops it.
    """
    ci = churned_index
    request = GcRequest(repo_path=ci.repo_path, mode=GcMode.WATCHED, compact=True)
    handle_gc(request, ci.index, allow_compact=True)

    hits = ci.index.fulltext_matches(ci.query, within=[ci.ref], top_k=5)
    assert len(hits) > 0, "search returned nothing after compaction"


def test_handle_gc_does_not_compact_without_opt_in(churned_index: ChurnedIndex) -> None:
    """`handle_gc` rewrites only when the caller passes `allow_compact`.

    The flag guards callers that do not own the connection exclusively.
    """
    ci = churned_index
    before_size = ci.index.disk_size_bytes()
    request = GcRequest(repo_path=ci.repo_path, mode=GcMode.WATCHED, compact=True)
    handle_gc(request, ci.index)  # allow_compact defaults False
    assert ci.index.disk_size_bytes() == before_size


def test_gc_compaction_failure_is_non_fatal(
    churned_index: ChurnedIndex,
    mocker: MockerFixture,
) -> None:
    """A failed rewrite leaves the index intact and gc still succeeds.

    The logical deletes commit before compaction, so an I/O error while
    the copy is made must not lose data: the live file is unchanged, the
    copy is deleted, and gc returns its counts.
    """
    ci = churned_index
    before_size = ci.index.disk_size_bytes()
    before_count = ci.index.chunk_counts_for_snapshot(at=ci.ref).total
    projects = Path(ci.index.db_path or "").parent / "index"
    before_files = sorted(projects.iterdir())
    mocker.patch("rbtr.index.store.Store._attach", side_effect=OSError("disk full"))

    request = GcRequest(repo_path=ci.repo_path, mode=GcMode.WATCHED, compact=True)
    response = handle_gc(request, ci.index, allow_compact=True)

    assert isinstance(response, GcResponse)  # swallowed, not raised
    assert ci.index.disk_size_bytes() == before_size  # rewrite did not take effect
    assert ci.index.chunk_counts_for_snapshot(at=ci.ref).total == before_count
    assert sorted(projects.iterdir()) == before_files


def test_search_survives_concurrent_compaction(churned_index: ChurnedIndex) -> None:
    """Concurrent searches keep working while a gc compacts — no errors.

    Compaction switches the project to a fresh file and keeps the old one
    attached for `retire_delay_s`, so a reader is never cut off mid-query:
    it finishes on the old file and binds the new one on its next call.
    Two register-pattern read shapes run here (`fulltext_matches` and
    `name_matches`) so a read spanning a switch is covered by more than
    one query. Any error (notably a `CatalogException` from a detached
    file) fails the test.
    """
    ci = churned_index
    errors: list[Exception] = []
    stop = threading.Event()

    def reader() -> None:
        try:
            while not stop.is_set():
                assert (
                    len(
                        ci.index.fulltext_matches(
                            ci.query,
                            within=[ci.ref],
                            top_k=5,
                        )
                    )
                    > 0
                )
                assert not ci.index.name_matches(
                    "calculate_retry_backoff_1",
                    within=[ci.ref],
                ).is_empty()
        except Exception as exc:  # noqa: BLE001 - re-asserted on main thread
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    try:
        for _ in range(3):
            request = GcRequest(repo_path=ci.repo_path, mode=GcMode.WATCHED, compact=True)
            handle_gc(request, ci.index, allow_compact=True)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=5)

    assert not any(t.is_alive() for t in threads), "a reader hung"
    assert not errors, f"readers errored during compaction: {errors}"


def test_compaction_deletes_the_old_file_once_its_delay_passes(
    churned_index: ChurnedIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The replaced file is detached and deleted when `retire_due` finds it due."""
    ci = churned_index
    projects = Path(ci.index.db_path or "").parent / "index"
    (old,) = projects.glob("*.duckdb")
    monkeypatch.setattr(config, "retire_delay_s", 0.0)

    ci.index.compact(ci.index.repo(ci.repo_path))
    assert old.exists(), "a read may still be using it"
    ci.index.retire_due()

    (new,) = projects.glob("*.duckdb")
    assert new != old
    assert len(ci.index.fulltext_matches(ci.query, within=[ci.ref], top_k=5)) > 0


def _write_compact_and_crash(db_path: Path, repo_path: str) -> None:
    """Child-process body: commit a write, compact, commit another, then die.

    Module-level because a spawned process can only run an importable
    function. `os._exit` skips DuckDB's close-time checkpoint and the old
    file's retirement, as a crash or a SIGKILL would: the first write sits
    in the old file's WAL and the second only in the new file's.
    """
    index = Index(db_path, writable=True)
    registered = index.register(repo_path)
    with index.session(registered) as ws:
        ws.add_watched_refs(["before"])
    index.compact(registered)
    with index.session(registered) as ws:
        ws.add_watched_refs(["after"])
    os._exit(0)


def test_writes_around_a_compaction_survive_a_crash(tmp_path: Path) -> None:
    """A crash between a switch and the old file's deletion loses and repeats nothing.

    The write committed before the switch is in the copy, the one after it
    is in the new file's own WAL, and the old file, which nothing names
    any more, is deleted when the index is opened again.
    """
    db_path = tmp_path / "index.duckdb"
    child = multiprocessing.get_context("spawn").Process(
        target=_write_compact_and_crash, args=(db_path, "/repo")
    )
    child.start()
    child.join(timeout=60)
    assert child.exitcode == 0
    assert len(list((tmp_path / "index").glob("*.duckdb"))) == 2, "the crash left both files"

    index = Index(db_path, writable=True)
    try:
        assert index.watched_refs(index.repos()[0]) == ["after", "before"]
        assert len(list((tmp_path / "index").glob("*.duckdb"))) == 1
    finally:
        index.close()


def test_compact_in_memory_keeps_the_data() -> None:
    """An in-memory index compacts the same way, into a fresh in-memory file."""
    index = Index(writable=True)
    repo = index.register("/repo")
    with index.session(repo) as ws:
        ws.add_watched_refs(["main"])
    index.compact(repo)
    assert index.watched_refs(repo) == ["main"]
    index.close()
