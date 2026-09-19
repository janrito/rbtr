"""CLI error contract for a failed daemon start in `Index.cli_cmd`.

`start_daemon` raises `RbtrError` (not `RuntimeError`), so
`Index.cli_cmd` must catch that type for its inline fallback to run.
Which outcome the fallback gets depends on *why* the start failed:

- any ordinary failure -> fall back to a real inline build;
- the DuckDB write lock held elsewhere -> refuse, because the inline
  build would contend for that same lock.

The start-failure exit code itself is covered end-to-end by
`tests/daemon/test_daemon_cli_smoke.py::test_start_with_db_lock_held_exits_cleanly`.
"""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest
from pytest_mock import MockerFixture

from rbtr.cli import Index
from rbtr.domain.models import SnapshotRef
from rbtr.errors import RbtrError
from rbtr.git import normalise_repo_path
from rbtr.index.store import IndexStore
from rbtr.tests.conftest import run_cli


@pytest.fixture
def start_fails(mocker: MockerFixture) -> None:
    """`start_daemon` raises `RbtrError` — the failure under test."""
    mocker.patch("rbtr.cli.start_daemon", side_effect=RbtrError("boom"))


@pytest.fixture
def inline_store(store: IndexStore, mocker: MockerFixture) -> IndexStore:
    """Point the inline fallback's `from_config` at the shared
    in-memory `store`, so the fallback runs a real build and the
    test can assert the symbols were written."""
    mocker.patch("rbtr.cli.IndexStore.from_config", return_value=store)
    return store


def test_index_falls_back_to_inline_when_start_fails(
    git_repo: pygit2.Repository,
    isolated_db: Path,
    start_fails: None,
    inline_store: IndexStore,
) -> None:
    """A failed daemon start falls back to a real inline build.

    `git_repo` gives real source to parse; `isolated_db` leaves the
    real `try_daemon` to find no daemon, so the auto-start branch
    runs with no transport patched.  After the fallback, the
    in-memory index holds the repo's symbols.
    """
    repo_path = str(git_repo.workdir)
    Index(
        refs=["HEAD"],
        repo_path=repo_path,
        daemon=True,
        embed=False,
        allow_missing_plugins=False,
    ).cli_cmd()

    repo_id = inline_store.get_repo_id(normalise_repo_path(repo_path))
    assert repo_id is not None, "inline fallback did not register the repo"
    commits = inline_store.list_indexed_snapshots(repo_id)
    assert len(commits) == 1, "inline fallback did not index HEAD"
    head = SnapshotRef(repo_id=repo_id, snapshot_sha=commits[0][0])
    assert inline_store.chunk_counts_for_snapshot(at=head).total > 0, "no symbols extracted"


def test_index_refuses_inline_build_when_db_is_locked(
    repo_path: str,
    isolated_db: Path,
) -> None:
    """A locked DB makes `rbtr index` fail honestly, not fall back.

    DuckDB's write lock is process-level, so holding it here makes both
    the spawned `daemon serve` and any inline build fail to open the
    store.  The command must exit 1 naming the lock, and must not claim
    it is falling back to inline execution -- which cannot work here.

    Runs the real CLI against a real lock rather than patching
    `start_daemon`: the contention being pinned is between processes,
    so the sibling test's in-process mocks cannot express it.
    """
    store = IndexStore.from_config(writable=True)  # take the exclusive lock
    try:
        result = run_cli(["index", "--repo-path", repo_path])

        assert result.returncode == 1, result.stderr
        assert "locked by another process" in result.stderr
        assert "Falling back to inline execution" not in result.stderr
    finally:
        store.close()
        run_cli(["daemon", "stop"])
