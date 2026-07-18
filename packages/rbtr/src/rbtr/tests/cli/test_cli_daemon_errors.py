"""CLI error-contract test: `start_daemon` raises `RbtrError` (not
`RuntimeError`), so `Index.cli_cmd` must catch that type for its
inline fallback to run.

The start-failure exit code itself is covered end-to-end by
`tests/daemon/test_daemon_cli_smoke.py::test_start_with_db_lock_held_exits_cleanly`.
"""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest
from pytest_mock import MockerFixture

from rbtr.cli import Index
from rbtr.errors import RbtrError
from rbtr.git import normalise_repo_path
from rbtr.index.store import IndexStore


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
        remove=False,
        remove_stale_refs=False,
        remove_stale_repos=False,
        daemon=True,
        embed=False,
        allow_missing_plugins=False,
    ).cli_cmd()

    repo_id = inline_store.get_repo_id(normalise_repo_path(repo_path))
    assert repo_id is not None, "inline fallback did not register the repo"
    commits = inline_store.list_indexed_commits(repo_id)
    assert len(commits) == 1, "inline fallback did not index HEAD"
    assert inline_store.count_chunks(commits[0][0], repo_id) > 0, "no symbols extracted"
