"""Behaviour of `handle_forget` — the daemon's forget-repo handler.

Forget is metadata-only: it removes a repo's references and the `repos`
row, leaving chunk reclamation to GC. These tests drive the handler
directly against a real in-memory store (no daemon, no patches).
"""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest

from rbtr.daemon.handlers import handle_forget
from rbtr.daemon.messages import ForgetRequest
from rbtr.errors import RbtrError
from rbtr.git import normalise_repo_path
from rbtr.index.store import IndexStore


@pytest.mark.parametrize("watched", [[], ["HEAD"]])
def test_forget_repo_when_nothing_beyond_head_watched(
    store: IndexStore, watched: list[str]
) -> None:
    """A repo is forgotten when nothing beyond HEAD is watched — whether the
    watch set is exactly HEAD or empty (an inline `--no-daemon` index)."""
    with store.session() as ws:
        ws.register_repo("/repo")
        if watched:
            ws.add_watched_refs(1, watched)

    resp = handle_forget(ForgetRequest(repo_path="/repo"), store)

    assert resp.forgotten == ["/repo"]
    assert store.get_repo_id("/repo") is None


def test_forget_refuses_repo_watching_extra_refs(store: IndexStore) -> None:
    """A repo still watching refs beyond HEAD is *not* forgotten — the
    caller must trim those refs first."""
    with store.session() as ws:
        ws.register_repo("/repo")
        ws.add_watched_refs(1, ["HEAD", "main"])

    with pytest.raises(RbtrError, match="HEAD"):
        handle_forget(ForgetRequest(repo_path="/repo"), store)

    assert store.get_repo_id("/repo") == 1  # untouched


def test_forget_stale_forgets_only_vanished_repos(store: IndexStore, tmp_path: Path) -> None:
    """`stale=True` forgets repos whose path no longer resolves and leaves
    live repos alone."""
    pygit2.init_repository(str(tmp_path / "live"), bare=False, initial_head="main")
    live = normalise_repo_path(str(tmp_path / "live"))
    gone = str(tmp_path / "gone")  # never created on disk
    with store.session() as ws:
        ws.register_repo(live)
        ws.register_repo(gone)

    resp = handle_forget(ForgetRequest(stale=True), store)

    assert resp.forgotten == [gone]
    assert store.get_repo_id(gone) is None
    assert store.get_repo_id(live) is not None  # live repo kept


def test_forget_stale_dry_run_reports_without_deleting(store: IndexStore, tmp_path: Path) -> None:
    """A dry run names the repos it *would* forget but deletes nothing."""
    gone = str(tmp_path / "gone")
    with store.session() as ws:
        ws.register_repo(gone)

    resp = handle_forget(ForgetRequest(stale=True, dry_run=True), store)

    assert resp.forgotten == [gone]
    assert resp.dry_run is True
    assert store.get_repo_id(gone) is not None  # still present
