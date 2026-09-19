"""Behaviour of `handle_forget` — the daemon's forget-repo handler.

Forget is metadata-only: it removes a repo's references and the `repos`
row, leaving chunk reclamation to GC. Repos whose checkout is gone are
`forget_stale_repos`'s business, tested with it. These tests drive the
handler directly against a real in-memory store (no daemon, no patches).
"""

from __future__ import annotations

import pytest

from rbtr.daemon.handlers import handle_forget
from rbtr.daemon.messages import ForgetRequest
from rbtr.errors import RbtrError
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
