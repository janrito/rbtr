"""Handler tests for `changed_symbols`.

Symbol-level classification is covered exhaustively by the index
suite (`test_diff_symbols.py`); these tests cover the handler's
own responsibilities: mapping the diff to labelled `ChangedSymbol`
items, and rejecting an unindexed ref with a clear error.
"""

from __future__ import annotations

import pytest

from rbtr.daemon.handlers import handle_changed_symbols
from rbtr.daemon.messages import ChangedSymbolsRequest
from rbtr.errors import RbtrError
from rbtr.index.models import ChangeKind
from rbtr.index.store import IndexStore


def test_changed_symbols_labels(
    seeded_store: IndexStore, fake_repo: str, daemon_commit: str, changed_head: str
) -> None:
    """A modified function and a new function surface with the right labels."""
    resp = handle_changed_symbols(
        ChangedSymbolsRequest(repo_path=fake_repo, base=daemon_commit, head=changed_head),
        seeded_store,
    )

    labelled = {(item.chunk.name, item.change) for item in resp.changes}
    assert labelled == {("load_config", ChangeKind.MODIFIED), ("helper", ChangeKind.ADDED)}


def test_changed_symbols_not_indexed(
    seeded_store: IndexStore, fake_repo: str, daemon_commit: str
) -> None:
    """An unindexed ref is a clear error, not an empty diff."""
    unindexed = "0" * 40

    with pytest.raises(RbtrError, match="not indexed"):
        handle_changed_symbols(
            ChangedSymbolsRequest(repo_path=fake_repo, base=daemon_commit, head=unindexed),
            seeded_store,
        )
