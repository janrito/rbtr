"""Tests for startup recovery.

Recovery sets `DaemonServer._wake` so the DB-polling worker
picks up un-embedded commits on startup.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from rbtr.daemon.server import DaemonServer
from rbtr.index.models import Snapshot
from rbtr.index.store import IndexStore

from ..index.conftest import make_chunk

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def recovery_store_unembedded() -> Generator[IndexStore]:
    """Store with one indexed but unembedded commit."""
    store = IndexStore(writable=True)
    with store.session() as ws:
        repo_id = ws.register_repo("/test/repo")
        ws.add_chunk(
            make_chunk("chunk1", name="foo", path="test.py", blob="blob1", repo_id=repo_id)
        )
        ws.insert_snapshots(
            [Snapshot(commit_sha="sha1", file_path="test.py", blob_sha="blob1")],
            repo_id=repo_id,
        )
        ws.mark_indexed(repo_id, "sha1")
    yield store
    store.close()


@pytest.fixture
def recovery_store_fully_embedded() -> Generator[IndexStore]:
    """Store with one indexed and fully embedded commit."""
    store = IndexStore(writable=True)
    with store.session() as ws:
        repo_id = ws.register_repo("/test/repo")
        ws.add_chunk(
            make_chunk("chunk1", name="foo", path="test.py", blob="blob1", repo_id=repo_id)
        )
        ws.insert_snapshots(
            [Snapshot(commit_sha="sha1", file_path="test.py", blob_sha="blob1")],
            repo_id=repo_id,
        )
        ws.update_embeddings(["chunk1"], [[0.1, 0.2, 0.3]], repo_id=repo_id)
        ws.mark_indexed(repo_id, "sha1")
    yield store
    store.close()


# ── Startup recovery ────────────────────────────────────────────────


def test_startup_recovery_enqueues_embed_jobs(
    recovery_store_unembedded: IndexStore,
    runtime_dir: Path,
) -> None:
    """Unembedded commit → wake event set."""
    server = DaemonServer(
        runtime_dir,
        store=recovery_store_unembedded,
        idle_poll_interval=60.0,
        busy_poll_interval=60.0,
    )
    assert server._wake.is_set()


def test_startup_recovery_skips_fully_embedded(
    recovery_store_fully_embedded: IndexStore,
    runtime_dir: Path,
) -> None:
    """Fully embedded commit → wake event not set."""
    server = DaemonServer(
        runtime_dir,
        store=recovery_store_fully_embedded,
        idle_poll_interval=60.0,
        busy_poll_interval=60.0,
    )
    assert not server._wake.is_set()
