"""Tests for /index command and footer index indicator."""

from __future__ import annotations

import queue
import tempfile

import pygit2
import pytest

from rbtr.engine import Engine, EngineState
from rbtr.engine.indexing import _build_index
from rbtr.engine.types import TaskType
from rbtr.events import (
    Event,
    IndexCleared,
    IndexReady,
    TableOutput,
)
from rbtr.models import BranchTarget

from .conftest import drain, make_repo_with_file, output_texts, wait_for_index

# ── Helpers ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _mock_embeddings(mocker):
    """Stub out the embedding step — no GGUF model needed in tests."""
    mocker.patch("rbtr.index.orchestrator._embed_missing")


def _make_engine(
    repo: pygit2.Repository,
) -> tuple[Engine, queue.Queue[Event], EngineState]:
    session = EngineState(repo=repo, owner="o", repo_name="r")
    events: queue.Queue[Event] = queue.Queue()
    engine = Engine(session, events)
    return engine, events, session


def _index_repo(engine: Engine, session: EngineState, repo: pygit2.Repository) -> None:
    """Set a review target and run indexing."""
    repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))
    session.review_target = BranchTarget(
        base_branch="main",
        head_branch="feature",
        updated_at=repo.head.peel(pygit2.Commit).commit_time,
    )
    _build_index(engine)


# ── /index status ────────────────────────────────────────────────────


def test_index_status_no_index() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo_with_file(tmp)
        engine, events, _ = _make_engine(repo)
        engine.run_task(TaskType.COMMAND, "/index")
        evts = drain(events)
        texts = output_texts(evts)
        assert any("No index loaded" in t for t in texts)


def test_index_status_with_index() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo_with_file(tmp)
        engine, events, session = _make_engine(repo)
        _index_repo(engine, session, repo)
        drain(events)

        engine.run_task(TaskType.COMMAND, "/index status")
        evts = drain(events)
        texts = output_texts(evts)
        tables = [e for e in evts if isinstance(e, TableOutput)]
        assert any("main" in t and "feature" in t for t in texts)
        assert any(t.title == "Chunks" for t in tables)
        assert any(t.title == "Edges" for t in tables)

        if session.index is not None:
            session.index.close()


def test_index_status_falls_back_to_base_ref() -> None:
    """When head ref has no data, /index status shows base ref data.

    Reproduces the real-world scenario: ``build_index("main")``
    succeeds but ``update_index("main", "feature")`` hasn't run or
    failed.  Status should show the base data with a note.
    """
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo_with_file(tmp)
        engine, events, session = _make_engine(repo)

        # Manually build base index only (no update_index for head).
        from datetime import UTC, datetime

        from rbtr.index.orchestrator import build_index
        from rbtr.index.store import IndexStore

        store = IndexStore()
        build_index(repo, "main", store)
        session.index = store
        session.review_target = BranchTarget(
            base_branch="main",
            head_branch="nonexistent-head",
            updated_at=datetime.now(tz=UTC),
        )
        drain(events)

        engine.run_task(TaskType.COMMAND, "/index status")
        evts = drain(events)
        texts = output_texts(evts)
        tables = [e for e in evts if isinstance(e, TableOutput)]

        # Should show base data, not empty tables.
        assert any(t.title == "Chunks" for t in tables)
        chunk_table = next(t for t in tables if t.title == "Chunks")
        total_row = chunk_table.rows[-1]
        assert int(total_row[1]) > 0, "Should show base chunks, not 0"

        # Should mention that head is not indexed.
        assert any("not indexed yet" in t for t in texts)

        store.close()


def test_index_status_default_subcommand() -> None:
    """/index with no args is the same as /index status."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo_with_file(tmp)
        engine, events, session = _make_engine(repo)
        _index_repo(engine, session, repo)
        drain(events)

        engine.run_task(TaskType.COMMAND, "/index")
        evts = drain(events)
        tables = [e for e in evts if isinstance(e, TableOutput)]
        assert any(t.title == "Chunks" for t in tables)

        if session.index is not None:
            session.index.close()


# ── /index clear ─────────────────────────────────────────────────────


def test_index_clear() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo_with_file(tmp)
        engine, events, session = _make_engine(repo)
        _index_repo(engine, session, repo)
        drain(events)

        assert session.index is not None
        assert session.index_ready is True

        engine.run_task(TaskType.COMMAND, "/index clear")
        evts = drain(events)
        texts = output_texts(evts)
        assert any("cleared" in t.lower() for t in texts)
        assert session.index is None
        assert session.index_ready is False
        assert any(isinstance(e, IndexCleared) for e in evts)


def test_index_clear_no_index() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo_with_file(tmp)
        engine, events, _ = _make_engine(repo)
        engine.run_task(TaskType.COMMAND, "/index clear")
        evts = drain(events)
        texts = output_texts(evts)
        assert any("No index file" in t for t in texts)


# ── /index rebuild ───────────────────────────────────────────────────


def test_index_rebuild() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo_with_file(tmp)
        engine, events, session = _make_engine(repo)
        _index_repo(engine, session, repo)
        drain(events)

        engine.run_task(TaskType.COMMAND, "/index rebuild")
        # Rebuild spawns a background thread — wait for completion.
        evts = drain(events)
        evts.extend(wait_for_index(events))
        texts = output_texts(evts)
        assert any("Rebuilding" in t for t in texts)
        assert any(isinstance(e, IndexReady) for e in evts)
        assert session.index is not None

        if session.index is not None:
            session.index.close()


def test_index_rebuild_no_target() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo_with_file(tmp)
        engine, events, _ = _make_engine(repo)
        engine.run_task(TaskType.COMMAND, "/index rebuild")
        evts = drain(events)
        texts = output_texts(evts)
        assert any("No review target" in t for t in texts)


# ── /index prune ─────────────────────────────────────────────────────


def test_index_prune_removes_orphans() -> None:
    """Manually inserting unreferenced chunks, then /index prune removes them."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo_with_file(tmp)
        engine, events, session = _make_engine(repo)
        _index_repo(engine, session, repo)
        drain(events)

        assert session.index is not None
        store = session.index

        # Insert an orphan chunk (not referenced by any snapshot).
        from rbtr.index.models import Chunk, ChunkKind

        orphan = Chunk(
            id="orphan_1",
            blob_sha="blob_orphan",
            file_path="orphan.py",
            kind=ChunkKind.FUNCTION,
            name="orphan_func",
            content="def orphan_func(): pass",
            line_start=1,
            line_end=1,
        )
        store.insert_chunks([orphan])
        assert store.count_orphan_chunks() == 1

        engine.run_task(TaskType.COMMAND, "/index prune")
        evts = drain(events)
        texts = output_texts(evts)
        assert any("1 orphan chunks" in t for t in texts)
        assert store.count_orphan_chunks() == 0

        store.close()


def test_index_prune_no_orphans() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo_with_file(tmp)
        engine, events, session = _make_engine(repo)
        _index_repo(engine, session, repo)
        drain(events)

        engine.run_task(TaskType.COMMAND, "/index prune")
        evts = drain(events)
        texts = output_texts(evts)
        assert any("No orphans" in t for t in texts)

        if session.index is not None:
            session.index.close()


def test_index_prune_no_index() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo_with_file(tmp)
        engine, events, _ = _make_engine(repo)
        engine.run_task(TaskType.COMMAND, "/index prune")
        evts = drain(events)
        texts = output_texts(evts)
        assert any("No index loaded" in t for t in texts)


# ── /index model ──────────────────────────────────────────────────────


def test_index_model_show_current() -> None:
    """/index model with no arg shows the current embedding model."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo_with_file(tmp)
        engine, events, _ = _make_engine(repo)
        engine.run_task(TaskType.COMMAND, "/index model")
        evts = drain(events)
        texts = output_texts(evts)
        assert any("Embedding model:" in t for t in texts)


def test_index_model_change(config_path) -> None:
    """/index model <id> persists the new model and clears embeddings."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo_with_file(tmp)
        engine, events, session = _make_engine(repo)
        _index_repo(engine, session, repo)
        drain(events)

        store = session.index
        assert store is not None

        # Seed an embedding so we can verify it gets cleared.
        chunks = store.get_chunks("main")
        assert len(chunks) > 0
        store.update_embedding(chunks[0].id, [0.1, 0.2, 0.3])
        store.checkpoint()

        new_model = "test-org/test-repo/test-model.gguf"
        engine.run_task(TaskType.COMMAND, f"/index model {new_model}")
        evts = drain(events)
        # Background re-embed task may also emit events — wait briefly.
        evts.extend(wait_for_index(events, timeout=5.0))
        texts = output_texts(evts)

        assert any("→" in t and new_model in t for t in texts)
        assert any("Cleared" in t for t in texts)

        # Config should be updated.
        from rbtr.config import config as cfg

        assert cfg.index.embedding_model == new_model

        # run_index reopens the store — use the new one.
        new_store = session.index
        assert new_store is not None

        # Embedding should be gone (cleared before re-index,
        # and _embed_missing is mocked so nothing re-embeds).
        refreshed = new_store.get_chunks("main")
        assert all(not c.embedding for c in refreshed)

        new_store.close()


def test_index_model_same_noop() -> None:
    """/index model with the current model is a no-op."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo_with_file(tmp)
        engine, events, _ = _make_engine(repo)

        from rbtr.config import config as cfg

        current = cfg.index.embedding_model
        engine.run_task(TaskType.COMMAND, f"/index model {current}")
        evts = drain(events)
        texts = output_texts(evts)
        assert any("Already using" in t for t in texts)


def test_index_model_no_index(config_path) -> None:
    """/index model without an active index still persists the config change."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo_with_file(tmp)
        engine, events, _ = _make_engine(repo)

        new_model = "org/repo/new-model.gguf"
        engine.run_task(TaskType.COMMAND, f"/index model {new_model}")
        evts = drain(events)
        texts = output_texts(evts)

        assert any("→" in t and new_model in t for t in texts)
        assert any("No index loaded" in t for t in texts)

        from rbtr.config import config as cfg

        assert cfg.index.embedding_model == new_model


# ── /index unknown ───────────────────────────────────────────────────


def test_index_unknown_subcommand() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo_with_file(tmp)
        engine, events, _ = _make_engine(repo)
        engine.run_task(TaskType.COMMAND, "/index foobar")
        evts = drain(events)
        texts = output_texts(evts)
        assert any("Unknown subcommand" in t for t in texts)


# ── /help includes /index ────────────────────────────────────────────


def test_help_lists_index_command() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo_with_file(tmp)
        engine, events, _ = _make_engine(repo)
        engine.run_task(TaskType.COMMAND, "/help")
        evts = drain(events)
        texts = output_texts(evts)
        assert any("/index" in t for t in texts)


# ── Footer format helper ─────────────────────────────────────────────


def test_format_count() -> None:
    from rbtr.tui import _format_count

    assert _format_count(42) == "42"
    assert _format_count(1_200) == "1.2k"
    assert _format_count(1_500_000) == "1.5M"
    assert _format_count(999) == "999"
    assert _format_count(1_000) == "1.0k"
