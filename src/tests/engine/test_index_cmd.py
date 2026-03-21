"""Tests for /index command and footer index indicator."""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest
from pytest_mock import MockerFixture

from rbtr.engine.core import Engine
from rbtr.engine.indexing import _build_index
from rbtr.engine.types import TaskType
from rbtr.events import IndexCleared, IndexReady, TableOutput
from rbtr.models import BranchTarget
from tests.helpers import drain, output_texts

from .conftest import wait_for_index

# ── Helpers ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _mock_embeddings(mocker: MockerFixture) -> None:
    """Stub out the embedding step — no GGUF model needed in tests."""
    mocker.patch("rbtr.index.orchestrator._embed_missing")


@pytest.fixture
def indexed_engine(repo_engine: Engine) -> Engine:
    """Engine with a repo that has been indexed (feature branch off main)."""
    engine = repo_engine
    repo = engine.state.repo
    assert repo is not None
    repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))
    engine.state.review_target = BranchTarget(
        base_branch="main",
        head_branch="feature",
        base_commit="main",
        head_commit="feature",
        updated_at=repo.head.peel(pygit2.Commit).commit_time,
    )
    _build_index(engine)
    return engine


# ── /index status ────────────────────────────────────────────────────


def test_index_status_no_index(repo_engine: Engine) -> None:
    engine = repo_engine
    engine.run_task(TaskType.COMMAND, "/index")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("No index loaded" in t for t in texts)


def test_index_status_with_index(indexed_engine: Engine) -> None:
    engine = indexed_engine
    drain(engine.events)

    engine.run_task(TaskType.COMMAND, "/index status")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    tables = [e for e in drained_events if isinstance(e, TableOutput)]
    assert any("main" in t and "feature" in t for t in texts)
    assert any(t.title == "Chunks" for t in tables)
    assert any(t.title == "Edges" for t in tables)

    if engine.state.index is not None:
        engine.state.index.close()


def test_index_status_falls_back_to_base_ref(repo_engine: Engine) -> None:
    """When head ref has no data, /index status shows base ref data.

    Reproduces the real-world scenario: `build_index("main")`
    succeeds but `update_index("main", "feature")` hasn't run or
    failed.  Status should show the base data with a note.
    """
    engine = repo_engine

    # Manually build base index only (no update_index for head).
    from datetime import UTC, datetime

    from rbtr.index.orchestrator import build_index
    from rbtr.index.store import IndexStore

    store = IndexStore()
    build_index(engine.state.repo, "main", store)
    engine.state.index = store
    engine.state.review_target = BranchTarget(
        base_branch="main",
        head_branch="nonexistent-head",
        base_commit="main",
        head_commit="nonexistent-head",
        updated_at=datetime.now(tz=UTC),
    )
    drain(engine.events)

    engine.run_task(TaskType.COMMAND, "/index status")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    tables = [e for e in drained_events if isinstance(e, TableOutput)]

    # Should show base data, not empty tables.
    assert any(t.title == "Chunks" for t in tables)
    chunk_table = next(t for t in tables if t.title == "Chunks")
    total_row = chunk_table.rows[-1]
    assert int(total_row[1]) > 0, "Should show base chunks, not 0"

    # Should mention that head is not indexed.
    assert any("not indexed yet" in t for t in texts)

    store.close()


def test_index_status_default_subcommand(indexed_engine: Engine) -> None:
    """/index with no args is the same as /index status."""
    engine = indexed_engine
    drain(engine.events)

    engine.run_task(TaskType.COMMAND, "/index")
    drained_events = drain(engine.events)
    tables = [e for e in drained_events if isinstance(e, TableOutput)]
    assert any(t.title == "Chunks" for t in tables)

    if engine.state.index is not None:
        engine.state.index.close()


# ── /index clear ─────────────────────────────────────────────────────


def test_index_clear(indexed_engine: Engine) -> None:
    engine = indexed_engine
    drain(engine.events)

    assert engine.state.index is not None
    assert engine.state.index_ready is True

    engine.run_task(TaskType.COMMAND, "/index clear")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("cleared" in t.lower() for t in texts)
    assert engine.state.index is None
    assert engine.state.index_ready is False
    assert any(isinstance(e, IndexCleared) for e in drained_events)


def test_index_clear_no_index(repo_engine: Engine) -> None:
    engine = repo_engine
    engine.run_task(TaskType.COMMAND, "/index clear")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("No index file" in t for t in texts)


# ── /index rebuild ───────────────────────────────────────────────────


def test_index_rebuild(indexed_engine: Engine) -> None:
    engine = indexed_engine
    drain(engine.events)

    engine.run_task(TaskType.COMMAND, "/index rebuild")
    # Rebuild spawns a background thread — wait for completion.
    drained_events = drain(engine.events)
    drained_events.extend(wait_for_index(engine.events))
    texts = output_texts(drained_events)
    assert any("Rebuilding" in t for t in texts)
    assert any(isinstance(e, IndexReady) for e in drained_events)
    assert engine.state.index is not None

    if engine.state.index is not None:
        engine.state.index.close()


def test_index_rebuild_no_target(repo_engine: Engine) -> None:
    engine = repo_engine
    engine.run_task(TaskType.COMMAND, "/index rebuild")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("No review target" in t for t in texts)


# ── /index prune ─────────────────────────────────────────────────────


def test_index_prune_removes_orphans(indexed_engine: Engine) -> None:
    """Manually inserting unreferenced chunks, then /index prune removes them."""
    engine = indexed_engine
    drain(engine.events)

    assert engine.state.index is not None
    store = engine.state.index

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
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("1 orphan chunks" in t for t in texts)
    assert store.count_orphan_chunks() == 0

    store.close()


def test_index_prune_no_orphans(indexed_engine: Engine) -> None:
    engine = indexed_engine
    drain(engine.events)

    engine.run_task(TaskType.COMMAND, "/index prune")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("No orphans" in t for t in texts)

    if engine.state.index is not None:
        engine.state.index.close()


def test_index_prune_no_index(repo_engine: Engine) -> None:
    engine = repo_engine
    engine.run_task(TaskType.COMMAND, "/index prune")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("No index loaded" in t for t in texts)


# ── /index model ──────────────────────────────────────────────────────


def test_index_model_show_current(repo_engine: Engine) -> None:
    """/index model with no arg shows the current embedding model."""
    engine = repo_engine
    engine.run_task(TaskType.COMMAND, "/index model")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("Embedding model:" in t for t in texts)


def test_index_model_change(config_path: Path, indexed_engine: Engine) -> None:
    """/index model <id> persists the new model and clears embeddings."""
    engine = indexed_engine
    drain(engine.events)

    store = engine.state.index
    assert store is not None

    # Seed an embedding so we can verify it gets cleared.
    chunks = store.get_chunks("main")
    assert len(chunks) > 0
    store.update_embedding(chunks[0].id, [0.1, 0.2, 0.3])
    store.checkpoint()

    new_model = "test-org/test-repo/test-model.gguf"
    engine.run_task(TaskType.COMMAND, f"/index model {new_model}")
    drained_events = drain(engine.events)
    # Background re-embed task may also emit events — wait briefly.
    drained_events.extend(wait_for_index(engine.events, timeout=5.0))
    texts = output_texts(drained_events)

    assert any("→" in t and new_model in t for t in texts)
    assert any("Cleared" in t for t in texts)

    # Config should be updated.
    from rbtr.config import config as cfg

    assert cfg.index.embedding_model == new_model

    # run_index reopens the store — use the new one.
    new_store = engine.state.index
    assert new_store is not None

    # Embedding should be gone (cleared before re-index,
    # and _embed_missing is mocked so nothing re-embeds).
    refreshed = new_store.get_chunks("main")
    assert all(not c.embedding for c in refreshed)

    new_store.close()


def test_index_model_same_noop(repo_engine: Engine) -> None:
    """/index model with the current model is a no-op."""
    engine = repo_engine

    from rbtr.config import config as cfg

    current = cfg.index.embedding_model
    engine.run_task(TaskType.COMMAND, f"/index model {current}")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("Already using" in t for t in texts)


def test_index_model_no_index(config_path: Path, repo_engine: Engine) -> None:
    """/index model without an active index still persists the config change."""
    engine = repo_engine

    new_model = "org/repo/new-model.gguf"
    engine.run_task(TaskType.COMMAND, f"/index model {new_model}")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)

    assert any("→" in t and new_model in t for t in texts)
    assert any("No index loaded" in t for t in texts)

    from rbtr.config import config as cfg

    assert cfg.index.embedding_model == new_model


# ── /index search ─────────────────────────────────────────────────────


def test_index_search_shows_results(indexed_engine: Engine) -> None:
    """/index search returns a table of ranked results."""
    engine = indexed_engine
    drain(engine.events)

    engine.run_task(TaskType.COMMAND, "/index search hello")
    drained_events = drain(engine.events)
    tables = [e for e in drained_events if isinstance(e, TableOutput)]
    assert any("Search:" in t.title for t in tables)
    search_table = next(t for t in tables if "Search:" in t.title)
    assert len(search_table.rows) >= 1
    # First column is score, second is kind.
    assert search_table.columns[0].header == "Score"
    assert search_table.columns[1].header == "Kind"

    if engine.state.index is not None:
        engine.state.index.close()


def test_index_search_no_results(indexed_engine: Engine) -> None:
    """/index search with gibberish returns a 'no results' message."""
    engine = indexed_engine
    drain(engine.events)

    engine.run_task(TaskType.COMMAND, "/index search zzz_nonexistent_xyz_999")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("No results" in t for t in texts)

    if engine.state.index is not None:
        engine.state.index.close()


def test_index_search_no_query(indexed_engine: Engine) -> None:
    """/index search with no query shows usage."""
    engine = indexed_engine
    drain(engine.events)

    engine.run_task(TaskType.COMMAND, "/index search")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("Usage:" in t for t in texts)

    if engine.state.index is not None:
        engine.state.index.close()


def test_index_search_no_index(repo_engine: Engine) -> None:
    """/index search without an index tells the user."""
    engine = repo_engine
    engine.run_task(TaskType.COMMAND, "/index search hello")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("No index loaded" in t for t in texts)


# ── /index search-diag ───────────────────────────────────────────────


def test_index_search_diag_shows_breakdown(indexed_engine: Engine) -> None:
    """/index search-diag returns a table with all signal columns."""
    engine = indexed_engine
    drain(engine.events)

    engine.run_task(TaskType.COMMAND, "/index search-diag hello")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    tables = [e for e in drained_events if isinstance(e, TableOutput)]

    # Should print the query classification header.
    assert any("class=" in t for t in texts)
    assert any("weights:" in t for t in texts)

    assert any("Diagnostics:" in t.title for t in tables)
    diag_table = next(t for t in tables if "Diagnostics:" in t.title)
    headers = [c.header for c in diag_table.columns]
    assert "Lex" in headers
    assert "Sem" in headers
    assert "Name" in headers
    assert "Kind" in headers
    assert "File" in headers
    assert "Imp" in headers
    assert "Prox" in headers
    assert "Chunk" in headers
    assert len(diag_table.rows) >= 1

    if engine.state.index is not None:
        engine.state.index.close()


def test_index_search_diag_no_results(indexed_engine: Engine) -> None:
    """/index search-diag with gibberish returns 'no results'."""
    engine = indexed_engine
    drain(engine.events)

    engine.run_task(TaskType.COMMAND, "/index search-diag zzz_nonexistent_xyz_999")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("No results" in t for t in texts)

    if engine.state.index is not None:
        engine.state.index.close()


def test_index_search_diag_no_query(indexed_engine: Engine) -> None:
    """/index search-diag with no query shows usage."""
    engine = indexed_engine
    drain(engine.events)

    engine.run_task(TaskType.COMMAND, "/index search-diag")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("Usage:" in t for t in texts)

    if engine.state.index is not None:
        engine.state.index.close()


def test_index_search_diag_no_index(repo_engine: Engine) -> None:
    """/index search-diag without an index tells the user."""
    engine = repo_engine
    engine.run_task(TaskType.COMMAND, "/index search-diag hello")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("No index loaded" in t for t in texts)


# ── /index unknown ───────────────────────────────────────────────────


def test_index_unknown_subcommand(repo_engine: Engine) -> None:
    engine = repo_engine
    engine.run_task(TaskType.COMMAND, "/index foobar")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("Unknown subcommand" in t for t in texts)


# ── /help includes /index ────────────────────────────────────────────


def test_help_lists_index_command(repo_engine: Engine) -> None:
    engine = repo_engine
    engine.run_task(TaskType.COMMAND, "/help")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("/index" in t for t in texts)


# ── Footer format helper ─────────────────────────────────────────────


def test_format_count() -> None:
    from rbtr.tui.footer import _format_count

    assert _format_count(42) == "42"
    assert _format_count(1_200) == "1.2k"
    assert _format_count(1_500_000) == "1.5M"
    assert _format_count(999) == "999"
    assert _format_count(1_000) == "1.0k"
