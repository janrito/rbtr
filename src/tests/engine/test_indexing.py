"""Tests for engine/indexing.py — background indexing on /review."""

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
    IndexProgress,
    IndexReady,
    IndexStarted,
    Output,
)
from rbtr.models import BranchTarget

from .conftest import drain, make_repo_with_file, wait_for_index


@pytest.fixture(autouse=True)
def _mock_embeddings(mocker):
    """Stub out the embedding step — no GGUF model needed in tests."""
    mocker.patch("rbtr.index.orchestrator._embed_missing")


# ── Helpers ──────────────────────────────────────────────────────────


def _make_engine(
    repo: pygit2.Repository,
) -> tuple[Engine, queue.Queue[Event], EngineState]:
    session = EngineState(repo=repo, owner="o", repo_name="r")
    events: queue.Queue[Event] = queue.Queue()
    engine = Engine(session, events)
    return engine, events, session


# ── run_index ────────────────────────────────────────────────────────


def test_index_emits_started_progress_ready() -> None:
    """run_index emits IndexStarted, IndexProgress, and IndexReady."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo_with_file(tmp)
        repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))

        engine, events, session = _make_engine(repo)
        session.review_target = BranchTarget(
            base_branch="main",
            head_branch="feature",
            updated_at=repo.head.peel(pygit2.Commit).commit_time,
        )

        _build_index(engine)
        evts = drain(events)

        started = [e for e in evts if isinstance(e, IndexStarted)]
        assert len(started) == 1
        assert started[0].total_files >= 1

        progress = [e for e in evts if isinstance(e, IndexProgress)]
        assert len(progress) >= 1
        phases = {e.phase for e in progress}
        assert "parsing" in phases

        ready = [e for e in evts if isinstance(e, IndexReady)]
        assert len(ready) == 1
        assert ready[0].chunk_count >= 1

        # Store is attached to session.
        assert session.index is not None
        session.index.close()


def test_index_sets_store_on_session() -> None:
    """After indexing, session.index is an IndexStore."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo_with_file(tmp)
        repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))

        engine, events, session = _make_engine(repo)
        session.review_target = BranchTarget(
            base_branch="main",
            head_branch="feature",
            updated_at=repo.head.peel(pygit2.Commit).commit_time,
        )

        _build_index(engine)
        drain(events)

        assert session.index is not None
        # Should be able to query chunks.
        chunks = session.index.get_chunks("feature")
        assert len(chunks) >= 1
        session.index.close()


def test_index_skipped_when_disabled(mocker) -> None:
    """Indexing is skipped when config.index.enabled is False."""
    mocker.patch("rbtr.engine.indexing.config.index.enabled", False)

    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo_with_file(tmp)
        repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))

        engine, events, session = _make_engine(repo)
        session.review_target = BranchTarget(
            base_branch="main",
            head_branch="feature",
            updated_at=repo.head.peel(pygit2.Commit).commit_time,
        )

        _build_index(engine)
        evts = drain(events)

        assert session.index is None
        assert not any(isinstance(e, IndexStarted) for e in evts)


def test_index_skipped_without_review_target() -> None:
    """No-op when there's no review target."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo_with_file(tmp)
        engine, events, session = _make_engine(repo)

        _build_index(engine)
        evts = drain(events)
        assert not evts
        assert session.index is None


_swift_grammar_available = False
try:
    import tree_sitter_swift as _ts_swift  # noqa: F401

    _swift_grammar_available = True
except ImportError:
    pass


@pytest.mark.skipif(_swift_grammar_available, reason="Swift grammar is installed")
def test_index_warns_missing_grammars() -> None:
    """Indexing warns about languages with no grammar installed."""
    with tempfile.TemporaryDirectory() as tmp:
        # Use a .swift file — swift has a registration but likely no grammar installed.
        repo = make_repo_with_file(tmp, filename="app.swift", content="let x = 1\n")
        repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))

        engine, events, session = _make_engine(repo)
        session.review_target = BranchTarget(
            base_branch="main",
            head_branch="feature",
            updated_at=repo.head.peel(pygit2.Commit).commit_time,
        )

        _build_index(engine)
        evts = drain(events)

        warnings = [e for e in evts if isinstance(e, Output) and "Missing grammars" in e.text]
        assert len(warnings) == 1
        assert "swift" in warnings[0].text

        if session.index is not None:
            session.index.close()


def test_review_branch_triggers_indexing() -> None:
    """End-to-end: /review <branch> triggers indexing and emits index events."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo_with_file(tmp, content="def greet():\n    pass\n")
        repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))

        session = EngineState(repo=repo, owner="o", repo_name="r", gh=None)
        events: queue.Queue[Event] = queue.Queue()
        engine = Engine(session, events)
        engine.run_task(TaskType.COMMAND, "/review feature")

        # Indexing now runs in a background thread — wait for it.
        evts = drain(events)
        evts.extend(wait_for_index(events))

        assert any(isinstance(e, IndexStarted) for e in evts)
        assert any(isinstance(e, IndexReady) for e in evts)

        if session.index is not None:
            session.index.close()


def test_review_two_args_triggers_indexing() -> None:
    """/review base target triggers indexing with correct base."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo_with_file(tmp)
        repo.branches.local.create("develop", repo.head.peel(pygit2.Commit))
        repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))

        session = EngineState(repo=repo, owner="o", repo_name="r", gh=None)
        events: queue.Queue[Event] = queue.Queue()
        engine = Engine(session, events)
        engine.run_task(TaskType.COMMAND, "/review develop feature")

        # Indexing now runs in a background thread — wait for it.
        evts = drain(events)
        evts.extend(wait_for_index(events))

        assert isinstance(session.review_target, BranchTarget)
        assert session.review_target.base_branch == "develop"

        assert any(isinstance(e, IndexStarted) for e in evts)
        assert any(isinstance(e, IndexReady) for e in evts)

        if session.index is not None:
            session.index.close()
