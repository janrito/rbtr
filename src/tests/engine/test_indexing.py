"""Tests for engine/indexing.py — background indexing on /review."""

from __future__ import annotations

import queue
import tempfile

import pygit2
import pytest

from rbtr.engine import Engine, Session
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


@pytest.fixture(autouse=True)
def _mock_embeddings(mocker):
    """Stub out the embedding step — no GGUF model needed in tests."""
    mocker.patch("rbtr.index.orchestrator._embed_missing")


# ── Helpers ──────────────────────────────────────────────────────────


def _make_engine(
    repo: pygit2.Repository,
) -> tuple[Engine, queue.Queue[Event], Session]:
    session = Session(repo=repo, owner="o", repo_name="r")
    events: queue.Queue[Event] = queue.Queue()
    engine = Engine(session, events)
    return engine, events, session


def _drain(events: queue.Queue[Event]) -> list[Event]:
    result: list[Event] = []
    while True:
        try:
            result.append(events.get_nowait())
        except queue.Empty:
            break
    return result


def _wait_for_index(events: queue.Queue[Event], timeout: float = 30.0) -> list[Event]:
    """Collect events until IndexReady or Output (index done/failed)."""
    collected: list[Event] = []
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            evt = events.get(timeout=0.1)
        except queue.Empty:
            continue
        collected.append(evt)
        if isinstance(evt, (IndexReady, Output)):
            # Drain any remaining events queued at the same time.
            collected.extend(_drain(events))
            break
    return collected


def _make_repo_with_file(
    tmp: str,
    filename: str = "hello.py",
    content: str = "def greet():\n    pass\n",
) -> pygit2.Repository:
    """Create a repo with main branch and one file."""
    repo = pygit2.init_repository(tmp)
    sig = pygit2.Signature("Test", "test@test.com")

    # Create a blob and tree with a file.
    blob_id = repo.create_blob(content.encode())
    tb = repo.TreeBuilder()
    tb.insert(filename, blob_id, pygit2.GIT_FILEMODE_BLOB)
    tree_id = tb.write()

    repo.create_commit("refs/heads/main", sig, sig, "init", tree_id, [])
    repo.set_head("refs/heads/main")
    return repo


# ── run_index ────────────────────────────────────────────────────────


def test_index_emits_started_progress_ready() -> None:
    """run_index emits IndexStarted, IndexProgress, and IndexReady."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo_with_file(tmp)
        repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))

        engine, events, session = _make_engine(repo)
        session.review_target = BranchTarget(
            base_branch="main",
            head_branch="feature",
            updated_at=repo.head.peel(pygit2.Commit).commit_time,
        )

        _build_index(engine)
        evts = _drain(events)

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
        repo = _make_repo_with_file(tmp)
        repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))

        engine, events, session = _make_engine(repo)
        session.review_target = BranchTarget(
            base_branch="main",
            head_branch="feature",
            updated_at=repo.head.peel(pygit2.Commit).commit_time,
        )

        _build_index(engine)
        _drain(events)

        assert session.index is not None
        # Should be able to query chunks.
        chunks = session.index.get_chunks("feature")
        assert len(chunks) >= 1
        session.index.close()


def test_index_skipped_when_disabled(mocker) -> None:
    """Indexing is skipped when config.index.enabled is False."""
    mocker.patch("rbtr.engine.indexing.config.index.enabled", False)

    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo_with_file(tmp)
        repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))

        engine, events, session = _make_engine(repo)
        session.review_target = BranchTarget(
            base_branch="main",
            head_branch="feature",
            updated_at=repo.head.peel(pygit2.Commit).commit_time,
        )

        _build_index(engine)
        evts = _drain(events)

        assert session.index is None
        assert not any(isinstance(e, IndexStarted) for e in evts)


def test_index_skipped_without_review_target() -> None:
    """No-op when there's no review target."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo_with_file(tmp)
        engine, events, session = _make_engine(repo)

        _build_index(engine)
        evts = _drain(events)
        assert not evts
        assert session.index is None


def test_index_warns_missing_grammars() -> None:
    """Indexing warns about languages with no grammar installed."""
    with tempfile.TemporaryDirectory() as tmp:
        # Use a .swift file — swift has a registration but likely no grammar installed.
        repo = _make_repo_with_file(tmp, filename="app.swift", content="let x = 1\n")
        repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))

        engine, events, session = _make_engine(repo)
        session.review_target = BranchTarget(
            base_branch="main",
            head_branch="feature",
            updated_at=repo.head.peel(pygit2.Commit).commit_time,
        )

        _build_index(engine)
        evts = _drain(events)

        warnings = [e for e in evts if isinstance(e, Output) and "Missing grammars" in e.text]
        assert len(warnings) == 1
        assert "swift" in warnings[0].text

        if session.index is not None:
            session.index.close()


def test_review_branch_triggers_indexing() -> None:
    """End-to-end: /review <branch> triggers indexing and emits index events."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo_with_file(tmp, content="def greet():\n    pass\n")
        repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))

        session = Session(repo=repo, owner="o", repo_name="r", gh=None)
        events: queue.Queue[Event] = queue.Queue()
        engine = Engine(session, events)
        engine.run_task(TaskType.COMMAND, "/review feature")

        # Indexing now runs in a background thread — wait for it.
        evts = _drain(events)
        evts.extend(_wait_for_index(events))

        assert any(isinstance(e, IndexStarted) for e in evts)
        assert any(isinstance(e, IndexReady) for e in evts)

        if session.index is not None:
            session.index.close()


def test_review_two_args_triggers_indexing() -> None:
    """/review base target triggers indexing with correct base."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo_with_file(tmp)
        repo.branches.local.create("develop", repo.head.peel(pygit2.Commit))
        repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))

        session = Session(repo=repo, owner="o", repo_name="r", gh=None)
        events: queue.Queue[Event] = queue.Queue()
        engine = Engine(session, events)
        engine.run_task(TaskType.COMMAND, "/review develop feature")

        # Indexing now runs in a background thread — wait for it.
        evts = _drain(events)
        evts.extend(_wait_for_index(events))

        assert isinstance(session.review_target, BranchTarget)
        assert session.review_target.base_branch == "develop"

        assert any(isinstance(e, IndexStarted) for e in evts)
        assert any(isinstance(e, IndexReady) for e in evts)

        if session.index is not None:
            session.index.close()
