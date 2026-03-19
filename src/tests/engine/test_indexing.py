"""Tests for engine/indexing.py — background indexing on /review."""

from __future__ import annotations

import pygit2
import pytest
from pytest_mock import MockerFixture

from rbtr.engine.core import Engine
from rbtr.engine.indexing import _build_index
from rbtr.engine.types import TaskType
from rbtr.events import (
    IndexProgress,
    IndexReady,
    IndexStarted,
    Output,
)
from rbtr.models import BranchTarget

from .conftest import drain, wait_for_index


@pytest.fixture(autouse=True)
def _mock_embeddings(mocker: MockerFixture) -> None:
    """Stub out the embedding step — no GGUF model needed in tests."""
    mocker.patch("rbtr.index.orchestrator._embed_missing")


# ── Helpers ──────────────────────────────────────────────────────────


@pytest.fixture
def review_engine(repo_engine: Engine) -> Engine:
    """Engine with a feature branch set as the review target."""
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
    return engine


# ── run_index ────────────────────────────────────────────────────────


def test_index_emits_started_progress_ready(review_engine: Engine) -> None:
    """run_index emits IndexStarted, IndexProgress, and IndexReady."""
    engine = review_engine

    _build_index(engine)
    drained_events = drain(engine.events)

    started = [e for e in drained_events if isinstance(e, IndexStarted)]
    assert len(started) == 1
    assert started[0].total_files >= 1

    progress = [e for e in drained_events if isinstance(e, IndexProgress)]
    assert len(progress) >= 1
    phases = {e.phase for e in progress}
    assert "parsing" in phases

    ready = [e for e in drained_events if isinstance(e, IndexReady)]
    assert len(ready) == 1
    assert ready[0].chunk_count >= 1

    # Store is attached to state.
    assert engine.state.index is not None
    engine.state.index.close()


def test_index_sets_store_on_state(review_engine: Engine) -> None:
    """After indexing, engine.state.index is an IndexStore."""
    engine = review_engine

    _build_index(engine)
    drain(engine.events)

    assert engine.state.index is not None
    # Should be able to query chunks.
    chunks = engine.state.index.get_chunks("feature")
    assert len(chunks) >= 1
    engine.state.index.close()


def test_index_skipped_when_disabled(mocker: MockerFixture, review_engine: Engine) -> None:
    """Indexing is skipped when config.index.enabled is False."""
    mocker.patch("rbtr.engine.indexing.config.index.enabled", False)

    engine = review_engine

    _build_index(engine)
    drained_events = drain(engine.events)

    assert engine.state.index is None
    assert not any(isinstance(e, IndexStarted) for e in drained_events)


def test_index_skipped_without_review_target(repo_engine: Engine) -> None:
    """No-op when there's no review target."""
    engine = repo_engine

    _build_index(engine)
    drained_events = drain(engine.events)
    assert not drained_events
    assert engine.state.index is None


_swift_grammar_available = False
try:
    import tree_sitter_swift as _ts_swift  # noqa: F401

    _swift_grammar_available = True
except ImportError:
    pass


@pytest.mark.skipif(_swift_grammar_available, reason="Swift grammar is installed")
def test_index_warns_missing_grammars(
    tmp_path: pytest.TempPathFactory, repo_engine: Engine
) -> None:
    """Indexing warns about languages with no grammar installed."""
    # Replace default hello.py with a .swift file the parser can't handle.
    repo = repo_engine.state.repo
    assert repo is not None
    sig = pygit2.Signature("Test", "test@test.com")
    blob = repo.create_blob(b"let x = 1\n")
    tb = repo.TreeBuilder()
    tb.insert("app.swift", blob, pygit2.GIT_FILEMODE_BLOB)
    parent = repo.head.peel(pygit2.Commit)
    repo.create_commit("refs/heads/main", sig, sig, "swift file", tb.write(), [parent.id])

    engine = repo_engine
    repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))
    engine.state.review_target = BranchTarget(
        base_branch="main",
        head_branch="feature",
        base_commit="main",
        head_commit="feature",
        updated_at=repo.head.peel(pygit2.Commit).commit_time,
    )

    _build_index(engine)
    drained_events = drain(engine.events)

    warnings = [e for e in drained_events if isinstance(e, Output) and "Missing grammars" in e.text]
    assert len(warnings) == 1
    assert "swift" in warnings[0].text

    if engine.state.index is not None:
        engine.state.index.close()


def test_review_branch_triggers_indexing(repo_engine: Engine) -> None:
    """End-to-end: /review <branch> triggers indexing and emits index events."""
    engine = repo_engine
    repo = engine.state.repo
    assert repo is not None
    repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))

    engine.run_task(TaskType.COMMAND, "/review feature")

    drained_events = drain(engine.events)
    drained_events.extend(wait_for_index(engine.events))

    assert any(isinstance(e, IndexStarted) for e in drained_events)
    assert any(isinstance(e, IndexReady) for e in drained_events)

    if engine.state.index is not None:
        engine.state.index.close()


def test_review_two_args_triggers_indexing(repo_engine: Engine) -> None:
    """/review base target triggers indexing with correct base."""
    engine = repo_engine
    repo = engine.state.repo
    assert repo is not None
    repo.branches.local.create("develop", repo.head.peel(pygit2.Commit))
    repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))

    engine.run_task(TaskType.COMMAND, "/review develop feature")

    drained_events = drain(engine.events)
    drained_events.extend(wait_for_index(engine.events))

    assert isinstance(engine.state.review_target, BranchTarget)
    assert engine.state.review_target.base_branch == "develop"

    assert any(isinstance(e, IndexStarted) for e in drained_events)
    assert any(isinstance(e, IndexReady) for e in drained_events)

    if engine.state.index is not None:
        engine.state.index.close()
