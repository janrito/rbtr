"""Shared fixtures for engine tests."""

from __future__ import annotations

import queue
import time
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import pygit2
import pytest

from rbtr.creds import OAuthCreds
from rbtr.engine.core import Engine
from rbtr.events import Event, IndexReady, Output
from rbtr.models import BranchSummary, BranchTarget, PRSummary, PRTarget
from rbtr.sessions.store import SessionStore
from rbtr.state import EngineState
from tests.helpers import drain

# ── Data fixtures ────────────────────────────────────────────────────


@pytest.fixture
def pr_fix_bug() -> PRSummary:
    return PRSummary(
        number=1,
        title="Fix bug",
        author="alice",
        base_branch="main",
        head_branch="fix-bug",
        updated_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


@pytest.fixture
def pr_add_feature() -> PRSummary:
    return PRSummary(
        number=42,
        title="Add feature",
        author="bob",
        base_branch="main",
        head_branch="add-feature",
        updated_at=datetime(2025, 6, 1, tzinfo=UTC),
    )


@pytest.fixture
def pr_refactor() -> PRSummary:
    return PRSummary(
        number=10,
        title="Refactor",
        author="alice",
        base_branch="develop",
        head_branch="refactor-x",
        updated_at=datetime(2025, 6, 1, tzinfo=UTC),
    )


@pytest.fixture
def branch_feature_x() -> BranchSummary:
    return BranchSummary(
        name="feature-x",
        last_commit_sha="abc123",
        last_commit_message="wip",
        updated_at=datetime(2025, 1, 2, tzinfo=UTC),
    )


@pytest.fixture
def target_pr_42() -> PRTarget:
    return PRTarget(
        number=42,
        title="Add feature",
        author="bob",
        base_branch="main",
        head_branch="add-feature",
        base_commit="main",
        head_commit="add-feature",
        updated_at=datetime(2025, 6, 1, tzinfo=UTC),
    )


@pytest.fixture
def target_branch() -> BranchTarget:
    return BranchTarget(
        base_branch="main",
        head_branch="feature-x",
        base_commit="main",
        head_commit="feature-x",
        updated_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


@pytest.fixture
def claude_oauth() -> OAuthCreds:
    return OAuthCreds(
        access_token="test-bearer-token",
        refresh_token="ref",
        expires_at=9999999999,
    )


@pytest.fixture
def chatgpt_oauth() -> OAuthCreds:
    return OAuthCreds(
        access_token="jwt-token",
        refresh_token="ref",
        expires_at=9999999999,
        account_id="acct_123",
    )


# ── Engine fixtures ──────────────────────────────────────────────────

# `engine`, `llm_ctx`, and `llm_engine` live in the root
# conftest — available to all test packages.


@pytest.fixture
def repo_engine(tmp_path: Path) -> Generator[Engine]:
    """Engine backed by a git repo in a temp directory.

    The repo has a single commit on `main` with `hello.py`.
    Tests can add branches/files via `engine.state.repo`.
    """
    repo = pygit2.init_repository(str(tmp_path))
    sig = pygit2.Signature("Test", "test@test.com")
    blob = repo.create_blob(b"def hello():\n    pass\n")
    tb = repo.TreeBuilder()
    tb.insert("hello.py", blob, pygit2.GIT_FILEMODE_BLOB)
    repo.create_commit("refs/heads/main", sig, sig, "init", tb.write(), [])
    repo.set_head("refs/heads/main")
    state = EngineState(owner="o", repo_name="r", repo=repo)
    with Engine(state, queue.Queue(), store=SessionStore()) as eng:
        yield eng


def wait_for_index(events: queue.Queue[Event], timeout: float = 30.0) -> list[Event]:
    """Collect events until IndexReady or Output (index done/failed)."""
    collected: list[Event] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            evt = events.get(timeout=0.1)
        except queue.Empty:
            continue
        collected.append(evt)
        if isinstance(evt, (IndexReady, Output)):
            collected.extend(drain(events))
            break
    return collected


# ── Git helpers ──────────────────────────────────────────────────────


def make_repo_with_file(
    tmp: str,
    filename: str = "hello.py",
    content: str = "def greet():\n    pass\n",
) -> pygit2.Repository:
    """Create a git repo with a main branch containing one file."""
    repo = pygit2.init_repository(tmp)
    sig = pygit2.Signature("Test", "test@test.com")
    blob_id = repo.create_blob(content.encode())
    tb = repo.TreeBuilder()
    tb.insert(filename, blob_id, pygit2.GIT_FILEMODE_BLOB)
    tree_id = tb.write()
    repo.create_commit("refs/heads/main", sig, sig, "init", tree_id, [])
    repo.set_head("refs/heads/main")
    return repo
