"""Shared helpers and test data for engine tests.

Provides:
- Engine construction helpers (``make_engine``, ``drain``, ``output_texts``)
- Git repo helpers (``make_repo_with_file``, ``wait_for_index``)
- Realistic, reusable model instances for PRs and branches
"""

from __future__ import annotations

import queue
import time
from datetime import UTC, datetime

import pygit2
from github import Github

from rbtr.creds import OAuthCreds
from rbtr.engine import Engine, Session
from rbtr.events import Event, IndexReady, Output
from rbtr.models import BranchSummary, BranchTarget, PRSummary, PRTarget

# ── Shared test data ─────────────────────────────────────────────────
#
# Realistic, semantically distinct review targets so every test that
# needs a PR or branch uses named constants instead of inline construction.

PR_FIX_BUG = PRSummary(
    number=1,
    title="Fix bug",
    author="alice",
    base_branch="main",
    head_branch="fix-bug",
    updated_at=datetime(2025, 1, 1, tzinfo=UTC),
)

PR_ADD_FEATURE = PRSummary(
    number=42,
    title="Add feature",
    author="bob",
    base_branch="main",
    head_branch="add-feature",
    updated_at=datetime(2025, 6, 1, tzinfo=UTC),
)

PR_REFACTOR = PRSummary(
    number=10,
    title="Refactor",
    author="alice",
    base_branch="develop",
    head_branch="refactor-x",
    updated_at=datetime(2025, 6, 1, tzinfo=UTC),
)

BRANCH_FEATURE_X = BranchSummary(
    name="feature-x",
    last_commit_sha="abc123",
    last_commit_message="wip",
    updated_at=datetime(2025, 1, 2, tzinfo=UTC),
)

TARGET_PR_42 = PRTarget(
    number=42,
    title="Add feature",
    author="bob",
    base_branch="main",
    head_branch="add-feature",
    updated_at=datetime(2025, 6, 1, tzinfo=UTC),
)

TARGET_BRANCH = BranchTarget(
    base_branch="main",
    head_branch="feature-x",
    updated_at=datetime(2025, 1, 1, tzinfo=UTC),
)

# ── Shared credential data ───────────────────────────────────────────

CLAUDE_OAUTH = OAuthCreds(
    access_token="test-bearer-token",
    refresh_token="ref",
    expires_at=9999999999,
)

CHATGPT_OAUTH = OAuthCreds(
    access_token="jwt-token",
    refresh_token="ref",
    expires_at=9999999999,
    account_id="acct_123",
)


# ── Engine helpers ───────────────────────────────────────────────────


def make_engine(
    *,
    owner: str = "testowner",
    repo_name: str = "testrepo",
    gh: Github | None = None,
    repo: pygit2.Repository | None = None,
) -> tuple[Engine, queue.Queue[Event], Session]:
    """Create an Engine with a pre-populated Session."""
    session = Session(owner=owner, repo_name=repo_name, gh=gh)
    if repo is not None:
        session.repo = repo
    events: queue.Queue[Event] = queue.Queue()
    engine = Engine(session, events)
    return engine, events, session


def drain(events: queue.Queue[Event]) -> list[Event]:
    """Drain all events from the queue into a list."""
    result: list[Event] = []
    while True:
        try:
            result.append(events.get_nowait())
        except queue.Empty:
            break
    return result


def output_texts(events: list[Event]) -> list[str]:
    """Extract text from Output events."""
    return [e.text for e in events if isinstance(e, Output)]


def has_event_type(events: list[Event], event_type: type) -> bool:
    """Check whether any event matches the given type."""
    return any(isinstance(e, event_type) for e in events)


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
