"""Shared helpers and test data for engine tests.

Provides:
- `engine` fixture — default engine with auto-cleanup
- `repo_engine` fixture — engine backed by a git repo in a temp dir
- Event helpers (`drain`, `output_texts`)
- Git repo helpers (`make_repo_with_file`, `wait_for_index`)
- Realistic, reusable model instances for PRs and branches
"""

from __future__ import annotations

import queue
import time
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import pygit2
import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from rbtr.creds import OAuthCreds
from rbtr.engine.core import Engine
from rbtr.events import Event, IndexReady, Output
from rbtr.llm.compact import _SummaryResult
from rbtr.models import BranchSummary, BranchTarget, PRSummary, PRTarget
from rbtr.sessions.store import SessionStore
from rbtr.state import EngineState

# Re-export top-level helpers so existing `from .conftest import …`
# lines in this package keep working without change.
from tests.conftest import drain, has_event_type, output_texts  # noqa: F401

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
    base_commit="main",
    head_commit="add-feature",
    updated_at=datetime(2025, 6, 1, tzinfo=UTC),
)

TARGET_BRANCH = BranchTarget(
    base_branch="main",
    head_branch="feature-x",
    base_commit="main",
    head_commit="feature-x",
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


# ── Message data builders ────────────────────────────────────────────

_USAGE = RequestUsage(input_tokens=0, output_tokens=0)


def _user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _assistant(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)], usage=_USAGE, model_name="test")


def _tool_return(name: str, content: str, *, call_id: str | None = None) -> ModelRequest:
    return ModelRequest(
        parts=[ToolReturnPart(tool_name=name, content=content, tool_call_id=call_id)]
    )


def _tool_call(
    name: str, args: dict[str, str] | None = None, *, call_id: str | None = None
) -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart(tool_name=name, args=args or {}, tool_call_id=call_id)],
        usage=_USAGE,
        model_name="test",
    )


def _thinking(text: str) -> ModelResponse:
    return ModelResponse(parts=[ThinkingPart(content=text)], usage=_USAGE, model_name="test")


def _turns(n: int) -> list[ModelRequest | ModelResponse]:
    """Create *n* user→assistant turn pairs."""
    msgs: list[ModelRequest | ModelResponse] = []
    for i in range(n):
        msgs.append(_user(f"question {i}"))
        msgs.append(_assistant(f"answer {i}"))
    return msgs


def _seed(engine: Engine, messages: list[ModelRequest | ModelResponse], **kwargs: object) -> None:
    """Seed messages into the engine's store."""
    engine._sync_store_context()
    engine.store.save_messages(engine.state.session_id, messages, **kwargs)  # type: ignore[arg-type]


# ── Compaction helpers ────────────────────────────────────────────────


def summary_result(text: str = "Summary.") -> _SummaryResult:
    """Build a `_SummaryResult` with zero cost — for mocking `_stream_summary`."""
    return _SummaryResult(text=text, input_tokens=0, output_tokens=0, cost=0.0)


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
