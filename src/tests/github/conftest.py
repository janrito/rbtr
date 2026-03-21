"""Shared fixtures for GitHub integration tests."""

from __future__ import annotations

import queue
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pygit2
import pytest
from github import Github
from github.PullRequest import PullRequest
from github.Repository import Repository
from pydantic_ai import RunContext
from pytest_mock import MockerFixture

from rbtr.engine.core import Engine
from rbtr.llm.deps import AgentDeps
from rbtr.models import PRTarget
from rbtr.sessions.store import SessionStore
from rbtr.state import EngineState

# ── Git helpers ──────────────────────────────────────────────────────


def _build_tree(
    repo: pygit2.Repository,
    files: dict[str, bytes],
) -> pygit2.Oid:
    """Build a nested tree from `{"dir/file.py": b"..."}` paths."""
    subtrees: dict[str, dict[str, bytes]] = {}
    blobs: dict[str, bytes] = {}
    for path, content in files.items():
        if "/" in path:
            top, rest = path.split("/", 1)
            subtrees.setdefault(top, {})[rest] = content
        else:
            blobs[path] = content
    tb = repo.TreeBuilder()
    for name, data in blobs.items():
        tb.insert(name, repo.create_blob(data), pygit2.GIT_FILEMODE_BLOB)
    for name, sub_files in subtrees.items():
        tb.insert(name, _build_tree(repo, sub_files), pygit2.GIT_FILEMODE_TREE)
    return tb.write()


def make_commit(
    repo: pygit2.Repository,
    files: dict[str, bytes],
    *,
    parents: list[pygit2.Oid] | None = None,
    ref: str = "refs/heads/main",
) -> pygit2.Oid:
    """Create a commit with the given file tree and return its OID."""
    tree_oid = _build_tree(repo, files)
    sig = pygit2.Signature("Test", "test@test.com")
    return repo.create_commit(ref, sig, sig, "commit", tree_oid, parents or [])


# ── PyGithub mock fixtures ──────────────────────────────────────────


@pytest.fixture
def gh(mocker: MockerFixture) -> Github:
    """Autospecced Github → Repository chain."""
    mock_gh = mocker.create_autospec(Github, instance=True)
    mock_repo = mocker.create_autospec(Repository, instance=True)
    mock_gh.get_repo.return_value = mock_repo
    return mock_gh


@pytest.fixture
def mock_pr(gh: Github, mocker: MockerFixture) -> PullRequest:
    """Configurable PullRequest mock.

    Defaults: no reviews, no comments, head/base SHAs set.
    Tests override return values directly on the mock.
    """
    pr = mocker.create_autospec(PullRequest, instance=True)
    gh.get_repo.return_value.get_pull.return_value = pr
    pr.get_reviews.return_value = []
    pr.get_review_comments.return_value = []
    pr.get_issue_comments.return_value = []
    pr.get_single_review_comments.return_value = []
    # Return no diff ranges by default — sync skips stale-comment
    # validation.  Tests that need diff ranges override this.
    pr.get_files.side_effect = NotImplementedError
    pr.head = mocker.MagicMock(sha="abc123", ref="feature")
    pr.base = mocker.MagicMock(sha="def456", ref="main")
    pr.create_review.return_value = mocker.MagicMock(
        id=200, html_url="https://github.com/owner/repo/pull/42#review"
    )
    return pr


def mock_review(
    review_id: int = 1,
    body: str = "LGTM",
    state: str = "APPROVED",
    user: str = "alice",
    user_type: str = "User",
    submitted_at: datetime | None = None,
) -> MagicMock:
    """Build a MagicMock resembling a PyGithub PullRequestReview."""
    r = MagicMock()
    r.id = review_id
    r.body = body
    r.state = state
    r.user.login = user
    r.user.type = user_type
    r.submitted_at = submitted_at or datetime(2025, 1, 15, 10, 0, tzinfo=UTC)
    return r


def mock_issue_comment(
    comment_id: int = 20,
    body: str = "General comment.",
    user: str = "alice",
    user_type: str = "User",
    created_at: datetime | None = None,
    reactions: list[MagicMock] | None = None,
) -> MagicMock:
    """Build a MagicMock resembling a PyGithub IssueComment."""
    c = MagicMock()
    c.id = comment_id
    c.body = body
    c.user.login = user
    c.user.type = user_type
    c.created_at = created_at or datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
    c.get_reactions.return_value = reactions or []
    return c


def mock_comment(
    comment_id: int = 200,
    path: str = "a.py",
    line: int | None = 10,
    body: str = "Fix this.",
    diff_hunk: str = "@@ -1,5 +1,5 @@",
    side: str | None = None,
    commit_id: str | None = None,
    user: str = "alice",
    user_type: str = "User",
    created_at: datetime | None = None,
    in_reply_to_id: int | None = None,
    reactions: list[MagicMock] | None = None,
) -> MagicMock:
    """Build a MagicMock resembling a PyGithub PullRequestComment."""
    c = MagicMock()
    c.id = comment_id
    c.path = path
    c.line = line
    c.body = body
    c.diff_hunk = diff_hunk
    c.side = side
    c.user.login = user
    c.user.type = user_type
    c.created_at = created_at or datetime(2025, 1, 15, 11, 0, tzinfo=UTC)
    c.in_reply_to_id = in_reply_to_id
    c.get_reactions.return_value = reactions or []
    c._rawData = {
        "id": comment_id,
        "path": path,
        "line": line,
        "body": body,
        "side": side,
        "commit_id": commit_id,
    }
    return c


@pytest.fixture
def pending_review(mock_pr: PullRequest) -> MagicMock:
    """Configure mock_pr to have one PENDING review by `reviewer`.

    Returns the review mock so tests can customise `.body` etc.
    """
    review = MagicMock()
    review.id = 99
    review.state = "PENDING"
    review.body = ""
    review.user.login = "reviewer"
    review.user.type = "User"
    mock_pr.get_reviews.return_value = [review]
    return review


# ── Engine fixtures ──────────────────────────────────────────────────


@pytest.fixture
def draft_engine(gh: Github, pr_target: PRTarget, workspace: Path) -> Generator[Engine]:
    """Real Engine wired to the autospecced `gh` mock.

    No git repo — for tests that only need GitHub API interaction.
    """
    state = EngineState(owner="owner", repo_name="repo")
    state.gh = gh
    state.gh_username = "reviewer"
    state.review_target = pr_target
    with Engine(state, queue.Queue(), store=SessionStore()) as eng:
        yield eng


@pytest.fixture
def repo_draft_engine(
    gh: Github,
    draft_pr_target: PRTarget,
    draft_repo: tuple[pygit2.Repository, pygit2.Oid, pygit2.Oid],
    workspace: Path,
) -> Generator[Engine]:
    """Engine with both a real git repo and the autospecced `gh` mock.

    For tests that need line translation (real repo) AND GitHub API
    interaction (mock gh) — e.g. sync with stale comment translation.
    """
    repo, _, _ = draft_repo
    state = EngineState(owner="owner", repo_name="repo", repo=repo)
    state.gh = gh
    state.gh_username = "reviewer"
    state.review_target = draft_pr_target
    with Engine(state, queue.Queue(), store=SessionStore()) as eng:
        yield eng


# ── Data fixtures ────────────────────────────────────────────────────


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point drafts_dir at a temp directory for draft persistence."""
    monkeypatch.setattr("rbtr.config.config.tools.drafts_dir", str(tmp_path / "drafts"))
    return tmp_path


@pytest.fixture
def draft_repo(
    tmp_path: Path,
) -> tuple[pygit2.Repository, pygit2.Oid, pygit2.Oid]:
    """Two-commit repo: base (main) → head (feature).

    Base: handler.py, utils.py, readme.md
    Head: handler.py (modified), utils.py, readme.md (deleted)
    """
    repo = pygit2.init_repository(str(tmp_path / "repo"))
    base = make_commit(
        repo,
        {
            "src/handler.py": b"def handle(request):\n    return 'ok'\n",
            "src/utils.py": b"def helper():\n    return 42\n",
            "readme.md": b"# Project\n\nDescription.\n",
        },
    )
    head = make_commit(
        repo,
        {
            "src/handler.py": b"def handle(request):\n    validate(request)\n    return 'ok'\n",
            "src/utils.py": b"def helper():\n    return 42\n",
        },
        parents=[base],
        ref="refs/heads/feature",
    )
    return repo, base, head


@pytest.fixture
def pr_target() -> PRTarget:
    """Simple PR target with hardcoded strings."""
    return PRTarget(
        number=42,
        title="Test PR",
        author="alice",
        base_branch="main",
        head_branch="feature",
        base_commit="main",
        head_commit="feature",
        head_sha="abc123face",
        updated_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


@pytest.fixture
def draft_pr_target(
    draft_repo: tuple[pygit2.Repository, pygit2.Oid, pygit2.Oid],
) -> PRTarget:
    """PRTarget wired to real draft_repo commit SHAs."""
    _, base, head = draft_repo
    return PRTarget(
        number=42,
        title="Add validation",
        author="alice",
        base_branch=str(base),
        head_branch="feature",
        base_commit=str(base),
        head_commit=str(head),
        head_sha=str(head),
        updated_at=datetime(2025, 6, 1, tzinfo=UTC),
    )


@pytest.fixture
def tool_ctx(
    draft_pr_target: PRTarget,
    draft_repo: tuple[pygit2.Repository, pygit2.Oid, pygit2.Oid],
    mocker: MockerFixture,
) -> Generator[RunContext[AgentDeps]]:
    """RunContext wired to the draft_repo — for LLM tool calls."""
    repo, _, _ = draft_repo

    state = EngineState()
    state.review_target = draft_pr_target
    state.repo = repo
    with SessionStore() as store:
        deps = AgentDeps(state=state, store=store)
        ctx = mocker.MagicMock(spec=RunContext)
        ctx.deps = deps
        yield ctx
