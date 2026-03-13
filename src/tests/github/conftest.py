"""Shared fake PyGithub objects and fixtures for GitHub integration tests.

All test files in this package use these stubs instead of the
real PyGithub classes.  Each fake implements only the methods
needed by the production code under test.
"""

from __future__ import annotations

import queue
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pygit2
import pytest
from pydantic_ai import RunContext
from pytest_mock import MockerFixture

from rbtr.engine import Engine
from rbtr.github.client import GitHubCtx
from rbtr.llm.deps import AgentDeps
from rbtr.models import PRTarget
from rbtr.sessions.store import SessionStore
from rbtr.state import EngineState


class FakeUser:
    def __init__(self, login: str = "alice", user_type: str = "User") -> None:
        self.login = login
        self.type = user_type


class FakeReaction:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeReview:
    """Stub for PullRequestReview."""

    def __init__(
        self,
        *,
        review_id: int = 100,
        body: str = "Looks good.",
        state: str = "APPROVED",
        user: FakeUser | None = None,
        submitted_at: datetime | None = None,
        html_url: str = "https://github.com/pr/1#review",
    ) -> None:
        self.id = review_id
        self.body = body
        self.state = state
        self.user = user or FakeUser()
        self.submitted_at = submitted_at or datetime(2025, 1, 15, 10, 0, tzinfo=UTC)
        self.html_url = html_url
        self.deleted = False

    def delete(self) -> None:
        self.deleted = True


class FakeInlineComment:
    """Stub for PullRequestComment (inline review comment)."""

    def __init__(
        self,
        *,
        comment_id: int = 10,
        body: str = "Fix this.",
        path: str = "src/handler.py",
        line: int | None = 42,
        diff_hunk: str = "@@ -40,5 +40,5 @@",
        user: FakeUser | None = None,
        created_at: datetime | None = None,
        in_reply_to_id: int | None = None,
        reactions: list[FakeReaction] | None = None,
    ) -> None:
        self.id = comment_id
        self.body = body
        self.path = path
        self.line = line
        self.diff_hunk = diff_hunk
        self.user = user or FakeUser()
        self.created_at = created_at or datetime(2025, 1, 15, 11, 0, tzinfo=UTC)
        self.in_reply_to_id = in_reply_to_id
        self._reactions = reactions or []
        # Matches GithubObject._rawData so get_pending_review can
        # read fields without triggering lazy-load completion.
        self._rawData: dict[str, Any] = {
            "id": comment_id,
            "path": path,
            "line": line,
            "body": body,
        }

    def set_raw(self, key: str, value: Any) -> FakeInlineComment:
        """Set a raw_data field and return self for chaining."""
        self._rawData[key] = value
        return self

    def get_reactions(self) -> list[FakeReaction]:
        return self._reactions


class FakeIssueComment:
    """Stub for IssueComment."""

    def __init__(
        self,
        *,
        comment_id: int = 20,
        body: str = "General comment.",
        user: FakeUser | None = None,
        created_at: datetime | None = None,
        reactions: list[FakeReaction] | None = None,
    ) -> None:
        self.id = comment_id
        self.body = body
        self.user = user or FakeUser()
        self.created_at = created_at or datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
        self._reactions = reactions or []

    def get_reactions(self) -> list[FakeReaction]:
        return self._reactions


class _FakePRRef:
    """Stub for PullRequest head/base ref — provides ``.sha`` and ``.ref``."""

    def __init__(self, sha: str = "", ref: str = "") -> None:
        self.sha = sha
        self.ref = ref


class FakePR:
    """Stub for PullRequest — supports all API surfaces used in tests."""

    def __init__(
        self,
        *,
        reviews: list[FakeReview] | None = None,
        inline_comments: list[FakeInlineComment] | None = None,
        issue_comments: list[FakeIssueComment] | None = None,
        review_comments_by_id: dict[int, list[FakeInlineComment]] | None = None,
        default_user: FakeUser | None = None,
        head_sha: str = "",
        base_sha: str = "",
    ) -> None:
        self._reviews = reviews or []
        self._inline_comments = inline_comments or []
        self._issue_comments = issue_comments or []
        self._review_comments_by_id = review_comments_by_id or {}
        self._default_user = default_user or FakeUser()
        self.created_reviews: list[dict[str, Any]] = []
        self.head = _FakePRRef(sha=head_sha)
        self.base = _FakePRRef(sha=base_sha)

    def get_reviews(self) -> list[FakeReview]:
        return self._reviews

    def get_review_comments(self) -> list[FakeInlineComment]:
        return self._inline_comments

    def get_issue_comments(self) -> list[FakeIssueComment]:
        return self._issue_comments

    def get_single_review_comments(self, review_id: int) -> list[FakeInlineComment]:
        return self._review_comments_by_id.get(review_id, [])

    def get_review(self, review_id: int) -> FakeReview:
        for r in self._reviews:
            if r.id == review_id:
                return r
        return FakeReview(review_id=review_id)

    def create_review(
        self,
        body: str = "",
        event: str = "",
        comments: list[Any] | None = None,
        commit: Any = None,
    ) -> FakeReview:
        review_id = 200 + len(self.created_reviews)
        self.created_reviews.append(
            {
                "body": body,
                "event": event,
                "comments": comments or [],
                "commit_id": commit.sha if commit is not None else None,
            }
        )

        # Inherit user from the most recent PENDING review so
        # get_pending_review re-fetch finds the created review.
        user = self._default_user
        for r in reversed(self._reviews):
            if r.state == "PENDING" and r.user:
                user = r.user
                break

        new_review = FakeReview(
            review_id=review_id,
            state=event if event else "PENDING",
            body=body,
            user=user,
        )
        self._reviews.append(new_review)

        # Build FakeInlineComments so re-fetch returns them.
        if comments:
            fake_comments: list[FakeInlineComment] = []
            for i, c in enumerate(comments):
                fake_comments.append(
                    FakeInlineComment(
                        comment_id=review_id * 1000 + i,
                        path=c["path"],
                        body=c["body"],
                        line=c.get("line"),
                    )
                )
            self._review_comments_by_id[review_id] = fake_comments

        return new_review


class FakeCommit:
    """Stub for github.Commit.Commit — only ``.sha`` is needed."""

    def __init__(self, sha: str) -> None:
        self.sha = sha


class FakeRepo:
    def __init__(self, pr: FakePR | None = None) -> None:
        self._pr = pr or FakePR()

    def get_pull(self, number: int) -> FakePR:
        return self._pr

    def get_commit(self, sha: str) -> FakeCommit:
        return FakeCommit(sha)


class FakeGithub:
    def __init__(self, repo: FakeRepo | None = None) -> None:
        self._repo = repo or FakeRepo()

    def get_repo(self, full_name: str) -> FakeRepo:
        return self._repo


def fake_ctx(
    gh: FakeGithub | None = None,
    owner: str = "owner",
    repo_name: str = "repo",
) -> GitHubCtx:
    """Build a ``GitHubCtx`` backed by fake objects for tests."""
    return GitHubCtx(
        gh=gh or FakeGithub(),  # type: ignore[arg-type]  # fake stub
        owner=owner,
        repo_name=repo_name,
    )


# ── Two-commit repo builder ─────────────────────────────────────────


def _build_tree(
    repo: pygit2.Repository,
    files: dict[str, bytes],
) -> pygit2.Oid:
    """Build a nested tree from ``{"dir/file.py": b"..."}`` paths."""
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


# ── Shared fixtures ──────────────────────────────────────────────────


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

    Base::

        src/handler.py  — def handle(request):\\n    return 'ok'\\n
        src/utils.py    — def helper():\\n    return 42\\n
        readme.md       — # Project\\n\\nDescription.\\n

    Head::

        src/handler.py  — def handle(request):\\n    validate(request)\\n    return 'ok'\\n
        src/utils.py    — unchanged
        readme.md       — deleted
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
def pr_target(
    draft_repo: tuple[pygit2.Repository, pygit2.Oid, pygit2.Oid],
) -> PRTarget:
    """PRTarget wired to the draft_repo commits."""
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
    pr_target: PRTarget,
    draft_repo: tuple[pygit2.Repository, pygit2.Oid, pygit2.Oid],
    mocker: MockerFixture,
) -> RunContext[AgentDeps]:
    """RunContext wired to the draft_repo — for LLM tool calls."""
    repo, _, _ = draft_repo

    # Each test gets a fresh EngineState — no global cache to reset.
    state = EngineState()
    state.review_target = pr_target
    state.repo = repo
    deps = AgentDeps(state=state, store=SessionStore())
    ctx = mocker.MagicMock(spec=RunContext)
    ctx.deps = deps
    return ctx


@pytest.fixture
def review_engine(
    pr_target: PRTarget,
    draft_repo: tuple[pygit2.Repository, pygit2.Oid, pygit2.Oid],
) -> Generator[Engine]:
    """Engine wired to draft_repo with a PR target — for sync/post/show."""
    repo, _, _ = draft_repo
    state = EngineState(owner="owner", repo_name="repo", repo=repo)
    state.review_target = pr_target
    state.gh_username = "reviewer"
    with Engine(state, queue.Queue(), store=SessionStore()) as eng:
        yield eng
