"""Tests for the anchor-based comment system.

Exercises side/commit_id metadata flowing through the full
lifecycle: add → edit/remove → sync → post → pull → display.

All tests use the shared ``draft_repo`` fixture (see conftest)
and drive behaviour through public APIs only.
"""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest
from pydantic_ai import RunContext

from rbtr.engine import Engine
from rbtr.engine.draft_cmd import _show_draft
from rbtr.engine.publish import post_review_draft, sync_review_draft
from rbtr.exceptions import RbtrError
from rbtr.github.client import get_pending_review, post_review
from rbtr.github.draft import load_draft, save_draft
from rbtr.llm.agent import AgentDeps
from rbtr.llm.tools.draft import (
    add_draft_comment,
    edit_draft_comment,
    remove_draft_comment,
)
from rbtr.models import InlineComment, PRTarget, ReviewDraft, ReviewEvent
from tests.conftest import drain, output_texts

from .conftest import (
    FakeGithub,
    FakeInlineComment,
    FakePR,
    FakeRepo,
    FakeReview,
    FakeUser,
    fake_ctx,
)

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def mixed_draft(workspace: Path) -> ReviewDraft:
    """Seed and return a draft with LEFT and RIGHT comments."""
    draft = ReviewDraft(
        summary="Mixed review.",
        comments=[
            InlineComment(
                path="a.py",
                line=10,
                side="LEFT",
                commit_id="aaa",
                body="Old code issue.",
            ),
            InlineComment(
                path="b.py",
                line=20,
                side="RIGHT",
                commit_id="bbb",
                body="New code issue.",
            ),
            InlineComment(
                path="c.py",
                line=30,
                side="RIGHT",
                commit_id="ccc",
                body="Third finding.",
            ),
        ],
    )
    save_draft(42, draft)
    return draft


# ── Adding comments — anchor → side + commit_id ─────────────────────


def test_add_head_anchor_right_side(workspace: Path, tool_ctx: RunContext[AgentDeps]) -> None:
    """Anchor on head file → RIGHT side, commit_id = head SHA."""
    add_draft_comment(tool_ctx, "src/handler.py", "validate(request)", "Bug.")
    draft = load_draft(42)
    assert draft is not None
    c = draft.comments[0]
    assert c.side == "RIGHT"
    assert c.commit_id == tool_ctx.deps.state.review_target.head_sha  # type: ignore[union-attr]
    assert c.line == 2


def test_add_base_anchor_left_side(workspace: Path, tool_ctx: RunContext[AgentDeps]) -> None:
    """Anchor with ref='base' → LEFT side, resolves against base."""
    add_draft_comment(tool_ctx, "src/handler.py", "return 'ok'", "Old code.", ref="base")
    draft = load_draft(42)
    assert draft is not None
    c = draft.comments[0]
    assert c.side == "LEFT"
    assert c.line == 2  # base line, before insertion


def test_add_head_anchor_shifted_line(workspace: Path, tool_ctx: RunContext[AgentDeps]) -> None:
    """Same code at different line numbers on each side."""
    add_draft_comment(tool_ctx, "src/handler.py", "return 'ok'", "Now line 3.")
    draft = load_draft(42)
    assert draft is not None
    assert draft.comments[0].line == 3  # shifted by insertion


def test_add_base_anchor_deleted_file(workspace: Path, tool_ctx: RunContext[AgentDeps]) -> None:
    """readme.md deleted at head — can still comment on base side."""
    result = add_draft_comment(tool_ctx, "readme.md", "# Project", "Why deleted?", ref="base")
    assert "Comment added" in result
    draft = load_draft(42)
    assert draft is not None
    assert draft.comments[0].side == "LEFT"
    assert draft.comments[0].path == "readme.md"


# ── Edit and remove — metadata preserved ─────────────────────────────


def test_edit_preserves_side_and_commit_id(
    workspace: Path, tool_ctx: RunContext[AgentDeps], mixed_draft: ReviewDraft
) -> None:
    edit_draft_comment(tool_ctx, "a.py", "Old code", body="Updated body.")
    draft = load_draft(42)
    assert draft is not None
    c = draft.comments[0]
    assert c.body == "Updated body."
    assert c.side == "LEFT"
    assert c.commit_id == "aaa"
    assert c.line == 10


def test_remove_preserves_remaining(
    workspace: Path, tool_ctx: RunContext[AgentDeps], mixed_draft: ReviewDraft
) -> None:
    remove_draft_comment(tool_ctx, "b.py", "New code")
    draft = load_draft(42)
    assert draft is not None
    assert len(draft.comments) == 2
    assert draft.comments[0].path == "a.py"
    assert draft.comments[1].path == "c.py"
    assert draft.summary == "Mixed review."


def test_full_add_edit_remove_cycle(workspace: Path, tool_ctx: RunContext[AgentDeps]) -> None:
    """Add → edit → remove cycle preserves metadata throughout."""
    add_draft_comment(tool_ctx, "src/handler.py", "validate(request)", "Initial.")
    edit_draft_comment(tool_ctx, "src/handler.py", "Initial", body="Revised.")

    draft = load_draft(42)
    assert draft is not None
    assert draft.comments[0].body == "Revised."
    assert draft.comments[0].side == "RIGHT"

    remove_draft_comment(tool_ctx, "src/handler.py", "Revised")
    draft = load_draft(42)
    assert draft is not None
    assert draft.comments == []


# ── Sync — translation + side forwarding ─────────────────────────────


def test_sync_translates_stale_comment_line(
    workspace: Path,
    review_engine: Engine,
    draft_repo: tuple[pygit2.Repository, pygit2.Oid, pygit2.Oid],
) -> None:
    """Comment at base line 2 gets translated to head line 3."""
    _, base, _head = draft_repo
    save_draft(
        42,
        ReviewDraft(
            summary="Review.",
            comments=[
                InlineComment(
                    path="src/handler.py",
                    line=2,
                    body="Return statement.",
                    commit_id=str(base),
                ),
            ],
        ),
    )
    pr = FakePR(reviews=[])
    review_engine.state.gh = FakeGithub(FakeRepo(pr))  # type: ignore[assignment]

    sync_review_draft(review_engine, 42)

    assert pr.created_reviews[0]["comments"][0]["line"] == 3


def test_sync_warns_and_skips_deleted_file(
    workspace: Path,
    review_engine: Engine,
    draft_repo: tuple[pygit2.Repository, pygit2.Oid, pygit2.Oid],
) -> None:
    """Comment on deleted file is warned about and skipped from push."""
    _, base, _head = draft_repo
    save_draft(
        42,
        ReviewDraft(
            summary="Review.",
            comments=[
                InlineComment(
                    path="readme.md",
                    line=1,
                    body="Gone.",
                    commit_id=str(base),
                ),
            ],
        ),
    )
    pr = FakePR(reviews=[])
    review_engine.state.gh = FakeGithub(FakeRepo(pr))  # type: ignore[assignment]

    sync_review_draft(review_engine, 42)

    assert any("deleted" in t.lower() for t in output_texts(drain(review_engine.events)))
    assert len(pr.created_reviews[0]["comments"]) == 0


def test_sync_sends_side_per_comment(
    workspace: Path,
    review_engine: Engine,
    draft_repo: tuple[pygit2.Repository, pygit2.Oid, pygit2.Oid],
) -> None:
    """LEFT and RIGHT sides are forwarded in the push payload."""
    _, _base, head = draft_repo
    save_draft(
        42,
        ReviewDraft(
            summary=".",
            comments=[
                # base line 2 ("return 'ok'") is in the LEFT diff hunk.
                InlineComment(
                    path="src/handler.py",
                    line=2,
                    side="LEFT",
                    body="Old.",
                    commit_id=str(head),
                ),
                # head line 2 ("validate(request)") is in the RIGHT diff hunk.
                InlineComment(
                    path="src/handler.py",
                    line=2,
                    side="RIGHT",
                    body="New.",
                    commit_id=str(head),
                ),
            ],
        ),
    )
    pr = FakePR(reviews=[])
    review_engine.state.gh = FakeGithub(FakeRepo(pr))  # type: ignore[assignment]

    sync_review_draft(review_engine, 42)

    comments = pr.created_reviews[0]["comments"]
    assert comments[0]["side"] == "LEFT"
    assert comments[1]["side"] == "RIGHT"


def test_sync_sends_commit_id(workspace: Path, review_engine: Engine, pr_target: PRTarget) -> None:
    """Push passes head_sha as commit_id to the GitHub API."""
    save_draft(42, ReviewDraft(summary=".", comments=[]))
    pr = FakePR(reviews=[])
    review_engine.state.gh = FakeGithub(FakeRepo(pr))  # type: ignore[assignment]

    sync_review_draft(review_engine, 42)

    assert pr.created_reviews[0]["commit_id"] == pr_target.head_sha


# ── File-level comments (line=0) ─────────────────────────────────────


def test_sync_file_level_comments_pass_validation(workspace: Path, review_engine: Engine) -> None:
    """File-level comments (line=0) are not rejected as stale."""
    save_draft(
        42,
        ReviewDraft(
            summary="Review.",
            comments=[
                InlineComment(path="dvc.yaml", line=0, body="Missing dep."),
                InlineComment(path="eval/evaluate.py", line=0, body="Merge marker."),
            ],
        ),
    )
    pr = FakePR(reviews=[])
    review_engine.state.gh = FakeGithub(FakeRepo(pr))  # type: ignore[assignment]

    sync_review_draft(review_engine, 42)

    assert len(pr.created_reviews) == 1
    pushed = pr.created_reviews[0]["comments"]
    assert len(pushed) == 2
    # File-level comments omit line and side from the API payload.
    assert "line" not in pushed[0]
    assert "side" not in pushed[0]


def test_post_file_level_comments_not_rejected(workspace: Path, review_engine: Engine) -> None:
    """post_review_draft does not raise for file-level comments."""
    draft = ReviewDraft(
        summary="Review.",
        comments=[InlineComment(path="dvc.yaml", line=0, body="Fix deps.")],
    )
    pr = FakePR(reviews=[])
    review_engine.state.gh = FakeGithub(FakeRepo(pr))  # type: ignore[assignment]

    # Should not raise — file-level comments skip validation.
    post_review_draft(review_engine, 42, draft, ReviewEvent.COMMENT)

    assert len(pr.created_reviews) == 1
    assert len(pr.created_reviews[0]["comments"]) == 1


# ── Post — translation errors + payload ──────────────────────────────


def test_post_errors_on_deleted_lines(
    workspace: Path,
    review_engine: Engine,
    draft_repo: tuple[pygit2.Repository, pygit2.Oid, pygit2.Oid],
) -> None:
    """Posting a draft with comments on deleted lines raises."""
    _, base, _head = draft_repo
    draft = ReviewDraft(
        summary="Review.",
        comments=[
            InlineComment(
                path="readme.md",
                line=1,
                body="Gone.",
                commit_id=str(base),
            ),
        ],
    )
    pr = FakePR(reviews=[])
    review_engine.state.gh = FakeGithub(FakeRepo(pr))  # type: ignore[assignment]

    with pytest.raises(RbtrError, match="deleted lines"):
        post_review_draft(review_engine, 42, draft, ReviewEvent.COMMENT)


def test_post_sends_side_and_commit_id() -> None:
    """post_review forwards side per-comment and commit_id per-review."""
    pr = FakePR()
    draft = ReviewDraft(
        summary="Review.",
        comments=[
            InlineComment(path="a.py", line=10, side="LEFT", body="Old."),
            InlineComment(path="a.py", line=20, side="RIGHT", body="New."),
        ],
    )
    post_review(
        fake_ctx(FakeGithub(FakeRepo(pr))),
        1,
        draft,
        ReviewEvent.COMMENT,
        commit_id="abc123",
    )

    review = pr.created_reviews[0]
    assert review["commit_id"] == "abc123"
    assert review["comments"][0]["side"] == "LEFT"
    assert review["comments"][1]["side"] == "RIGHT"


# ── Pull — reading side + commit_id from GitHub ─────────────────────


def test_pull_reads_side_and_commit_id() -> None:
    comment = FakeInlineComment(comment_id=10, path="a.py", line=10, body="Fix.")
    comment.set_raw("side", "LEFT").set_raw("commit_id", "sha999")
    pr = FakePR(
        reviews=[FakeReview(review_id=1, state="PENDING", user=FakeUser("me"))],
        review_comments_by_id={1: [comment]},
    )

    result = get_pending_review(fake_ctx(FakeGithub(FakeRepo(pr))), 1, "me")
    assert result is not None
    assert result.comments[0].side == "LEFT"
    assert result.comments[0].commit_id == "sha999"


def test_pull_defaults_when_fields_absent() -> None:
    comment = FakeInlineComment(comment_id=10, path="a.py", line=10, body="Fix.")
    pr = FakePR(
        reviews=[FakeReview(review_id=1, state="PENDING", user=FakeUser("me"))],
        review_comments_by_id={1: [comment]},
    )

    result = get_pending_review(fake_ctx(FakeGithub(FakeRepo(pr))), 1, "me")
    assert result is not None
    assert result.comments[0].side == "RIGHT"
    assert result.comments[0].commit_id == ""


# ── Display — _show_draft tags ───────────────────────────────────────


def test_show_draft_left_side_tag(workspace: Path, review_engine: Engine) -> None:
    save_draft(
        42,
        ReviewDraft(
            summary=".",
            comments=[InlineComment(path="a.py", line=5, side="LEFT", body="Old.")],
        ),
    )
    _show_draft(review_engine, 42)
    texts = output_texts(drain(review_engine.events))
    assert any("(base)" in t for t in texts)


def test_show_draft_stale_commit_tag(workspace: Path, review_engine: Engine) -> None:
    save_draft(
        42,
        ReviewDraft(
            summary=".",
            comments=[
                InlineComment(
                    path="a.py",
                    line=5,
                    body="Old.",
                    commit_id="aabbccdd11223344",
                ),
            ],
        ),
    )
    _show_draft(review_engine, 42)
    texts = output_texts(drain(review_engine.events))
    assert any("stale:" in t for t in texts)
    assert any("aabbccd" in t for t in texts)


def test_show_draft_file_level_no_line(workspace: Path, review_engine: Engine) -> None:
    """File-level comments (line=0) display path without :0."""
    save_draft(
        42,
        ReviewDraft(
            summary=".",
            comments=[InlineComment(path="dvc.yaml", line=0, body="Missing dep.")],
        ),
    )
    _show_draft(review_engine, 42)
    texts = output_texts(drain(review_engine.events))
    combined = "\n".join(texts)
    # File heading shows path; no ":0" line reference.
    assert "dvc.yaml" in combined
    assert ":0" not in combined


def test_show_draft_current_commit_no_tag(
    workspace: Path, review_engine: Engine, pr_target: PRTarget
) -> None:
    save_draft(
        42,
        ReviewDraft(
            summary=".",
            comments=[
                InlineComment(
                    path="a.py",
                    line=5,
                    body="Fresh.",
                    commit_id=pr_target.head_sha,
                ),
            ],
        ),
    )
    _show_draft(review_engine, 42)
    texts = output_texts(drain(review_engine.events))
    combined = "\n".join(texts)
    assert "stale:" not in combined
    assert "(base)" not in combined
