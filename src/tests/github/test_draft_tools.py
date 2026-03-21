"""Tests for draft management LLM tools — add, edit, remove, set_summary.

The `add_draft_comment` tests use a real git repo so that
`resolve_anchor` and diff-range validation work end-to-end.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pygit2
import pytest
from pydantic_ai import RunContext

from rbtr.git.objects import DiffLineRanges
from rbtr.github.draft import load_draft, save_draft, snap_to_commentable_line
from rbtr.llm.deps import AgentDeps
from rbtr.llm.tools.draft import (
    _get_diff_ranges,
    add_draft_comment,
    edit_draft_comment,
    read_draft,
    remove_draft_comment,
    set_draft_summary,
)
from rbtr.models import InlineComment, PRTarget, ReviewDraft
from rbtr.sessions.store import SessionStore
from rbtr.state import EngineState

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def ctx(
    draft_pr_target: PRTarget,
    draft_repo: tuple[pygit2.Repository, pygit2.Oid, pygit2.Oid],
) -> Generator[RunContext[AgentDeps]]:
    """RunContext with a real repo and PR target."""
    repo, _, _ = draft_repo

    # Each test gets a fresh EngineState — no global cache to reset.
    state = EngineState()
    state.review_target = draft_pr_target
    state.repo = repo
    with SessionStore() as store:
        deps = AgentDeps(state=state, store=store)
        mock_ctx = MagicMock(spec=RunContext)
        mock_ctx.deps = deps
        yield mock_ctx


@pytest.fixture
def ctx_no_repo(draft_pr_target: PRTarget) -> Generator[RunContext[AgentDeps]]:
    """RunContext with a PR target but no repo (for edit/remove tests)."""
    state = EngineState()
    state.review_target = draft_pr_target
    with SessionStore() as store:
        deps = AgentDeps(state=state, store=store)
        mock_ctx = MagicMock(spec=RunContext)
        mock_ctx.deps = deps
        yield mock_ctx


# ── add_draft_comment ───────────────────────────────────────────────

# head handler.py:
#   line 1: def handle(request):
#   line 2:     validate(request)
#   line 3:     return 'ok'


def test_add_comment(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    result = add_draft_comment(ctx, "src/handler.py", "validate(request)", "**blocker:** Bug here.")
    assert "Comment added" in result
    assert "1 comment" in result

    draft = load_draft(42)
    assert draft is not None
    assert len(draft.comments) == 1
    c = draft.comments[0]
    assert c.path == "src/handler.py"
    assert c.line == 2
    assert c.side == "RIGHT"
    assert c.commit_id != ""
    assert c.body == "**blocker:** Bug here."


def test_add_comment_with_suggestion(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    result = add_draft_comment(ctx, "src/handler.py", "validate(request)", "Use this.", "fixed()")
    assert "Comment added" in result

    draft = load_draft(42)
    assert draft is not None
    assert draft.comments[0].suggestion == "fixed()"


def test_add_multiple_comments(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    add_draft_comment(ctx, "src/handler.py", "validate(request)", "First.")
    add_draft_comment(ctx, "src/handler.py", "return 'ok'", "Second.")

    draft = load_draft(42)
    assert draft is not None
    assert len(draft.comments) == 2
    assert draft.comments[0].line == 2
    assert draft.comments[1].line == 3


def test_add_comment_anchor_not_found(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    result = add_draft_comment(ctx, "src/handler.py", "nonexistent_code()", "Body.")
    assert "Cannot add comment" in result
    assert "not found" in result.lower()

    # No comment saved.
    assert load_draft(42) is None


def test_add_comment_anchor_ambiguous(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    # "request" appears twice in handler.py (lines 1 and 2).
    result = add_draft_comment(ctx, "src/handler.py", "request", "Body.")
    assert "Cannot add comment" in result
    assert "matches" in result.lower()


def test_add_comment_file_not_in_diff(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    # utils.py exists but wasn't changed in the diff.
    result = add_draft_comment(ctx, "src/utils.py", "helper()", "Body.")
    assert "Cannot add comment" in result
    assert "not in the PR diff" in result


def test_add_comment_anchor_outside_diff_hunk_snaps(
    workspace: Path, ctx: RunContext[AgentDeps]
) -> None:
    # "def handle(request):" is line 1 — may or may not be in a diff
    # hunk.  If not, the tool snaps to the nearest commentable line
    # instead of rejecting.
    result = add_draft_comment(ctx, "src/handler.py", "def handle(request):", "Body.")
    assert "Comment added" in result


@pytest.mark.parametrize(
    ("line", "expected_line", "has_note"),
    [
        (10, 10, False),  # exact match
        (12, 10, True),  # snaps to nearest
        (99, 10, True),  # snaps from far away
    ],
    ids=["exact", "near", "far"],
)
def test_snap_to_commentable_line(line: int, expected_line: int, has_note: bool) -> None:
    ranges: DiffLineRanges = {"a.py": {5, 10}}
    result_line, error, note = snap_to_commentable_line(ranges, "a.py", line)
    assert result_line == expected_line
    assert error is None
    assert bool(note) == has_note


def test_snap_rejects_file_not_in_diff() -> None:
    ranges: DiffLineRanges = {"a.py": {5, 10}}
    _, error, _ = snap_to_commentable_line(ranges, "b.py", 5)
    assert error is not None
    assert "not in the PR diff" in error


def test_snap_skips_when_no_ranges() -> None:
    result_line, error, note = snap_to_commentable_line({}, "a.py", 42)
    assert result_line == 42
    assert error is None
    assert note == ""


# ── diff_range_cache on EngineState ──────────────────────────────────


def test_diff_range_cache_populated_on_first_add(
    workspace: Path, ctx: RunContext[AgentDeps]
) -> None:
    """First `add_draft_comment` populates `state.diff_range_cache`."""
    state = ctx.deps.state
    assert state.diff_range_cache is None

    add_draft_comment(ctx, "src/handler.py", "validate(request)", "Body.")

    assert state.diff_range_cache is not None
    key, right, left = state.diff_range_cache
    target = state.review_target
    assert target is not None
    assert key == (target.base_commit, target.head_commit)
    assert isinstance(right, dict)
    assert isinstance(left, dict)


def test_diff_range_cache_reused_across_calls(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    """Subsequent calls reuse the cached ranges (same object identity)."""
    _get_diff_ranges(ctx)
    cache_after_first = ctx.deps.state.diff_range_cache

    _get_diff_ranges(ctx)
    cache_after_second = ctx.deps.state.diff_range_cache

    # Same object — not re-fetched.
    assert cache_after_first is cache_after_second


def test_diff_range_cache_invalidated_on_key_change(
    workspace: Path, ctx: RunContext[AgentDeps]
) -> None:
    """Cache is rebuilt when the review target's commits change."""
    _get_diff_ranges(ctx)
    old_cache = ctx.deps.state.diff_range_cache
    assert old_cache is not None

    # Simulate a refs refresh — change the key by swapping base/head.
    target = ctx.deps.state.review_target
    assert isinstance(target, PRTarget)
    ctx.deps.state.review_target = target.model_copy(
        update={
            "base_commit": target.head_commit,
            "head_commit": target.base_commit,
        }
    )

    _get_diff_ranges(ctx)
    new_cache = ctx.deps.state.diff_range_cache

    # Cache was rebuilt — different key.
    assert new_cache is not old_cache
    assert new_cache is not None
    assert new_cache[0] != old_cache[0]


def test_fresh_engine_state_has_no_cache() -> None:
    """New `EngineState` starts with no diff range cache."""
    state = EngineState()
    assert state.diff_range_cache is None


def test_add_comment_ref_base(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    # Base handler.py line 2: "    return 'ok'" — in the diff (old side).
    result = add_draft_comment(
        ctx, "src/handler.py", "return 'ok'", "This was the old code.", ref="base"
    )
    assert "Comment added" in result

    draft = load_draft(42)
    assert draft is not None
    c = draft.comments[0]
    assert c.side == "LEFT"
    assert c.line == 2


def test_add_comment_ref_invalid(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    result = add_draft_comment(ctx, "src/handler.py", "validate(request)", "Body.", ref="other")
    assert "head" in result
    assert "base" in result


def test_add_comment_file_not_found(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    result = add_draft_comment(ctx, "no/such/file.py", "anything", "Body.")
    assert "Cannot add comment" in result
    assert "not found" in result.lower()


# ── edit_draft_comment ──────────────────────────────────────────────


@pytest.fixture
def seeded_draft(workspace: Path) -> ReviewDraft:
    """Save and return a two-comment draft for edit/remove tests."""
    draft = ReviewDraft(
        summary="Review.",
        comments=[
            InlineComment(path="a.py", line=10, body="Original."),
            InlineComment(path="b.py", line=20, body="Also original."),
        ],
    )
    save_draft(42, draft)
    return draft


def test_edit_comment_body(
    workspace: Path, ctx_no_repo: RunContext[AgentDeps], seeded_draft: ReviewDraft
) -> None:
    result = edit_draft_comment(ctx_no_repo, "a.py", "Original", body="Updated body.")
    assert "updated" in result.lower()

    draft = load_draft(42)
    assert draft is not None
    assert draft.comments[0].body == "Updated body."
    assert draft.comments[0].path == "a.py"


def test_edit_comment_clear_suggestion(workspace: Path, ctx_no_repo: RunContext[AgentDeps]) -> None:
    draft = ReviewDraft(
        summary="",
        comments=[
            InlineComment(path="a.py", line=10, body="Fix.", suggestion="code"),
        ],
    )
    save_draft(42, draft)

    edit_draft_comment(ctx_no_repo, "a.py", "Fix", suggestion="")
    loaded = load_draft(42)
    assert loaded is not None
    assert loaded.comments[0].suggestion == ""


def test_edit_comment_not_found(
    workspace: Path, ctx_no_repo: RunContext[AgentDeps], seeded_draft: ReviewDraft
) -> None:
    result = edit_draft_comment(ctx_no_repo, "a.py", "nonexistent", body="Nope.")
    assert "Cannot edit" in result


def test_edit_comment_ambiguous(workspace: Path, ctx_no_repo: RunContext[AgentDeps]) -> None:
    draft = ReviewDraft(
        summary="",
        comments=[
            InlineComment(path="a.py", line=10, body="**nit:** First issue."),
            InlineComment(path="a.py", line=20, body="**nit:** Second issue."),
        ],
    )
    save_draft(42, draft)

    result = edit_draft_comment(ctx_no_repo, "a.py", "**nit:**", body="New.")
    assert "Cannot edit" in result
    assert "matches" in result.lower()


def test_edit_comment_empty_draft(workspace: Path, ctx_no_repo: RunContext[AgentDeps]) -> None:
    result = edit_draft_comment(ctx_no_repo, "a.py", "anything", body="Nope.")
    assert "no comments" in result


# ── remove_draft_comment ────────────────────────────────────────────


def test_remove_comment(
    workspace: Path, ctx_no_repo: RunContext[AgentDeps], seeded_draft: ReviewDraft
) -> None:
    result = remove_draft_comment(ctx_no_repo, "a.py", "Original")
    assert "Removed" in result

    draft = load_draft(42)
    assert draft is not None
    assert len(draft.comments) == 1
    assert draft.comments[0].path == "b.py"


def test_remove_last_comment(workspace: Path, ctx_no_repo: RunContext[AgentDeps]) -> None:
    draft = ReviewDraft(
        summary="Summary.",
        comments=[InlineComment(path="a.py", line=1, body="Only.")],
    )
    save_draft(42, draft)

    result = remove_draft_comment(ctx_no_repo, "a.py", "Only")
    assert "0 comment" in result

    loaded = load_draft(42)
    assert loaded is not None
    assert loaded.comments == []
    assert loaded.summary == "Summary."


def test_remove_not_found(
    workspace: Path, ctx_no_repo: RunContext[AgentDeps], seeded_draft: ReviewDraft
) -> None:
    result = remove_draft_comment(ctx_no_repo, "a.py", "nonexistent")
    assert "Cannot remove" in result


def test_remove_ambiguous(workspace: Path, ctx_no_repo: RunContext[AgentDeps]) -> None:
    draft = ReviewDraft(
        summary="",
        comments=[
            InlineComment(path="a.py", line=10, body="**nit:** First."),
            InlineComment(path="a.py", line=20, body="**nit:** Second."),
        ],
    )
    save_draft(42, draft)

    result = remove_draft_comment(ctx_no_repo, "a.py", "**nit:**")
    assert "Cannot remove" in result
    assert "matches" in result.lower()


def test_remove_synced_comment_tombstones(
    workspace: Path, ctx_no_repo: RunContext[AgentDeps]
) -> None:
    """Removing a synced comment (has github_id) creates a tombstone."""
    draft = ReviewDraft(
        summary="",
        comments=[
            InlineComment(path="a.py", line=1, body="Synced.", github_id=100, comment_hash="abc"),
        ],
    )
    save_draft(42, draft)

    result = remove_draft_comment(ctx_no_repo, "a.py", "Synced")
    assert "Removed" in result

    loaded = load_draft(42)
    assert loaded is not None
    # Comment is kept as a tombstone (empty body, github_id preserved).
    assert len(loaded.comments) == 1
    assert loaded.comments[0].body == ""
    assert loaded.comments[0].suggestion == ""
    assert loaded.comments[0].github_id == 100


def test_remove_empty_draft(workspace: Path, ctx_no_repo: RunContext[AgentDeps]) -> None:
    result = remove_draft_comment(ctx_no_repo, "a.py", "anything")
    assert "no comments" in result


# ── set_draft_summary ───────────────────────────────────────────────


def test_set_summary(workspace: Path, ctx_no_repo: RunContext[AgentDeps]) -> None:
    result = set_draft_summary(ctx_no_repo, "Great PR overall.")
    assert "updated" in result

    draft = load_draft(42)
    assert draft is not None
    assert draft.summary == "Great PR overall."


def test_set_summary_preserves_comments(
    workspace: Path, ctx_no_repo: RunContext[AgentDeps], seeded_draft: ReviewDraft
) -> None:
    set_draft_summary(ctx_no_repo, "New summary.")

    draft = load_draft(42)
    assert draft is not None
    assert draft.summary == "New summary."
    assert len(draft.comments) == 2


def test_set_summary_overwrites(workspace: Path, ctx_no_repo: RunContext[AgentDeps]) -> None:
    set_draft_summary(ctx_no_repo, "First.")
    set_draft_summary(ctx_no_repo, "Second.")

    draft = load_draft(42)
    assert draft is not None
    assert draft.summary == "Second."


# ── read_draft ──────────────────────────────────────────────────────


def test_read_draft_no_draft(workspace: Path, ctx_no_repo: RunContext[AgentDeps]) -> None:
    assert read_draft(ctx_no_repo) == "No draft yet."


def test_read_draft_returns_raw_yaml(workspace: Path, ctx_no_repo: RunContext[AgentDeps]) -> None:
    """read_draft returns the raw YAML file content, not a formatted view."""
    save_draft(
        42,
        ReviewDraft(
            summary="Looks good overall.",
            comments=[
                InlineComment(path="src/api.py", line=10, body="Fix this."),
                InlineComment(
                    path="src/api.py",
                    line=20,
                    body="Consider using a context manager.",
                    suggestion="with open(f) as h:",
                ),
            ],
        ),
    )
    result = read_draft(ctx_no_repo)
    # Raw YAML contains the field names as keys.
    assert "summary:" in result
    assert "comments:" in result
    assert "path: src/api.py" in result
    assert "body:" in result
    # Full body text present, not truncated.
    assert "Fix this." in result
    assert "Consider using a context manager." in result
    assert "with open(f) as h:" in result


def test_read_draft_roundtrip_with_edit(
    workspace: Path, ctx_no_repo: RunContext[AgentDeps]
) -> None:
    """Body text from read_draft can be used as edit_draft_comment's comment arg."""
    save_draft(
        42,
        ReviewDraft(
            summary="Review.",
            comments=[
                InlineComment(path="a.py", line=5, body="This function is too long."),
                InlineComment(path="b.py", line=8, body="Missing error handling."),
            ],
        ),
    )
    raw = read_draft(ctx_no_repo)

    # Pick a substring from the raw output to use as the comment locator.
    assert "too long" in raw
    result = edit_draft_comment(ctx_no_repo, "a.py", "too long", body="Split into helpers.")
    assert "updated" in result.lower()

    draft = load_draft(42)
    assert draft is not None
    assert draft.comments[0].body == "Split into helpers."
    # Other comment untouched.
    assert draft.comments[1].body == "Missing error handling."


def test_read_draft_pagination(
    workspace: Path, ctx_no_repo: RunContext[AgentDeps], monkeypatch: pytest.MonkeyPatch
) -> None:
    """read_draft paginates with offset/max_lines."""
    save_draft(
        42,
        ReviewDraft(
            summary="A multiline summary.\nWith two lines.",
            comments=[
                InlineComment(path="a.py", line=1, body="Comment A."),
                InlineComment(path="b.py", line=2, body="Comment B."),
                InlineComment(path="c.py", line=3, body="Comment C."),
            ],
        ),
    )
    # Read with a small page to force truncation.
    page1 = read_draft(ctx_no_repo, max_lines=3)
    assert "... limited" in page1
    assert "offset=" in page1

    # Second page has content.
    assert read_draft(ctx_no_repo, offset=3, max_lines=3).strip()

    # Full file fits with a large limit.
    from rbtr.github.draft import draft_path

    total_lines = len(draft_path(42).read_text().splitlines())
    full = read_draft(ctx_no_repo, max_lines=total_lines + 10)
    assert "... limited" not in full


def test_read_draft_offset_exceeds(workspace: Path, ctx_no_repo: RunContext[AgentDeps]) -> None:
    save_draft(42, ReviewDraft(summary="Short."))
    result = read_draft(ctx_no_repo, offset=9999)
    assert "exceeds" in result.lower()


def test_read_draft_multiline_body_preserved(
    workspace: Path, ctx_no_repo: RunContext[AgentDeps]
) -> None:
    """Multiline markdown bodies are preserved exactly in read_draft output."""
    body = "First paragraph.\n\nSecond paragraph with `code`.\n\n- bullet 1\n- bullet 2"
    save_draft(
        42,
        ReviewDraft(
            comments=[InlineComment(path="x.py", line=1, body=body)],
        ),
    )
    raw = read_draft(ctx_no_repo, max_lines=1000)
    # The body is present in full — check key parts.
    assert "First paragraph." in raw
    assert "Second paragraph with `code`." in raw
    assert "- bullet 1" in raw
    assert "- bullet 2" in raw


def test_read_draft_content_survives_full_lifecycle(
    workspace: Path, ctx: RunContext[AgentDeps]
) -> None:
    """add → read → edit → read — content is consistent throughout."""
    # Add via tool.
    add_draft_comment(ctx, "src/handler.py", "validate(request)", "Bug here.")
    raw1 = read_draft(ctx)
    assert "Bug here." in raw1

    # Edit via tool using substring from raw output.
    edit_draft_comment(ctx, "src/handler.py", "Bug here", body="Not a bug, my mistake.")
    raw2 = read_draft(ctx)
    assert "Not a bug, my mistake." in raw2
    assert "Bug here." not in raw2


# ── Anchor comments — side + commit_id ───────────────────────────────
#
# These tests use `tool_ctx` from conftest (not the local `ctx`)
# because they need the conftest `draft_repo` which has a deleted
# file (readme.md) for LEFT-side anchor testing.


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
    assert c.line == 2


def test_add_head_anchor_shifted_line(workspace: Path, tool_ctx: RunContext[AgentDeps]) -> None:
    """Same code at different line numbers on each side."""
    add_draft_comment(tool_ctx, "src/handler.py", "return 'ok'", "Now line 3.")
    draft = load_draft(42)
    assert draft is not None
    assert draft.comments[0].line == 3


def test_add_base_anchor_deleted_file(workspace: Path, tool_ctx: RunContext[AgentDeps]) -> None:
    """readme.md deleted at head — can still comment on base side."""
    result = add_draft_comment(tool_ctx, "readme.md", "# Project", "Why deleted?", ref="base")
    assert "Comment added" in result
    draft = load_draft(42)
    assert draft is not None
    assert draft.comments[0].side == "LEFT"
    assert draft.comments[0].path == "readme.md"


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
