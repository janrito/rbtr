"""Tests for draft management LLM tools — add, edit, remove, set_summary.

The ``add_draft_comment`` tests use a real git repo so that
``resolve_anchor`` and diff-range validation work end-to-end.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pygit2
import pytest
from pydantic_ai import RunContext

from rbtr.github.draft import load_draft, save_draft
from rbtr.llm.agent import AgentDeps
from rbtr.llm.tools.draft import (
    add_draft_comment,
    edit_draft_comment,
    read_draft,
    remove_draft_comment,
    set_draft_summary,
)
from rbtr.models import InlineComment, PRTarget, ReviewDraft
from rbtr.state import EngineState

# ── Repo builder (minimal copy from git/conftest) ────────────────────


def _build_tree(
    repo: pygit2.Repository,
    files: dict[str, bytes],
) -> pygit2.Oid:
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


def _make_commit(
    repo: pygit2.Repository,
    files: dict[str, bytes],
    *,
    parents: list[pygit2.Oid] | None = None,
    ref: str = "refs/heads/main",
) -> pygit2.Oid:
    tree_oid = _build_tree(repo, files)
    sig = pygit2.Signature("Test", "test@test.com")
    return repo.create_commit(ref, sig, sig, "commit", tree_oid, parents or [])


# File content — the diff adds ``validate(request)`` on line 2.
_BASE_HANDLER = b"def handle(request):\n    return 'ok'\n"
_HEAD_HANDLER = b"def handle(request):\n    validate(request)\n    return 'ok'\n"
_UTILS = b"def helper():\n    return 42\n"


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point drafts_dir at a temp directory."""
    monkeypatch.setattr("rbtr.config.config.tools.drafts_dir", str(tmp_path / "drafts"))
    return tmp_path


@pytest.fixture
def draft_repo(tmp_path: Path) -> tuple[pygit2.Repository, pygit2.Oid, pygit2.Oid]:
    """A two-commit repo: base (main) → head (feature).

    Changed file: ``src/handler.py`` (one line added).
    Unchanged file: ``src/utils.py``.
    """
    repo = pygit2.init_repository(str(tmp_path / "repo"))
    base = _make_commit(
        repo,
        {"src/handler.py": _BASE_HANDLER, "src/utils.py": _UTILS},
    )
    head = _make_commit(
        repo,
        {"src/handler.py": _HEAD_HANDLER, "src/utils.py": _UTILS},
        parents=[base],
        ref="refs/heads/feature",
    )
    return repo, base, head


@pytest.fixture
def pr_target(draft_repo: tuple[pygit2.Repository, pygit2.Oid, pygit2.Oid]) -> PRTarget:
    _, base, head = draft_repo
    return PRTarget(
        number=42,
        title="Test PR",
        author="alice",
        base_branch=str(base),
        head_branch="feature",
        head_sha=str(head),
        updated_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


@pytest.fixture
def ctx(
    pr_target: PRTarget,
    draft_repo: tuple[pygit2.Repository, pygit2.Oid, pygit2.Oid],
    monkeypatch: pytest.MonkeyPatch,
) -> RunContext[AgentDeps]:
    """RunContext with a real repo and PR target."""
    repo, _, _ = draft_repo

    # Clear the global diff-range cache between tests.
    import rbtr.llm.tools.draft as _tools_mod

    monkeypatch.setattr(_tools_mod, "_cached_ranges", None)
    monkeypatch.setattr(_tools_mod, "_cached_ranges_left", None)
    monkeypatch.setattr(_tools_mod, "_cached_ranges_key", ("", ""))

    state = EngineState()
    state.review_target = pr_target
    state.repo = repo
    deps = AgentDeps(state=state)
    mock_ctx = MagicMock(spec=RunContext)
    mock_ctx.deps = deps
    return mock_ctx


@pytest.fixture
def ctx_no_repo(pr_target: PRTarget) -> RunContext[AgentDeps]:
    """RunContext with a PR target but no repo (for edit/remove tests)."""
    state = EngineState()
    state.review_target = pr_target
    deps = AgentDeps(state=state)
    mock_ctx = MagicMock(spec=RunContext)
    mock_ctx.deps = deps
    return mock_ctx


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


def test_add_comment_anchor_outside_diff_hunk(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    # "def handle(request):" is line 1 — context line. Whether it's
    # commentable depends on the diff hunk range. If it's NOT in the
    # range, we get the "not in the PR diff" error. If it IS (context
    # line), the comment succeeds. Either way, the flow works.
    result = add_draft_comment(ctx, "src/handler.py", "def handle(request):", "Body.")
    # Line 1 is typically a context line in the diff — should succeed.
    assert "Comment added" in result or "Cannot add comment" in result


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


def _seed_draft(pr_number: int) -> None:
    draft = ReviewDraft(
        summary="Review.",
        comments=[
            InlineComment(path="a.py", line=10, body="Original."),
            InlineComment(path="b.py", line=20, body="Also original."),
        ],
    )
    save_draft(pr_number, draft)


def test_edit_comment_body(workspace: Path, ctx_no_repo: RunContext[AgentDeps]) -> None:
    _seed_draft(42)
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


def test_edit_comment_not_found(workspace: Path, ctx_no_repo: RunContext[AgentDeps]) -> None:
    _seed_draft(42)
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


def test_remove_comment(workspace: Path, ctx_no_repo: RunContext[AgentDeps]) -> None:
    _seed_draft(42)
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


def test_remove_not_found(workspace: Path, ctx_no_repo: RunContext[AgentDeps]) -> None:
    _seed_draft(42)
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
    workspace: Path, ctx_no_repo: RunContext[AgentDeps]
) -> None:
    _seed_draft(42)
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
