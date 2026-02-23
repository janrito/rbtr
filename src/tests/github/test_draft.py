"""Tests for rbtr.github.draft — draft persistence and merge logic."""

from pathlib import Path

import pytest

from rbtr.github.draft import (
    delete_draft,
    get_unsynced_comments,
    load_draft,
    merge_remote,
    save_draft,
)
from rbtr.models import InlineComment, ReviewDraft, ReviewEvent

# ── Test data ────────────────────────────────────────────────────────

COMMENT_A = InlineComment(
    path="src/handler.py",
    line=42,
    body="This will throw on an empty list.",
)

COMMENT_B = InlineComment(
    path="src/handler.py",
    line=87,
    body="Consider extracting this.",
    suggestion="def _parse(raw):\n    return Item(raw)",
)

COMMENT_C = InlineComment(
    path="src/utils.py",
    line=10,
    body="Unused import.",
)

DRAFT = ReviewDraft(
    summary="Overall good, a few issues.",
    comments=[COMMENT_A, COMMENT_B],
)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point WORKSPACE_DIR at a temp directory."""
    monkeypatch.setattr("rbtr.github.draft.WORKSPACE_DIR", tmp_path)
    return tmp_path


# ── Roundtrip ────────────────────────────────────────────────────────


def test_save_and_load_roundtrip(workspace: Path) -> None:
    save_draft(99, DRAFT)
    loaded = load_draft(99)
    assert loaded is not None
    assert loaded == DRAFT


def test_load_nonexistent_returns_none(workspace: Path) -> None:
    assert load_draft(999) is None


def test_save_creates_parent_dirs(workspace: Path) -> None:
    save_draft(1, DRAFT)
    assert (workspace / "REVIEW-DRAFT-1.toml").exists()


def test_toml_is_human_readable(workspace: Path) -> None:
    save_draft(42, DRAFT)
    content = (workspace / "REVIEW-DRAFT-42.toml").read_text()
    assert "src/handler.py" in content
    assert "[[comments]]" in content
    assert "This will throw on an empty list." in content


def test_roundtrip_preserves_suggestion(workspace: Path) -> None:
    save_draft(1, DRAFT)
    loaded = load_draft(1)
    assert loaded is not None
    assert loaded.comments[1].suggestion == COMMENT_B.suggestion


def test_roundtrip_preserves_event(workspace: Path) -> None:
    draft = DRAFT.model_copy(update={"event": ReviewEvent.REQUEST_CHANGES})
    save_draft(1, draft)
    loaded = load_draft(1)
    assert loaded is not None
    assert loaded.event == ReviewEvent.REQUEST_CHANGES


def test_save_overwrites_existing(workspace: Path) -> None:
    save_draft(1, DRAFT)
    updated = DRAFT.model_copy(update={"summary": "Revised."})
    save_draft(1, updated)
    loaded = load_draft(1)
    assert loaded is not None
    assert loaded.summary == "Revised."


# ── Delete ───────────────────────────────────────────────────────────


def test_delete_existing(workspace: Path) -> None:
    save_draft(1, DRAFT)
    assert delete_draft(1) is True
    assert load_draft(1) is None


def test_delete_nonexistent(workspace: Path) -> None:
    assert delete_draft(999) is False


# ── Merge ────────────────────────────────────────────────────────────


def test_merge_into_none_creates_draft() -> None:
    result = merge_remote(None, [COMMENT_A, COMMENT_C])
    assert len(result.comments) == 2
    assert result.summary == ""


def test_merge_appends_new_comments() -> None:
    result = merge_remote(DRAFT, [COMMENT_C])
    assert len(result.comments) == 3
    assert result.comments[-1] == COMMENT_C


def test_merge_skips_duplicates() -> None:
    """Comments with the same (path, line) as local are not appended."""
    remote_duplicate = InlineComment(
        path=COMMENT_A.path,
        line=COMMENT_A.line,
        body="Different body, same location.",
    )
    result = merge_remote(DRAFT, [remote_duplicate])
    assert len(result.comments) == len(DRAFT.comments)
    # Local body is preserved, not overwritten.
    match_comment = next(
        c for c in result.comments if c.path == COMMENT_A.path and c.line == COMMENT_A.line
    )
    assert match_comment.body == COMMENT_A.body


def test_merge_preserves_local_summary() -> None:
    result = merge_remote(DRAFT, [COMMENT_C])
    assert result.summary == DRAFT.summary


def test_merge_no_new_returns_same_draft() -> None:
    result = merge_remote(DRAFT, [COMMENT_A])
    assert result is DRAFT


def test_merge_mixed_new_and_duplicate() -> None:
    remote_dup = InlineComment(
        path=COMMENT_A.path,
        line=COMMENT_A.line,
        body="Dup.",
    )
    result = merge_remote(DRAFT, [remote_dup, COMMENT_C])
    assert len(result.comments) == 3
    assert result.comments[-1] == COMMENT_C


# ── get_unsynced_comments ────────────────────────────────────────────


def test_unsynced_empty_when_all_present() -> None:
    """Remote comments that match local (path, line) are not unsynced."""
    remote = [InlineComment(path=COMMENT_A.path, line=COMMENT_A.line, body="Remote version.")]
    assert get_unsynced_comments(DRAFT, remote) == []


def test_unsynced_detects_missing() -> None:
    """Remote comments at new (path, line) are returned as unsynced."""
    remote = [
        InlineComment(path=COMMENT_A.path, line=COMMENT_A.line, body="Known."),
        COMMENT_C,
    ]
    unsynced = get_unsynced_comments(DRAFT, remote)
    assert len(unsynced) == 1
    assert unsynced[0] == COMMENT_C


def test_unsynced_empty_draft() -> None:
    """All remote comments are unsynced when the local draft is empty."""
    empty = ReviewDraft(summary="", comments=[])
    unsynced = get_unsynced_comments(empty, [COMMENT_A])
    assert len(unsynced) == 1
