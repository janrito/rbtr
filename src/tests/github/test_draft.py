"""Tests for rbtr.github.draft — persistence, matching, and sync status."""

from pathlib import Path

import pytest

from rbtr.github.draft import (
    _comment_hash,
    comment_sync_status,
    delete_draft,
    load_draft,
    match_comments,
    save_draft,
    stamp_synced,
)
from rbtr.models import InlineComment, ReviewDraft

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


def _h(c: InlineComment) -> str:
    """Shortcut for _comment_hash in tests."""
    return _comment_hash(c)


def _synced(c: InlineComment) -> InlineComment:
    """Return a copy with comment_hash set (as if after a push)."""
    return c.model_copy(update={"comment_hash": _h(c)})


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point drafts_dir at a temp directory."""
    drafts = tmp_path / "drafts"
    monkeypatch.setattr("rbtr.config.config.tools.drafts_dir", str(drafts))
    return drafts


# ── Roundtrip ────────────────────────────────────────────────────────


def test_save_and_load_roundtrip(workspace: Path) -> None:
    save_draft(99, DRAFT)
    loaded = load_draft(99)
    assert loaded is not None
    assert loaded.summary == DRAFT.summary
    assert len(loaded.comments) == len(DRAFT.comments)
    assert loaded.comments[0].body == COMMENT_A.body
    assert loaded.comments[1].suggestion == COMMENT_B.suggestion


def test_load_nonexistent_returns_none(workspace: Path) -> None:
    assert load_draft(999) is None


def test_save_creates_parent_dirs(workspace: Path) -> None:
    save_draft(1, DRAFT)
    assert (workspace / "1.yaml").exists()


def test_yaml_is_human_readable(workspace: Path) -> None:
    save_draft(42, DRAFT)
    content = (workspace / "42.yaml").read_text()
    assert "src/handler.py" in content
    assert "- path:" in content
    assert "This will throw on an empty list." in content


def test_roundtrip_preserves_suggestion(workspace: Path) -> None:
    save_draft(1, DRAFT)
    loaded = load_draft(1)
    assert loaded is not None
    assert loaded.comments[1].suggestion == COMMENT_B.suggestion


def test_save_overwrites_existing(workspace: Path) -> None:
    save_draft(1, DRAFT)
    updated = DRAFT.model_copy(update={"summary": "Revised."})
    save_draft(1, updated)
    loaded = load_draft(1)
    assert loaded is not None
    assert loaded.summary == "Revised."


def test_roundtrip_preserves_github_id(workspace: Path) -> None:
    c = COMMENT_A.model_copy(update={"github_id": 12345})
    draft = ReviewDraft(comments=[c])
    save_draft(1, draft)
    loaded = load_draft(1)
    assert loaded is not None
    assert loaded.comments[0].github_id == 12345


def test_roundtrip_preserves_sync_fields(workspace: Path) -> None:
    c = InlineComment(path="a.py", line=5, body="Fix.", github_id=100, comment_hash="abc123")
    draft = ReviewDraft(
        summary="Old summary.",
        comments=[c],
        github_review_id=99,
        summary_hash="def456",
    )
    save_draft(1, draft)
    loaded = load_draft(1)
    assert loaded is not None
    assert loaded.github_review_id == 99
    assert loaded.summary_hash == "def456"
    assert loaded.comments[0].comment_hash == "abc123"


# ── Delete ───────────────────────────────────────────────────────────


def test_delete_existing(workspace: Path) -> None:
    save_draft(1, DRAFT)
    assert delete_draft(1) is True
    assert load_draft(1) is None


def test_delete_nonexistent(workspace: Path) -> None:
    assert delete_draft(999) is False


# ── Matching: tier 1 (github_id) ────────────────────────────────────


def test_match_by_github_id() -> None:
    """Tier 1: local and remote share the same github_id."""
    c = _synced(InlineComment(path="a.py", line=10, body="Local.", github_id=100))
    local = [c]
    remote = [InlineComment(path="a.py", line=10, body="Local.", github_id=100)]

    result = match_comments(local, remote)
    assert len(result.comments) == 1
    assert result.comments[0].github_id == 100
    assert result.warnings == []


def test_match_detects_remote_edit() -> None:
    """Remote body changed, local clean → accept remote edit."""
    original = _synced(InlineComment(path="a.py", line=10, body="Original.", github_id=100))
    local = [original]
    remote = [InlineComment(path="a.py", line=10, body="Edited on GitHub.", github_id=100)]

    result = match_comments(local, remote)
    assert result.comments[0].body == "Edited on GitHub."
    assert result.warnings == []


def test_match_keeps_local_edit() -> None:
    """Local body changed, remote unchanged → keep local."""
    original = InlineComment(path="a.py", line=10, body="Original.", github_id=100)
    synced_h = _h(original)
    local = [
        InlineComment(
            path="a.py",
            line=10,
            body="Edited locally.",
            github_id=100,
            comment_hash=synced_h,
        )
    ]
    remote = [original.model_copy(update={"github_id": 100})]

    result = match_comments(local, remote)
    assert result.comments[0].body == "Edited locally."
    assert result.warnings == []


def test_match_conflict_keeps_local() -> None:
    """Both sides changed → conflict, keep local, warn."""
    original = InlineComment(path="a.py", line=10, body="Original.", github_id=100)
    synced_h = _h(original)
    local = [
        InlineComment(
            path="a.py",
            line=10,
            body="Local edit.",
            github_id=100,
            comment_hash=synced_h,
        )
    ]
    remote = [InlineComment(path="a.py", line=10, body="Remote edit.", github_id=100)]

    result = match_comments(local, remote)
    assert result.comments[0].body == "Local edit."
    assert len(result.warnings) == 1
    assert "Conflict" in result.warnings[0]
    assert "Remote was:" in result.warnings[0]


def test_match_remote_deletion() -> None:
    """Local has github_id with comment_hash, but remote doesn't → deleted."""
    c = _synced(InlineComment(path="a.py", line=10, body="Deleted.", github_id=100))
    local = [c]
    remote: list[InlineComment] = []

    result = match_comments(local, remote)
    assert len(result.comments) == 0
    assert any("deleted on GitHub" in w for w in result.warnings)


# ── Matching: tier 2 (content) ───────────────────────────────────────


def test_match_by_content() -> None:
    """Tier 2: local has no github_id, matches remote by content."""
    local = [InlineComment(path="a.py", line=10, body="Fix this.")]
    remote = [InlineComment(path="a.py", line=10, body="Fix this.", github_id=200)]

    result = match_comments(local, remote)
    assert len(result.comments) == 1
    assert result.comments[0].github_id == 200


def test_match_by_content_with_suggestion() -> None:
    """Content match includes suggestion block in comparison."""
    local = [InlineComment(path="a.py", line=10, body="Use this.", suggestion="better()")]
    remote = [
        InlineComment(
            path="a.py",
            line=10,
            body="Use this.",
            suggestion="better()",
            github_id=300,
        )
    ]

    result = match_comments(local, remote)
    assert result.comments[0].github_id == 300


def test_content_match_ambiguous_skipped() -> None:
    """If multiple remote comments have same content, don't match."""
    local = [InlineComment(path="a.py", line=10, body="Fix.")]
    remote = [
        InlineComment(path="a.py", line=10, body="Fix.", github_id=100),
        InlineComment(path="a.py", line=10, body="Fix.", github_id=101),
    ]

    result = match_comments(local, remote)
    # Local kept without github_id, both remotes imported.
    assert len(result.comments) == 3
    gids = {c.github_id for c in result.comments}
    assert None in gids  # original local
    assert 100 in gids
    assert 101 in gids


# ── Matching: unmatched ──────────────────────────────────────────────


def test_new_remote_imported() -> None:
    """Remote comment with no local match → imported."""
    local: list[InlineComment] = []
    remote = [InlineComment(path="b.py", line=20, body="New remote.", github_id=500)]

    result = match_comments(local, remote)
    assert len(result.comments) == 1
    assert result.comments[0].github_id == 500
    assert result.comments[0].body == "New remote."
    assert any("imported" in w for w in result.warnings)


def test_new_local_kept() -> None:
    """Local comment without github_id and no match → kept as new."""
    local = [InlineComment(path="c.py", line=5, body="New local.")]
    remote: list[InlineComment] = []

    result = match_comments(local, remote)
    assert len(result.comments) == 1
    assert result.comments[0].github_id is None


def test_mixed_match_and_import() -> None:
    """Mix of matched, locally-new, and remotely-new comments."""
    matched = _synced(InlineComment(path="a.py", line=10, body="Matched.", github_id=100))
    local = [
        matched,
        InlineComment(path="b.py", line=20, body="Locally new."),
    ]
    remote = [
        InlineComment(path="a.py", line=10, body="Matched.", github_id=100),
        InlineComment(path="c.py", line=30, body="Remotely new.", github_id=200),
    ]

    result = match_comments(local, remote)
    assert len(result.comments) == 3
    bodies = {c.body for c in result.comments}
    assert bodies == {"Matched.", "Locally new.", "Remotely new."}


# ── stamp_synced ─────────────────────────────────────────────────────


def test_stamp_synced() -> None:
    comments = [
        InlineComment(path="a.py", line=10, body="Fix.", github_id=100),
        InlineComment(path="b.py", line=20, body="Nit.", suggestion="better()", github_id=200),
        InlineComment(path="c.py", line=30, body="New."),  # no github_id
    ]
    draft = ReviewDraft(summary="Summary.", comments=comments)
    stamped = stamp_synced(draft)

    assert stamped.summary_hash != ""
    # Every comment gets a comment_hash, even those without github_id.
    for c in stamped.comments:
        assert c.comment_hash == _h(c)


# ── comment_sync_status ──────────────────────────────────────────────


def test_status_new() -> None:
    c = InlineComment(path="a.py", line=1, body="New.")
    assert comment_sync_status(c) == "★"


def test_status_clean() -> None:
    c = _synced(InlineComment(path="a.py", line=1, body="Clean.", github_id=100))
    assert comment_sync_status(c) == "✓"


def test_status_dirty_body() -> None:
    original = InlineComment(path="a.py", line=1, body="Original.", github_id=100)
    edited = InlineComment(
        path="a.py", line=1, body="Edited.", github_id=100, comment_hash=_h(original)
    )
    assert comment_sync_status(edited) == "✎"


def test_status_dirty_line() -> None:
    original = InlineComment(path="a.py", line=1, body="Same.", github_id=100)
    moved = InlineComment(
        path="a.py", line=99, body="Same.", github_id=100, comment_hash=_h(original)
    )
    assert comment_sync_status(moved) == "✎"


# ── _comment_hash ────────────────────────────────────────────────────


def test_hash_deterministic() -> None:
    c = InlineComment(path="a.py", line=1, body="Fix.")
    assert _h(c) == _h(c)


def test_hash_differs_on_body() -> None:
    a = InlineComment(path="a.py", line=1, body="Fix.")
    b = InlineComment(path="a.py", line=1, body="Different.")
    assert _h(a) != _h(b)


def test_hash_differs_on_line() -> None:
    a = InlineComment(path="a.py", line=1, body="Fix.")
    b = InlineComment(path="a.py", line=2, body="Fix.")
    assert _h(a) != _h(b)
