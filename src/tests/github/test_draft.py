"""Tests for rbtr.github.draft — persistence, matching, and sync status."""

from pathlib import Path

import pytest

from rbtr.engine.draft_cmd import _resolve_event
from rbtr.git.objects import DiffLineRanges
from rbtr.github.client import format_comment_body, parse_comment_body
from rbtr.github.draft import (
    _comment_hash,
    comment_sync_status,
    delete_draft,
    is_tombstone,
    load_draft,
    match_comments,
    partition_comments,
    save_draft,
    stamp_synced,
)
from rbtr.models import InlineComment, ReviewDraft, ReviewEvent

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def comment_a() -> InlineComment:
    return InlineComment(
        path="src/handler.py",
        line=42,
        body="This will throw on an empty list.",
    )


@pytest.fixture
def comment_b() -> InlineComment:
    return InlineComment(
        path="src/handler.py",
        line=87,
        body="Consider extracting this.",
        suggestion="def _parse(raw):\n    return Item(raw)",
    )


@pytest.fixture
def draft(comment_a: InlineComment, comment_b: InlineComment) -> ReviewDraft:
    return ReviewDraft(
        summary="Overall good, a few issues.",
        comments=[comment_a, comment_b],
    )


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point drafts_dir at a temp directory."""
    drafts = tmp_path / "drafts"
    monkeypatch.setattr("rbtr.config.config.tools.drafts_dir", str(drafts))
    return drafts


# ── Roundtrip ────────────────────────────────────────────────────────


def test_save_and_load_roundtrip(
    workspace: Path, comment_a: InlineComment, comment_b: InlineComment, draft: ReviewDraft
) -> None:
    save_draft(99, draft)
    loaded = load_draft(99)
    assert loaded is not None
    assert loaded.summary == draft.summary
    assert len(loaded.comments) == len(draft.comments)
    assert loaded.comments[0].body == comment_a.body
    assert loaded.comments[1].suggestion == comment_b.suggestion


def test_load_nonexistent_returns_none(workspace: Path, draft: ReviewDraft) -> None:
    assert load_draft(999) is None


def test_save_creates_parent_dirs(
    workspace: Path, comment_b: InlineComment, draft: ReviewDraft
) -> None:
    save_draft(1, draft)
    assert (workspace / "1.yaml").exists()


def test_yaml_is_human_readable(
    workspace: Path, comment_b: InlineComment, draft: ReviewDraft
) -> None:
    save_draft(42, draft)
    content = (workspace / "42.yaml").read_text()
    assert "src/handler.py" in content
    assert "- path:" in content
    assert "This will throw on an empty list." in content


def test_roundtrip_preserves_suggestion(
    workspace: Path, comment_a: InlineComment, comment_b: InlineComment, draft: ReviewDraft
) -> None:
    save_draft(1, draft)
    loaded = load_draft(1)
    assert loaded is not None
    assert loaded.comments[1].suggestion == comment_b.suggestion


def test_save_overwrites_existing(
    workspace: Path, comment_a: InlineComment, draft: ReviewDraft
) -> None:
    save_draft(1, draft)
    updated = draft.model_copy(update={"summary": "Revised."})
    save_draft(1, updated)
    loaded = load_draft(1)
    assert loaded is not None
    assert loaded.summary == "Revised."


def test_roundtrip_preserves_github_id(
    workspace: Path, comment_a: InlineComment, draft: ReviewDraft
) -> None:
    c = comment_a.model_copy(update={"github_id": 12345})
    draft = ReviewDraft(comments=[c])
    save_draft(1, draft)
    loaded = load_draft(1)
    assert loaded is not None
    assert loaded.comments[0].github_id == 12345


def test_roundtrip_preserves_sync_fields(workspace: Path, draft: ReviewDraft) -> None:
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


def test_delete_existing(workspace: Path, draft: ReviewDraft) -> None:
    save_draft(1, draft)
    assert delete_draft(1) is True
    assert load_draft(1) is None


def test_delete_nonexistent(workspace: Path) -> None:
    assert delete_draft(999) is False


# ── Matching: tier 1 (github_id) ────────────────────────────────────


def test_match_by_github_id() -> None:
    """Tier 1: local and remote share the same github_id."""
    c = InlineComment(path="a.py", line=10, body="Local.", github_id=100)
    c = c.model_copy(update={"comment_hash": _comment_hash(c)})
    local = [c]
    remote = [InlineComment(path="a.py", line=10, body="Local.", github_id=100)]

    result = match_comments(local, remote)
    assert len(result.comments) == 1
    assert result.comments[0].github_id == 100
    assert result.warnings == []


def test_match_detects_remote_edit() -> None:
    """Remote body changed, local clean → accept remote edit."""
    original = InlineComment(path="a.py", line=10, body="Original.", github_id=100)
    original = original.model_copy(update={"comment_hash": _comment_hash(original)})
    local = [original]
    remote = [InlineComment(path="a.py", line=10, body="Edited on GitHub.", github_id=100)]

    result = match_comments(local, remote)
    assert result.comments[0].body == "Edited on GitHub."
    assert result.warnings == []


def test_match_keeps_local_edit() -> None:
    """Local body changed, remote unchanged → keep local."""
    original = InlineComment(path="a.py", line=10, body="Original.", github_id=100)
    synced_h = _comment_hash(original)
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
    synced_h = _comment_hash(original)
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
    c = InlineComment(path="a.py", line=10, body="Deleted.", github_id=100)
    c = c.model_copy(update={"comment_hash": _comment_hash(c)})
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
    matched = InlineComment(path="a.py", line=10, body="Matched.", github_id=100)
    matched = matched.model_copy(update={"comment_hash": _comment_hash(matched)})
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


def test_stamp_synced(draft: ReviewDraft) -> None:
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
        assert c.comment_hash == _comment_hash(c)


# ── comment_sync_status ──────────────────────────────────────────────


def test_status_new() -> None:
    c = InlineComment(path="a.py", line=1, body="New.")
    assert comment_sync_status(c) == "★"


def test_status_clean() -> None:
    c = InlineComment(path="a.py", line=1, body="Clean.", github_id=100)
    c = c.model_copy(update={"comment_hash": _comment_hash(c)})
    assert comment_sync_status(c) == "✓"


def test_status_dirty_body() -> None:
    original = InlineComment(path="a.py", line=1, body="Original.", github_id=100)
    edited = InlineComment(
        path="a.py", line=1, body="Edited.", github_id=100, comment_hash=_comment_hash(original)
    )
    assert comment_sync_status(edited) == "✎"


def test_status_dirty_line() -> None:
    original = InlineComment(path="a.py", line=1, body="Same.", github_id=100)
    moved = InlineComment(
        path="a.py", line=99, body="Same.", github_id=100, comment_hash=_comment_hash(original)
    )
    assert comment_sync_status(moved) == "✎"


# ── _comment_hash ────────────────────────────────────────────────────


def test_hash_deterministic() -> None:
    c = InlineComment(path="a.py", line=1, body="Fix.")
    assert _comment_hash(c) == _comment_hash(c)


def test_hash_differs_on_body(comment_b: InlineComment) -> None:
    a = InlineComment(path="a.py", line=1, body="Fix.")
    b = InlineComment(path="a.py", line=1, body="Different.")
    assert _comment_hash(a) != _comment_hash(b)


def test_hash_differs_on_line(comment_b: InlineComment) -> None:
    a = InlineComment(path="a.py", line=1, body="Fix.")
    b = InlineComment(path="a.py", line=2, body="Fix.")
    assert _comment_hash(a) != _comment_hash(b)


def test_hash_excludes_side_and_commit_id(comment_b: InlineComment) -> None:
    base = InlineComment(path="a.py", line=1, body="Fix.")
    with_side = base.model_copy(update={"side": "LEFT", "commit_id": "abc123"})
    assert _comment_hash(base) == _comment_hash(with_side)


# ── parse_comment_body / format_comment_body ─────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected_body", "expected_suggestion"),
    [
        ("Fix this bug.", "Fix this bug.", ""),
        (
            "Use this instead.\n\n```suggestion\nbetter()\n```",
            "Use this instead.",
            "better()",
        ),
        (
            "Fix.\n\n```suggestion\nline1\nline2\n```",
            "Fix.",
            "line1\nline2",
        ),
        (
            "Fix.\n\n```suggestion\norphan code",
            "Fix.",
            "orphan code",
        ),
    ],
    ids=["plain", "single-line", "multiline", "unclosed-fence"],
)
def test_parse_comment_body(raw: str, expected_body: str, expected_suggestion: str) -> None:
    body, suggestion = parse_comment_body(raw)
    assert body == expected_body
    assert suggestion == expected_suggestion


@pytest.mark.parametrize(
    ("body", "suggestion", "expected_contains"),
    [
        ("Fix this.", "", "Fix this."),
        ("Use this.", "better()", "```suggestion\nbetter()\n```"),
    ],
    ids=["plain", "with-suggestion"],
)
def test_format_comment_body(body: str, suggestion: str, expected_contains: str) -> None:
    c = InlineComment(path="a.py", line=1, body=body, suggestion=suggestion)
    result = format_comment_body(c)
    assert expected_contains in result


def test_format_comment_body_empty_suggestion() -> None:
    c = InlineComment(path="a.py", line=1, body="Comment.", suggestion="")
    assert format_comment_body(c) == "Comment."


def test_format_and_parse_roundtrip_plain(comment_b: InlineComment) -> None:
    c = InlineComment(path="a.py", line=1, body="Fix this.")
    body, suggestion = parse_comment_body(format_comment_body(c))
    assert body == "Fix this."
    assert suggestion == ""


def test_format_and_parse_roundtrip_with_suggestion(comment_b: InlineComment) -> None:
    c = InlineComment(path="a.py", line=1, body="Use this.", suggestion="better()")
    body, suggestion = parse_comment_body(format_comment_body(c))
    assert body == "Use this."
    assert suggestion == "better()"


# ── partition_comments ───────────────────────────────────────────────


def test_partition_empty_ranges_reject_line_comments() -> None:
    comments = [InlineComment(path="a.py", line=10, body="x")]
    valid, invalid = partition_comments(comments, {}, {})
    assert valid == []
    assert invalid == comments


def test_partition_valid_comment_passes() -> None:
    ranges: DiffLineRanges = {"a.py": {5, 10, 15}}
    comments = [InlineComment(path="a.py", line=10, body="x")]
    valid, invalid = partition_comments(comments, ranges, {})
    assert len(valid) == 1
    assert len(invalid) == 0


@pytest.mark.parametrize(
    ("path", "line", "reason"),
    [
        ("a.py", 99, "wrong line"),
        ("b.py", 5, "wrong path"),
    ],
    ids=["wrong-line", "wrong-path"],
)
def test_partition_invalid_comment(path: str, line: int, reason: str) -> None:
    ranges: DiffLineRanges = {"a.py": {5, 10, 15}}
    comments = [InlineComment(path=path, line=line, body="x")]
    valid, invalid = partition_comments(comments, ranges, {})
    assert len(valid) == 0
    assert len(invalid) == 1


def test_partition_mixed_valid_and_invalid() -> None:
    ranges: DiffLineRanges = {"a.py": {10}, "b.py": {20}}
    comments = [
        InlineComment(path="a.py", line=10, body="ok"),
        InlineComment(path="a.py", line=99, body="stale"),
        InlineComment(path="b.py", line=20, body="ok2"),
        InlineComment(path="c.py", line=1, body="gone"),
    ]
    valid, invalid = partition_comments(comments, ranges, {})
    assert {c.body for c in valid} == {"ok", "ok2"}
    assert {c.body for c in invalid} == {"stale", "gone"}


def test_partition_left_side_uses_left_ranges() -> None:
    comments = [InlineComment(path="a.py", line=7, side="LEFT", body="old")]
    valid, invalid = partition_comments(comments, {"a.py": {10}}, {"a.py": {7}})
    assert len(valid) == 1
    assert len(invalid) == 0


def test_partition_file_level_always_valid() -> None:
    comments = [InlineComment(path="a.py", line=0, body="file-level")]
    valid, invalid = partition_comments(comments, {}, {})
    assert len(valid) == 1
    assert len(invalid) == 0


# ── is_tombstone ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("body", "github_id", "expected"),
    [
        ("", 100, True),
        ("Fix.", 100, False),
        ("", None, False),
    ],
    ids=["tombstone", "has-body", "no-github-id"],
)
def test_is_tombstone(body: str, github_id: int | None, expected: bool) -> None:
    c = InlineComment(path="a.py", line=1, body=body, github_id=github_id)
    assert is_tombstone(c) is expected


def test_tombstone_sync_status() -> None:
    c = InlineComment(path="a.py", line=1, body="", github_id=100, comment_hash="abc")
    assert comment_sync_status(c) == "✗"


def test_match_tombstone_beats_remote() -> None:
    """Tombstoned local comment wins over remote version."""
    local = [
        InlineComment(path="a.py", line=1, body="", github_id=50, comment_hash="abc"),
    ]
    remote = [
        InlineComment(path="a.py", line=1, body="Remote content.", github_id=50),
    ]
    result = match_comments(local, remote)
    assert len(result.comments) == 1
    assert result.comments[0].body == ""
    assert result.comments[0].github_id == 50


# ── _resolve_event ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        ("", ReviewEvent.COMMENT),
        ("comment", ReviewEvent.COMMENT),
        ("approve", ReviewEvent.APPROVE),
        ("request_changes", ReviewEvent.REQUEST_CHANGES),
        ("changes", ReviewEvent.REQUEST_CHANGES),
    ],
)
def test_resolve_event(arg: str, expected: ReviewEvent) -> None:
    assert _resolve_event(arg) == expected


@pytest.mark.parametrize("arg", ["merge", "yolo"])
def test_resolve_event_invalid(arg: str) -> None:
    assert _resolve_event(arg) is None
