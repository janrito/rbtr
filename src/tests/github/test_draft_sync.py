"""Tests for draft sync and comment matching — realistic API-shaped data.

Exercises the full lifecycle that `/draft sync` performs:
create → read → match → merge → delete → push → re-read.

Test data is modelled on captured GitHub API responses.  Key
properties from real fixtures:

- Per-review endpoint returns `line: null`, `side: null` for
  ALL review comments (pending AND submitted).
- `position` and `diff_hunk` are always present.
- `diff_hunk` runs from the `@@` header through the commented
  line — `_position_to_line` walks it to recover `(line, side)`.
- Pending comments return 404 on `PATCH /pulls/comments/{id}`.
- Pending comments CAN be individually `DELETE`-d.
- After delete-and-recreate, comment IDs change but content and
  `position`/`diff_hunk` remain identical.

These tests do NOT depend on fixture files on disk — they inline
the data patterns.
"""

from __future__ import annotations

import queue
from pathlib import Path

import pytest

from rbtr.engine.publish import sync_review_draft
from rbtr.events import ContextMarkerReady, Event, FlushPanel, Output, OutputLevel
from rbtr.github.client import (
    _position_to_line,
    format_comment_body,
    get_pending_review,
    parse_comment_body,
)
from rbtr.github.draft import (
    _comment_hash,
    load_draft,
    match_comments,
    save_draft,
    stamp_synced,
)
from rbtr.models import InlineComment, PRTarget, ReviewDraft
from rbtr.state import EngineState

from .conftest import (
    FakeGithub,
    FakeInlineComment,
    FakePR,
    FakeRepo,
    FakeReview,
    FakeUser,
    fake_ctx,
)

# ── Realistic diff hunks (from captured fixtures) ────────────────────
#
# New file: position 1 → line 1, RIGHT
NEW_FILE_HUNK = "@@ -0,0 +1,5 @@\n+outs:"

# Modified file: 3 context lines + 1 addition, position 4 → line 172, RIGHT
MODIFIED_FILE_HUNK = (
    "@@ -169,6 +169,7 @@ jobs:\n"
    "           BEAUHURST_API_KEY: ${{ secrets.BEAUHURST_API_KEY}}\n"
    "           BEAUHURST_USERNAME: ${{ secrets.BEAUHURST_USERNAME}}\n"
    "           BRAVE_SEARCH_API_KEY: ${{ secrets.BRAVE_SEARCH_API_KEY }}\n"
    "+          BEAUHURST_API_BASE_URL: ${{ secrets.BEAUHURST_API_BASE_URL}}"
)

# Deletion on old side: position 2 → line 170, LEFT
DELETION_HUNK = (
    "@@ -169,4 +169,3 @@ jobs:\n"
    "           BEAUHURST_API_KEY: ${{ secrets.BEAUHURST_API_KEY}}\n"
    "-          BEAUHURST_USERNAME: ${{ secrets.BEAUHURST_USERNAME}}"
)


def _pending_comment(
    *,
    comment_id: int = 10,
    path: str = "src/handler.py",
    body: str = "Fix this.",
    diff_hunk: str = NEW_FILE_HUNK,
    position: int = 1,
) -> FakeInlineComment:
    """Build a FakeInlineComment shaped like the real GitHub API.

    Per-review endpoint returns `line: null`, `side: null`.
    The production code falls back to `_position_to_line(diff_hunk)`.
    """
    c = FakeInlineComment(
        comment_id=comment_id,
        path=path,
        body=body,
        line=None,  # GitHub per-review endpoint always returns null
        diff_hunk=diff_hunk,
    )
    # Override _rawData to match real API shape.
    c._rawData = {
        "id": comment_id,
        "path": path,
        "body": body,
        "line": None,
        "original_line": None,
        "side": None,
        "position": position,
        "original_position": position,
        "diff_hunk": diff_hunk,
        "commit_id": "abc123",
    }
    return c


# ── _position_to_line on real hunks ──────────────────────────────────


def test_position_to_line_new_file() -> None:
    """New file hunk: single addition line → (1, RIGHT)."""
    line, side = _position_to_line(NEW_FILE_HUNK)
    assert line == 1
    assert side == "RIGHT"


def test_position_to_line_modified_file() -> None:
    """Modified file: 3 context + 1 addition → (172, RIGHT)."""
    line, side = _position_to_line(MODIFIED_FILE_HUNK)
    assert line == 172
    assert side == "RIGHT"


def test_position_to_line_deletion() -> None:
    """Deletion hunk: context + deletion → (170, LEFT)."""
    line, side = _position_to_line(DELETION_HUNK)
    assert line == 170
    assert side == "LEFT"


# ── get_pending_review with realistic _rawData ───────────────────────


def test_get_pending_review_recovers_line_from_hunk() -> None:
    """When line is null (pending review), recover from diff_hunk."""
    comments = [
        _pending_comment(
            comment_id=100,
            path="new_file.dvc",
            body="Comment on new file.",
            diff_hunk=NEW_FILE_HUNK,
            position=1,
        ),
        _pending_comment(
            comment_id=101,
            path="workflows/cml.yaml",
            body="Comment on modified file.",
            diff_hunk=MODIFIED_FILE_HUNK,
            position=4,
        ),
    ]
    pr = FakePR(
        reviews=[FakeReview(review_id=50, state="PENDING", user=FakeUser("me"))],
        review_comments_by_id={50: comments},
    )
    result = get_pending_review(fake_ctx(FakeGithub(FakeRepo(pr))), 1, "me")

    assert result is not None
    assert len(result.comments) == 2

    # New file: line 1, side RIGHT.
    assert result.comments[0].line == 1
    assert result.comments[0].side == "RIGHT"
    assert result.comments[0].path == "new_file.dvc"

    # Modified file: line 172, side RIGHT.
    assert result.comments[1].line == 172
    assert result.comments[1].side == "RIGHT"


def test_get_pending_review_reads_commit_id() -> None:
    """commit_id is read from rawData."""
    c = _pending_comment(comment_id=200, body="test")
    c._rawData["commit_id"] = "deadbeef"
    pr = FakePR(
        reviews=[FakeReview(review_id=50, state="PENDING", user=FakeUser("me"))],
        review_comments_by_id={50: [c]},
    )
    result = get_pending_review(fake_ctx(FakeGithub(FakeRepo(pr))), 1, "me")
    assert result is not None
    assert result.comments[0].commit_id == "deadbeef"


def test_get_pending_review_parses_suggestion_from_body() -> None:
    """Suggestion blocks are split from the body."""
    raw = "Use this instead.\n\n```suggestion\nbetter()\n```"
    c = _pending_comment(comment_id=300, body=raw)
    c._rawData["body"] = raw
    pr = FakePR(
        reviews=[FakeReview(review_id=50, state="PENDING", user=FakeUser("me"))],
        review_comments_by_id={50: [c]},
    )
    result = get_pending_review(fake_ctx(FakeGithub(FakeRepo(pr))), 1, "me")
    assert result is not None
    assert result.comments[0].body == "Use this instead."
    assert result.comments[0].suggestion == "better()"


def test_get_pending_review_deletion_hunk_gives_left_side() -> None:
    """A comment on a deleted line should resolve to side=LEFT."""
    c = _pending_comment(
        comment_id=400,
        body="This line was removed.",
        diff_hunk=DELETION_HUNK,
        position=2,
    )
    pr = FakePR(
        reviews=[FakeReview(review_id=50, state="PENDING", user=FakeUser("me"))],
        review_comments_by_id={50: [c]},
    )
    result = get_pending_review(fake_ctx(FakeGithub(FakeRepo(pr))), 1, "me")
    assert result is not None
    assert result.comments[0].line == 170
    assert result.comments[0].side == "LEFT"


# ── match_comments: scenario A — baseline (no changes) ───────────────


def test_match_baseline_no_changes() -> None:
    """Synced local and identical remote → all matched, no warnings."""
    local_a = InlineComment(path="a.py", line=1, body="Comment A.", github_id=100)
    local_b = InlineComment(path="b.py", line=172, body="Comment B.", github_id=101)
    synced = [
        stamp_synced(ReviewDraft(comments=[local_a, local_b])).comments[0],
        stamp_synced(ReviewDraft(comments=[local_a, local_b])).comments[1],
    ]

    remote = [
        InlineComment(path="a.py", line=1, body="Comment A.", github_id=100),
        InlineComment(path="b.py", line=172, body="Comment B.", github_id=101),
    ]
    result = match_comments(synced, remote)
    assert len(result.comments) == 2
    assert result.warnings == []
    assert all(c.github_id is not None for c in result.comments)


# ── match_comments: scenario B — remote edit ─────────────────────────


def test_match_remote_edit_accepted_when_local_clean() -> None:
    """Remote body changed, local unchanged → accept remote edit."""
    original = InlineComment(path="a.py", line=1, body="Original.", github_id=100)
    stamped = stamp_synced(ReviewDraft(comments=[original])).comments[0]

    remote = [InlineComment(path="a.py", line=1, body="Edited on GitHub.", github_id=100)]
    result = match_comments([stamped], remote)

    assert result.comments[0].body == "Edited on GitHub."
    assert result.warnings == []


def test_match_remote_edit_conflicts_with_local_edit() -> None:
    """Both sides changed → conflict, keep local, warn."""
    original = InlineComment(path="a.py", line=1, body="Original.", github_id=100)
    base_hash = _comment_hash(original)
    local_edited = InlineComment(
        path="a.py",
        line=1,
        body="Local rewrite.",
        github_id=100,
        comment_hash=base_hash,
    )
    remote = [InlineComment(path="a.py", line=1, body="Remote rewrite.", github_id=100)]

    result = match_comments([local_edited], remote)
    assert result.comments[0].body == "Local rewrite."
    assert len(result.warnings) == 1
    assert "Conflict" in result.warnings[0]


# ── match_comments: scenario C — remote deletion ─────────────────────


def test_match_remote_deletion() -> None:
    """Remote comment gone, local has stale github_id → removed."""
    original = InlineComment(path="a.py", line=1, body="Will be deleted.", github_id=100)
    stamped = stamp_synced(ReviewDraft(comments=[original])).comments[0]

    # Remote has fewer comments — 100 is missing.
    remote = [InlineComment(path="b.py", line=10, body="Survivor.", github_id=200)]

    result = match_comments([stamped], remote)
    # The deleted comment is dropped; the remote survivor is imported.
    bodies = {c.body for c in result.comments}
    assert "Will be deleted." not in bodies
    assert "Survivor." in bodies
    assert any("deleted on GitHub" in w for w in result.warnings)


def test_match_remote_deletion_with_other_local_comments() -> None:
    """Remote deletes one comment; other local comments survive."""
    kept = InlineComment(path="a.py", line=1, body="Kept.", github_id=100)
    deleted = InlineComment(path="b.py", line=10, body="Deleted.", github_id=200)
    stamped = stamp_synced(ReviewDraft(comments=[kept, deleted])).comments

    # Only comment 100 exists remotely.
    remote = [InlineComment(path="a.py", line=1, body="Kept.", github_id=100)]

    result = match_comments(list(stamped), remote)
    assert len(result.comments) == 1
    assert result.comments[0].body == "Kept."
    assert any("deleted on GitHub" in w for w in result.warnings)


# ── match_comments: scenario D — local-only changes ──────────────────


def test_match_local_edit_kept_when_remote_clean() -> None:
    """Local edit, remote unchanged → keep local, push on next sync."""
    original = InlineComment(path="a.py", line=1, body="Original.", github_id=100)
    base_hash = _comment_hash(original)
    local_edited = InlineComment(
        path="a.py",
        line=1,
        body="Locally improved.",
        github_id=100,
        comment_hash=base_hash,
    )
    remote = [InlineComment(path="a.py", line=1, body="Original.", github_id=100)]

    result = match_comments([local_edited], remote)
    assert result.comments[0].body == "Locally improved."
    assert result.warnings == []


def test_match_local_deletion_pushes_fewer_comments() -> None:
    """Local removes a comment; match produces a shorter list."""
    kept = InlineComment(path="a.py", line=1, body="Kept.", github_id=100)
    stamped = stamp_synced(ReviewDraft(comments=[kept])).comments

    # Remote still has both, but local only has one.
    remote = [
        InlineComment(path="a.py", line=1, body="Kept.", github_id=100),
        InlineComment(path="b.py", line=10, body="Extra.", github_id=200),
    ]
    result = match_comments(list(stamped), remote)
    # The unmatched remote is imported — this is correct because
    # we can't distinguish "local deleted" from "remote added".
    assert len(result.comments) == 2
    assert any("imported" in w for w in result.warnings)


def test_match_new_local_comment_with_no_remote() -> None:
    """Purely local comment (no github_id) stays as-is."""
    local = [InlineComment(path="new.py", line=5, body="Brand new.")]
    result = match_comments(local, [])
    assert len(result.comments) == 1
    assert result.comments[0].github_id is None
    assert result.warnings == []


# ── match_comments: scenario E — both changed ────────────────────────


def test_match_both_changed_different_comments() -> None:
    """Local edits comment A, remote edits comment B → both accepted."""
    original_a = InlineComment(path="a.py", line=1, body="A original.", github_id=100)
    original_b = InlineComment(path="b.py", line=10, body="B original.", github_id=200)
    base_hash_a = _comment_hash(original_a)
    base_hash_b = _comment_hash(original_b)

    local = [
        # A was locally edited.
        InlineComment(
            path="a.py",
            line=1,
            body="A locally edited.",
            github_id=100,
            comment_hash=base_hash_a,
        ),
        # B unchanged locally.
        InlineComment(
            path="b.py",
            line=10,
            body="B original.",
            github_id=200,
            comment_hash=base_hash_b,
        ),
    ]
    remote = [
        # A unchanged remotely.
        InlineComment(path="a.py", line=1, body="A original.", github_id=100),
        # B was remotely edited.
        InlineComment(path="b.py", line=10, body="B remotely edited.", github_id=200),
    ]

    result = match_comments(local, remote)
    by_path = {c.path: c for c in result.comments}
    assert by_path["a.py"].body == "A locally edited."  # local edit preserved
    assert by_path["b.py"].body == "B remotely edited."  # remote edit accepted
    assert result.warnings == []


def test_match_both_changed_same_comment_conflicts() -> None:
    """Both sides edit the same comment → conflict, local wins."""
    original = InlineComment(path="a.py", line=1, body="Original.", github_id=100)
    base_hash = _comment_hash(original)

    local = [
        InlineComment(
            path="a.py",
            line=1,
            body="Local version.",
            github_id=100,
            comment_hash=base_hash,
        )
    ]
    remote = [InlineComment(path="a.py", line=1, body="Remote version.", github_id=100)]

    result = match_comments(local, remote)
    assert result.comments[0].body == "Local version."
    assert len(result.warnings) == 1
    assert "Conflict" in result.warnings[0]


# ── Tier-2 content matching (after delete-and-recreate) ──────────────


def test_tier2_matches_by_content_after_recreate() -> None:
    """After push, re-fetched comments get new IDs but same content."""
    # Local has comments without github_id (just pushed, ids unknown).
    local = [
        InlineComment(path="a.py", line=1, body="Comment A."),
        InlineComment(path="b.py", line=172, body="Comment B."),
    ]
    # Remote has the same content with new github_ids.
    remote = [
        InlineComment(path="a.py", line=1, body="Comment A.", github_id=500),
        InlineComment(path="b.py", line=172, body="Comment B.", github_id=501),
    ]
    result = match_comments(local, remote)
    assert len(result.comments) == 2
    assert result.comments[0].github_id == 500
    assert result.comments[1].github_id == 501
    assert result.warnings == []


def test_tier2_ambiguous_content_not_matched() -> None:
    """Same content on same line → ambiguous, don't match."""
    local = [InlineComment(path="a.py", line=1, body="Dupe.")]
    remote = [
        InlineComment(path="a.py", line=1, body="Dupe.", github_id=600),
        InlineComment(path="a.py", line=1, body="Dupe.", github_id=601),
    ]
    result = match_comments(local, remote)
    # Local unmatched + both remotes imported.
    assert len(result.comments) == 3


def test_tier2_matches_with_suggestion() -> None:
    """Content match includes the suggestion block."""
    local = [InlineComment(path="a.py", line=1, body="Fix.", suggestion="better()")]
    remote = [
        InlineComment(
            path="a.py",
            line=1,
            body="Fix.",
            suggestion="better()",
            github_id=700,
        )
    ]
    result = match_comments(local, remote)
    assert result.comments[0].github_id == 700


# ── stamp_synced and dirty detection ─────────────────────────────────


def test_stamp_sets_hashes_on_all_comments() -> None:
    """stamp_synced writes comment_hash and summary_hash."""
    draft = ReviewDraft(
        summary="Good work.",
        comments=[
            InlineComment(path="a.py", line=1, body="Fix.", github_id=100),
            InlineComment(path="b.py", line=10, body="Nit."),
        ],
    )
    stamped = stamp_synced(draft)
    assert stamped.summary_hash != ""
    for c in stamped.comments:
        assert c.comment_hash == _comment_hash(c)


def test_dirty_detection_after_local_edit() -> None:
    """Editing body after stamp makes hash diverge."""
    original = InlineComment(path="a.py", line=1, body="Original.", github_id=100)
    stamped = stamp_synced(ReviewDraft(comments=[original])).comments[0]

    edited = stamped.model_copy(update={"body": "Edited."})
    assert _comment_hash(edited) != edited.comment_hash


def test_dirty_detection_after_line_change() -> None:
    """Moving a comment to a different line makes hash diverge."""
    original = InlineComment(path="a.py", line=1, body="Same.", github_id=100)
    stamped = stamp_synced(ReviewDraft(comments=[original])).comments[0]

    moved = stamped.model_copy(update={"line": 99})
    assert _comment_hash(moved) != moved.comment_hash


def test_side_and_commit_id_not_in_hash() -> None:
    """side and commit_id are excluded from the content hash."""
    a = InlineComment(path="a.py", line=1, body="X.", side="RIGHT", commit_id="abc")
    b = InlineComment(path="a.py", line=1, body="X.", side="LEFT", commit_id="def")
    assert _comment_hash(a) == _comment_hash(b)


# ── format_comment_body / parse_comment_body round-trip ──────────────


def test_format_and_parse_roundtrip_plain() -> None:
    """Plain body round-trips through format → parse."""
    c = InlineComment(path="a.py", line=1, body="Fix this bug.")
    formatted = format_comment_body(c)
    body, suggestion = parse_comment_body(formatted)
    assert body == "Fix this bug."
    assert suggestion == ""


def test_format_and_parse_roundtrip_with_suggestion() -> None:
    """Body + suggestion round-trips through format → parse."""
    c = InlineComment(path="a.py", line=1, body="Use this.", suggestion="better()")
    formatted = format_comment_body(c)
    body, suggestion = parse_comment_body(formatted)
    assert body == "Use this."
    assert suggestion == "better()"


# ── Full sync orchestration ──────────────────────────────────────────
#
# These tests exercise sync_review_draft through the _FakeEngine,
# modelling the full delete → push → re-fetch cycle.


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("rbtr.config.config.tools.drafts_dir", str(tmp_path / "drafts"))
    return tmp_path


def _make_synced_draft(
    summary: str,
    comments: list[InlineComment],
    review_id: int = 99,
) -> ReviewDraft:
    """Build a draft as if it was just synced (hashes set)."""
    draft = ReviewDraft(
        summary=summary,
        comments=comments,
        github_review_id=review_id,
    )
    return stamp_synced(draft)


def _make_fake_engine(
    *,
    gh: FakeGithub,
    gh_username: str = "reviewer",
) -> _FakeEngine:
    return _FakeEngine(gh=gh, gh_username=gh_username)


class _FakeEngine:
    """Minimal engine stub for sync_review_draft tests."""

    def __init__(self, *, gh: FakeGithub, gh_username: str = "reviewer") -> None:
        from datetime import UTC, datetime

        self.state = EngineState()
        self.state.review_target = PRTarget(
            number=42,
            title="Test PR",
            author="alice",
            base_branch="main",
            head_branch="feature",
            base_commit="main",
            head_commit="feature",
            updated_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        self.state.gh = gh  # type: ignore[assignment]
        self.state.gh_username = gh_username
        self.state.owner = "owner"
        self.state.repo_name = "repo"
        self._events: queue.Queue[Event] = queue.Queue()

    def _emit(self, event: Event) -> None:
        self._events.put(event)

    def _out(self, text: str) -> None:
        self._emit(Output(text=text))

    def _warn(self, text: str) -> None:
        self._emit(Output(text=text, level=OutputLevel.WARNING))

    def _clear(self) -> None:
        self._emit(FlushPanel(discard=True))

    def _context(self, marker: str, content: str) -> None:
        self._emit(ContextMarkerReady(marker=marker, content=content))

    def _check_cancel(self) -> None:
        pass

    def collected_text(self) -> str:
        lines: list[str] = []
        while not self._events.empty():
            ev = self._events.get_nowait()
            if isinstance(ev, Output):
                lines.append(ev.text)
        return "\n".join(lines)


def test_sync_full_cycle_with_realistic_rawdata(workspace: Path) -> None:
    """End-to-end sync with null line/side in remote data.

    Models the real API: comments created with line+side come back
    from the per-review endpoint with line=null, side=null, and
    _position_to_line recovers the correct values.
    """
    # Local draft with one comment.
    save_draft(
        42,
        ReviewDraft(
            summary="Local review.",
            comments=[InlineComment(path="a.py", line=1, body="Local comment.")],
        ),
    )

    # Remote has a different comment (null line/side like real API).
    remote_c = _pending_comment(
        comment_id=500,
        path="b.yaml",
        body="Remote comment.",
        diff_hunk=MODIFIED_FILE_HUNK,
        position=4,
    )
    pr = FakePR(
        reviews=[FakeReview(review_id=99, state="PENDING", user=FakeUser("reviewer"))],
        review_comments_by_id={99: [remote_c]},
    )
    engine = _make_fake_engine(gh=FakeGithub(FakeRepo(pr)))

    sync_review_draft(engine, 42)  # type: ignore[arg-type]

    draft = load_draft(42)
    assert draft is not None
    assert len(draft.comments) == 2

    # The remote comment should have recovered line=172 from the hunk.
    remote_imported = next(c for c in draft.comments if c.path == "b.yaml")
    assert remote_imported.line == 172
    assert remote_imported.side == "RIGHT"


def test_sync_remote_edit_accepted(workspace: Path) -> None:
    """Scenario B: remote edits a comment, local is clean → accept."""
    original = InlineComment(path="a.py", line=1, body="Original.", github_id=100)
    save_draft(42, _make_synced_draft("Summary.", [original]))

    remote_c = _pending_comment(comment_id=100, path="a.py", body="Edited remotely.")
    pr = FakePR(
        reviews=[FakeReview(review_id=99, state="PENDING", user=FakeUser("reviewer"))],
        review_comments_by_id={99: [remote_c]},
    )
    engine = _make_fake_engine(gh=FakeGithub(FakeRepo(pr)))

    sync_review_draft(engine, 42)  # type: ignore[arg-type]

    draft = load_draft(42)
    assert draft is not None
    assert draft.comments[0].body == "Edited remotely."


def test_sync_remote_deletion_removes_comment(workspace: Path) -> None:
    """Scenario C: remote deletes a comment → removed locally."""
    original = InlineComment(path="a.py", line=1, body="Will die.", github_id=100)
    save_draft(42, _make_synced_draft("Summary.", [original]))

    # Remote has no comments at all.
    pr = FakePR(
        reviews=[
            FakeReview(review_id=99, state="PENDING", body="Summary.", user=FakeUser("reviewer"))
        ],
        review_comments_by_id={99: []},
    )
    engine = _make_fake_engine(gh=FakeGithub(FakeRepo(pr)))

    sync_review_draft(engine, 42)  # type: ignore[arg-type]

    draft = load_draft(42)
    assert draft is not None
    assert len(draft.comments) == 0
    assert "deleted on GitHub" in engine.collected_text()


def test_sync_local_edit_pushes(workspace: Path) -> None:
    """Scenario D: local edit, remote unchanged → push local."""
    original = InlineComment(path="a.py", line=1, body="Original.", github_id=100)
    base_hash = _comment_hash(original)
    edited = InlineComment(
        path="a.py",
        line=1,
        body="Locally improved.",
        github_id=100,
        comment_hash=base_hash,
    )
    save_draft(42, ReviewDraft(summary="Summary.", comments=[edited], github_review_id=99))

    # Remote still has original.
    remote_c = _pending_comment(comment_id=100, path="a.py", body="Original.")
    pr = FakePR(
        reviews=[FakeReview(review_id=99, state="PENDING", user=FakeUser("reviewer"))],
        review_comments_by_id={99: [remote_c]},
    )
    engine = _make_fake_engine(gh=FakeGithub(FakeRepo(pr)))

    sync_review_draft(engine, 42)  # type: ignore[arg-type]

    # The push should contain the locally edited body.
    assert len(pr.created_reviews) == 1
    pushed_bodies = [c["body"] for c in pr.created_reviews[0]["comments"]]
    assert "Locally improved." in pushed_bodies


def test_sync_both_changed_different_comments(workspace: Path) -> None:
    """Scenario E: local edits A, remote edits B → both accepted."""
    original_a = InlineComment(path="a.py", line=1, body="A original.", github_id=100)
    original_b = InlineComment(path="b.py", line=10, body="B original.", github_id=200)
    synced = _make_synced_draft("Summary.", [original_a, original_b])

    # Local edits A.
    local = synced.model_copy(
        update={
            "comments": [
                synced.comments[0].model_copy(update={"body": "A locally edited."}),
                synced.comments[1],  # B unchanged
            ]
        }
    )
    save_draft(42, local)

    # Remote edits B.
    remote_a = _pending_comment(comment_id=100, path="a.py", body="A original.")
    remote_b = _pending_comment(comment_id=200, path="b.py", body="B remotely edited.")
    pr = FakePR(
        reviews=[FakeReview(review_id=99, state="PENDING", user=FakeUser("reviewer"))],
        review_comments_by_id={99: [remote_a, remote_b]},
    )
    engine = _make_fake_engine(gh=FakeGithub(FakeRepo(pr)))

    sync_review_draft(engine, 42)  # type: ignore[arg-type]

    loaded = load_draft(42)
    assert loaded is not None
    by_path = {c.path: c for c in loaded.comments}
    assert by_path["a.py"].body == "A locally edited."
    assert by_path["b.py"].body == "B remotely edited."
    assert "Conflict" not in engine.collected_text()


def test_sync_conflict_keeps_local_warns(workspace: Path) -> None:
    """Both sides edit the same comment → conflict, keep local."""
    original = InlineComment(path="a.py", line=1, body="Base.", github_id=100)
    synced = _make_synced_draft("Summary.", [original])
    local = synced.model_copy(
        update={"comments": [synced.comments[0].model_copy(update={"body": "Local version."})]}
    )
    save_draft(42, local)

    remote_c = _pending_comment(comment_id=100, path="a.py", body="Remote version.")
    pr = FakePR(
        reviews=[FakeReview(review_id=99, state="PENDING", user=FakeUser("reviewer"))],
        review_comments_by_id={99: [remote_c]},
    )
    engine = _make_fake_engine(gh=FakeGithub(FakeRepo(pr)))

    sync_review_draft(engine, 42)  # type: ignore[arg-type]

    loaded = load_draft(42)
    assert loaded is not None
    assert loaded.comments[0].body == "Local version."
    assert "Conflict" in engine.collected_text()


def test_sync_reassigns_github_ids_after_recreate(workspace: Path) -> None:
    """After delete+recreate, new github_ids are assigned via tier-2 match.

    FakePR.create_review auto-generates comments from the pushed
    data.  The re-fetch picks up the auto-generated IDs via tier-2
    content matching (same path + line + body).
    """
    save_draft(
        42,
        ReviewDraft(
            summary="Review.",
            comments=[InlineComment(path="a.py", line=1, body="Comment.")],
        ),
    )

    # No initial remote review — local-only push.
    pr = FakePR(reviews=[], default_user=FakeUser("reviewer"))
    engine = _make_fake_engine(gh=FakeGithub(FakeRepo(pr)))

    sync_review_draft(engine, 42)  # type: ignore[arg-type]

    draft = load_draft(42)
    assert draft is not None
    # create_review auto-generates review 200 → comment id 200000.
    # The comment should have picked up the new github_id via
    # tier-2 content matching.
    assert draft.comments[0].github_id is not None
    assert draft.comments[0].github_id != 0


def test_sync_no_remote_no_local_is_noop(workspace: Path) -> None:
    """No local draft, no remote review → nothing happens."""
    pr = FakePR(reviews=[])
    engine = _make_fake_engine(gh=FakeGithub(FakeRepo(pr)))

    sync_review_draft(engine, 42)  # type: ignore[arg-type]

    assert load_draft(42) is None
    assert "Nothing to sync" in engine.collected_text()


def test_sync_remote_summary_adopted_when_local_empty(workspace: Path) -> None:
    """Remote summary fills empty local summary."""
    save_draft(42, ReviewDraft(comments=[InlineComment(path="a.py", line=1, body="X.")]))

    remote_c = _pending_comment(comment_id=50, path="a.py", body="X.")
    pr = FakePR(
        reviews=[
            FakeReview(
                review_id=99,
                state="PENDING",
                body="Remote summary.",
                user=FakeUser("reviewer"),
            )
        ],
        review_comments_by_id={99: [remote_c]},
    )
    engine = _make_fake_engine(gh=FakeGithub(FakeRepo(pr)))

    sync_review_draft(engine, 42)  # type: ignore[arg-type]

    draft = load_draft(42)
    assert draft is not None
    assert draft.summary == "Remote summary."


def test_sync_local_summary_takes_priority(workspace: Path) -> None:
    """Local summary is not overwritten by remote."""
    save_draft(
        42,
        ReviewDraft(
            summary="Local summary.",
            comments=[InlineComment(path="a.py", line=1, body="X.")],
        ),
    )

    remote_c = _pending_comment(comment_id=50, path="a.py", body="X.")
    pr = FakePR(
        reviews=[
            FakeReview(
                review_id=99,
                state="PENDING",
                body="Remote summary.",
                user=FakeUser("reviewer"),
            )
        ],
        review_comments_by_id={99: [remote_c]},
    )
    engine = _make_fake_engine(gh=FakeGithub(FakeRepo(pr)))

    sync_review_draft(engine, 42)  # type: ignore[arg-type]

    draft = load_draft(42)
    assert draft is not None
    assert draft.summary == "Local summary."


# ── Tombstone (local deletion of synced comments) ────────────────────


def test_tombstone_detection() -> None:
    """is_tombstone identifies empty-body comments with github_id."""
    from rbtr.github.draft import is_tombstone

    # Tombstone: has github_id, empty body.
    assert is_tombstone(InlineComment(path="a.py", line=1, body="", github_id=100))
    # Not a tombstone: has body.
    assert not is_tombstone(InlineComment(path="a.py", line=1, body="Fix.", github_id=100))
    # Not a tombstone: no github_id (local-only).
    assert not is_tombstone(InlineComment(path="a.py", line=1, body="", github_id=None))


def test_tombstone_sync_status() -> None:
    """Tombstoned comments show ✗ status."""
    from rbtr.github.draft import comment_sync_status

    tombstone = InlineComment(path="a.py", line=1, body="", github_id=100, comment_hash="abc")
    assert comment_sync_status(tombstone) == "✗"


def test_tombstone_excluded_from_push(workspace: Path) -> None:
    """Tombstoned comments are not pushed to GitHub."""
    live = InlineComment(path="a.py", line=1, body="Keep this.", github_id=100)
    original_b = InlineComment(path="b.py", line=5, body="Was here.", github_id=200)

    # Sync first to set hashes, then tombstone B (like remove_draft_comment does).
    draft = _make_synced_draft("Summary.", [live, original_b])
    tombstoned_b = draft.comments[1].model_copy(update={"body": "", "suggestion": ""})
    draft = draft.model_copy(update={"comments": [draft.comments[0], tombstoned_b]})
    save_draft(42, draft)

    # Remote has both comments.
    remote_a = _pending_comment(comment_id=100, path="a.py", body="Keep this.")
    remote_b = _pending_comment(comment_id=200, path="b.py", body="Was here.")
    pr = FakePR(
        reviews=[FakeReview(review_id=99, state="PENDING", user=FakeUser("reviewer"))],
        review_comments_by_id={99: [remote_a, remote_b]},
    )
    engine = _make_fake_engine(gh=FakeGithub(FakeRepo(pr)))

    sync_review_draft(engine, 42)  # type: ignore[arg-type]

    # Only the live comment should have been pushed.
    assert len(pr.created_reviews) == 1
    pushed_bodies = [c["body"] for c in pr.created_reviews[0]["comments"]]
    assert "Keep this." in pushed_bodies
    assert "" not in pushed_bodies

    # Tombstone should not be in the saved draft.
    loaded = load_draft(42)
    assert loaded is not None
    assert len(loaded.comments) == 1
    assert loaded.comments[0].body == "Keep this."
    assert "Deleting 1 comment" in engine.collected_text()


def test_tombstone_with_remote_edit_still_deleted(workspace: Path) -> None:
    """Even if remote edited the comment, local tombstone wins."""
    original = InlineComment(path="a.py", line=1, body="Original.", github_id=100)
    base_hash = _comment_hash(original)
    # Tombstone it locally.
    tombstone = InlineComment(
        path="a.py",
        line=1,
        body="",
        github_id=100,
        comment_hash=base_hash,
    )
    save_draft(42, ReviewDraft(summary="S.", comments=[tombstone], github_review_id=99))

    # Remote has an edited version.
    remote_c = _pending_comment(comment_id=100, path="a.py", body="Edited on GH.")
    pr = FakePR(
        reviews=[FakeReview(review_id=99, state="PENDING", user=FakeUser("reviewer"))],
        review_comments_by_id={99: [remote_c]},
    )
    engine = _make_fake_engine(gh=FakeGithub(FakeRepo(pr)))

    sync_review_draft(engine, 42)  # type: ignore[arg-type]

    # The tombstone should cause the comment to be excluded from
    # push and dropped from the draft.
    loaded = load_draft(42)
    assert loaded is not None
    assert len(loaded.comments) == 0


def test_match_tombstone_beats_remote() -> None:
    """In three-way merge, tombstone (local dirty) beats remote unchanged."""
    original = InlineComment(path="a.py", line=1, body="Original.", github_id=100)
    base_hash = _comment_hash(original)
    tombstone = InlineComment(
        path="a.py",
        line=1,
        body="",
        github_id=100,
        comment_hash=base_hash,
    )
    remote = [InlineComment(path="a.py", line=1, body="Original.", github_id=100)]

    result = match_comments([tombstone], remote)
    # Tombstone wins (local dirty, remote clean → keep local).
    assert result.comments[0].body == ""
    assert result.comments[0].github_id == 100
