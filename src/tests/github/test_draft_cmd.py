"""Tests for /draft command dispatch — show, sync, post, clear.

Tests are organised around a shared dataset (``PR``, ``DRAFT``,
``DRAFT_WITH_SUGGESTION``) and a ``FakeEngine`` that captures events.
The command layer is tested via ``cmd_draft`` — GitHub API interaction
is always behind ``engine.publish`` and mocked at that boundary.
"""

from __future__ import annotations

import queue
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rbtr.engine.draft_cmd import (
    POST_EVENTS,
    SUBCOMMANDS,
    _resolve_event,
    cmd_draft,
)
from rbtr.events import Event, FlushPanel, MarkdownOutput, Output
from rbtr.github.draft import save_draft
from rbtr.models import InlineComment, PRTarget, ReviewDraft, ReviewEvent
from rbtr.state import EngineState

# ── Shared test data ─────────────────────────────────────────────────

PR = PRTarget(
    number=42,
    title="Add retry logic",
    author="alice",
    base_branch="main",
    head_branch="feature/retry",
    updated_at=datetime(2025, 6, 15, tzinfo=UTC),
)

DRAFT = ReviewDraft(
    summary="Looks good with minor issues.",
    comments=[
        InlineComment(path="src/client.py", line=42, body="**blocker:** Retry without backoff."),
        InlineComment(path="src/config.py", line=8, body="**nit:** Unused import."),
    ],
)

DRAFT_WITH_SUGGESTION = ReviewDraft(
    summary="",
    comments=[
        InlineComment(
            path="src/client.py",
            line=42,
            body="Use exponential backoff.",
            suggestion="time.sleep(2 ** attempt)",
        ),
    ],
)

EMPTY_DRAFT = ReviewDraft(summary="", comments=[])


# ── FakeEngine ───────────────────────────────────────────────────────


class FakeEngine:
    """Minimal engine stub that captures events for assertions."""

    def __init__(self, *, gh: Any = None, gh_username: str = "") -> None:
        self.state = EngineState()
        self.state.review_target = PR
        self.state.gh = gh
        self.state.gh_username = gh_username
        self.state.owner = "owner"
        self.state.repo_name = "repo"
        self._events: queue.Queue[Event] = queue.Queue()

    def _emit(self, event: Event) -> None:
        self._events.put(event)

    def _out(self, text: str, style: str = "") -> None:
        self._emit(Output(text=text, style=style))

    def _warn(self, text: str) -> None:
        self._emit(Output(text=text, style="rbtr.out.warning"))

    def _error(self, text: str) -> None:
        self._emit(Output(text=text, style="rbtr.out.error"))

    def _markdown(self, text: str) -> None:
        self._emit(MarkdownOutput(text=text))

    def _flush(self) -> None:
        self._emit(FlushPanel())

    def _clear(self) -> None:
        self._emit(FlushPanel(discard=True))

    def _check_cancel(self) -> None:
        pass

    def collected_text(self) -> str:
        """Drain events and concatenate Output + MarkdownOutput text."""
        lines: list[str] = []
        while not self._events.empty():
            ev = self._events.get_nowait()
            if isinstance(ev, (Output, MarkdownOutput)):
                lines.append(ev.text)
        return "\n".join(lines)

    def collected_events(self) -> list[Event]:
        """Drain and return all events."""
        events: list[Event] = []
        while not self._events.empty():
            events.append(self._events.get_nowait())
        return events


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("rbtr.config.config.tools.drafts_dir", str(tmp_path / "drafts"))
    return tmp_path


# ── No PR selected ───────────────────────────────────────────────────


def test_no_pr_selected() -> None:
    engine = FakeEngine()
    engine.state.review_target = None
    cmd_draft(engine, "")  # type: ignore[arg-type]  # FakeEngine stub
    assert "No PR selected" in engine.collected_text()


# ── /draft (show) ────────────────────────────────────────────────────


def test_show_no_draft(workspace: Path) -> None:
    engine = FakeEngine()
    cmd_draft(engine, "")  # type: ignore[arg-type]  # FakeEngine stub
    assert "No draft" in engine.collected_text()


def test_show_draft_with_comments(workspace: Path) -> None:
    save_draft(42, DRAFT)
    engine = FakeEngine()
    cmd_draft(engine, "")  # type: ignore[arg-type]  # FakeEngine stub
    text = engine.collected_text()
    # Summary at the bottom.
    assert "## Summary" in text
    assert "Looks good" in text
    # Comments section.
    assert "2 comments" in text
    # File headings.
    assert "### src/client.py" in text
    assert "### src/config.py" in text


def test_show_draft_with_suggestion(workspace: Path) -> None:
    save_draft(42, DRAFT_WITH_SUGGESTION)
    engine = FakeEngine()
    cmd_draft(engine, "")  # type: ignore[arg-type]  # FakeEngine stub
    text = engine.collected_text()
    assert "```suggestion" in text


def test_show_empty_summary(workspace: Path) -> None:
    save_draft(42, EMPTY_DRAFT)
    engine = FakeEngine()
    cmd_draft(engine, "")  # type: ignore[arg-type]  # FakeEngine stub
    text = engine.collected_text()
    assert "(empty)" in text
    assert "No inline comments" in text


# ── /draft sync ──────────────────────────────────────────────────────


@patch("rbtr.engine.draft_cmd.sync_review_draft")
def test_sync_delegates_to_review(mock_sync: MagicMock, workspace: Path) -> None:
    engine = FakeEngine(gh=MagicMock(), gh_username="reviewer")
    cmd_draft(engine, "sync")  # type: ignore[arg-type]  # FakeEngine stub
    mock_sync.assert_called_once_with(engine, 42)


# ── /draft post ──────────────────────────────────────────────────────


def test_post_no_draft(workspace: Path) -> None:
    engine = FakeEngine(gh=MagicMock(), gh_username="reviewer")
    cmd_draft(engine, "post")  # type: ignore[arg-type]  # FakeEngine stub
    assert "No draft to post" in engine.collected_text()


def test_post_empty_draft(workspace: Path) -> None:
    save_draft(42, EMPTY_DRAFT)
    engine = FakeEngine(gh=MagicMock(), gh_username="reviewer")
    cmd_draft(engine, "post")  # type: ignore[arg-type]  # FakeEngine stub
    assert "Draft is empty" in engine.collected_text()


def test_post_invalid_event(workspace: Path) -> None:
    save_draft(42, DRAFT)
    engine = FakeEngine(gh=MagicMock(), gh_username="reviewer")
    cmd_draft(engine, "post merge")  # type: ignore[arg-type]  # FakeEngine stub
    assert "Unknown event type" in engine.collected_text()


@patch("rbtr.engine.draft_cmd.post_review_draft")
def test_post_delegates_to_review(mock_post: MagicMock, workspace: Path) -> None:
    """cmd_draft delegates to post_review_draft with correct args."""
    save_draft(42, DRAFT)
    mock_post.return_value = True

    engine = FakeEngine(gh=MagicMock(), gh_username="reviewer")
    cmd_draft(engine, "post")  # type: ignore[arg-type]  # FakeEngine stub

    mock_post.assert_called_once()
    args = mock_post.call_args[0]
    assert args[1] == 42  # pr_number
    assert args[2].summary == DRAFT.summary  # draft
    assert args[3] == ReviewEvent.COMMENT  # event


@patch("rbtr.engine.draft_cmd.post_review_draft")
def test_post_approve_passes_event(mock_post: MagicMock, workspace: Path) -> None:
    save_draft(42, DRAFT)
    mock_post.return_value = True

    engine = FakeEngine(gh=MagicMock(), gh_username="reviewer")
    cmd_draft(engine, "post approve")  # type: ignore[arg-type]  # FakeEngine stub

    assert mock_post.call_args[0][3] == ReviewEvent.APPROVE


@patch("rbtr.engine.draft_cmd.post_review_draft")
def test_post_request_changes_passes_event(mock_post: MagicMock, workspace: Path) -> None:
    save_draft(42, DRAFT)
    mock_post.return_value = True

    engine = FakeEngine(gh=MagicMock(), gh_username="reviewer")
    cmd_draft(engine, "post request_changes")  # type: ignore[arg-type]  # FakeEngine stub

    assert mock_post.call_args[0][3] == ReviewEvent.REQUEST_CHANGES


# ── /draft clear ─────────────────────────────────────────────────────


@patch("rbtr.engine.draft_cmd.clear_review_draft")
def test_clear_delegates_to_review(mock_clear: MagicMock, workspace: Path) -> None:
    engine = FakeEngine()
    cmd_draft(engine, "clear")  # type: ignore[arg-type]  # FakeEngine stub
    mock_clear.assert_called_once_with(engine, 42)


# ── Unknown subcommand ───────────────────────────────────────────────


def test_unknown_subcommand(workspace: Path) -> None:
    engine = FakeEngine()
    cmd_draft(engine, "bogus")  # type: ignore[arg-type]  # FakeEngine stub
    text = engine.collected_text()
    assert "Unknown subcommand" in text
    assert "Usage" in text


# ── Module-level constants ───────────────────────────────────────────


def test_subcommands_list() -> None:
    """SUBCOMMANDS and POST_EVENTS are non-empty for tab completion."""
    assert len(SUBCOMMANDS) >= 2
    assert len(POST_EVENTS) >= 2
    assert all(isinstance(name, str) and isinstance(desc, str) for name, desc in SUBCOMMANDS)


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


def test_resolve_event_invalid() -> None:
    assert _resolve_event("merge") is None


# ── /draft display layout ───────────────────────────────────────────


def test_show_summary_at_bottom(workspace: Path) -> None:
    """Summary section appears after comments in the output."""
    save_draft(42, DRAFT)
    engine = FakeEngine()
    cmd_draft(engine, "")  # type: ignore[arg-type]  # FakeEngine stub
    text = engine.collected_text()
    comments_pos = text.find("2 comments")
    summary_heading_pos = text.find("## Summary")
    assert comments_pos < summary_heading_pos


def test_show_comments_grouped_by_file(workspace: Path) -> None:
    """Comments on the same file appear under a single file heading."""
    draft = ReviewDraft(
        summary="Overview.",
        comments=[
            InlineComment(path="src/api.py", line=10, body="First comment."),
            InlineComment(path="src/db.py", line=5, body="DB comment."),
            InlineComment(path="src/api.py", line=20, body="Second comment on same file."),
        ],
    )
    save_draft(42, draft)
    engine = FakeEngine()
    cmd_draft(engine, "")  # type: ignore[arg-type]  # FakeEngine stub
    text = engine.collected_text()
    # File headings present.
    assert "### src/api.py" in text
    assert "### src/db.py" in text
    # Both api.py comments under the same heading.
    api_heading_pos = text.find("### src/api.py")
    db_heading_pos = text.find("### src/db.py")
    first_pos = text.find("First comment.")
    second_pos = text.find("Second comment on same file.")
    assert api_heading_pos < first_pos < second_pos
    # db.py section is separate.
    assert api_heading_pos < db_heading_pos


def test_show_markdown_body_emitted(workspace: Path) -> None:
    """Comment bodies are emitted as MarkdownOutput, not plain Output."""
    save_draft(
        42,
        ReviewDraft(
            summary="Summary text.",
            comments=[
                InlineComment(path="a.py", line=1, body="**bold** and `code`."),
            ],
        ),
    )
    engine = FakeEngine()
    cmd_draft(engine, "")  # type: ignore[arg-type]  # FakeEngine stub
    events = engine.collected_events()
    md_events = [e for e in events if isinstance(e, MarkdownOutput)]
    # At least: the ## heading, ### heading, comment body, ## Summary heading, summary body.
    md_texts = [e.text for e in md_events]
    assert any("**bold** and `code`." in t for t in md_texts)
    assert any("## Summary" in t for t in md_texts)
    assert any("Summary text." in t for t in md_texts)


def test_show_suggestion_as_code_block(workspace: Path) -> None:
    """Suggestions are rendered as ```suggestion code blocks."""
    save_draft(42, DRAFT_WITH_SUGGESTION)
    engine = FakeEngine()
    cmd_draft(engine, "")  # type: ignore[arg-type]  # FakeEngine stub
    events = engine.collected_events()
    md_texts = [e.text for e in events if isinstance(e, MarkdownOutput)]
    assert any("```suggestion" in t and "time.sleep(2 ** attempt)" in t for t in md_texts)


def test_show_tombstone_marker(workspace: Path) -> None:
    """Tombstoned comments show deletion marker."""
    draft = ReviewDraft(
        summary=".",
        comments=[
            InlineComment(
                path="a.py",
                line=5,
                body="",
                github_id=100,
                comment_hash="abc",
            ),
        ],
    )
    save_draft(42, draft)
    engine = FakeEngine()
    cmd_draft(engine, "")  # type: ignore[arg-type]  # FakeEngine stub
    text = engine.collected_text()
    assert "deleted" in text.lower()
    assert "✗" in text


def test_show_status_indicators(workspace: Path) -> None:
    """New, clean, and dirty comments show correct indicators."""
    draft = ReviewDraft(
        summary=".",
        comments=[
            # New (never synced) — ★
            InlineComment(path="a.py", line=1, body="New comment."),
            # Clean (hash matches) — ✓
            InlineComment(
                path="b.py",
                line=2,
                body="Clean comment.",
                comment_hash="",  # Will be set below.
            ),
        ],
    )
    # Compute proper hash for the clean comment.
    from rbtr.github.draft import _comment_hash

    draft.comments[1] = draft.comments[1].model_copy(
        update={"comment_hash": _comment_hash(draft.comments[1])}
    )
    save_draft(42, draft)
    engine = FakeEngine()
    cmd_draft(engine, "")  # type: ignore[arg-type]  # FakeEngine stub
    text = engine.collected_text()
    assert "★" in text
    assert "✓" in text


def test_show_no_comments_message(workspace: Path) -> None:
    """Draft with summary but no comments shows 'No inline comments'."""
    save_draft(42, ReviewDraft(summary="Just a summary."))
    engine = FakeEngine()
    cmd_draft(engine, "")  # type: ignore[arg-type]  # FakeEngine stub
    text = engine.collected_text()
    assert "No inline comments" in text
    assert "Just a summary." in text


def test_show_empty_summary_label(workspace: Path) -> None:
    """Draft with comments but no summary shows (empty) for summary."""
    save_draft(
        42,
        ReviewDraft(comments=[InlineComment(path="a.py", line=1, body="Comment.")]),
    )
    engine = FakeEngine()
    cmd_draft(engine, "")  # type: ignore[arg-type]  # FakeEngine stub
    text = engine.collected_text()
    assert "(empty)" in text
