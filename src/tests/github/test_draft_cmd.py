"""Tests for /draft command dispatch — show, sync, post, clear.

Tests are organised around a shared dataset (``PR``, ``DRAFT``,
``DRAFT_WITH_SUGGESTION``) and a ``FakeEngine`` that captures events.
The command layer is tested via ``cmd_draft`` — GitHub API interaction
is always behind ``engine.review`` and mocked at that boundary.
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
from rbtr.engine.state import EngineState
from rbtr.events import Event, FlushPanel, Output
from rbtr.github.draft import save_draft
from rbtr.models import InlineComment, PRTarget, ReviewDraft, ReviewEvent

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

    def _flush(self) -> None:
        self._emit(FlushPanel())

    def _clear(self) -> None:
        self._emit(FlushPanel(discard=True))

    def _check_cancel(self) -> None:
        pass

    def collected_text(self) -> str:
        """Drain events and concatenate Output text."""
        lines: list[str] = []
        while not self._events.empty():
            ev = self._events.get_nowait()
            if isinstance(ev, Output):
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
    monkeypatch.setattr("rbtr.github.draft.WORKSPACE_DIR", tmp_path)
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
    assert "Looks good" in text
    assert "2 comments" in text
    assert "src/client.py:42" in text
    assert "src/config.py:8" in text


def test_show_draft_with_suggestion(workspace: Path) -> None:
    save_draft(42, DRAFT_WITH_SUGGESTION)
    engine = FakeEngine()
    cmd_draft(engine, "")  # type: ignore[arg-type]  # FakeEngine stub
    assert "has suggestion" in engine.collected_text()


def test_show_empty_summary(workspace: Path) -> None:
    save_draft(42, EMPTY_DRAFT)
    engine = FakeEngine()
    cmd_draft(engine, "")  # type: ignore[arg-type]  # FakeEngine stub
    text = engine.collected_text()
    assert "Summary: (empty)" in text
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
