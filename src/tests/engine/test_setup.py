"""Tests for Engine._run_setup — credential detection, endpoint listing,
saved model loading, and the unexpected-error handler in run_task.

Covers the untested branches in engine/core.py: setup with various
credential combinations, endpoint listing, saved model preference,
and the generic Exception handler.
"""

from __future__ import annotations

import tempfile

import pygit2

from rbtr.config import config
from rbtr.creds import creds
from rbtr.engine import Engine, Session, TaskType
from rbtr.events import Event

from .conftest import CHATGPT_OAUTH, CLAUDE_OAUTH, drain, make_engine, output_texts


def _setup_repo(monkeypatch, tmp) -> None:
    """Create a git repo at *tmp* and chdir into it."""
    repo = pygit2.init_repository(tmp)
    sig = pygit2.Signature("Test", "test@test.com")
    tree = repo.TreeBuilder().write()
    repo.create_commit("refs/heads/main", sig, sig, "init", tree, [])
    repo.set_head("refs/heads/main")
    repo.remotes.create("origin", "git@github.com:acme/widgets.git")
    monkeypatch.chdir(tmp)


# ── Setup with credentials ───────────────────────────────────────────


def test_setup_detects_github_token(monkeypatch, creds_path, config_path) -> None:
    """Setup with a stored GitHub token authenticates automatically."""
    creds.update(github_token="ghp_test123")

    # Mock Github so get_user().login succeeds without a real API call.
    from unittest.mock import MagicMock

    fake_gh = MagicMock()
    fake_gh.get_user.return_value.login = "testuser"
    monkeypatch.setattr("rbtr.engine.core.Github", lambda **_kw: fake_gh)

    with tempfile.TemporaryDirectory() as tmp:
        _setup_repo(monkeypatch, tmp)
        session = Session()
        events: __import__("queue").Queue[Event] = __import__("queue").Queue()
        engine = Engine(session, events)

        engine.run_task(TaskType.SETUP, "")
        texts = output_texts(drain(events))

        assert session.gh is not None
        assert session.gh_username == "testuser"
        assert any("Authenticated with GitHub" in t for t in texts)


def test_setup_detects_claude_oauth(monkeypatch, creds_path, config_path) -> None:
    """Setup with stored Claude OAuth marks provider as connected."""
    creds.update(claude=CLAUDE_OAUTH)
    with tempfile.TemporaryDirectory() as tmp:
        _setup_repo(monkeypatch, tmp)
        session = Session()
        events: __import__("queue").Queue[Event] = __import__("queue").Queue()
        engine = Engine(session, events)

        engine.run_task(TaskType.SETUP, "")
        texts = output_texts(drain(events))

        assert session.claude_connected is True
        assert any("Connected to Anthropic" in t for t in texts)


def test_setup_detects_chatgpt_oauth(monkeypatch, creds_path, config_path) -> None:
    """Setup with stored ChatGPT OAuth marks provider as connected."""
    creds.update(chatgpt=CHATGPT_OAUTH)
    with tempfile.TemporaryDirectory() as tmp:
        _setup_repo(monkeypatch, tmp)
        session = Session()
        events: __import__("queue").Queue[Event] = __import__("queue").Queue()
        engine = Engine(session, events)

        engine.run_task(TaskType.SETUP, "")
        texts = output_texts(drain(events))

        assert session.chatgpt_connected is True
        assert any("Connected to ChatGPT" in t for t in texts)


def test_setup_detects_openai_key(monkeypatch, creds_path, config_path) -> None:
    """Setup with stored OpenAI key marks provider as connected."""
    creds.update(openai_api_key="sk-test")
    with tempfile.TemporaryDirectory() as tmp:
        _setup_repo(monkeypatch, tmp)
        session = Session()
        events: __import__("queue").Queue[Event] = __import__("queue").Queue()
        engine = Engine(session, events)

        engine.run_task(TaskType.SETUP, "")
        texts = output_texts(drain(events))

        assert session.openai_connected is True
        assert any("Connected to OpenAI" in t for t in texts)


# ── Endpoints ────────────────────────────────────────────────────────


def test_setup_lists_endpoints(monkeypatch, creds_path, config_path) -> None:
    """Setup lists stored endpoints."""
    from rbtr.providers.endpoint import save_endpoint

    save_endpoint("ollama", "http://localhost:11434/v1", "")

    with tempfile.TemporaryDirectory() as tmp:
        _setup_repo(monkeypatch, tmp)
        session = Session()
        events: __import__("queue").Queue[Event] = __import__("queue").Queue()
        engine = Engine(session, events)

        engine.run_task(TaskType.SETUP, "")
        texts = output_texts(drain(events))

        assert any("ollama" in t and "localhost:11434" in t for t in texts)
        # With an endpoint present, "No LLM connected" should NOT appear
        assert not any("No LLM connected" in t for t in texts)


# ── Saved model ──────────────────────────────────────────────────────


def test_setup_loads_saved_model(monkeypatch, creds_path, config_path) -> None:
    """Setup restores saved model preference from config."""
    config.update(model="claude/claude-sonnet-4-20250514")

    with tempfile.TemporaryDirectory() as tmp:
        _setup_repo(monkeypatch, tmp)
        session = Session()
        events: __import__("queue").Queue[Event] = __import__("queue").Queue()
        engine = Engine(session, events)

        engine.run_task(TaskType.SETUP, "")
        drain(events)

        assert session.model_name == "claude/claude-sonnet-4-20250514"


# ── Error handling ───────────────────────────────────────────────────


def test_run_task_unexpected_error_emits_failure(config_path) -> None:
    """Generic exception in a task emits error output and TaskFinished(success=False)."""
    engine, events, _ = make_engine()

    # Force an unexpected error during a command
    def _explode(eng, args):
        raise RuntimeError("kaboom")

    import rbtr.engine.core as core_mod

    original = core_mod.cmd_connect
    core_mod.cmd_connect = _explode
    try:
        engine.run_task(TaskType.COMMAND, "/connect claude")
        evts = drain(events)

        assert evts[-1].success is False
        assert evts[-1].cancelled is False
        texts = output_texts(evts)
        assert any("kaboom" in t for t in texts)
    finally:
        core_mod.cmd_connect = original


def test_copy_to_clipboard_does_not_raise() -> None:
    """_copy_to_clipboard swallows errors gracefully."""
    # Just ensure it doesn't raise on any platform
    Engine._copy_to_clipboard("test text")
