"""Tests for Engine._run_setup — credential detection, endpoint listing,
saved model loading, and the unexpected-error handler in run_task.

Covers the untested branches in engine/core.py: setup with various
credential combinations, endpoint listing, saved model preference,
and the generic Exception handler.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.config import config
from rbtr.creds import creds
from rbtr.engine import Engine, TaskType

from .conftest import CHATGPT_OAUTH, CLAUDE_OAUTH, drain, output_texts


@pytest.fixture
def setup_engine(monkeypatch: pytest.MonkeyPatch, repo_engine: Engine) -> Engine:
    """Engine chdir'd into its repo with a remote — ready for _run_setup discovery."""
    repo = repo_engine.state.repo
    assert repo is not None
    repo.remotes.create("origin", "git@github.com:acme/widgets.git")
    monkeypatch.chdir(repo.workdir)
    repo_engine.state.owner = ""
    repo_engine.state.repo_name = ""
    return repo_engine


# ── Setup with credentials ───────────────────────────────────────────


def test_setup_detects_github_token(
    monkeypatch: pytest.MonkeyPatch, creds_path: Path, config_path: Path, setup_engine: Engine
) -> None:
    """Setup with a stored GitHub token authenticates automatically."""
    creds.update(github_token="ghp_test123")

    from unittest.mock import MagicMock

    fake_gh = MagicMock()
    fake_gh.get_user.return_value.login = "testuser"
    monkeypatch.setattr("rbtr.engine.core.Github", lambda **_kw: fake_gh)

    engine = setup_engine

    engine.run_task(TaskType.SETUP, "")
    texts = output_texts(drain(engine.events))

    assert engine.state.gh is not None
    assert engine.state.gh_username == "testuser"
    assert any("Authenticated with GitHub" in t for t in texts)


def test_setup_detects_claude_oauth(
    creds_path: Path, config_path: Path, setup_engine: Engine
) -> None:
    """Setup with stored Claude OAuth marks provider as connected."""
    creds.update(claude=CLAUDE_OAUTH)
    engine = setup_engine

    engine.run_task(TaskType.SETUP, "")
    texts = output_texts(drain(engine.events))

    assert engine.state.claude_connected is True
    assert any("Connected to Anthropic" in t for t in texts)


def test_setup_detects_chatgpt_oauth(
    creds_path: Path, config_path: Path, setup_engine: Engine
) -> None:
    """Setup with stored ChatGPT OAuth marks provider as connected."""
    creds.update(chatgpt=CHATGPT_OAUTH)
    engine = setup_engine

    engine.run_task(TaskType.SETUP, "")
    texts = output_texts(drain(engine.events))

    assert engine.state.chatgpt_connected is True
    assert any("Connected to ChatGPT" in t for t in texts)


def test_setup_detects_openai_key(
    creds_path: Path, config_path: Path, setup_engine: Engine
) -> None:
    """Setup with stored OpenAI key marks provider as connected."""
    creds.update(openai_api_key="sk-test")
    engine = setup_engine

    engine.run_task(TaskType.SETUP, "")
    texts = output_texts(drain(engine.events))

    assert engine.state.openai_connected is True
    assert any("Connected to OpenAI" in t for t in texts)


# ── Endpoints ────────────────────────────────────────────────────────


def test_setup_lists_endpoints(creds_path: Path, config_path: Path, setup_engine: Engine) -> None:
    """Setup lists stored endpoints."""
    from rbtr.providers.endpoint import save_endpoint

    save_endpoint("ollama", "http://localhost:11434/v1", "")

    engine = setup_engine

    engine.run_task(TaskType.SETUP, "")
    texts = output_texts(drain(engine.events))

    assert any("ollama" in t and "localhost:11434" in t for t in texts)
    assert not any("No LLM connected" in t for t in texts)


# ── Saved model ──────────────────────────────────────────────────────


def test_setup_loads_saved_model(creds_path: Path, config_path: Path, setup_engine: Engine) -> None:
    """Setup restores saved model preference from config."""
    config.update(model="claude/claude-sonnet-4-20250514")

    engine = setup_engine

    engine.run_task(TaskType.SETUP, "")
    drain(engine.events)

    assert engine.state.model_name == "claude/claude-sonnet-4-20250514"


# ── Error handling ───────────────────────────────────────────────────


def test_run_task_unexpected_error_emits_failure(config_path: Path, engine: Engine) -> None:
    """Generic exception in a task emits error output and TaskFinished(success=False)."""

    def _explode(eng, args):
        raise RuntimeError("kaboom")

    import rbtr.engine.core as core_mod

    original = core_mod.cmd_connect
    core_mod.cmd_connect = _explode
    try:
        engine.run_task(TaskType.COMMAND, "/connect claude")
        drained_events = drain(engine.events)

        assert drained_events[-1].success is False
        assert drained_events[-1].cancelled is False
        texts = output_texts(drained_events)
        assert any("kaboom" in t for t in texts)
    finally:
        core_mod.cmd_connect = original


def test_copy_to_clipboard_does_not_raise() -> None:
    """_copy_to_clipboard swallows errors gracefully."""
    Engine._copy_to_clipboard("test text")
