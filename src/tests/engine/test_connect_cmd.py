"""Tests for /connect — Claude, ChatGPT, and endpoint auth flows.

OpenAI and GitHub /connect paths are already covered in test_engine.py.
This file covers the remaining untested flows: Claude two-phase OAuth,
ChatGPT auto + manual fallback, and endpoint persistence.
"""

from __future__ import annotations

from rbtr.creds import creds
from rbtr.engine import TaskType
from rbtr.events import LinkOutput
from rbtr.exceptions import RbtrError
from rbtr.providers.claude import PendingLogin as ClaudePending
from rbtr.providers.openai_codex import PendingLogin as ChatGPTPending

from .conftest import CHATGPT_OAUTH, CLAUDE_OAUTH, drain, has_event_type, make_engine, output_texts

# ── /connect dispatch ────────────────────────────────────────────────


def test_connect_no_args_shows_usage() -> None:
    engine, events, _ = make_engine()
    engine.run_task(TaskType.COMMAND, "/connect")
    texts = output_texts(drain(events))

    assert any("Usage" in t for t in texts)
    assert any("github" in t for t in texts)


def test_connect_unknown_service_warns() -> None:
    engine, events, _ = make_engine()
    engine.run_task(TaskType.COMMAND, "/connect bogus")
    texts = output_texts(drain(events))

    assert any("Unknown service" in t for t in texts)


# ── /connect claude ──────────────────────────────────────────────────

_CLAUDE_PENDING = ClaudePending(code_verifier="test-verifier")
_AUTHORIZE_URL = "https://console.anthropic.com/oauth/authorize?code=test"


def test_connect_claude_phase1_emits_link(creds_path, mocker) -> None:
    """Phase 1: begin_login → stores pending, emits link."""
    engine, events, session = make_engine()
    mocker.patch(
        "rbtr.engine.connect.claude_provider.begin_login",
        return_value=(_AUTHORIZE_URL, _CLAUDE_PENDING),
    )

    engine.run_task(TaskType.COMMAND, "/connect claude")
    evts = drain(events)

    assert session.claude_pending_login is _CLAUDE_PENDING
    assert has_event_type(evts, LinkOutput)
    texts = output_texts(evts)
    assert any("/connect claude" in t for t in texts)


def test_connect_claude_phase2_completes_login(creds_path, mocker) -> None:
    """Phase 2: parse code → complete login → sets connected."""
    engine, events, session = make_engine()
    session.claude_pending_login = _CLAUDE_PENDING

    mocker.patch(
        "rbtr.engine.connect.claude_provider.parse_auth_code",
        return_value=("auth-code", "state-value"),
    )
    mocker.patch(
        "rbtr.engine.connect.claude_provider.complete_login",
        return_value=CLAUDE_OAUTH,
    )
    mocker.patch("rbtr.engine.connect.get_models")

    engine.run_task(TaskType.COMMAND, "/connect claude auth-code#state-value")
    evts = drain(events)

    assert session.claude_connected is True
    assert session.claude_pending_login is None
    assert creds.claude.access_token == CLAUDE_OAUTH.access_token
    texts = output_texts(evts)
    assert any("Connected to Anthropic" in t for t in texts)


def test_connect_claude_phase2_without_pending_warns() -> None:
    """Phase 2 without pending login warns."""
    engine, events, session = make_engine()
    assert session.claude_pending_login is None

    engine.run_task(TaskType.COMMAND, "/connect claude some-code")
    texts = output_texts(drain(events))

    assert any("No pending" in t for t in texts)


def test_connect_claude_phase2_failure_clears_pending(mocker) -> None:
    """Phase 2 failure warns and clears pending state."""
    engine, events, session = make_engine()
    session.claude_pending_login = _CLAUDE_PENDING

    mocker.patch(
        "rbtr.engine.connect.claude_provider.parse_auth_code",
        side_effect=ValueError("bad code"),
    )

    engine.run_task(TaskType.COMMAND, "/connect claude bad-code")
    texts = output_texts(drain(events))

    assert session.claude_pending_login is None
    assert any("failed" in t.lower() for t in texts)


def test_connect_claude_already_connected(creds_path) -> None:
    """/connect claude when already authenticated says so."""
    creds.update(claude=CLAUDE_OAUTH)
    engine, events, session = make_engine()

    engine.run_task(TaskType.COMMAND, "/connect claude")
    texts = output_texts(drain(events))

    assert session.claude_connected is True
    assert any("Already connected" in t for t in texts)


# ── /connect chatgpt ─────────────────────────────────────────────────

_CHATGPT_PENDING = ChatGPTPending(code_verifier="test-verifier", state="test-state")
_CHATGPT_AUTH_URL = "https://auth.openai.com/authorize?code=test"


def test_connect_chatgpt_auto_flow(creds_path, mocker) -> None:
    """Automatic localhost callback flow → sets connected."""
    engine, events, session = make_engine()

    mocker.patch(
        "rbtr.engine.connect.codex_provider.authenticate",
        return_value=CHATGPT_OAUTH,
    )
    mocker.patch("rbtr.engine.connect.get_models")

    engine.run_task(TaskType.COMMAND, "/connect chatgpt")
    evts = drain(events)

    assert session.chatgpt_connected is True
    assert creds.chatgpt.access_token == CHATGPT_OAUTH.access_token
    texts = output_texts(evts)
    assert any("Connected to ChatGPT" in t for t in texts)


def test_connect_chatgpt_port_busy_fallback(creds_path, mocker) -> None:
    """Port-busy error falls back to manual paste flow."""
    engine, events, session = make_engine()

    mocker.patch(
        "rbtr.engine.connect.codex_provider.authenticate",
        side_effect=RbtrError("Port 1455 is busy"),
    )
    mocker.patch(
        "rbtr.engine.connect.codex_provider.begin_login",
        return_value=(_CHATGPT_AUTH_URL, _CHATGPT_PENDING),
    )

    engine.run_task(TaskType.COMMAND, "/connect chatgpt")
    evts = drain(events)

    assert session.chatgpt_pending_login is _CHATGPT_PENDING
    assert has_event_type(evts, LinkOutput)
    texts = output_texts(evts)
    assert any("/connect chatgpt" in t for t in texts)


def test_connect_chatgpt_non_port_error_warns(creds_path, mocker) -> None:
    """Non-port RbtrError warns without fallback."""
    engine, events, session = make_engine()

    mocker.patch(
        "rbtr.engine.connect.codex_provider.authenticate",
        side_effect=RbtrError("token exchange failed"),
    )

    engine.run_task(TaskType.COMMAND, "/connect chatgpt")
    texts = output_texts(drain(events))

    assert session.chatgpt_connected is False
    assert any("failed" in t.lower() for t in texts)


def test_connect_chatgpt_generic_error_warns(creds_path, mocker) -> None:
    """Generic exception warns without fallback."""
    engine, events, session = make_engine()

    mocker.patch(
        "rbtr.engine.connect.codex_provider.authenticate",
        side_effect=RuntimeError("unexpected"),
    )

    engine.run_task(TaskType.COMMAND, "/connect chatgpt")
    texts = output_texts(drain(events))

    assert session.chatgpt_connected is False
    assert any("failed" in t.lower() for t in texts)


def test_connect_chatgpt_phase2_completes_login(creds_path, mocker) -> None:
    """Phase 2: user pastes redirect URL → completes login."""
    engine, events, session = make_engine()
    session.chatgpt_pending_login = _CHATGPT_PENDING

    mocker.patch(
        "rbtr.engine.connect.codex_provider.complete_login",
        return_value=CHATGPT_OAUTH,
    )
    mocker.patch("rbtr.engine.connect.get_models")

    engine.run_task(TaskType.COMMAND, "/connect chatgpt http://localhost:1455/callback?code=x")
    evts = drain(events)

    assert session.chatgpt_connected is True
    assert session.chatgpt_pending_login is None
    texts = output_texts(evts)
    assert any("Connected to ChatGPT" in t for t in texts)


def test_connect_chatgpt_phase2_without_pending_warns() -> None:
    """Phase 2 without pending login warns."""
    engine, events, _ = make_engine()

    engine.run_task(TaskType.COMMAND, "/connect chatgpt http://localhost:1455/callback?code=x")
    texts = output_texts(drain(events))

    assert any("No pending" in t for t in texts)


def test_connect_chatgpt_phase2_failure_clears_pending(mocker) -> None:
    """Phase 2 failure warns and clears pending state."""
    engine, events, session = make_engine()
    session.chatgpt_pending_login = _CHATGPT_PENDING

    mocker.patch(
        "rbtr.engine.connect.codex_provider.complete_login",
        side_effect=ValueError("bad redirect URL"),
    )

    engine.run_task(TaskType.COMMAND, "/connect chatgpt http://bad-url")
    texts = output_texts(drain(events))

    assert session.chatgpt_pending_login is None
    assert any("failed" in t.lower() for t in texts)


def test_connect_chatgpt_already_connected(creds_path) -> None:
    """/connect chatgpt when already authenticated says so."""
    creds.update(chatgpt=CHATGPT_OAUTH)
    engine, events, session = make_engine()

    engine.run_task(TaskType.COMMAND, "/connect chatgpt")
    texts = output_texts(drain(events))

    assert session.chatgpt_connected is True
    assert any("Already connected" in t for t in texts)


# ── /connect endpoint ────────────────────────────────────────────────


def test_connect_endpoint_no_args_shows_usage(config_path, creds_path) -> None:
    """Bare /connect endpoint shows usage instructions."""
    engine, events, _ = make_engine()
    engine.run_task(TaskType.COMMAND, "/connect endpoint")
    texts = output_texts(drain(events))

    assert any("Usage" in t for t in texts)
    assert any("base_url" in t for t in texts)


def test_connect_endpoint_saves_and_confirms(config_path, creds_path, mocker) -> None:
    """/connect endpoint name url key saves the endpoint."""
    mocker.patch("rbtr.engine.connect.get_models")
    engine, events, _ = make_engine()
    engine.run_task(
        TaskType.COMMAND,
        "/connect endpoint myendpoint http://localhost:11434/v1 sk-test",
    )
    texts = output_texts(drain(events))

    assert any("connected" in t.lower() for t in texts)
    assert any("/model myendpoint/" in t for t in texts)

    # Verify persistence
    from rbtr.providers.endpoint import load_endpoint

    ep = load_endpoint("myendpoint")
    assert ep is not None
    assert ep.base_url == "http://localhost:11434/v1"


def test_connect_endpoint_invalid_name_warns(config_path, creds_path) -> None:
    """/connect endpoint with invalid name warns."""
    engine, events, _ = make_engine()
    engine.run_task(
        TaskType.COMMAND,
        "/connect endpoint BAD NAME http://localhost:11434/v1",
    )
    texts = output_texts(drain(events))

    assert any("Invalid" in t for t in texts)


def test_connect_endpoint_lists_existing(config_path, creds_path) -> None:
    """/connect endpoint with no args lists existing endpoints."""
    from rbtr.providers.endpoint import save_endpoint

    save_endpoint("myep", "http://localhost:11434/v1", "")

    engine, events, _ = make_engine()
    engine.run_task(TaskType.COMMAND, "/connect endpoint")
    texts = output_texts(drain(events))

    assert any("myep" in t for t in texts)
