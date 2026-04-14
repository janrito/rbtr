"""Tests for /connect — OpenAI, Claude, ChatGPT, and endpoint auth flows."""

from __future__ import annotations

from pathlib import Path

from pytest_mock import MockerFixture

from rbtr_legacy.creds import OAuthCreds, creds
from rbtr_legacy.engine.core import Engine
from rbtr_legacy.engine.types import TaskType
from rbtr_legacy.events import LinkOutput, TaskFinished
from rbtr_legacy.exceptions import PortBusyError, RbtrError
from rbtr_legacy.oauth import PendingLogin
from rbtr_legacy.providers import BuiltinProvider
from rbtr_legacy.providers.endpoint import load_endpoint, save_endpoint
from tests.helpers import drain, has_event_type, output_texts

# ── /connect openai ───────────────────────────────────────────────────


def test_connect_openai_saves_key(creds_path: Path, engine: Engine) -> None:
    engine.run_task(TaskType.COMMAND, "/connect openai sk-test-key-123")
    drained_events = drain(engine.events)
    assert isinstance(drained_events[-1], TaskFinished)
    assert drained_events[-1].success is True
    assert BuiltinProvider.OPENAI in engine.state.connected_providers
    assert creds.openai_api_key == "sk-test-key-123"
    texts = output_texts(drained_events)
    assert any("Connected to OpenAI" in t for t in texts)


def test_connect_openai_already_connected(creds_path: Path, engine: Engine) -> None:
    creds.update(openai_api_key="sk-existing")
    engine.run_task(TaskType.COMMAND, "/connect openai")
    drained_events = drain(engine.events)
    assert isinstance(drained_events[-1], TaskFinished)
    assert drained_events[-1].success is True
    assert BuiltinProvider.OPENAI in engine.state.connected_providers
    texts = output_texts(drained_events)
    assert any("Already connected" in t for t in texts)


def test_connect_openai_rejects_bad_key(creds_path: Path, engine: Engine) -> None:
    engine.run_task(TaskType.COMMAND, "/connect openai bad-key-format")
    drained_events = drain(engine.events)
    assert isinstance(drained_events[-1], TaskFinished)
    assert drained_events[-1].success is True
    assert BuiltinProvider.OPENAI not in engine.state.connected_providers
    texts = output_texts(drained_events)
    assert any("Invalid" in t for t in texts)


def test_connect_openai_no_key_shows_usage(creds_path: Path, engine: Engine) -> None:
    engine.run_task(TaskType.COMMAND, "/connect openai")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("Usage" in t for t in texts)


def test_connect_openai_replaces_existing_key(creds_path: Path, engine: Engine) -> None:
    """Providing a key when one exists replaces it."""
    creds.update(openai_api_key="sk-old")
    engine.run_task(TaskType.COMMAND, "/connect openai sk-new-key")
    drained_events = drain(engine.events)
    assert isinstance(drained_events[-1], TaskFinished)
    assert drained_events[-1].success is True
    assert BuiltinProvider.OPENAI in engine.state.connected_providers
    assert creds.openai_api_key == "sk-new-key"


# ── /connect dispatch ────────────────────────────────────────────────


def test_connect_no_args_shows_usage(engine: Engine, claude_oauth: OAuthCreds) -> None:
    engine.run_task(TaskType.COMMAND, "/connect")
    texts = output_texts(drain(engine.events))

    assert any("Usage" in t for t in texts)
    assert any("github" in t for t in texts)


def test_connect_unknown_service_warns(engine: Engine, claude_oauth: OAuthCreds) -> None:
    engine.run_task(TaskType.COMMAND, "/connect bogus")
    texts = output_texts(drain(engine.events))

    assert any("Unknown service" in t for t in texts)


# ── /connect claude ──────────────────────────────────────────────────

_CLAUDE_PENDING = PendingLogin(code_verifier="test-verifier")
_CLAUDE_AUTH_URL = "https://claude.ai/oauth/authorize?code=test"


def test_connect_claude_auto_flow(
    creds_path: Path, mocker: MockerFixture, engine: Engine, claude_oauth: OAuthCreds
) -> None:
    """Automatic localhost callback flow → sets connected."""
    mocker.patch(
        "rbtr_legacy.engine.connect_cmd.claude_provider.authenticate",
        return_value=claude_oauth,
    )
    mocker.patch("rbtr_legacy.engine.connect_cmd.get_models")

    engine.run_task(TaskType.COMMAND, "/connect claude")
    drained_events = drain(engine.events)

    assert BuiltinProvider.CLAUDE in engine.state.connected_providers
    assert creds.claude.access_token == claude_oauth.access_token
    texts = output_texts(drained_events)
    assert any("Connected to Anthropic" in t for t in texts)


def test_connect_claude_port_busy_falls_back_to_manual(
    creds_path: Path, mocker: MockerFixture, engine: Engine, claude_oauth: OAuthCreds
) -> None:
    """Port busy → falls back to manual URL paste flow."""
    mocker.patch(
        "rbtr_legacy.engine.connect_cmd.claude_provider.authenticate",
        side_effect=PortBusyError("port busy"),
    )
    mocker.patch(
        "rbtr_legacy.engine.connect_cmd.claude_provider.begin_login",
        return_value=(_CLAUDE_AUTH_URL, _CLAUDE_PENDING),
    )

    engine.run_task(TaskType.COMMAND, "/connect claude")
    drained_events = drain(engine.events)

    assert engine.state.pending_logins.get(BuiltinProvider.CLAUDE) is _CLAUDE_PENDING
    assert has_event_type(drained_events, LinkOutput)
    texts = output_texts(drained_events)
    assert any("/connect claude" in t for t in texts)


def test_connect_claude_manual_phase2_completes(
    creds_path: Path, mocker: MockerFixture, engine: Engine, claude_oauth: OAuthCreds
) -> None:
    """Manual fallback phase 2: paste URL → complete login."""
    engine.state.pending_logins[BuiltinProvider.CLAUDE] = _CLAUDE_PENDING

    mocker.patch(
        "rbtr_legacy.engine.connect_cmd.claude_provider.complete_login",
        return_value=claude_oauth,
    )
    mocker.patch("rbtr_legacy.engine.connect_cmd.get_models")

    engine.run_task(
        TaskType.COMMAND, "/connect claude http://localhost:53692/callback?code=abc&state=xyz"
    )

    assert BuiltinProvider.CLAUDE in engine.state.connected_providers
    assert creds.claude.access_token == claude_oauth.access_token


def test_connect_claude_already_connected_restarts_login(
    creds_path: Path,
    mocker: MockerFixture,
    engine: Engine,
    chatgpt_oauth: OAuthCreds,
    claude_oauth: OAuthCreds,
) -> None:
    """/connect claude when already authenticated restarts the login flow."""
    creds.update(claude=claude_oauth)

    mocker.patch(
        "rbtr_legacy.engine.connect_cmd.claude_provider.authenticate",
        return_value=claude_oauth,
    )
    mocker.patch("rbtr_legacy.engine.connect_cmd.get_models")

    engine.run_task(TaskType.COMMAND, "/connect claude")
    texts = output_texts(drain(engine.events))

    assert BuiltinProvider.CLAUDE in engine.state.connected_providers
    assert any("Connected to" in t for t in texts)
    assert not any("Already connected" in t for t in texts)


# ── /connect chatgpt ─────────────────────────────────────────────────

_CHATGPT_PENDING = PendingLogin(code_verifier="test-verifier", state="test-state")
_CHATGPT_AUTH_URL = "https://auth.openai.com/authorize?code=test"


def test_connect_chatgpt_auto_flow(
    creds_path: Path, mocker: MockerFixture, engine: Engine, chatgpt_oauth: OAuthCreds
) -> None:
    """Automatic localhost callback flow → sets connected."""

    mocker.patch(
        "rbtr_legacy.engine.connect_cmd.codex_provider.authenticate",
        return_value=chatgpt_oauth,
    )
    mocker.patch("rbtr_legacy.engine.connect_cmd.get_models")

    engine.run_task(TaskType.COMMAND, "/connect chatgpt")
    drained_events = drain(engine.events)

    assert BuiltinProvider.CHATGPT in engine.state.connected_providers
    assert creds.chatgpt.access_token == chatgpt_oauth.access_token
    texts = output_texts(drained_events)
    assert any("Connected to ChatGPT" in t for t in texts)


def test_connect_chatgpt_port_busy_fallback(
    creds_path: Path, mocker: MockerFixture, engine: Engine
) -> None:
    """Port-busy error falls back to manual paste flow."""

    mocker.patch(
        "rbtr_legacy.engine.connect_cmd.codex_provider.authenticate",
        side_effect=PortBusyError("Port 1455 is busy"),
    )
    mocker.patch(
        "rbtr_legacy.engine.connect_cmd.codex_provider.begin_login",
        return_value=(_CHATGPT_AUTH_URL, _CHATGPT_PENDING),
    )

    engine.run_task(TaskType.COMMAND, "/connect chatgpt")
    drained_events = drain(engine.events)

    assert engine.state.pending_logins.get(BuiltinProvider.CHATGPT) is _CHATGPT_PENDING
    assert has_event_type(drained_events, LinkOutput)
    texts = output_texts(drained_events)
    assert any("/connect chatgpt" in t for t in texts)


def test_connect_chatgpt_non_port_error_warns(
    creds_path: Path, mocker: MockerFixture, engine: Engine
) -> None:
    """Non-port RbtrError warns without fallback."""

    mocker.patch(
        "rbtr_legacy.engine.connect_cmd.codex_provider.authenticate",
        side_effect=RbtrError("token exchange failed"),
    )

    engine.run_task(TaskType.COMMAND, "/connect chatgpt")
    texts = output_texts(drain(engine.events))

    assert BuiltinProvider.CHATGPT not in engine.state.connected_providers
    assert any("failed" in t.lower() for t in texts)


def test_connect_chatgpt_generic_error_warns(
    creds_path: Path, mocker: MockerFixture, engine: Engine, chatgpt_oauth: OAuthCreds
) -> None:
    """Generic exception warns without fallback."""

    mocker.patch(
        "rbtr_legacy.engine.connect_cmd.codex_provider.authenticate",
        side_effect=RuntimeError("unexpected"),
    )

    engine.run_task(TaskType.COMMAND, "/connect chatgpt")
    texts = output_texts(drain(engine.events))

    assert BuiltinProvider.CHATGPT not in engine.state.connected_providers
    assert any("failed" in t.lower() for t in texts)


def test_connect_chatgpt_phase2_completes_login(
    creds_path: Path, mocker: MockerFixture, engine: Engine, chatgpt_oauth: OAuthCreds
) -> None:
    """Phase 2: user pastes redirect URL → completes login."""
    engine.state.pending_logins[BuiltinProvider.CHATGPT] = _CHATGPT_PENDING

    mocker.patch(
        "rbtr_legacy.engine.connect_cmd.codex_provider.complete_login",
        return_value=chatgpt_oauth,
    )
    mocker.patch("rbtr_legacy.engine.connect_cmd.get_models")

    engine.run_task(TaskType.COMMAND, "/connect chatgpt http://localhost:1455/callback?code=x")
    drained_events = drain(engine.events)

    assert BuiltinProvider.CHATGPT in engine.state.connected_providers
    assert engine.state.pending_logins.get(BuiltinProvider.CHATGPT) is None
    texts = output_texts(drained_events)
    assert any("Connected to ChatGPT" in t for t in texts)


def test_connect_chatgpt_phase2_without_pending_warns(engine: Engine) -> None:
    """Phase 2 without pending login warns."""

    engine.run_task(TaskType.COMMAND, "/connect chatgpt http://localhost:1455/callback?code=x")
    texts = output_texts(drain(engine.events))

    assert any("No pending" in t for t in texts)


def test_connect_chatgpt_phase2_failure_clears_pending(
    mocker: MockerFixture, engine: Engine, chatgpt_oauth: OAuthCreds
) -> None:
    """Phase 2 failure warns and clears pending state."""
    engine.state.pending_logins[BuiltinProvider.CHATGPT] = _CHATGPT_PENDING

    mocker.patch(
        "rbtr_legacy.engine.connect_cmd.codex_provider.complete_login",
        side_effect=ValueError("bad redirect URL"),
    )

    engine.run_task(TaskType.COMMAND, "/connect chatgpt http://bad-url")
    texts = output_texts(drain(engine.events))

    assert engine.state.pending_logins.get(BuiltinProvider.CHATGPT) is None
    assert any("failed" in t.lower() for t in texts)


def test_connect_chatgpt_already_connected_restarts_login(
    creds_path: Path, mocker: MockerFixture, engine: Engine, chatgpt_oauth: OAuthCreds
) -> None:
    """/connect chatgpt when already authenticated restarts the login flow."""
    creds.update(chatgpt=chatgpt_oauth)

    mocker.patch(
        "rbtr_legacy.engine.connect_cmd.codex_provider.authenticate",
        return_value=chatgpt_oauth,
    )
    mocker.patch("rbtr_legacy.engine.connect_cmd.get_models")

    engine.run_task(TaskType.COMMAND, "/connect chatgpt")
    texts = output_texts(drain(engine.events))

    # Should re-authenticate, not short-circuit.
    assert BuiltinProvider.CHATGPT in engine.state.connected_providers
    assert any("Connected to" in t for t in texts)
    assert not any("Already connected" in t for t in texts)


# ── /connect endpoint ────────────────────────────────────────────────


def test_connect_endpoint_no_args_shows_usage(
    config_path: Path, creds_path: Path, engine: Engine
) -> None:
    """Bare /connect endpoint shows usage instructions."""
    engine.run_task(TaskType.COMMAND, "/connect endpoint")
    texts = output_texts(drain(engine.events))

    assert any("Usage" in t for t in texts)
    assert any("base_url" in t for t in texts)


def test_connect_endpoint_saves_and_confirms(
    config_path: Path, creds_path: Path, mocker: MockerFixture, engine: Engine
) -> None:
    """/connect endpoint name url key saves the endpoint."""
    mocker.patch("rbtr_legacy.engine.connect_cmd.get_models")
    engine.run_task(
        TaskType.COMMAND,
        "/connect endpoint myendpoint http://localhost:11434/v1 sk-test",
    )
    texts = output_texts(drain(engine.events))

    assert any("connected" in t.lower() for t in texts)
    assert any("/model myendpoint/" in t for t in texts)
    assert "myendpoint" in engine.state.connected_providers

    # Verify persistence

    ep = load_endpoint("myendpoint")
    assert ep is not None
    assert ep.base_url == "http://localhost:11434/v1"


def test_connect_endpoint_invalid_name_warns(
    config_path: Path, creds_path: Path, engine: Engine
) -> None:
    """/connect endpoint with invalid name warns."""
    engine.run_task(
        TaskType.COMMAND,
        "/connect endpoint BAD NAME http://localhost:11434/v1",
    )
    texts = output_texts(drain(engine.events))

    assert any("Invalid" in t for t in texts)


def test_connect_endpoint_lists_existing(
    config_path: Path, creds_path: Path, engine: Engine
) -> None:
    """/connect endpoint with no args lists existing endpoints."""

    save_endpoint("myep", "http://localhost:11434/v1", "")

    engine.run_task(TaskType.COMMAND, "/connect endpoint")
    texts = output_texts(drain(engine.events))

    assert any("myep" in t for t in texts)
