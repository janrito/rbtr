"""Handler for /connect — service authentication flows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from github import Auth, Github

from rbtr.config import config
from rbtr.creds import creds
from rbtr.events import LinkOutput
from rbtr.exceptions import RbtrError
from rbtr.github import auth
from rbtr.oauth import oauth_is_set
from rbtr.providers import (
    claude as claude_provider,
    endpoint as endpoint_provider,
    openai_codex as codex_provider,
)
from rbtr.styles import CODE_HIGHLIGHT, LINK_STYLE

from .model import get_models
from .types import Service, TaskCancelled

if TYPE_CHECKING:
    from .core import Engine


def cmd_connect(engine: Engine, args: str) -> None:
    """Dispatch /connect to the appropriate service handler."""
    parts = args.strip().split(maxsplit=1)
    name = parts[0].lower() if parts else ""
    extra = parts[1].strip() if len(parts) > 1 else ""

    try:
        service = Service(name) if name else None
    except ValueError:
        engine._warn(f"Unknown service: {name}. Supported: {', '.join(s.key for s in Service)}")
        return

    match service:
        case None:
            engine._out("Usage: /connect <service>")
            for s in Service:
                engine._out(f"  /connect {s.key:<16} — {s.description}")
        case Service.GITHUB:
            _connect_github(engine)
        case Service.CLAUDE:
            _connect_claude(engine, extra)
        case Service.CHATGPT:
            _connect_chatgpt(engine, extra)
        case Service.OPENAI:
            _connect_openai(engine, extra)
        case Service.ENDPOINT:
            _connect_endpoint(engine, extra)


def _connect_github(engine: Engine) -> None:

    creds.update(github_token="")
    try:
        device = auth.request_device_code()
        user_code = device["user_code"]
        verification_uri = device["verification_uri"]
        device_code = device["device_code"]
        interval = int(device["interval"])

        engine._emit(
            LinkOutput(
                markup=(
                    f"Open [link={verification_uri}][{LINK_STYLE}]{verification_uri}"
                    f"[/{LINK_STYLE}][/link] and enter code: "
                    f"[{CODE_HIGHLIGHT}]{user_code}[/{CODE_HIGHLIGHT}]"
                )
            )
        )
        engine._copy_to_clipboard(user_code)
        engine._out("Code copied to clipboard.")
        engine._flush()

        engine._out("Waiting for authorization…")
        token = auth.poll_for_token(device_code, interval, cancel=engine._cancel)
        engine._check_cancel()
        engine._clear()

        creds.update(github_token=token)
        gh = Github(auth=Auth.Token(token), timeout=config.github.timeout)
        engine.state.gh = gh
        engine.state.gh_username = gh.get_user().login
        engine._out("Authenticated with GitHub.")
    except TaskCancelled:
        raise
    except Exception as e:
        engine._check_cancel()
        engine._warn(f"Authentication failed: {e}")


def _connect_claude(engine: Engine, auth_code: str) -> None:

    # Phase 2: user pasted the code#state from the browser
    if auth_code:
        pending = engine.state.claude_pending_login
        if pending is None:
            engine._warn("No pending Claude login. Run /connect claude first to start.")
            return
        try:
            code, state = claude_provider.parse_auth_code(auth_code)
            oauth_creds = claude_provider.complete_login(code, state, pending)
            creds.update(claude=oauth_creds)
            engine.state.claude_connected = True
            engine.state.claude_pending_login = None
            engine._out("Connected to Anthropic. LLM is ready.")
            get_models(engine)
        except TaskCancelled:
            raise
        except Exception as e:
            engine._check_cancel()
            engine.state.claude_pending_login = None
            engine._warn(f"Anthropic connection failed: {e}")
        return

    # Check for existing credentials
    if oauth_is_set(creds.claude):
        engine.state.claude_connected = True
        engine._out("Already connected to Anthropic.")
        return

    # Phase 1: generate PKCE, open browser, show instructions
    authorize_url, pending = claude_provider.begin_login()
    engine.state.claude_pending_login = pending

    engine._emit(
        LinkOutput(
            markup=(
                f"Opening browser to sign in with your Claude account…\n"
                f"If the browser didn't open, visit: "
                f"[link={authorize_url}][{LINK_STYLE}]{authorize_url}[/{LINK_STYLE}][/link]"
            )
        )
    )
    engine._out("")
    engine._out("After authorizing, paste the code shown in the browser:")
    engine._out("  /connect claude <code>")


def _connect_chatgpt(engine: Engine, extra: str) -> None:

    # Phase 2 (manual fallback): user pasted the redirect URL
    if extra:
        pending = engine.state.chatgpt_pending_login
        if pending is None:
            engine._warn("No pending ChatGPT login. Run /connect chatgpt first to start.")
            return
        try:
            oauth_creds = codex_provider.complete_login(extra, pending)
            creds.update(chatgpt=oauth_creds)
            engine.state.chatgpt_connected = True
            engine.state.chatgpt_pending_login = None
            engine._out("Connected to ChatGPT. LLM is ready.")
            get_models(engine)
        except TaskCancelled:
            raise
        except Exception as e:
            engine._check_cancel()
            engine.state.chatgpt_pending_login = None
            engine._warn(f"ChatGPT connection failed: {e}")
        return

    # Check for existing credentials
    if oauth_is_set(creds.chatgpt):
        engine.state.chatgpt_connected = True
        engine._out("Already connected to ChatGPT.")
        return

    # Phase 1: try automatic localhost callback
    engine._out("Opening browser to sign in with your ChatGPT account…")
    engine._flush()

    try:
        engine._out("Waiting for authorization…")
        oauth_creds = codex_provider.authenticate(cancel=engine._cancel)
        engine._check_cancel()
        engine._clear()

        creds.update(chatgpt=oauth_creds)
        engine.state.chatgpt_connected = True
        engine._out("Connected to ChatGPT. LLM is ready.")
        get_models(engine)
    except TaskCancelled:
        raise
    except RbtrError as e:
        engine._check_cancel()
        engine._clear()
        if "busy" in str(e).lower() or "1455" in str(e):
            # Port busy — fall back to manual paste flow
            authorize_url, pending = codex_provider.begin_login()
            engine.state.chatgpt_pending_login = pending
            engine._emit(
                LinkOutput(
                    markup=(
                        f"Port 1455 is busy. Opening browser manually…\n"
                        f"If the browser didn't open, visit: "
                        f"[link={authorize_url}][{LINK_STYLE}]"
                        f"{authorize_url}[/{LINK_STYLE}][/link]"
                    )
                )
            )
            engine._out("")
            engine._out("After authorizing, paste the redirect URL from your browser:")
            engine._out("  /connect chatgpt <url>")
        else:
            engine._warn(f"ChatGPT connection failed: {e}")
    except Exception as e:
        engine._check_cancel()
        engine._clear()
        engine._warn(f"ChatGPT connection failed: {e}")


def _connect_openai(engine: Engine, api_key: str) -> None:

    # Check for existing key
    if creds.openai_api_key and not api_key:
        engine.state.openai_connected = True
        engine._out("Already connected to OpenAI.")
        return

    if not api_key:
        engine._out("Usage: /connect openai <api_key>")
        engine._out("Get your API key from https://platform.openai.com/api-keys")
        return

    # Validate the key format (sk-... or sk-proj-...)
    if not api_key.startswith("sk-"):
        engine._warn("Invalid API key format. OpenAI keys start with 'sk-'.")
        return

    creds.update(openai_api_key=api_key)
    engine.state.openai_connected = True
    engine._out("Connected to OpenAI. LLM is ready.")
    get_models(engine)


def _connect_endpoint(engine: Engine, args: str) -> None:

    parts = args.split()
    if len(parts) < 2:
        engine._out("Usage: /connect endpoint <name> <base_url> [api_key]")
        engine._out("")
        engine._out("  name     — Short identifier (lowercase, e.g. deepinfra)")
        engine._out("  base_url — OpenAI-compatible API base URL")
        engine._out("  api_key  — API key (optional for local endpoints)")
        engine._out("")
        engine._out("Examples:")
        engine._out("  /connect endpoint deepinfra https://api.deepinfra.com/v1/openai sk-...")
        engine._out("  /connect endpoint ollama http://localhost:11434/v1")
        # List existing endpoints
        endpoints = endpoint_provider.list_endpoints()
        if endpoints:
            engine._out("")
            engine._out("Connected endpoints:")
            for ep in endpoints:
                engine._out(f"  {ep.name:<16}{ep.base_url}")
        return

    name = parts[0]
    base_url = parts[1]
    api_key = parts[2] if len(parts) > 2 else ""

    try:
        endpoint_provider.save_endpoint(name, base_url, api_key)
        engine._out(f"Endpoint {name!r} connected ({base_url}).")
        engine._out(f"Set a model with: /model {name}/<model-id>")
        get_models(engine)
    except RbtrError as e:
        engine._warn(str(e))
