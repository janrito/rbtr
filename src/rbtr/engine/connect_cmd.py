"""Handler for /connect — service authentication flows."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from github import Auth, Github

from rbtr.config import config
from rbtr.creds import OAuthCreds, creds
from rbtr.events import LinkOutput
from rbtr.exceptions import PortBusyError, RbtrError, TaskCancelled
from rbtr.github import auth as github_auth
from rbtr.oauth import PendingLogin
from rbtr.providers import (
    PROVIDERS,
    BuiltinProvider,
    claude as claude_provider,
    endpoint as endpoint_provider,
    google as google_provider,
    openai_codex as codex_provider,
)
from rbtr.styles import CODE_HIGHLIGHT, LINK_STYLE

from .model_cmd import get_models
from .types import Service

if TYPE_CHECKING:
    from .core import Engine


def cmd_connect(engine: Engine, args: str) -> None:
    """Dispatch /connect to the appropriate service handler."""
    parts = args.strip().split(maxsplit=1)
    name = parts[0].lower() if parts else ""
    extra = parts[1].strip() if len(parts) > 1 else ""

    if not name:
        _show_connect_help(engine)
        return

    # Builtin LLM providers.
    try:
        provider = BuiltinProvider(name)
        _connect_provider(engine, provider, extra)
        return
    except ValueError:
        pass

    # Custom endpoints.
    if name == "endpoint":
        _connect_endpoint(engine, extra)
        return

    # Non-provider services (github).
    try:
        service = Service(name)
    except ValueError:
        all_names = [p.value for p in BuiltinProvider] + ["endpoint"] + [s.key for s in Service]
        engine._warn(f"Unknown service: {name}. Supported: {', '.join(all_names)}")
        return

    match service:
        case Service.GITHUB:
            _connect_github(engine)


def _show_connect_help(engine: Engine) -> None:
    """Print /connect usage with all providers and services."""
    engine._out("Usage: /connect <service>")
    for p, prov in PROVIDERS.items():
        desc = "Connect with API key" if hasattr(prov, "CRED_FIELD") else prov.LABEL
        engine._out(f"  /connect {p.value:<16} — {desc}")
    engine._out(f"  /connect {'endpoint':<16} — OpenAI-compatible endpoint")
    for s in Service:
        engine._out(f"  /connect {s.key:<16} — {s.description}")


def _connect_provider(engine: Engine, provider: BuiltinProvider, extra: str) -> None:
    """Route a builtin provider to its connect handler."""
    match provider:
        case BuiltinProvider.CLAUDE:
            _connect_claude(engine, extra)
            return
        case BuiltinProvider.CHATGPT:
            _connect_auto_oauth(
                engine,
                provider,
                extra,
                authenticate=codex_provider.authenticate,
                begin_login=codex_provider.begin_login,
                complete_login=codex_provider.complete_login,
            )
            return
        case BuiltinProvider.GOOGLE:
            _connect_auto_oauth(
                engine,
                provider,
                extra,
                authenticate=google_provider.authenticate,
                begin_login=google_provider.begin_login,
                complete_login=google_provider.complete_login,
            )
            return

    # API-key providers use the generic handler.
    prov = PROVIDERS[provider]
    if hasattr(prov, "CRED_FIELD"):
        _connect_api_key(engine, provider, extra)
        return


# ── API-key providers ────────────────────────────────────────────────


def _connect_api_key(
    engine: Engine,
    provider: BuiltinProvider,
    api_key: str,
) -> None:
    """Generic connect handler for API-key providers."""
    prov = PROVIDERS[provider]
    cred_field: str = prov.CRED_FIELD  # type: ignore[attr-defined]  # guarded by hasattr in caller

    current_key = getattr(creds, cred_field, "")
    if current_key and not api_key:
        engine.state.connected_providers.add(provider)
        engine._out(f"Already connected to {prov.LABEL}.")
        return

    if not api_key:
        engine._out(f"Usage: /connect {provider.value} <api_key>")
        return

    key_prefix: str | None = getattr(prov, "KEY_PREFIX", None)
    if key_prefix and not api_key.startswith(key_prefix):
        engine._warn(f"Invalid API key format. {prov.LABEL} keys start with '{key_prefix}'.")
        return

    creds.update(**{cred_field: api_key})
    engine.state.connected_providers.add(provider)
    engine._out(f"Connected to {prov.LABEL}. LLM is ready.")
    get_models(engine)


# ── Auto-OAuth providers (localhost callback + manual fallback) ──────


def _connect_auto_oauth(
    engine: Engine,
    provider: BuiltinProvider,
    extra: str,
    *,
    authenticate: Callable[..., OAuthCreds],
    begin_login: Callable[[], tuple[str, PendingLogin]],
    complete_login: Callable[[str, PendingLogin], OAuthCreds],
) -> None:
    """Connect handler for OAuth providers that use a localhost callback.

    Shared by ChatGPT and Google.  Handles the three-case flow:
      1. *extra* is non-empty → phase 2 (complete pending login)
      2. Already connected → short-circuit
      3. No args → phase 1 (try auto, fall back to manual on port busy)
    """
    label = PROVIDERS[provider].LABEL

    # Phase 2: user pasted the redirect URL.
    if extra:
        pending = engine.state.pending_logins.get(provider)
        if pending is None:
            engine._warn(f"No pending {label} login. Run /connect {provider.value} first.")
            return
        try:
            oc = complete_login(extra, pending)
            creds.update(**{provider.value: oc})
            engine.state.connected_providers.add(provider)
            engine.state.pending_logins.pop(provider, None)
            engine._out(f"Connected to {label}. LLM is ready.")
            get_models(engine)
        except TaskCancelled:
            raise
        except Exception as e:
            engine._check_cancel()
            engine.state.pending_logins.pop(provider, None)
            engine._warn(f"{label} connection failed: {e}")
        return

    # Already connected.
    if PROVIDERS[provider].is_connected():
        engine.state.connected_providers.add(provider)
        engine._out(f"Already connected to {label}.")
        return

    # Phase 1: try automatic localhost callback.
    engine._out(f"Opening browser to sign in with your {label} account…")
    engine._flush()

    try:
        engine._out("Waiting for authorization…")
        oc = authenticate(cancel=engine._cancel)
        engine._check_cancel()
        engine._clear()

        creds.update(**{provider.value: oc})
        engine.state.connected_providers.add(provider)
        engine._out(f"Connected to {label}. LLM is ready.")
        get_models(engine)
    except TaskCancelled:
        raise
    except PortBusyError:
        engine._check_cancel()
        engine._clear()
        url, pending = begin_login()
        engine.state.pending_logins[provider] = pending
        engine._emit(
            LinkOutput(
                markup=(
                    f"Callback port is busy. Opening browser manually…\n"
                    f"If the browser didn't open, visit: "
                    f"[link={url}][{LINK_STYLE}]{url}[/{LINK_STYLE}][/link]"
                )
            )
        )
        engine._out("")
        engine._out("After authorizing, paste the redirect URL from your browser:")
        engine._out(f"  /connect {provider.value} <url>")
    except Exception as e:
        engine._check_cancel()
        engine._clear()
        engine._warn(f"{label} connection failed: {e}")


# ── Claude (manual-only OAuth) ───────────────────────────────────────


def _connect_claude(engine: Engine, auth_code: str) -> None:
    """Connect to Claude — manual copy-paste OAuth flow."""
    prov = BuiltinProvider.CLAUDE
    label = PROVIDERS[prov].LABEL

    # Phase 2: user pasted the code#state from the browser.
    if auth_code:
        pending = engine.state.pending_logins.get(prov)
        if pending is None:
            engine._warn(f"No pending {label} login. Run /connect {prov.value} first.")
            return
        try:
            code, state = claude_provider.parse_auth_code(auth_code)
            oc = claude_provider.complete_login(code, state, pending)
            creds.update(claude=oc)
            engine.state.connected_providers.add(prov)
            engine.state.pending_logins.pop(prov, None)
            engine._out(f"Connected to {label}. LLM is ready.")
            get_models(engine)
        except TaskCancelled:
            raise
        except Exception as e:
            engine._check_cancel()
            engine.state.pending_logins.pop(prov, None)
            engine._warn(f"{label} connection failed: {e}")
        return

    # Already connected.
    if PROVIDERS[prov].is_connected():
        engine.state.connected_providers.add(prov)
        engine._out(f"Already connected to {label}.")
        return

    # Phase 1: generate PKCE, open browser, show instructions.
    url, pending = claude_provider.begin_login()
    engine.state.pending_logins[prov] = pending

    engine._emit(
        LinkOutput(
            markup=(
                f"Opening browser to sign in with your {label} account…\n"
                f"If the browser didn't open, visit: "
                f"[link={url}][{LINK_STYLE}]{url}[/{LINK_STYLE}][/link]"
            )
        )
    )
    engine._out("")
    engine._out("After authorizing, paste the code shown in the browser:")
    engine._out(f"  /connect {prov.value} <code>")


# ── GitHub ───────────────────────────────────────────────────────────


def _connect_github(engine: Engine) -> None:
    creds.update(github_token="")
    try:
        device = github_auth.request_device_code()
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
        token = github_auth.poll_for_token(device_code, interval, cancel=engine._cancel)
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


# ── Custom endpoints ─────────────────────────────────────────────────


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
        engine.state.connected_providers.add(name)
        engine._out(f"Endpoint {name!r} connected ({base_url}).")
        engine._out(f"Set a model with: /model {name}/<model-id>")
        get_models(engine)
    except RbtrError as e:
        engine._warn(str(e))
