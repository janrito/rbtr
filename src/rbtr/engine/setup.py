"""Engine setup — repository discovery and provider initialisation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pygit2
from github import Auth, Github

from rbtr.config import config
from rbtr.creds import creds
from rbtr.exceptions import RbtrError
from rbtr.oauth import oauth_is_set
from rbtr.providers import endpoint as endpoint_provider

if TYPE_CHECKING:
    from .core import Engine

log = logging.getLogger(__name__)


def run_setup(engine: Engine) -> None:
    """Discover the repository, authenticate providers, load preferences."""
    # Deferred: open_repo calls pygit2.discover_repository which needs CWD set.
    from rbtr.git import open_repo
    from rbtr.github.client import parse_github_remote

    try:
        repo = open_repo()
        owner, repo_name = parse_github_remote(repo)
    except RbtrError as e:
        engine._error(str(e))
        return

    engine.state.repo = repo
    engine.state.owner = owner
    engine.state.repo_name = repo_name
    engine.state.session_label = _make_session_label(owner, repo_name, repo)
    engine._out(f"Repository: {owner}/{repo_name}")

    if creds.github_token:
        gh = Github(auth=Auth.Token(creds.github_token), timeout=config.github.timeout)
        engine.state.gh = gh
        engine._out("GitHub token loaded.")
    else:
        engine._out("Not authenticated. Use /connect github to authenticate.")

    if oauth_is_set(creds.claude):
        engine.state.claude_connected = True
        engine._out("Connected to Anthropic.")

    if oauth_is_set(creds.chatgpt):
        engine.state.chatgpt_connected = True
        engine._out("Connected to ChatGPT.")

    if creds.openai_api_key:
        engine.state.openai_connected = True
        engine._out("Connected to OpenAI.")

    endpoints = endpoint_provider.list_endpoints()
    for ep in endpoints:
        engine._out(f"Endpoint: {ep.name} ({ep.base_url})")

    if not (engine.state.has_llm or endpoints):
        engine._out("No LLM connected. Use /connect claude, chatgpt, or openai.")

    # Load saved model preference (context window resolves lazily
    # when the model cache is first populated).
    saved_model = config.model
    if saved_model:
        engine.state.model_name = saved_model

    engine._out("Type a message for the LLM, /help for commands, !cmd for shell")


def ensure_gh_username(engine: Engine) -> str:
    """Return the GitHub username, fetching lazily on first call.

    Creates the API call only when the username is actually needed
    (e.g. draft sync, review posting) rather than at startup.
    Returns an empty string if no GitHub connection is available.
    """
    if engine.state.gh_username:
        return engine.state.gh_username
    if engine.state.gh is None:
        return ""
    try:
        username = engine.state.gh.get_user().login
        engine.state.gh_username = username
        return username
    except Exception:
        log.warning("Failed to fetch GitHub username", exc_info=True)
        return ""


def _make_session_label(owner: str, repo_name: str, repo: pygit2.Repository) -> str:
    """Build a human-readable session label from repo context.

    Format: ``owner/repo — ref`` where *ref* is the current branch
    name or short commit hash.
    """
    ref = ""
    if not repo.head_is_unborn:
        ref = str(repo.head.target)[:8] if repo.head_is_detached else repo.head.shorthand
    return f"{owner}/{repo_name} — {ref}" if ref else f"{owner}/{repo_name}"
