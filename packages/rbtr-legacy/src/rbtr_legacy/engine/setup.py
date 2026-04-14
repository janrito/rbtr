"""Engine setup — repository discovery and provider initialisation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pygit2
from github import Auth, Github

from rbtr_legacy.config import config
from rbtr_legacy.creds import creds
from rbtr_legacy.exceptions import RbtrError
from rbtr_legacy.providers import PROVIDERS, endpoint as endpoint_provider
from rbtr_legacy.skills import load_skills

if TYPE_CHECKING:
    from .core import Engine

log = logging.getLogger(__name__)

_BANNER = r"""
        ██        ██
       ░██       ░██
 ██████░██      ██████ ██████
░░██░░█░██████ ░░░██░ ░░██░░█
 ░██ ░ ░██░░░██  ░██   ░██ ░
 ░██   ░██  ░██  ░██   ░██
░███   ░██████   ░░██ ░███
░░░    ░░░░░      ░░  ░░░

"""


def run_setup(engine: Engine) -> None:
    """Discover the repository, authenticate providers, load preferences."""
    # Deferred: open_repo calls pygit2.discover_repository which needs CWD set.
    from rbtr_legacy.git import open_repo
    from rbtr_legacy.github.client import parse_github_remote

    try:
        repo = open_repo()
        owner, repo_name = parse_github_remote(repo)
    except RbtrError as e:
        engine._error(str(e))
        return

    engine.state.repo = repo
    engine.state.owner = owner
    engine.state.repo_name = repo_name

    for line in _BANNER.splitlines():
        engine._out(line)

    engine._out(f"Repository: {owner}/{repo_name}")

    if creds.github_token:
        gh = Github(auth=Auth.Token(creds.github_token), timeout=config.github.timeout)
        engine.state.gh = gh
        engine._out("GitHub token loaded.")
    else:
        engine._out("Not authenticated. Use /connect github to authenticate.")

    for provider, prov in PROVIDERS.items():
        if prov.is_connected():
            engine.state.connected_providers.add(provider)
            engine._out(f"Connected to {prov.LABEL}.")

    for ep in endpoint_provider.list_endpoints():
        engine.state.connected_providers.add(ep.name)
        engine._out(f"Endpoint: {ep.name} ({ep.base_url})")

    if not engine.state.has_llm:
        engine._out("No LLM connected. Use /connect to add a provider.")

    # Load saved model preference (context window resolves lazily
    # when the model cache is first populated).
    saved_model = config.model
    if saved_model:
        engine.state.model_name = saved_model

    # Discover skills from all default and configured directories.
    repo = engine.state.repo
    project_root = repo.workdir.rstrip("/") if repo and repo.workdir else None
    registry = load_skills(config.skills, project_root=project_root)
    engine.state.skill_registry = registry
    if registry:
        engine._out(f"Skills: {len(registry)} discovered")

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

    Format: `owner/repo — ref` where *ref* is the current branch
    name or short commit hash.
    """
    ref = ""
    if not repo.head_is_unborn:
        ref = str(repo.head.target)[:8] if repo.head_is_detached else repo.head.shorthand
    return f"{owner}/{repo_name} — {ref}" if ref else f"{owner}/{repo_name}"
