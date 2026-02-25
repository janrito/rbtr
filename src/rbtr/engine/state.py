"""Mutable engine state — shared between engine and UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pygit2
from github import Github
from pydantic_ai.messages import ModelMessage

from rbtr.github.client import GitHubCtx
from rbtr.models import DiscussionEntry, Target
from rbtr.providers import claude as claude_provider, openai_codex as codex_provider
from rbtr.usage import SessionUsage

if TYPE_CHECKING:
    from rbtr.index.store import IndexStore


@dataclass
class EngineState:
    """Mutable state for the current rbtr session."""

    # Session persistence — ID set on init, label set on setup.
    session_id: str = ""
    session_label: str = ""
    repo: pygit2.Repository | None = None
    owner: str = ""
    repo_name: str = ""
    gh: Github | None = None
    gh_username: str = ""
    review_target: Target | None = None
    # Code index store — populated on /review, cleared on /new.
    index: IndexStore | None = None
    # Whether background indexing has completed.  Set True by the
    # indexing thread on IndexReady, False on new /review.
    index_ready: bool = False
    claude_connected: bool = False
    claude_pending_login: claude_provider.PendingLogin | None = None
    openai_connected: bool = False
    chatgpt_connected: bool = False
    chatgpt_pending_login: codex_provider.PendingLogin | None = None
    model_name: str | None = None
    # Whether the active model supports thinking effort settings.
    # None = unknown (no message sent yet), True/False after first LLM call.
    effort_supported: bool | None = None
    # Cached model list for Tab completion — populated by /model and
    # provider connect.  List of (provider_name, [prefixed_model_ids]).
    cached_models: list[tuple[str, list[str]]] = field(default_factory=list)
    # Timestamp of last model cache refresh (epoch seconds).
    models_fetched_at: float = 0.0
    # Cached review targets for Tab completion — populated by /review list.
    # (display_label, completion_text) pairs, e.g. ("#42 Fix bug", "42")
    # or ("feature-x", "feature-x").
    cached_review_targets: list[tuple[str, str]] = field(default_factory=list)
    # Transient cache of the conversation history — loaded from DB
    # before each Agent.iter(), set after each turn.  The DB is the
    # source of truth.  Direct reads are acceptable for compaction
    # checks and error-retry flows within a single turn.
    message_history: list[ModelMessage] = field(default_factory=list)
    # Cumulative token usage and cost for the current conversation.
    usage: SessionUsage = field(default_factory=SessionUsage)
    # Cached PR discussion — fetched once per PR, cleared on new /review.
    discussion_cache: list[DiscussionEntry] | None = None

    @property
    def gh_ctx(self) -> GitHubCtx | None:
        """Build a :class:`GitHubCtx` from session state, or ``None``."""
        if self.gh is None:
            return None
        return GitHubCtx(gh=self.gh, owner=self.owner, repo_name=self.repo_name)

    @property
    def has_llm(self) -> bool:
        """Whether any LLM provider is connected."""
        return self.claude_connected or self.chatgpt_connected or self.openai_connected
