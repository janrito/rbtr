"""Mutable session state — shared between engine and UI."""

from __future__ import annotations

from dataclasses import dataclass, field

import pygit2
from github import Github

from rbtr.models import Target
from rbtr.providers import claude as claude_provider, openai_codex as codex_provider
from rbtr.usage import SessionUsage


@dataclass
class Session:
    """Mutable state for the current rbtr session."""

    repo: pygit2.Repository | None = None
    owner: str = ""
    repo_name: str = ""
    gh: Github | None = None
    review_target: Target | None = None
    claude_connected: bool = False
    claude_pending_login: claude_provider.PendingLogin | None = None
    openai_connected: bool = False
    chatgpt_connected: bool = False
    chatgpt_pending_login: codex_provider.PendingLogin | None = None
    model_name: str | None = None
    # Cached model list for Tab completion — populated by /model and
    # provider connect.  List of (provider_name, [prefixed_model_ids]).
    cached_models: list[tuple[str, list[str]]] = field(default_factory=list)
    # Timestamp of last model cache refresh (epoch seconds).
    models_fetched_at: float = 0.0
    # Conversation history — independent of the model so switching
    # models preserves context. Passed to Agent.iter(message_history=…).
    message_history: list[object] = field(default_factory=list)
    # Cumulative token usage and cost for the current conversation.
    usage: SessionUsage = field(default_factory=SessionUsage)

    @property
    def has_llm(self) -> bool:
        """Whether any LLM provider is connected."""
        return self.claude_connected or self.chatgpt_connected or self.openai_connected
