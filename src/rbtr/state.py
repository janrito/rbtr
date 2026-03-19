"""Mutable session state — shared between engine, LLM pipeline, and UI."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pygit2

from rbtr.git.objects import DiffLineRanges
from rbtr.models import DiscussionEntry, Target
from rbtr.oauth import PendingLogin
from rbtr.usage import SessionUsage

if TYPE_CHECKING:
    from github import Github

    from rbtr.github.client import GitHubCtx
    from rbtr.index.store import IndexStore
    from rbtr.providers import BuiltinProvider
    from rbtr.skills.registry import SkillRegistry


@dataclass
class EngineState:
    """Mutable state for the current rbtr session."""

    # Session persistence — ID set on init, label set on setup.
    session_started_at: float = field(default_factory=time.time)
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
    connected_providers: set[str] = field(default_factory=set)
    pending_logins: dict[BuiltinProvider, PendingLogin] = field(default_factory=dict)
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
    # Cumulative token usage and cost for the current conversation.
    usage: SessionUsage = field(default_factory=SessionUsage)
    # Cached PR discussion — fetched once per PR, cleared on new /review.
    discussion_cache: list[DiscussionEntry] | None = None
    # Discovered skills — populated at startup and on /reload.
    skill_registry: SkillRegistry | None = None
    # Cached diff line ranges for review comment validation.
    # Tuple of `((base_commit, head_commit), right_ranges, left_ranges)`.
    # Self-invalidating: callers compare the key to the current target
    # commits and rebuild when stale.  Set to `None` when no review
    # target is active.
    diff_range_cache: tuple[tuple[str, str], DiffLineRanges, DiffLineRanges] | None = None

    @property
    def gh_ctx(self) -> GitHubCtx | None:
        """Build a `GitHubCtx` from session state, or `None`."""
        if self.gh is None:
            return None
        from rbtr.github.client import GitHubCtx  # deferred: avoids PyGithub at import time

        return GitHubCtx(gh=self.gh, owner=self.owner, repo_name=self.repo_name)

    @property
    def has_llm(self) -> bool:
        """Whether any LLM provider is connected."""
        return bool(self.connected_providers)

    @property
    def repo_scope(self) -> str | None:
        """Fact scope key for the current repo, or `None`."""
        if self.owner and self.repo_name:
            return f"{self.owner}/{self.repo_name}"
        return None
