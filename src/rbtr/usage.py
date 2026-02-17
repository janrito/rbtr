"""Token usage tracking and context-window awareness.

Pure accumulator — all model metadata (context window, cost) comes from
PydanticAI's ``ModelResponse``, which already calls ``genai-prices``
internally.  This module never imports provider or pricing libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

DEFAULT_CONTEXT_WINDOW = 128_000


class ThresholdStatus(StrEnum):
    """How close the conversation is to the context limit."""

    OK = "ok"
    WARNING = "warning"  # >=70% of context used
    CRITICAL = "critical"  # >=90% of context used


class MessageCountStatus(StrEnum):
    """How long the conversation is by message count."""

    OK = "ok"
    WARNING = "warning"  # >25 messages
    CRITICAL = "critical"  # >50 messages


@dataclass
class SessionUsage:
    """Cumulative token usage for the current conversation.

    The engine calls :meth:`record_run` after each LLM interaction,
    passing token counts from ``RunUsage`` and cost/context-window
    from ``ModelResponse.cost()``.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_cost: float = 0.0
    # Last recorded input_tokens from a single run — approximates
    # the current context size (what the model actually received).
    last_input_tokens: int = 0
    # Context window size reported by the model's last response.
    # Updated every run from ModelResponse.cost().model.context_window.
    context_window: int = DEFAULT_CONTEXT_WINDOW
    # Number of user-model exchanges in this conversation.
    message_count: int = 0
    # Whether the last run had pricing data available.
    cost_available: bool = True
    # Whether the context window size is known from model metadata.
    # When False, context_window is the DEFAULT_CONTEXT_WINDOW assumption.
    context_window_known: bool = False

    def record_run(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        cost: float = 0.0,
        cost_available: bool = True,
        context_window: int | None = None,
    ) -> None:
        """Add tokens from a completed LLM run.

        *cost* and *context_window* come from ``ModelResponse.cost()``
        and default to zero / unchanged when pricing is unavailable.
        When *context_window* is ``None``, fall back to the default
        rather than carrying over a previous model's value.
        """
        self.message_count += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cache_read_tokens += cache_read_tokens
        self.cache_write_tokens += cache_write_tokens
        self.last_input_tokens = input_tokens
        self.total_cost += cost
        self.cost_available = cost_available

        if context_window is not None:
            self.context_window = context_window
            self.context_window_known = True
        else:
            self.context_window = DEFAULT_CONTEXT_WINDOW
            self.context_window_known = False

    @property
    def context_used_pct(self) -> float:
        """Percentage of context window consumed (0-100)."""
        if self.context_window == 0:
            return 0.0
        return (self.last_input_tokens / self.context_window) * 100

    @property
    def threshold_status(self) -> ThresholdStatus:
        """Check how close the conversation is to the context limit."""
        pct = self.context_used_pct
        if pct >= 90:
            return ThresholdStatus.CRITICAL
        if pct >= 70:
            return ThresholdStatus.WARNING
        return ThresholdStatus.OK

    @property
    def message_count_status(self) -> MessageCountStatus:
        """Check how long the conversation is by message count."""
        if self.message_count > 50:
            return MessageCountStatus.CRITICAL
        if self.message_count > 25:
            return MessageCountStatus.WARNING
        return MessageCountStatus.OK

    def reset(self) -> None:
        """Clear all tracked usage (e.g. on /new)."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.total_cost = 0.0
        self.last_input_tokens = 0
        self.message_count = 0
        self.context_window = DEFAULT_CONTEXT_WINDOW
        self.cost_available = True
        self.context_window_known = False


def format_tokens(count: int) -> str:
    """Format a token count for display: ``1.2k``, ``150k``, ``1.2M``."""
    if count < 1_000:
        return str(count)
    if count < 100_000:
        return f"{count / 1_000:.1f}k"
    if count < 1_000_000:
        return f"{count // 1_000}k"
    return f"{count / 1_000_000:.1f}M"


def format_cost(usd: float) -> str:
    """Format a USD cost for display."""
    if usd < 0.01:
        return f"${usd:.4f}"
    return f"${usd:.2f}"
