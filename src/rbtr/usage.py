"""Token usage tracking and context-window awareness.

Pure accumulator — all model metadata (context window, cost) comes from
PydanticAI's ``ModelResponse``, which already calls ``genai-prices``
internally.  This module never imports provider or pricing libraries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
class _LiveBase:
    """Snapshot of cumulative counters at the start of an agent run.

    ``_update_live_usage`` adds the run's incremental ``RunUsage``
    on top of these to produce accurate lifetime totals mid-run.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


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
    # Number of user turns (agent invocations) in this conversation.
    turn_count: int = 0
    # Number of LLM responses (ModelResponse / API calls).
    response_count: int = 0
    # Whether the last run had pricing data available.
    cost_available: bool = True
    # Whether the context window size is known from model metadata.
    # When False, context_window is the DEFAULT_CONTEXT_WINDOW assumption.
    context_window_known: bool = False
    # Number of compactions performed in this session.
    compaction_count: int = 0
    # Overhead — compaction and extraction costs, tracked separately
    # from conversation costs.  Included in total_cost for the footer;
    # shown independently in /stats.
    compaction_input_tokens: int = 0
    compaction_output_tokens: int = 0
    compaction_cost: float = 0.0
    extraction_input_tokens: int = 0
    extraction_output_tokens: int = 0
    extraction_cost: float = 0.0
    # Baseline snapshot taken at the start of each agent run, used by
    # _update_live_usage to compute accurate lifetime totals mid-run.
    live_base: _LiveBase = field(default_factory=_LiveBase)

    def snapshot_base(self) -> None:
        """Save current cumulative counters as the live-update baseline.

        Call this at the start of each agent run, before streaming.
        ``_update_live_usage`` then adds the run's incremental
        ``RunUsage`` on top to produce accurate lifetime totals
        for the footer while tool calls are in progress.
        """
        self.live_base = _LiveBase(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens,
        )

    def record_run(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        last_input_tokens: int | None = None,
        cost: float = 0.0,
        cost_available: bool = True,
        context_window: int | None = None,
        new_responses: int = 0,
    ) -> None:
        """Add tokens from a completed LLM run.

        *input_tokens* is the cumulative total across all requests in
        the run (PydanticAI sums them).  *last_input_tokens* is the
        input tokens from the **last** request only — the actual prompt
        size that determines context-window consumption.  When
        ``None``, falls back to *input_tokens* (assumes single-request
        run).

        *cost* and *context_window* come from ``ModelResponse.cost()``
        and default to zero / unchanged when pricing is unavailable.
        When *context_window* is ``None``, fall back to the default
        rather than carrying over a previous model's value.

        *new_responses* is the number of ``ModelResponse`` messages
        (LLM API calls) produced in this run.
        """
        self.turn_count += 1
        self.response_count += new_responses
        self.input_tokens = self.live_base.input_tokens + input_tokens
        self.output_tokens = self.live_base.output_tokens + output_tokens
        self.cache_read_tokens = self.live_base.cache_read_tokens + cache_read_tokens
        self.cache_write_tokens = self.live_base.cache_write_tokens + cache_write_tokens
        # For context-% display we need the single-request prompt size,
        # not the cumulative sum.  Fall back to cumulative when the
        # per-request value is unavailable.
        self.last_input_tokens = (
            last_input_tokens if last_input_tokens is not None else input_tokens
        )
        self.total_cost += cost
        self.cost_available = cost_available

        if context_window is not None:
            self.context_window = context_window
            self.context_window_known = True
        else:
            self.context_window = DEFAULT_CONTEXT_WINDOW
            self.context_window_known = False

        # Snapshot so the next run or live-update starts from the
        # correct baseline.
        self.snapshot_base()

    def record_compaction(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ) -> None:
        """Record tokens and cost from a compaction summary call."""
        self.compaction_input_tokens += input_tokens
        self.compaction_output_tokens += output_tokens
        self.compaction_cost += cost
        self.total_cost += cost

    def record_extraction(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ) -> None:
        """Record tokens and cost from a memory extraction call."""
        self.extraction_input_tokens += input_tokens
        self.extraction_output_tokens += output_tokens
        self.extraction_cost += cost
        self.total_cost += cost

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
        """Check how long the conversation is by turn count."""
        if self.turn_count > 50:
            return MessageCountStatus.CRITICAL
        if self.turn_count > 25:
            return MessageCountStatus.WARNING
        return MessageCountStatus.OK

    def restore(
        self,
        *,
        turn_count: int,
        response_count: int,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        compaction_input_tokens: int = 0,
        compaction_output_tokens: int = 0,
        compaction_cost: float = 0.0,
        extraction_input_tokens: int = 0,
        extraction_output_tokens: int = 0,
        extraction_cost: float = 0.0,
    ) -> None:
        """Restore counters from DB stats (e.g. on session resume)."""
        self.turn_count = turn_count
        self.response_count = response_count
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_cost = cost + compaction_cost + extraction_cost
        self.compaction_input_tokens = compaction_input_tokens
        self.compaction_output_tokens = compaction_output_tokens
        self.compaction_cost = compaction_cost
        self.extraction_input_tokens = extraction_input_tokens
        self.extraction_output_tokens = extraction_output_tokens
        self.extraction_cost = extraction_cost
        self.snapshot_base()

    def reset(self) -> None:
        """Clear all tracked usage (e.g. on /new)."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.total_cost = 0.0
        self.last_input_tokens = 0
        self.turn_count = 0
        self.response_count = 0
        self.context_window = DEFAULT_CONTEXT_WINDOW
        self.cost_available = True
        self.context_window_known = False
        self.compaction_count = 0
        self.compaction_input_tokens = 0
        self.compaction_output_tokens = 0
        self.compaction_cost = 0.0
        self.extraction_input_tokens = 0
        self.extraction_output_tokens = 0
        self.extraction_cost = 0.0
        self.live_base = _LiveBase()


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
