"""Footer rendering — pure functions, no UI state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Group
from rich.text import Text

from rbtr.config import ThinkingEffort, config
from rbtr.models import BranchTarget, PRTarget, SnapshotTarget

if TYPE_CHECKING:
    from rbtr.state import EngineState
from rbtr.styles import (
    ERROR,
    FOOTER,
    USAGE_CRITICAL,
    USAGE_MESSAGES,
    USAGE_OK,
    USAGE_UNCERTAIN,
    USAGE_WARNING,
)
from rbtr.usage import (
    MessageCountStatus,
    ThresholdStatus,
    format_cost,
    format_tokens,
)


def render_footer(
    state: EngineState,
    width: int,
    *,
    index_ready: bool,
    index_chunks: int,
    index_phase: str,
    index_indexed: int,
    index_total: int,
    spinner_frame: str,
) -> Group:
    """Build the 1- or 2-line footer bar."""
    # ── Left side ────────────────────────────────────────────────
    repo = f" {state.owner}/{state.repo_name}" if state.owner else " rbtr"

    target = state.review_target
    match target:
        case PRTarget(number=n, base_branch=base, head_branch=head):
            review = f" PR #{n} · {base} → {head}"
        case BranchTarget(base_branch=base, head_branch=head):
            review = f" {base} → {head}"
        case SnapshotTarget(ref_label=label):
            review = f" {label}"
        case None:
            review = ""
    if not state.gh and not review:
        review = " ✗ not authenticated"

    # Index status — appended to the review target on line 2.
    if index_ready:
        index_status = f" · ● {_format_count(index_chunks)}"
    elif index_total > 0:
        label = index_phase or "indexing"
        index_status = f" · {spinner_frame} {label} {index_indexed}/{index_total}"
    else:
        index_status = ""
    if review and index_status:
        review += index_status

    # ── Right side ───────────────────────────────────────────────
    model = state.model_name or ""
    usage = state.usage
    has_usage = usage.input_tokens > 0 or usage.output_tokens > 0

    # Line 1: repo left, model + thinking effort right.
    effort = config.thinking_effort
    supported = state.effort_supported
    if model and effort is not ThinkingEffort.NONE:
        if supported is False:
            effort_label = "off"
            effort_style: str | None = ERROR
        else:
            effort_label = effort
            effort_style = None
        line1 = Text(style=FOOTER)
        line1.append(repo)
        model_right = f"{model} ∴ {effort_label} "
        pad = width - len(repo) - len(model_right)
        line1.append(" " * max(pad, 2))
        line1.append(f"{model} ∴ ")
        line1.append(effort_label, style=effort_style)
        line1.append(" ")
    elif model:
        line1 = _footer_line(repo, f"{model} ", width)
    else:
        line1 = _footer_line(repo, "", width)

    # Single-line footer when there's nothing for line 2.
    if not review and not has_usage:
        return Group(line1)

    # Line 2: review target left, usage stats right.
    ctx = ""
    msgs = ""
    token_parts: list[str] = []
    if has_usage:
        msgs = f"|{usage.turn_count}:{usage.response_count}|"
        ctx_pct = f"{usage.context_used_pct:.0f}%"
        ctx_size = format_tokens(usage.context_window)
        ctx = f"{ctx_pct} of {ctx_size}"
        token_parts.append(f"↑ {format_tokens(usage.input_tokens)}")
        token_parts.append(f"↓ {format_tokens(usage.output_tokens)}")
        if usage.cache_read_tokens:
            token_parts.append(f"↯ {format_tokens(usage.cache_read_tokens)}")
        token_parts.append(format_cost(usage.total_cost))

    right2 = ("  ".join([msgs, ctx, *token_parts]) + " ") if has_usage else ""
    left2 = review or " "

    line2 = Text(style=FOOTER)
    line2.append(left2)
    pad = width - len(left2) - len(right2)
    line2.append(" " * max(pad, 2))

    if has_usage:
        match usage.message_count_status:
            case MessageCountStatus.OK:
                msgs_style = USAGE_MESSAGES
            case MessageCountStatus.WARNING:
                msgs_style = USAGE_WARNING
            case MessageCountStatus.CRITICAL:
                msgs_style = USAGE_CRITICAL
        line2.append(msgs, style=msgs_style)
        line2.append("  ")

        match usage.threshold_status:
            case ThresholdStatus.OK:
                pct_style = USAGE_OK
            case ThresholdStatus.WARNING:
                pct_style = USAGE_WARNING
            case ThresholdStatus.CRITICAL:
                pct_style = USAGE_CRITICAL
        line2.append(ctx_pct, style=pct_style)
        line2.append(" of ")
        total_style = USAGE_UNCERTAIN if not usage.context_window_known else None
        line2.append(format_tokens(usage.context_window), style=total_style)
        if token_parts:
            rest, cost = token_parts[:-1], token_parts[-1]
            if rest:
                line2.append("  " + "  ".join(rest))
            line2.append("  ")
            cost_style = USAGE_UNCERTAIN if not usage.cost_available else None
            line2.append(cost, style=cost_style)
        line2.append(" ")
    return Group(line1, line2)


def _footer_line(left: str, right: str, width: int) -> Text:
    """Build a single footer line with left/right alignment."""
    t = Text(style=FOOTER)
    t.append(left)
    pad = width - len(left) - len(right)
    t.append(" " * max(pad, 2))
    if right:
        t.append(right)
    return t


def _format_count(n: int) -> str:
    """Format a count with k/M suffix for the footer."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)
