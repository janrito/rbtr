"""GitHub discussion tool — read PR reviews, inline comments, and general comments."""

from __future__ import annotations

from pydantic_ai import RunContext

from rbtr.config import config
from rbtr.github.client import get_pr_discussion as fetch_pr_discussion
from rbtr.llm.deps import AgentDeps
from rbtr.llm.tools.common import limited, require_pr, review_toolset
from rbtr.models import DiscussionEntry, DiscussionEntryKind, PRTarget


def format_discussion_entry(entry: DiscussionEntry) -> str:
    """Format a single discussion entry as text for LLM consumption.

    Produces a self-contained block with header (timestamp, author,
    kind-specific metadata), optional diff hunk, body, and reactions.
    """
    ts = entry.created_at.strftime("%Y-%m-%d %H:%M")
    bot_tag = " [bot]" if entry.is_bot else ""
    header_parts: list[str] = [f"[{ts}] @{entry.author}{bot_tag}"]

    match entry.kind:
        case DiscussionEntryKind.REVIEW:
            header_parts.append(f"({entry.review_state})")
        case DiscussionEntryKind.INLINE:
            location = entry.path
            if entry.line is not None:
                location += f":{entry.line}"
            header_parts.append(f"on {location}")
            if entry.in_reply_to_id is not None:
                header_parts.append("(reply)")
        case DiscussionEntryKind.COMMENT:
            pass

    header = " ".join(header_parts)
    parts: list[str] = [header]

    if entry.diff_hunk:
        parts.append(f"```\n{entry.diff_hunk}\n```")

    if entry.body:
        parts.append(entry.body)

    if entry.reactions:
        reaction_str = "  ".join(f"{emoji} {count}" for emoji, count in entry.reactions.items())
        parts.append(f"Reactions: {reaction_str}")

    return "\n".join(parts)


@review_toolset.tool(prepare=require_pr)
def get_pr_discussion(
    ctx: RunContext[AgentDeps],
    offset: int = 0,
    max_results: int | None = None,
) -> str:
    """Read the existing discussion on the current pull request.

    Fetches all reviews, inline comments, and general comments
    from GitHub, sorted chronologically (oldest first).

    Args:
        offset: Number of entries to skip (default 0).  Use to
            fetch the next page when a previous call was limited.
        max_results: Maximum entries to return per call
            (defaults to `tools.max_results` config value).
    """
    state = ctx.deps.state
    target = state.review_target
    gh_ctx = state.gh_ctx
    if not isinstance(target, PRTarget) or gh_ctx is None:
        return "No PR selected or not authenticated with GitHub."

    # Fetch and cache on state for the duration of the review.
    if state.discussion_cache is None:
        state.discussion_cache = fetch_pr_discussion(gh_ctx, target.number)
    entries = state.discussion_cache

    if not entries:
        return "No discussion on this PR yet."

    limit = (
        min(max_results, config.tools.max_results)
        if max_results is not None
        else config.tools.max_results
    )
    total = len(entries)
    page: list[DiscussionEntry] = entries[offset : offset + limit]
    if not page:
        return f"Offset {offset} exceeds {total} entries."

    lines = [format_discussion_entry(entry) for entry in page]
    result = "\n\n---\n\n".join(lines)
    shown = offset + len(page)
    if shown < total:
        result += limited(shown, total, hint=f"offset={shown} to continue")
    return result
