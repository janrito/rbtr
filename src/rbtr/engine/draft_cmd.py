"""Handler for /draft — view, sync, and post review drafts.

This module is a thin command dispatcher.  It parses subcommands
and arguments, validates local state (draft exists, event type
valid), and delegates GitHub interaction to ``engine.review``.
It never imports ``github.client`` or catches ``GithubException``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.github.draft import comment_sync_status, is_tombstone, load_draft
from rbtr.models import DiffSide, InlineComment, PRTarget, ReviewEvent

from .review import clear_review_draft, post_review_draft, sync_review_draft

if TYPE_CHECKING:
    from .core import Engine

# Subcommands exposed for tab completion.
SUBCOMMANDS: list[tuple[str, str]] = [
    ("sync", "Sync draft with GitHub (pull remote, push local)"),
    ("post", "Submit review to GitHub (/draft post [event])"),
    ("clear", "Delete local draft and remote pending review"),
]

# Event types for tab completion after "/draft post ".
POST_EVENTS: list[tuple[str, str]] = [
    ("comment", "Post as a comment (default)"),
    ("approve", "Approve the PR"),
    ("request_changes", "Request changes"),
]


def cmd_draft(engine: Engine, args: str) -> None:
    """Dispatch /draft subcommands."""
    target = engine.state.review_target
    if not isinstance(target, PRTarget):
        engine._warn("No PR selected. Use /review <number> first.")
        return

    parts = args.strip().split(maxsplit=1)
    sub = parts[0].lower() if parts and parts[0] else ""

    match sub:
        case "":
            _show_draft(engine, target.number)
        case "sync":
            _sync_draft(engine, target.number)
        case "post":
            event_arg = parts[1].strip().lower() if len(parts) > 1 else ""
            _post_draft(engine, target.number, event_arg)
        case "clear":
            clear_review_draft(engine, target.number)
        case _:
            engine._warn(f"Unknown subcommand: {sub}")
            engine._out("Usage: /draft [sync | post [event] | clear]")


def _show_draft(engine: Engine, pr_number: int) -> None:
    """Display the current draft with rich markdown rendering.

    Layout:
    - ``## N comments`` heading, then each file as ``### path``
      with its comments rendered as markdown.
    - ``## Summary`` at the bottom (most prominent piece last).
    """
    draft = load_draft(pr_number)
    if draft is None:
        engine._out("No draft for this PR. The LLM can create one during review.")
        return

    target = engine.state.review_target
    if not isinstance(target, PRTarget):
        return  # unreachable — cmd_draft guards this
    head_sha = target.head_sha

    # ── Comments ─────────────────────────────────────────────────
    if draft.comments:
        live = [c for c in draft.comments if not is_tombstone(c)]
        tombstones = len(draft.comments) - len(live)
        header = f"## {len(live)} comment{'s' if len(live) != 1 else ''}"
        if tombstones:
            header += f" ({tombstones} pending deletion)"
        engine._markdown(header)

        # Group by file path, preserving insertion order.
        by_file: dict[str, list[tuple[int, InlineComment]]] = {}
        for i, comment in enumerate(draft.comments, 1):
            by_file.setdefault(comment.path, []).append((i, comment))

        for path, group in by_file.items():
            engine._markdown(f"### {path}")

            for i, comment in group:
                status = comment_sync_status(comment)
                if is_tombstone(comment):
                    loc = f":{comment.line}" if comment.line > 0 else ""
                    engine._out(f"  {status} {i}. {loc} — (will be deleted on next sync)")
                    continue

                side_tag = " (base)" if comment.side is DiffSide.LEFT else ""
                stale_tag = ""
                if comment.commit_id and head_sha and comment.commit_id != head_sha:
                    stale_tag = f" ⚠ stale: {comment.commit_id[:7]}"
                line_ref = f":{comment.line}" if comment.line > 0 else ""
                engine._out(f"  {status} {i}. L{line_ref}{side_tag}{stale_tag}")

                engine._markdown(comment.body)

                if comment.suggestion:
                    engine._markdown(f"```suggestion\n{comment.suggestion}\n```")
    else:
        engine._out("No inline comments.")

    # ── Summary (last — most prominent) ──────────────────────────
    engine._markdown("## Summary")
    if draft.summary:
        engine._markdown(draft.summary)
    else:
        engine._out("(empty)")


def _sync_draft(engine: Engine, pr_number: int) -> None:
    """Bidirectional sync: pull remote pending comments, push local back."""
    sync_review_draft(engine, pr_number)


def _post_draft(engine: Engine, pr_number: int, event_arg: str) -> None:
    """Validate local state and delegate posting to review.py."""
    # Load local draft.
    draft = load_draft(pr_number)
    if draft is None:
        engine._warn("No draft to post. Build one during the review first.")
        return
    if not draft.summary and not draft.comments:
        engine._warn("Draft is empty — add a summary or comments before posting.")
        return

    # Resolve event type.
    event = _resolve_event(event_arg)
    if event is None:
        engine._warn(
            f"Unknown event type: {event_arg}. Use: comment (default), approve, or request_changes."
        )
        return

    post_review_draft(engine, pr_number, draft, event)


def _resolve_event(arg: str) -> ReviewEvent | None:
    """Map a user-provided event string to a ReviewEvent."""
    match arg:
        case "" | "comment":
            return ReviewEvent.COMMENT
        case "approve":
            return ReviewEvent.APPROVE
        case "request_changes" | "changes":
            return ReviewEvent.REQUEST_CHANGES
        case _:
            return None
