"""Handler for /draft — view, sync, and post review drafts.

This module is a thin command dispatcher.  It parses subcommands
and arguments, validates local state (draft exists, event type
valid), and delegates GitHub interaction to `engine.publish`.
It never imports `github.client` or catches `GithubException`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rbtr_legacy.git.objects import read_blob
from rbtr_legacy.github.draft import comment_sync_status, is_tombstone, load_draft
from rbtr_legacy.llm.memory import extract_facts_from_ctx
from rbtr_legacy.models import DiffSide, InlineComment, PRTarget, ReviewEvent
from rbtr_legacy.sessions.overhead import FactExtractionSource

from .publish import clear_review_draft, post_review_draft, sync_review_draft

if TYPE_CHECKING:
    import pygit2

    from .core import Engine

log = logging.getLogger(__name__)

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


def _file_context(
    repo: pygit2.Repository,
    ref: str,
    path: str,
    line: int,
    *,
    radius: int = 2,
) -> str | None:
    """Return a few numbered lines around *line* from *path* at *ref*.

    Returns `None` when the file can't be read (missing, binary,
    or line out of range).  The *radius* controls how many lines
    above and below the target line to include.
    """
    blob = read_blob(repo, ref, path)
    if blob is None:
        return None
    data: bytes = blob.data
    # Skip binary files.
    if b"\x00" in data[:8192]:
        return None
    all_lines = data.decode(errors="replace").splitlines()
    if line < 1 or line > len(all_lines):
        return None
    start = max(line - radius, 1)
    end = min(line + radius, len(all_lines))
    width = len(str(end))
    numbered = "\n".join(f"  {n:{width}d}│ {all_lines[n - 1]}" for n in range(start, end + 1))
    return numbered


def _show_draft(engine: Engine, pr_number: int) -> None:
    """Display the current draft with rich markdown rendering.

    Layout:
    - `## N comments` heading, then each file as `### path`
      with its comments rendered as markdown.
    - `## Summary` at the bottom (most prominent piece last).
    """
    draft = load_draft(pr_number)
    if draft is None:
        engine._out("No draft for this PR. The LLM can create one during review.")
        return

    target = engine.state.review_target
    if not isinstance(target, PRTarget):
        return  # unreachable — cmd_draft guards this
    head_sha = target.head_sha
    repo = engine.state.repo

    # ── Comments ─────────────────────────────────────────────────
    if draft.comments:
        live = [c for c in draft.comments if not is_tombstone(c)]
        tombstones = len(draft.comments) - len(live)
        header = f"## {len(live)} comment{'s' if len(live) != 1 else ''}"
        if tombstones:
            header += f" ({tombstones} pending deletion)"
        engine._markdown(header)
        engine._out("")

        # Group by file path, preserving insertion order.
        by_file: dict[str, list[tuple[int, InlineComment]]] = {}
        for i, comment in enumerate(draft.comments, 1):
            by_file.setdefault(comment.path, []).append((i, comment))

        for file_idx, (path, group) in enumerate(by_file.items()):
            if file_idx > 0:
                engine._out("")
            engine._markdown(f"### {path}")
            engine._out("")

            for idx, (i, comment) in enumerate(group):
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

                # Show a few lines of file context around the comment.
                if repo and comment.line > 0:
                    ref = comment.commit_id or head_sha or target.head_commit
                    if comment.side is DiffSide.LEFT:
                        ref = target.base_commit
                    snippet = _file_context(repo, ref, comment.path, comment.line)
                    if snippet:
                        engine._out(snippet)

                engine._markdown(comment.body)

                if comment.suggestion:
                    engine._markdown(f"```suggestion\n{comment.suggestion}\n```")

                # Blank line between comments within the same file.
                if idx < len(group) - 1:
                    engine._out("")
    else:
        engine._out("No inline comments.")

    # ── Summary (last — most prominent) ──────────────────────────
    engine._out("")
    engine._markdown("## Summary")
    engine._out("")
    if draft.summary:
        engine._markdown(draft.summary)
    else:
        engine._out("(empty)")

    live = [c for c in draft.comments if not is_tombstone(c)]
    has_summary = "present" if draft.summary else "absent"
    engine._context(
        f"[/draft → {len(live)} comments]",
        f"Viewed review draft: {len(live)} comments, summary {has_summary}.",
    )


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

    # Extract facts from the full session — a completed review is
    # the richest source of project knowledge.
    ctx = engine._llm_context()
    messages = engine.store.load_messages(ctx.state.session_id)
    if messages:
        extract_facts_from_ctx(ctx, messages, source=FactExtractionSource.POST)


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
