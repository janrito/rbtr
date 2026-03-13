"""Draft management tools — add, edit, remove comments and set summary."""

from __future__ import annotations

import logging

from pydantic_ai import RunContext

from rbtr.config import config
from rbtr.git.objects import (
    DiffLineRanges,
    diff_line_ranges,
    resolve_anchor,
)
from rbtr.github import client
from rbtr.github.draft import (
    draft_path,
    draft_transaction,
    find_comment,
    load_draft,
    save_draft,
    snap_to_commentable_line,
)
from rbtr.llm.deps import AgentDeps
from rbtr.llm.tools.common import limited, review_toolset
from rbtr.models import DiffSide, InlineComment, PRTarget, ReviewDraft

log = logging.getLogger(__name__)


def _pr_number(ctx: RunContext[AgentDeps]) -> int:
    """Return the PR number from the review target."""
    target = ctx.deps.state.review_target
    if not isinstance(target, PRTarget):  # pragma: no cover — guarded by prepare
        msg = "no PR target"
        raise RuntimeError(msg)
    return target.number


def _load_or_create_draft(pr_number: int) -> ReviewDraft:
    """Load the draft from disk or create an empty one."""
    return load_draft(pr_number) or ReviewDraft()


def _get_diff_ranges(
    ctx: RunContext[AgentDeps],
    *,
    side: DiffSide = DiffSide.RIGHT,
) -> DiffLineRanges:
    """Return commentable line ranges for the current review target.

    Tries GitHub's ``pull.get_files()`` patch data first (the
    authoritative source for which lines the review API accepts).
    Falls back to the local pygit2 three-dot diff when the GitHub
    context is unavailable or the API call fails.

    Results are cached on ``state.diff_range_cache`` keyed by
    ``(base_commit, head_commit)`` so repeated ``add_draft_comment``
    calls in the same turn don't re-fetch.  The cache self-invalidates
    when the key changes (e.g. after a refs refresh).
    """
    state = ctx.deps.state
    target = state.review_target
    if target is None:
        return {}
    key = (target.base_commit, target.head_commit)
    cache = state.diff_range_cache
    if cache is None or cache[0] != key:
        right, left = _fetch_diff_ranges(ctx, key)
        state.diff_range_cache = (key, right, left)
    else:
        _, right, left = cache
    return left if side is DiffSide.LEFT else right


def _fetch_diff_ranges(
    ctx: RunContext[AgentDeps],
    key: tuple[str, str],
) -> tuple[DiffLineRanges, DiffLineRanges]:
    """Fetch diff ranges, preferring GitHub data over local git.

    Returns ``(right_ranges, left_ranges)``.
    """
    # Try GitHub first — authoritative for what the review API accepts.
    gh_ctx = ctx.deps.state.gh_ctx
    target = ctx.deps.state.review_target
    if gh_ctx is not None and isinstance(target, PRTarget):
        try:
            return client.get_pr_diff_ranges(gh_ctx, target.number)
        except Exception:
            log.debug("GitHub diff ranges unavailable, falling back to local git")

    # Fall back to local git.
    repo = ctx.deps.state.repo
    if target is None or repo is None:
        return {}, {}
    base, head = key
    try:
        return diff_line_ranges(repo, base, head)
    except KeyError:
        return {}, {}


@review_toolset.tool
def add_draft_comment(
    ctx: RunContext[AgentDeps],
    path: str,
    anchor: str,
    body: str,
    suggestion: str = "",
    ref: str = "head",
) -> str:
    """Add an inline comment to the review draft.

    Args:
        path: File path relative to repo root
            (e.g. `src/api/handler.py`).  Must be a file
            that was changed in the PR diff.
        anchor: An exact substring of the file content at
            `ref`.  Must match exactly one location — the
            comment is placed on the **last line** of the match.
        body: Markdown body of the comment.
        suggestion: Optional replacement code.
        ref: Which version of the file to resolve the anchor
            against (defaults to `"head"`).
    """
    if ref not in ("head", "base"):
        return 'ref must be "head" or "base".'

    pr = _pr_number(ctx)
    state = ctx.deps.state
    target = state.review_target
    repo = state.repo

    if not isinstance(target, PRTarget):  # pragma: no cover — guarded by prepare
        return "No PR target."
    if repo is None:  # pragma: no cover — guarded by prepare
        return "No repository."

    # Determine side and resolve ref.
    if ref == "head":
        side = DiffSide.RIGHT
        resolved_ref = target.head_commit
    else:
        side = DiffSide.LEFT
        resolved_ref = target.base_commit

    # Resolve anchor → line number.
    result = resolve_anchor(repo, resolved_ref, path, anchor)
    if isinstance(result, str):
        return f"Cannot add comment: {result}"
    line = result

    # Ensure line is commentable — snap to nearest valid line if needed.
    ranges = _get_diff_ranges(ctx, side=side)
    line, error, note = snap_to_commentable_line(ranges, path, line)
    if error:
        return f"Cannot add comment: {error}"

    comment = InlineComment(
        path=path,
        line=line,
        side=side,
        commit_id=target.head_sha,
        body=body,
        suggestion=suggestion,
    )
    with draft_transaction():
        draft = _load_or_create_draft(pr)
        draft = draft.model_copy(update={"comments": [*draft.comments, comment]})
        save_draft(pr, draft)

    placed = f"Comment added ({path}:{line}). Draft has {len(draft.comments)} comment(s)."
    if note:
        placed = f"{note} {placed}"
    return placed


@review_toolset.tool
def edit_draft_comment(
    ctx: RunContext[AgentDeps],
    path: str,
    comment: str,
    body: str = "",
    suggestion: str | None = None,
) -> str:
    """Edit an existing comment in the review draft.

    Args:
        path: File path the comment belongs to
            (e.g. `src/api/handler.py`).
        comment: A substring that uniquely identifies the
            comment's body on this file.
        body: New markdown body.  Empty string keeps the
            current body.
        suggestion: New replacement code.  `None` keeps the
            current value.  Empty string clears it.
    """

    pr = _pr_number(ctx)
    with draft_transaction():
        draft = _load_or_create_draft(pr)

        if not draft.comments:
            return "Draft has no comments to edit."

        result = find_comment(draft.comments, path, comment)
        if isinstance(result, str):
            return f"Cannot edit comment: {result}"

        index, matched = result
        updates: dict[str, str] = {}

        if body:
            updates["body"] = body
        if suggestion is not None:
            updates["suggestion"] = suggestion

        updated = matched.model_copy(update=updates)
        new_comments = list(draft.comments)
        new_comments[index] = updated
        draft = draft.model_copy(update={"comments": new_comments})
        save_draft(pr, draft)

        return f"Comment updated ({updated.path}:{updated.line})."


@review_toolset.tool
def remove_draft_comment(
    ctx: RunContext[AgentDeps],
    path: str,
    comment: str,
) -> str:
    """Remove a comment from the review draft.

    Locate a comment by file path and a substring of its body,
    then remove it from the draft.

    Args:
        path: File path the comment belongs to
            (e.g. `src/api/handler.py`).
        comment: A substring that uniquely identifies the
            comment's body on this file.
    """

    pr = _pr_number(ctx)
    with draft_transaction():
        draft = _load_or_create_draft(pr)

        if not draft.comments:
            return "Draft has no comments to remove."

        result = find_comment(draft.comments, path, comment)
        if isinstance(result, str):
            return f"Cannot remove comment: {result}"

        index, removed = result

        if removed.github_id is not None:
            # Synced comment — tombstone it so the next push excludes
            # it and the remote copy is not re-imported on pull.
            tombstoned = removed.model_copy(update={"body": "", "suggestion": ""})
            new_comments = list(draft.comments)
            new_comments[index] = tombstoned
        else:
            # Never synced — safe to drop entirely.
            new_comments = [c for i, c in enumerate(draft.comments) if i != index]

        draft = draft.model_copy(update={"comments": new_comments})
        save_draft(pr, draft)

        return (
            f"Removed comment ({removed.path}:{removed.line}). "
            f"Draft has {len(new_comments)} comment(s)."
        )


@review_toolset.tool
def set_draft_summary(
    ctx: RunContext[AgentDeps],
    summary: str,
) -> str:
    """Set or replace the top-level summary of the review draft.

    This is the main body of the review that appears at the top
    when posted to GitHub — a high-level assessment of the PR.

    Args:
        summary: Markdown text for the review summary.
    """

    pr = _pr_number(ctx)
    with draft_transaction():
        draft = _load_or_create_draft(pr)
        draft = draft.model_copy(update={"summary": summary})
        save_draft(pr, draft)

    return f"Review summary updated ({len(summary)} chars)."


@review_toolset.tool
def read_draft(
    ctx: RunContext[AgentDeps],
    offset: int = 0,
    max_lines: int | None = None,
) -> str:
    """Read the raw review draft.

    Args:
        offset: Number of lines to skip (0-indexed, default 0).
        max_lines: Maximum number of lines to return
            (defaults to `tools.max_lines` config value).
    """
    pr = _pr_number(ctx)
    path = draft_path(pr)
    if not path.exists():
        return "No draft yet."

    text = path.read_text()
    lines = text.splitlines()
    total = len(lines)
    capped = (
        min(max_lines, config.tools.max_lines) if max_lines is not None else config.tools.max_lines
    )
    page = lines[offset : offset + capped]
    if not page:
        return f"Offset {offset} exceeds {total} lines."
    result = "\n".join(page)
    shown = offset + len(page)
    if shown < total:
        result += limited(shown, total, hint=f"offset={shown} to continue")
    return result
