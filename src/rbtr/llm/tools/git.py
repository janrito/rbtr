"""Git tools — changed_files, diff, commit_log."""

from __future__ import annotations

from pydantic_ai import RunContext

from rbtr.config import config
from rbtr.git import changed_files as _changed_files
from rbtr.git.objects import (
    DiffResult,
    DiffStats,
    commit_log_between,
    diff_refs,
    diff_single,
)
from rbtr.llm.deps import AgentDeps
from rbtr.llm.tools.common import get_repo, limited, repo_toolset


@repo_toolset.tool
def changed_files(
    ctx: RunContext[AgentDeps],
    offset: int = 0,
    max_results: int | None = None,
) -> str:
    """List file paths changed between the review target's base and head.

    Args:
        offset: Number of entries to skip (default 0).
        max_results: Maximum entries to return per call
            (defaults to `tools.max_results` config value).
    """
    repo = get_repo(ctx)
    target = ctx.deps.state.review_target
    if target is None:
        return "No review target selected."

    try:
        paths = _changed_files(repo, target.base_commit, target.head_commit)
    except KeyError as exc:
        return f"Could not resolve refs: {exc}"

    if not paths:
        return "No files changed between base and head."

    sorted_paths = sorted(paths)
    total = len(sorted_paths)
    limit = (
        min(max_results, config.tools.max_results)
        if max_results is not None
        else config.tools.max_results
    )
    page = sorted_paths[offset : offset + limit]
    if not page:
        return f"Offset {offset} exceeds {total} changed files."

    header = f"Changed files ({total}):"
    listing = "\n".join(f"  {p}" for p in page)
    result = f"{header}\n{listing}"
    shown = offset + len(page)
    if shown < total:
        result += limited(shown, total, hint=f"offset={shown} to continue")
    return result


@repo_toolset.tool
def diff(
    ctx: RunContext[AgentDeps],
    path: str = "",
    ref: str = "",
    offset: int = 0,
    max_lines: int | None = None,
) -> str:
    """See the line-level patch for a file or the full review target.

    Use this when you need to read exact added/removed lines — e.g.
    to verify a specific code change or quote a diff hunk.  Prefer
    `changed_symbols` when you first need to understand *what*
    changed structurally before diving into raw patches.

    Args:
        path: File path to restrict the diff to
            (e.g. `src/api/handler.py`).  Empty string
            (default) shows the full diff.
        ref: A commit SHA, branch name, or `base..head` range.
            Empty string (default) diffs the review target.
        offset: Number of output lines to skip (0-indexed,
            default 0).
        max_lines: Maximum lines of diff output to return
            (defaults to `tools.max_lines` config value).
    """
    repo = get_repo(ctx)
    target = ctx.deps.state.review_target
    capped = (
        min(max_lines, config.tools.max_lines) if max_lines is not None else config.tools.max_lines
    )

    try:
        if not ref:
            if target is None:
                return "No review target selected."
            result = diff_refs(repo, target.base_commit, target.head_commit, path=path)
        elif ".." in ref:
            parts = ref.split("..", 1)
            result = diff_refs(repo, parts[0], parts[1], path=path)
        else:
            result = diff_single(repo, ref, path=path)
    except (KeyError, ValueError) as exc:
        return exc.args[0] if exc.args else str(exc)

    return _paginate_diff(result, offset, capped)


def _format_diff_stats(s: DiffStats) -> str:
    """Format `DiffStats` into a one-line summary."""
    return f"{s.files_changed} files changed, +{s.insertions} -{s.deletions}"


def _paginate_diff(dr: DiffResult, offset: int, max_lines: int) -> str:
    """Apply pagination to a `DiffResult`."""
    total = len(dr.patch_lines)
    page = dr.patch_lines[offset : offset + max_lines]

    text = _format_diff_stats(dr.stats) + "\n\n" + "\n".join(page)
    shown_end = offset + len(page)
    if shown_end < total:
        text += limited(shown_end, total, hint=f"offset={shown_end} to continue")
    return text


@repo_toolset.tool
def commit_log(
    ctx: RunContext[AgentDeps],
    offset: int = 0,
    max_results: int | None = None,
) -> str:
    """Show the commit log between the review target's base and head.

    Args:
        offset: Number of commits to skip (default 0).
        max_results: Maximum commits to return per call
            (defaults to `tools.max_results` config value).
    """
    repo = get_repo(ctx)
    target = ctx.deps.state.review_target
    if target is None:
        return "No review target selected."

    try:
        entries = commit_log_between(repo, target.base_commit, target.head_commit)
    except KeyError as exc:
        return exc.args[0] if exc.args else str(exc)

    if not entries:
        return "No commits between base and head (branches are identical)."

    all_lines = [f"{e.sha[:8]} {e.author}: {e.message}" for e in entries]
    total = len(all_lines)
    limit = (
        min(max_results, config.tools.max_results)
        if max_results is not None
        else config.tools.max_results
    )
    page = all_lines[offset : offset + limit]
    if not page:
        return f"Offset {offset} exceeds {total} commits."
    result = "\n".join(page)
    shown = offset + len(page)
    if shown < total:
        result += limited(shown, total, hint=f"offset={shown} to continue")
    return result
